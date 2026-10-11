"""PPO with parallel CPU physics workers and a GPU learner.

Each worker process owns several LunarRescueEnv instances and a recorder. Every
iteration the learner broadcasts the current policy (weights, observation
normaliser, version number); workers step their environments with it for
``horizon`` control steps, record every transition, and return the batch. The
learner computes values and advantages, updates the actor-critic, and repeats.

Recorded episodes carry the policy version for every action, so an attempt that
spans a policy update is labelled with its full lineage.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TrainConfig:
    run_id: str = "run"
    out_root: str = "/runs"
    workers: int = 30
    envs_per_worker: int = 6
    horizon: int = 64
    total_steps: int = 300_000_000
    hours: float = 6.0
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 8
    entropy: float = 0.002
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    init_std: float = 0.3
    hidden: tuple = (512, 256, 128)
    checkpoint_every: int = 50
    curriculum: bool = True
    lock_telescoping: bool = False     # comparison policy: segments held at nominal length
    seed: int = 0
    curriculum_step: float = 0.05
    curriculum_threshold: float = 0.55
    log_every: int = 5
    resume: bool = True            # continue from the latest checkpoint of this run (crash / preemption)
    worker_timeout_s: float = 600.0
    level_cap_on_resume: float = -1.0   # >= 0: lower restored curriculum levels to at most this value


# ---------------------------------------------------------------------- model
def build_model(actor_dim: int, critic_dim: int, act_dim: int, hidden, init_std: float):
    import torch
    import torch.nn as nn

    def mlp(i, o):
        layers, d = [], i
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers.append(nn.Linear(d, o))
        return nn.Sequential(*layers)

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = mlp(actor_dim, act_dim)
            self.critic = mlp(critic_dim, 1)
            self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))
            with torch.no_grad():
                self.actor[-1].weight.mul_(0.01)
                self.actor[-1].bias.zero_()

        def dist(self, obs):
            mu = self.actor(obs)
            return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    return ActorCritic()


class RunningNorm:
    def __init__(self, dim: int):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)

    def state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}


# ---------------------------------------------------------------------- worker
def _worker(idx: int, cfg: TrainConfig, conn, level_shared) -> None:
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import torch
        torch.set_num_threads(1)
        from ..env.lunar import OBSTACLES, LunarRescueEnv
        from ..record.recorder import WorkerRecorder

        level = {k: float(level_shared[i]) for i, k in enumerate(OBSTACLES)}
        envs = [LunarRescueEnv(seed=cfg.seed * 1000 + idx * 37 + j, pool_size=2, level=level)
                for j in range(cfg.envs_per_worker)]
        rec = WorkerRecorder(cfg.out_root, cfg.run_id, idx)
        obs = [e.reset() for e in envs]
        for j, e in enumerate(envs):
            rec.start(j, e.episode_meta(), 0, e.slot.data.qpos)
        act_dim = envs[0].slot.model.nu
        slides = envs[0].slot.amap.is_slide
        conn.send(("ready", len(obs[0]["actor"]), len(obs[0]["critic"]), act_dim))
        model = None
        episodes_since_refresh = 0
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                rec.close(msg[1])
                conn.send(("stopped", idx))
                return
            _, state, norm_a, version, global_step, new_level = msg
            if model is None:
                model = build_model(len(obs[0]["actor"]), len(obs[0]["critic"]), act_dim, cfg.hidden, cfg.init_std)
            model.load_state_dict(state)
            if new_level is not None and episodes_since_refresh > 40 and any(
                    abs(new_level[k] - min(sl.level.get(k, 0.0) for sl in envs[0].pool)) >= 0.1 for k in new_level):
                for e in envs:
                    try:
                        e.refresh(new_level, n=1)
                    except Exception:  # noqa: BLE001  (keep the existing course if a new one fails)
                        pass
                episodes_since_refresh = 0
            H, N = cfg.horizon, len(envs)
            buf_a = np.zeros((H, N, len(obs[0]["actor"])), np.float32)
            buf_c = np.zeros((H, N, len(obs[0]["critic"])), np.float32)
            buf_act = np.zeros((H, N, act_dim), np.float32)
            buf_logp = np.zeros((H, N), np.float32)
            buf_r = np.zeros((H, N), np.float32)
            buf_done = np.zeros((H, N), np.float32)
            buf_trunc = np.zeros((H, N), np.float32)
            trunc_obs = np.zeros((H, N, len(obs[0]["critic"])), np.float32)
            finished = []
            for t in range(H):
                a_raw = np.stack([o["actor"] for o in obs])
                c_raw = np.stack([o["critic"] for o in obs])
                a_raw = np.nan_to_num(a_raw, nan=0.0, posinf=0.0, neginf=0.0)
                c_raw = np.nan_to_num(c_raw, nan=0.0, posinf=0.0, neginf=0.0)
                buf_a[t], buf_c[t] = a_raw, c_raw
                with torch.no_grad():
                    dist = model.dist(torch.from_numpy(norm_a(a_raw)))
                    act = dist.sample()
                    logp = dist.log_prob(act).sum(-1)
                act_np = act.numpy()
                buf_act[t], buf_logp[t] = act_np, logp.numpy()
                for j, e in enumerate(envs):
                    a = act_np[j].copy()
                    if cfg.lock_telescoping:
                        a[slides] = 0.0
                    try:
                        o, r, term, trunc, info = e.step(a)
                    except Exception as exc:  # noqa: BLE001  (physics error: record it, reset this env only)
                        o, r, term, trunc = obs[j], 0.0, True, False
                        info = {"terms": {}, "reason": "physics_error", "x": float(e.slot.data.qpos[0]),
                                "passed": dict(getattr(e, "passed", {})), "max_penetration": 0.0,
                                "error": repr(exc)[:200]}
                        if not np.all(np.isfinite(o["actor"])):
                            o = {"actor": np.zeros_like(obs[j]["actor"]), "critic": np.zeros_like(obs[j]["critic"])}
                    rec.step(j, np.clip(a, -1, 1), r, info["terms"], version, info["x"], e.slot.data.qpos)
                    buf_r[t, j] = r
                    if term or trunc:
                        info["max_penetration_episode"] = e.max_pen
                        status = "complete" if term else "truncated"
                        row = rec.finish(j, status, info["reason"], global_step + t * N + j, info)
                        finished.append({k: row[k] for k in ("return", "steps", "reason", "max_x", "success",
                                                              "passed", "spawn_index", "spawn_x", "level", "episode_id")})
                        buf_done[t, j] = 1.0
                        if trunc:
                            buf_trunc[t, j] = 1.0
                            trunc_obs[t, j] = o["critic"]
                        o = e.reset()
                        rec.start(j, e.episode_meta(), global_step + t * N + j, e.slot.data.qpos)
                        episodes_since_refresh += 1
                    obs[j] = o
            last_c = np.stack([o["critic"] for o in obs])
            conn.send(("batch", idx, buf_a, buf_c, buf_act, buf_logp, buf_r, buf_done, buf_trunc, trunc_obs,
                       last_c, finished))
    except Exception:  # noqa: BLE001
        conn.send(("error", idx, traceback.format_exc()))


# ---------------------------------------------------------------------- learner
def train(cfg: TrainConfig, on_log=None) -> dict:
    import torch

    from ..env.lunar import OBSTACLES

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.out_root) / cfg.run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    ctx = mp.get_context("spawn")
    level = {k: (0.0 if cfg.curriculum else 1.0) for k in OBSTACLES}
    level_shared = ctx.Array("d", [level[k] for k in OBSTACLES])
    pipes, procs = [], []
    for i in range(cfg.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(i, cfg, b, level_shared), daemon=True)
        p.start()
        pipes.append(a)
        procs.append(p)
    dims = None
    for c in pipes:
        msg = c.recv()
        if msg[0] == "error":
            raise RuntimeError(f"worker {msg[1]} failed:\n{msg[2]}")
        dims = msg[1:]
    actor_dim, critic_dim, act_dim = dims
    model = build_model(actor_dim, critic_dim, act_dim, cfg.hidden, cfg.init_std).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    norm_a, norm_c = RunningNorm(actor_dim), RunningNorm(critic_dim)
    log_path = run_dir / "train_log.jsonl"
    started = run_dir / "started_at.txt"
    if not started.exists():
        started.write_text(str(time.time()))
    t_start = float(started.read_text())          # the deadline counts time before any restart
    global_step, version = 0, 0
    lr = cfg.lr
    ckpts = sorted((run_dir / "checkpoints").glob("policy_v*.pt"))
    if cfg.resume and ckpts:
        ck = torch.load(ckpts[-1], map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        for n, key in ((norm_a, "norm_actor"), (norm_c, "norm_critic")):
            n.mean, n.var, n.count = ck[key]["mean"], ck[key]["var"], ck[key]["count"]
        version, global_step = int(ck["version"]), int(ck["global_step"])
        lr = float(ck.get("lr", cfg.lr))
        for i, k in enumerate(OBSTACLES):
            level[k] = float(ck["level"].get(k, level[k]))
            if cfg.level_cap_on_resume >= 0:
                level[k] = min(level[k], cfg.level_cap_on_resume)
            level_shared[i] = level[k]
        level_changed = dict(level)
        with open(log_path, "a") as f:
            f.write(json.dumps({"event": "resumed", "from": ckpts[-1].name, "version": version,
                                "global_step": global_step, "time": time.time()}) + "\n")
    recent: list[dict] = []
    stats_hist = []
    level_changed = locals().get("level_changed")
    skipped_updates = 0
    try:
        while global_step < cfg.total_steps and (time.time() - t_start) < cfg.hours * 3600:
            it_t = time.time()
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            for c in pipes:
                c.send(("run", state, _frozen(norm_a), version, global_step, level_changed))
            level_changed = None
            batches = []
            for c in pipes:
                if not c.poll(cfg.worker_timeout_s):
                    raise RuntimeError("a worker stopped responding (resume will restart from the last checkpoint)")
                msg = c.recv()
                if msg[0] == "error":
                    raise RuntimeError(f"worker {msg[1]} failed:\n{msg[2]}")
                batches.append(msg)
            roll_t = time.time() - it_t
            A = np.concatenate([b[2] for b in batches], axis=1)
            C = np.concatenate([b[3] for b in batches], axis=1)
            ACT = np.concatenate([b[4] for b in batches], axis=1)
            LOGP = np.concatenate([b[5] for b in batches], axis=1)
            R = np.concatenate([b[6] for b in batches], axis=1)
            D = np.concatenate([b[7] for b in batches], axis=1)
            TR = np.concatenate([b[8] for b in batches], axis=1)
            TOBS = np.concatenate([b[9] for b in batches], axis=1)
            LAST = np.concatenate([b[10] for b in batches], axis=0)
            for b in batches:
                recent += b[11]
            H, N = R.shape
            norm_a.update(A.reshape(-1, actor_dim))
            norm_c.update(C.reshape(-1, critic_dim))
            with torch.no_grad():
                v = model.critic(torch.from_numpy(norm_c(C.reshape(-1, critic_dim))).to(dev)).cpu().numpy().reshape(H, N)
                v_last = model.critic(torch.from_numpy(norm_c(LAST)).to(dev)).cpu().numpy().reshape(N)
                v_trunc = model.critic(torch.from_numpy(norm_c(TOBS.reshape(-1, critic_dim))).to(dev)).cpu().numpy().reshape(H, N)
            # truncated episodes bootstrap from the value of their final state
            R_adj = R + cfg.gamma * v_trunc * TR
            adv = np.zeros_like(R)
            gae = np.zeros(N, np.float32)
            for t in reversed(range(H)):
                nv = v_last if t == H - 1 else v[t + 1]
                nonterm = 1.0 - D[t]
                delta = R_adj[t] + cfg.gamma * nv * nonterm - v[t]
                gae = delta + cfg.gamma * cfg.lam * nonterm * gae
                adv[t] = gae
            ret = adv + v
            obs_a = torch.from_numpy(norm_a(A.reshape(-1, actor_dim))).to(dev)
            obs_c = torch.from_numpy(norm_c(C.reshape(-1, critic_dim))).to(dev)
            act_t = torch.from_numpy(ACT.reshape(-1, act_dim)).to(dev)
            logp_old = torch.from_numpy(LOGP.reshape(-1)).to(dev)
            adv_t = torch.from_numpy(adv.reshape(-1)).to(dev)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            ret_t = torch.from_numpy(ret.reshape(-1)).to(dev)
            n = len(ret_t)
            mb = n // cfg.minibatches
            kl_v, pl_v, vl_v, ent_v = 0.0, 0.0, 0.0, 0.0
            for _ in range(cfg.epochs):
                perm = torch.randperm(n, device=dev)
                for k in range(cfg.minibatches):
                    idx = perm[k * mb:(k + 1) * mb]
                    dist = model.dist(obs_a[idx])
                    logp = dist.log_prob(act_t[idx]).sum(-1)
                    ratio = (logp - logp_old[idx]).exp()
                    pg = -torch.min(ratio * adv_t[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv_t[idx]).mean()
                    vl = (model.critic(obs_c[idx]).squeeze(-1) - ret_t[idx]).pow(2).mean()
                    ent = dist.entropy().sum(-1).mean()
                    loss = pg + cfg.value_coef * vl - cfg.entropy * ent
                    opt.zero_grad()
                    if not torch.isfinite(loss):
                        skipped_updates += 1          # never apply a NaN/inf update to the policy
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        kl = (logp_old[idx] - logp).mean().item()
                    kl_v, pl_v, vl_v, ent_v = kl, pg.item(), vl.item(), ent.item()
            # adaptive learning rate from the KL of the last minibatch
            if kl_v > 2 * cfg.target_kl:
                lr = max(lr / 1.5, 1e-5)
            elif kl_v < cfg.target_kl / 2:
                lr = min(lr * 1.5, 1e-3)
            for g in opt.param_groups:
                g["lr"] = lr
            version += 1
            global_step += H * N
            # curriculum: raise an obstacle's difficulty once most attempts that reach it get past it
            if cfg.curriculum and len(recent) >= 200:
                changed = False
                for i, k in enumerate(OBSTACLES):
                    reached = [e for e in recent if _reached(e, k)]
                    if len(reached) >= 40:
                        rate = np.mean([e["passed"].get(k, False) for e in reached])
                        if rate > cfg.curriculum_threshold and level[k] < 1.0:
                            level[k] = min(1.0, level[k] + cfg.curriculum_step)
                            level_shared[i] = level[k]
                            changed = True
                if changed:
                    level_changed = dict(level)
                    recent = recent[-50:]          # judge the new level on new attempts only
            if version % cfg.log_every == 0 or version == 1:
                last = recent[-400:]
                row = {
                    "version": version, "global_step": global_step, "elapsed_s": round(time.time() - t_start, 1),
                    "steps_per_s": round(H * N / (time.time() - it_t), 1), "rollout_s": round(roll_t, 2),
                    "episodes_total": len(recent), "lr": lr, "kl": kl_v, "policy_loss": pl_v, "value_loss": vl_v,
                    "entropy": ent_v, "action_std": float(model.log_std.exp().mean().item()),
                    "mean_return": float(np.mean([e["return"] for e in last])) if last else None,
                    "mean_max_x": float(np.mean([e["max_x"] for e in last])) if last else None,
                    "mean_progress_m": float(np.mean([e["max_x"] - e["spawn_x"] for e in last])) if last else None,
                    "best_progress_m": float(np.max([e["max_x"] - e["spawn_x"] for e in last])) if last else None,
                    "pass_rates": {k: float(np.mean([e["passed"].get(k, False) for e in last if _reached(e, k)] or [0.0]))
                                   for k in OBSTACLES},
                    "success_rate": float(np.mean([e["success"] for e in last])) if last else None,
                    "reasons": _count([e["reason"] for e in last]), "level": dict(level),
                    "skipped_updates": skipped_updates,
                }
                stats_hist.append(row)
                with open(log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                if on_log:
                    on_log(row)
            if version % cfg.checkpoint_every == 0:
                _save_ckpt(run_dir, model, norm_a, norm_c, version, global_step, level, cfg, opt, lr)
            if len(recent) > 5000:
                recent = recent[-3000:]
    finally:
        _save_ckpt(run_dir, model, norm_a, norm_c, version, global_step, level, cfg, opt, lr)
        for c in pipes:
            try:
                c.send(("stop", global_step))
            except Exception:  # noqa: BLE001
                pass
        for c in pipes:
            try:
                if c.poll(120):
                    c.recv()
            except Exception:  # noqa: BLE001
                pass
        for p in procs:
            p.join(timeout=30)
    return {"version": version, "global_step": global_step, "log": stats_hist[-3:]}


def _reached(e: dict, k: str) -> bool:
    """An attempt reached obstacle ``k`` if it spawned in front of it or passed the one before."""
    order = ["debris", "hab", "fissure", "scarp"]
    first = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}[e["spawn_index"]]
    i = order.index(k)
    return i == first or (i > first and e["passed"].get(order[i - 1], False))


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


class _Frozen:
    def __init__(self, n: RunningNorm):
        self.mean, self.std = n.mean.copy(), np.sqrt(n.var + 1e-8)

    def __call__(self, x):
        return np.clip((x - self.mean) / self.std, -10, 10).astype(np.float32)


def _frozen(n: RunningNorm) -> _Frozen:
    return _Frozen(n)


def _save_ckpt(run_dir: Path, model, norm_a, norm_c, version, global_step, level, cfg, opt=None, lr=None) -> None:
    import torch
    path = run_dir / "checkpoints" / f"policy_v{version:06d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "norm_actor": norm_a.state(), "norm_critic": norm_c.state(),
                "version": version, "global_step": global_step, "level": level, "config": asdict(cfg),
                "optimizer": opt.state_dict() if opt is not None else None, "lr": lr}, tmp)
    os.replace(tmp, path)
