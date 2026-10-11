"""PPO for the learned layer above the frozen MoCapAct prior.

Same structure as ``ascent.train.ppo`` (parallel CPU workers, every episode recorded with its
policy lineage, resume from the latest checkpoint), adapted to ``ReshapeEnv``:

* each worker batches the frozen prior across its environments,
* the curriculum raises a level when most attempts that reach an obstacle get past it,
* the teacher weight ``teach`` falls linearly to zero over ``teach_steps`` training steps.
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
    env: dict = field(default_factory=dict)          # EnvCfg overrides
    workers: int = 30
    envs_per_worker: int = 6
    horizon: int = 64
    total_steps: int = 500_000_000
    hours: float = 3.0
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 8
    entropy: float = 0.0
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    std_command: float = 0.3         # speed, heading
    std_tele: float = 0.3
    std_dz: float = 0.2
    std_dq: float = 0.2
    hidden: tuple = (512, 256, 128)
    checkpoint_every: int = 50
    curriculum: bool = True
    start_level: float = 0.0
    curriculum_step: float = 0.2
    curriculum_steps: dict = field(default_factory=lambda: {"scarp": 0.05})   # finer steps for named obstacles
    curriculum_floor: float = 0.2     # an obstacle that fewer attempts than this get past is eased again
    curriculum_min_attempts: int = 40  # attempts at an obstacle needed before its level may change
    curriculum_threshold: float = 0.6
    teach_steps: int = 8_000_000      # the teacher's imitation weight reaches zero here
    teach_floor: float = 0.0
    seed: int = 0
    log_every: int = 5
    resume: bool = True
    init_from: str = ""              # checkpoint path to start from when the run has none of its own
    worker_timeout_s: float = 900.0
    learner_threads: int = 8


# ---------------------------------------------------------------------- model
def build_model(actor_dim: int, critic_dim: int, act_dim: int, hidden, init_std):
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
            self.log_std = nn.Parameter(torch.log(torch.as_tensor(np.asarray(init_std, np.float32))))
            with torch.no_grad():
                self.actor[-1].weight.mul_(0.01)
                self.actor[-1].bias.zero_()

        def dist(self, obs):
            mu = self.actor(obs)
            return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    return ActorCritic()


def init_std_vector(cfg: TrainConfig, act_dim: int, n_tele: int = 5, n_z: int = 60) -> np.ndarray:
    std = np.full(act_dim, cfg.std_dq, np.float32)
    std[:2] = cfg.std_command
    std[2:2 + n_tele] = cfg.std_tele
    std[2 + n_tele:2 + n_tele + n_z] = cfg.std_dz
    return std


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


class _Frozen:
    def __init__(self, n: RunningNorm):
        self.mean, self.std = n.mean.copy(), np.sqrt(n.var + 1e-8)

    def __call__(self, x):
        return np.clip((x - self.mean) / self.std, -10, 10).astype(np.float32)


def make_env_cfg(overrides: dict):
    from .env import EnvCfg
    cfg = EnvCfg()
    for k, v in (overrides or {}).items():
        setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
    return cfg


# ---------------------------------------------------------------------- worker
def _worker(idx: int, cfg: TrainConfig, conn, level0: dict) -> None:
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import torch
        torch.set_num_threads(1)
        from ..record.recorder import WorkerRecorder
        from .env import CONTROL_DT, TERM_KEYS, ReshapeEnv, frozen_walk

        ecfg = make_env_cfg(cfg.env)
        frozen = frozen_walk()
        envs = [ReshapeEnv(ecfg, seed=cfg.seed * 100003 + idx * 37 + j, level=level0)
                for j in range(cfg.envs_per_worker)]
        rec = WorkerRecorder(cfg.out_root, cfg.run_id, idx, playback_every=ecfg.record_every, term_keys=TERM_KEYS,
                             control_hz=1.0 / CONTROL_DT)
        obs = [e.reset() for e in envs]
        for j, e in enumerate(envs):
            rec.start(j, e.episode_meta(), 0, e.qpos)
        act_dim = envs[0].act_dim
        n_a, n_c = len(obs[0]["actor"]), len(obs[0]["critic"])
        lo = [np.full(5, np.inf) for _ in envs]
        hi = [np.full(5, -np.inf) for _ in envs]
        bumps = [0.0 for _ in envs]
        conn.send(("ready", n_a, n_c, act_dim))
        model, level = None, dict(level0)
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                rec.close(msg[1])
                conn.send(("stopped", idx))
                return
            _, state, norm_a, version, global_step, new_level, teach = msg
            if model is None:
                model = build_model(n_a, n_c, act_dim, cfg.hidden, init_std_vector(cfg, act_dim))
            model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
            if new_level is not None:
                level = dict(new_level)
            for e in envs:
                e.teach = teach
                if ecfg.mode == "flat":
                    e.level.update(level)
            H, N = cfg.horizon, len(envs)
            buf_a = np.zeros((H, N, n_a), np.float32)
            buf_c = np.zeros((H, N, n_c), np.float32)
            buf_act = np.zeros((H, N, act_dim), np.float32)
            buf_logp = np.zeros((H, N), np.float32)
            buf_r = np.zeros((H, N), np.float32)
            buf_done = np.zeros((H, N), np.float32)
            buf_trunc = np.zeros((H, N), np.float32)
            trunc_obs = np.zeros((H, N, n_c), np.float32)
            finished = []
            for t in range(H):
                a_raw = np.stack([o["actor"] for o in obs])
                c_raw = np.stack([o["critic"] for o in obs])
                buf_a[t], buf_c[t] = a_raw, c_raw
                with torch.no_grad():
                    dist = model.dist(torch.from_numpy(norm_a(a_raw)))
                    act = dist.sample()
                    logp = dist.log_prob(act).sum(-1)
                act_np = act.numpy()
                buf_act[t], buf_logp[t] = act_np, logp.numpy()
                walk = [e.walk_obs(act_np[j]) for j, e in enumerate(envs)]
                q = frozen(walk, np.stack([e.dz() for e in envs]))
                for j, e in enumerate(envs):
                    o, r, term, trunc, info = e.step_physics(q[j])
                    rec.step(j, np.clip(act_np[j], -1, 1), r, info["terms"], version, info["x"], e.qpos)
                    lo[j] = np.minimum(lo[j], info["ratios"])
                    hi[j] = np.maximum(hi[j], info["ratios"])
                    bumps[j] += info["bump"]
                    buf_r[t, j] = r
                    if term or trunc:
                        info["extra"] = {"ratio_min": np.round(lo[j], 3).tolist(), "ratio_max": np.round(hi[j], 3).tolist(),
                                         "bump_steps": bumps[j], "teach": teach}
                        row = rec.finish(j, "complete" if term else "truncated", info["reason"],
                                         global_step + t * N + j, info)
                        finished.append({k: row[k] for k in ("return", "steps", "reason", "max_x", "success", "passed",
                                                              "attempted", "spawn_index", "spawn_x", "level",
                                                              "episode_id", "ratio_min", "ratio_max", "bump_steps")})
                        buf_done[t, j] = 1.0
                        if trunc:
                            buf_trunc[t, j] = 1.0
                            trunc_obs[t, j] = o["critic"]
                        if ecfg.mode != "flat" and any(abs(level[k] - e.level_built.get(k, 0.0)) >= 0.049 for k in level):
                            try:
                                e.refresh(level)
                            except Exception:  # noqa: BLE001  (keep the existing layout if a new one fails)
                                pass
                        o = e.reset()
                        rec.start(j, e.episode_meta(), global_step + t * N + j, e.qpos)
                        lo[j], hi[j], bumps[j] = np.full(5, np.inf), np.full(5, -np.inf), 0.0
                    obs[j] = o
            last_c = np.stack([o["critic"] for o in obs])
            conn.send(("batch", idx, buf_a, buf_c, buf_act, buf_logp, buf_r, buf_done, buf_trunc, trunc_obs,
                       last_c, finished))
    except Exception:  # noqa: BLE001
        conn.send(("error", idx, traceback.format_exc()))


# ---------------------------------------------------------------------- learner
def train(cfg: TrainConfig, on_log=None) -> dict:
    import torch

    from .env import COURSE_LEVELS, FLAT_LEVELS

    torch.set_num_threads(cfg.learner_threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.out_root) / cfg.run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    names = FLAT_LEVELS if cfg.env.get("mode") == "flat" else COURSE_LEVELS
    level = {k: (cfg.start_level if cfg.curriculum else 1.0) for k in names}

    ckpts = sorted((run_dir / "checkpoints").glob("policy_v*.pt"))
    ck = None
    if cfg.resume and ckpts:
        ck = torch.load(ckpts[-1], map_location=dev)
        for k in names:
            level[k] = float(ck["level"].get(k, level[k]))

    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    for i in range(cfg.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(i, cfg, b, dict(level)), daemon=True)
        p.start()
        pipes.append(a)
        procs.append(p)
    dims = None
    for c in pipes:
        if not c.poll(cfg.worker_timeout_s):
            raise RuntimeError("a worker did not start")
        msg = c.recv()
        if msg[0] == "error":
            raise RuntimeError(f"worker {msg[1]} failed:\n{msg[2]}")
        dims = msg[1:]
    actor_dim, critic_dim, act_dim = dims
    model = build_model(actor_dim, critic_dim, act_dim, cfg.hidden, init_std_vector(cfg, act_dim)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    norm_a, norm_c = RunningNorm(actor_dim), RunningNorm(critic_dim)
    log_path = run_dir / "train_log.jsonl"
    started = run_dir / "started_at.txt"
    if not started.exists():
        started.write_text(str(time.time()))
    t_start = float(started.read_text())          # the deadline counts time before any restart
    global_step, version, lr = 0, 0, cfg.lr
    if ck is not None:
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        for n, key in ((norm_a, "norm_actor"), (norm_c, "norm_critic")):
            n.mean, n.var, n.count = ck[key]["mean"], ck[key]["var"], ck[key]["count"]
        version, global_step = int(ck["version"]), int(ck["global_step"])
        lr = float(ck.get("lr") or cfg.lr)
        with open(log_path, "a") as f:
            f.write(json.dumps({"event": "resumed", "from": ckpts[-1].name, "version": version,
                                "global_step": global_step, "time": time.time()}) + "\n")
    elif cfg.init_from:
        src = torch.load(cfg.init_from, map_location=dev)
        model.load_state_dict(src["model"])
        for n, key in ((norm_a, "norm_actor"), (norm_c, "norm_critic")):
            n.mean, n.var, n.count = src[key]["mean"], src[key]["var"], src[key]["count"]
        with open(log_path, "a") as f:
            f.write(json.dumps({"event": "initialised_from", "path": cfg.init_from,
                                "source_version": int(src["version"]), "time": time.time()}) + "\n")
    recent: list[dict] = []
    stats_hist = []
    level_changed = dict(level)
    skipped_updates = 0
    try:
        while global_step < cfg.total_steps and (time.time() - t_start) < cfg.hours * 3600:
            it_t = time.time()
            teach = max(cfg.teach_floor, 1.0 - global_step / max(1, cfg.teach_steps))
            # plain arrays: sharing torch tensors with 30 workers exhausts the resource-sharer socket
            state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            for c in pipes:
                c.send(("run", state, _Frozen(norm_a), version, global_step, level_changed, teach))
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
            R_adj = R + cfg.gamma * v_trunc * TR          # truncated episodes bootstrap from their final state
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
            kl_v, pl_v, vl_v = 0.0, 0.0, 0.0
            for _ in range(cfg.epochs):
                perm = torch.randperm(n, device=dev)
                for k in range(cfg.minibatches):
                    idx = perm[k * mb:(k + 1) * mb]
                    dist = model.dist(obs_a[idx])
                    logp = dist.log_prob(act_t[idx]).sum(-1)
                    ratio = (logp - logp_old[idx]).exp()
                    pg = -torch.min(ratio * adv_t[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv_t[idx]).mean()
                    vl = (model.critic(obs_c[idx]).squeeze(-1) - ret_t[idx]).pow(2).mean()
                    loss = pg + cfg.value_coef * vl - cfg.entropy * dist.entropy().sum(-1).mean()
                    opt.zero_grad()
                    if not torch.isfinite(loss):
                        skipped_updates += 1          # never apply a NaN/inf update to the policy
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        kl_v, pl_v, vl_v = (logp_old[idx] - logp).mean().item(), pg.item(), vl.item()
            if kl_v > 2 * cfg.target_kl:              # adaptive learning rate from the last minibatch's KL
                lr = max(lr / 1.5, 1e-5)
            elif kl_v < cfg.target_kl / 2:
                lr = min(lr * 1.5, 1e-3)
            for g in opt.param_groups:
                g["lr"] = lr
            version += 1
            global_step += H * N
            # curriculum: raise a level once most attempts that reach that obstacle get past it
            if cfg.curriculum and len(recent) >= max(150, cfg.curriculum_min_attempts):
                changed = False
                for k in names:
                    tried = [e for e in recent if e["attempted"].get(k)]
                    if len(tried) >= cfg.curriculum_min_attempts:
                        rate = float(np.mean([e["passed"].get(k, False) for e in tried]))
                        step = float(cfg.curriculum_steps.get(k, cfg.curriculum_step))
                        if rate > cfg.curriculum_threshold and level[k] < 1.0:
                            level[k] = min(1.0, level[k] + step)
                            changed = True
                        elif rate < cfg.curriculum_floor and level[k] > 0.0:
                            level[k] = max(0.0, level[k] - step)
                            changed = True
                if changed:
                    level_changed = dict(level)
                    recent = recent[-40:]             # judge the new level on new attempts only
            if version % cfg.log_every == 0 or version == 1:
                last = recent[-400:]
                tried = {k: [e for e in last if e["attempted"].get(k)] for k in names}
                span = np.mean([np.subtract(e["ratio_max"], e["ratio_min"]) for e in last], axis=0).round(3).tolist() if last else None
                row = {
                    "version": version, "global_step": global_step, "elapsed_s": round(time.time() - t_start, 1),
                    "steps_per_s": round(H * N / (time.time() - it_t), 1), "rollout_s": round(roll_t, 2),
                    "episodes_window": len(last), "lr": lr, "kl": kl_v, "policy_loss": pl_v, "value_loss": vl_v,
                    "action_std": model.log_std.exp().detach().cpu().numpy()[[0, 2, 7, -1]].round(3).tolist(),
                    "teach": round(teach, 3),
                    "mean_return": float(np.mean([e["return"] for e in last])) if last else None,
                    "mean_steps": float(np.mean([e["steps"] for e in last])) if last else None,
                    "mean_progress_m": float(np.mean([e["max_x"] - e["spawn_x"] for e in last])) if last else None,
                    "best_max_x": float(np.max([e["max_x"] for e in last])) if last else None,
                    "pass_rates": {k: round(float(np.mean([e["passed"].get(k, False) for e in v])), 3) if v else None
                                   for k, v in tried.items()},
                    "attempts": {k: len(v) for k, v in tried.items()},
                    "success_rate": float(np.mean([e["success"] for e in last])) if last else None,
                    "bump_steps": float(np.mean([e["bump_steps"] for e in last])) if last else None,
                    "shape_span": span, "reasons": _count([e["reason"] for e in last]), "level": dict(level),
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


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _save_ckpt(run_dir: Path, model, norm_a, norm_c, version, global_step, level, cfg, opt=None, lr=None) -> None:
    import torch
    path = run_dir / "checkpoints" / f"policy_v{version:06d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "norm_actor": norm_a.state(), "norm_critic": norm_c.state(),
                "version": version, "global_step": global_step, "level": level, "config": asdict(cfg),
                "optimizer": opt.state_dict() if opt is not None else None, "lr": lr,
                "dims": [model.actor[0].in_features, model.critic[0].in_features, model.actor[-1].out_features]}, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------- evaluation
def load_policy(path: str):
    """A deterministic policy function (actor observation -> action) from a checkpoint."""
    import torch
    ck = torch.load(path, map_location="cpu")
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in TrainConfig.__dataclass_fields__})
    a_dim, c_dim, act_dim = ck["dims"]
    model = build_model(a_dim, c_dim, act_dim, cfg.hidden, init_std_vector(cfg, act_dim))
    model.load_state_dict(ck["model"])
    mean, std = ck["norm_actor"]["mean"], np.sqrt(ck["norm_actor"]["var"] + 1e-8)

    def act(obs_actor: np.ndarray) -> np.ndarray:
        x = np.clip((obs_actor - mean) / std, -10, 10).astype(np.float32)
        with torch.no_grad():
            return model.actor(torch.from_numpy(x[None]))[0].numpy()

    return act, ck


def rollout(env, policy=None, spawn: int = 0, max_steps: int = 1500, frame_every: int = 0, camera=None,
            size=(360, 640)) -> dict:
    """One episode. ``policy=None`` runs the frozen prior alone (zero action)."""
    obs = env.reset(spawn=spawn)
    qpos, frames, ratios, xs, heads, teacher = [env.qpos.astype(np.float32).copy()], [], [], [], [], []
    info, ret, terms = {"reason": ""}, 0.0, {}
    for k in range(max_steps):
        a = np.zeros(env.act_dim) if policy is None else policy(obs["actor"])
        obs, r, term, trunc, info = env.step(a)
        ret += r
        for name, v in info["terms"].items():
            terms[name] = terms.get(name, 0.0) + v
        qpos.append(env.qpos.astype(np.float32).copy())
        ratios.append(info["ratios"]); xs.append(info["x"]); heads.append(info["head_top"]); teacher.append(info["teacher"])
        if frame_every and k % frame_every == 0:
            frames.append(env.env.physics.render(height=size[0], width=size[1], camera_id=camera))
        if term or trunc:
            break
    return {"reason": info["reason"], "steps": len(xs), "return": ret, "max_x": float(max(xs)), "start_x": env.start_x,
            "passed": info.get("passed"), "attempted": info.get("attempted"), "terms": terms,
            "ratios": np.array(ratios), "x": np.array(xs), "head_top": np.array(heads), "teacher": np.array(teacher),
            "qpos": np.stack(qpos), "frames": frames}
