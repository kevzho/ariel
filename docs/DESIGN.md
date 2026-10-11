# Design

Decisions agreed with Guhan on October 10, 2026. [PROJECT.md](PROJECT.md) holds the
requirements; this file records the chosen body, world and look. Measured results
belong in [STATUS.md](STATUS.md).

## Robot: a telescoping rescue humanoid

A 1.55 m, ~62 kg humanoid whose limb bones and torso telescope. The policy commands
every actuator on every control step.

| Group | Rotary joints (movement) | Telescoping segments (reshaping) |
|---|---|---|
| Spine | waist yaw, pitch, roll | torso 0.85-1.25x |
| Each leg | hip yaw, roll, pitch; knee; ankle pitch, roll | thigh, shin 0.8-1.4x |
| Each arm | shoulder pitch, roll, yaw; elbow | upper arm, forearm 0.8-1.4x |
| Total | 23 | 9 |

- Rotary joints are native position servos with torque limits and a back-EMF damping term that reproduces a linear torque-speed curve.
- Telescoping segments are sliding joints with force limits and a speed cap of
  about 0.25 m/s, so a change takes visible time and cannot act as a catapult.
- Mass and motor limits stay fixed as lengths change. Long legs give reach and
  stride but raise the centre of mass and need more torque; a compact body is
  stable but cannot reach.
- Hands are mitts for crawling, bracing and pulling up.
- Rough dimensions at nominal length: pelvis height 0.89 m, shoulder height 1.34 m.

Look: Artemis-era flight hardware. White thermal panels, gold-foil insulation on joint
housings, black anodized structure, polished inner tubes that show when segments
extend, a dark sensor visor with a lidar puck, light regolith dust. Markings use a
fictional ASCENT mission patch and HackPHS branding; no real agency insignia.

## World: a damaged base after a moonquake

Lunar south pole, gravity 1.62 m/s². A corridor about 3 m wide and 25 m long:

1. Cracked sintered landing pad (start), with the fissure running through it.
2. Debris and cargo field: crates, cable reels, panels, a broken comms mast. Some
   crates and small rocks are free physics objects.
3. Collapsed habitat module lying across the route; its low gap is the crawlspace.
4. Fissure from the quake.
5. Toppled solar tower forming the ledge/climb.
6. Broken regolith and boulders to a stranded astronaut sitting against a boulder,
   suit beacon pulsing.

HackPHS branding (logo from hackphs.tech, `assets/branding/hackphs-logo.svg`) appears
on the base structures. Background: a parked lander, crater rims, Earth low on the
horizon, black sky.

Lighting: cinematic realism. Physically correct hard sunlight at a low south-pole
elevation, black sky, plus subtle Earthshine and bounce light so shadows keep detail.

Training uses procedurally varied layouts from the same element set. A fixed showcase
course and separate held-out layouts are reserved for evaluation.

## Rendering

- Every training episode: a side-on tracking camera rendered with MuJoCo from the
  recorded state, consistent across episodes for comparison.
- Showcase: cinematic multi-angle Blender (Cycles) renders from the same recorded
  states.
- Robot link and structure meshes are authored once and shared by MuJoCo, Blender
  and a glTF export for the web viewer.

## Learning to move, not to exploit

Any physical strategy is allowed (walking, crawling, hopping, climbing, using loose
debris). The design removes the shortcuts that usually replace real movement:

| Exploit | Prevention |
|---|---|
| Diving or lunging for progress, back-and-forth farming | Progress reward only for a new furthest pelvis position (frontier), measured at the pelvis, not hands or feet |
| Vibration or contact jitter "skating" | Realistic motor bandwidth and torque-speed limits, action-rate and power costs, randomized friction and masses |
| Telescoping catapult | Hard speed cap on segments, force limits, cost on slide work |
| Passing through obstacles | Solid collision geometry, 4 ms physics step at low lunar speeds, penetration audit; deep penetration ends the episode |
| Ending the episode early to avoid costs | No per-step survival penalty; costs are small relative to progress |
| Reaching across the goal with an arm | Success requires the pelvis near the astronaut and the robot settled for one second |
| Memorizing one course | Procedural layouts in training; held-out layouts for evaluation |
| Privileged information | The policy sees only onboard sensing (joints, segment lengths, IMU, contacts, ground and ceiling scans); privileged state is critic-only |

Shape is never rewarded directly. Telescoping costs only energy, so it is used when it
helps the task. A comparison policy trains the same robot with telescoping locked.

Every episode records audit metrics (peak penetration, peak contact force, peak joint
speed, energy, airborne fraction), and every episode is watchable, so suspected
exploits can be checked directly.

## Compute

All simulation, training and rendering runs on Modal (profile `tejas-16842`). The
desktop is used for writing code and compiling models only.
