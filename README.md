# Autonomous Drone Navigation in Procedurally Generated 3D Cities using Deep RL

A drone learns, by reinforcement learning alone, to fly from a random start to a
random delivery target through a procedurally generated 3D city — avoiding
buildings, managing a battery, and generalising to city layouts it has never
seen.

The baseline algorithm is **DQN, implemented from scratch in PyTorch** (no
Stable-Baselines3, no RL library). The repository is built as a
baseline-vs-proposed experiment harness: a new algorithm plugs in behind one
interface, and every other variable — environment, reward, observation space,
seeds, evaluation protocol — stays fixed.

Everything is Python. Simulation and visualisation use **PyBullet**; no Unity,
Unreal, ML-Agents, AirSim or Isaac Sim.

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Commands](#commands)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [Camera controls](#camera-controls)
- [Results](#results)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Research roadmap](#research-roadmap)

---

## Quick start

### Windows (PowerShell)

PyBullet publishes **no Windows wheels**, so it compiles from source and needs a
C++ toolchain. Python 3.13+ is not supported by PyBullet; use 3.10–3.12.

```powershell
winget install Python.Python.3.12
winget install Microsoft.VisualStudio.2022.BuildTools --override `
  "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools"

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# Build PyBullet against numpy (see note below) — takes ~10 minutes
pip install --no-build-isolation --force-reinstall pybullet
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
pip install --no-build-isolation --force-reinstall pybullet
```

> **Why `--no-build-isolation` for PyBullet?**
> PyBullet only compiles its numpy fast path if numpy is importable *at build
> time*. pip's isolated build environment has no numpy, so a normal
> `pip install pybullet` silently produces a build where `getCameraImage`
> returns a flat Python tuple of ~1M integers per frame. Converting that costs
> ~270 ms per frame and makes video recording ~20× slower. Verify with:
> ```python
> import pybullet; print(pybullet.isNumpyEnabled())   # want 1
> ```
> Training and evaluation do not need PyBullet at all — only visualisation does.

### Verify the install

```powershell
pytest -q                                              # full test suite
python scripts/generate_city.py --seeds 0 1 2 --check-determinism
python scripts/generate_city.py --seed 7 --gui --hold 0    # opens the 3D window
```

---

## How it works

```
              CityConfig + seed
                     |
                     v
        Procedural city  (list of AABB buildings)
                     |
        +------------+-------------+
        |                          |
        v                          v
  Analytic core              PyBullet scene
  (geometry.py)              (renderer.py)
  collisions, rays            display only
        |                          |
        |  <-- same geometry, verified equal -->
        v
   DroneCityEnv (gymnasium)
        |
   observation (21-d) ---> DQNAgent ---> action (9 discrete)
        |                     |
     reward <-----------------+
        |                     |
        +--> RewardBreakdown  +--> ReplayBuffer --> training
```

**Physics is analytic, rendering is PyBullet.** The drone is kinematic, so the
only geometric queries needed are *sphere-vs-boxes* (collision) and
*ray-vs-boxes* (sensors). Both are solved in closed form with vectorised NumPy,
which is exact, deterministic across platforms, and about 4× faster than the
equivalent engine calls — the environment runs at **~4,200 steps/s**. PyBullet
renders the identical geometry, and `tests/test_geometry.py` asserts the
analytic rays agree with PyBullet's own `rayTestBatch` to within 2 cm. A demo is
therefore a faithful replay of training dynamics, not a re-simulation.

**Observation (21 values, all normalised).** Drone position, target position
relative to the drone *in its body frame*, distance to target, body-frame
velocity, heading as (cos, sin), altitude, battery, and 7 range-sensor rays
(front, front-left, front-right, left, right, up, down). Body-frame encoding
makes the policy invariant to absolute world orientation. The agent is never
given the city layout or a shortest path.

**Termination vs truncation.** Reaching the target, crashing, flat battery and
leaving the volume set `terminated`. The step limit sets `truncated`. Only
`terminated` is stored as the replay-buffer done flag, so a time-limit cutoff
still bootstraps — treating it as terminal would teach the agent the world ends
at `max_steps`.

---

## Commands

| Task | Command |
|---|---|
| Inspect a generated city | `python scripts/generate_city.py --seed 7 --gui --hold 0` |
| Random baseline | `python scripts/random_agent.py --episodes 200` |
| Train | `python scripts/train.py --config configs/dqn.yaml` |
| Watch a checkpoint fly | `python scripts/demo.py --checkpoint outputs/checkpoints/dqn_best.pt` |
| Learning progression | `python scripts/compare_learning.py` |
| Evaluate | `python scripts/evaluate.py --checkpoint outputs/checkpoints/dqn_best.pt --episodes 100` |
| Record clips | `python scripts/record_video.py --all-stages` |
| Figures | `python scripts/plot_results.py` |
| TensorBoard | `tensorboard --logdir outputs/logs` |

Training is **headless by design** — no window is opened, so thousands of
episodes run at full speed. Visualisation happens afterwards by replaying
saved checkpoints.

Any config value can be overridden without editing YAML:

```powershell
python scripts/train.py --config configs/dqn.yaml `
  --set dqn.learning_rate=1e-4 reward.collision=-150 city.build_probability=0.5
```

---

## Pretrained checkpoints

Four trained models are committed, so a fresh clone can run the demos without
retraining:

| File | What it is |
|---|---|
| `outputs/checkpoints/dqn_best.pt` | Baseline DQN, highest validation success rate |
| `outputs/checkpoints/dqn_final.pt` | Baseline DQN, weights at the end of training |
| `outputs/checkpoints/her_dqn_best.pt` | HER-DQN, highest validation success rate |
| `outputs/checkpoints/her_dqn_final.pt` | HER-DQN, weights at the end of training |

Everything else in `outputs/checkpoints/` (the periodic `*_episode_N.pt`
snapshots and `*_last.pt`) stays gitignored; `.gitignore` whitelists exactly
the four files above.

These work on a fresh clone as-is:

```powershell
python scripts/demo.py --checkpoint outputs/checkpoints/dqn_best.pt
python scripts/demo.py --checkpoint outputs/checkpoints/her_dqn_best.pt
python scripts/evaluate.py --checkpoint outputs/checkpoints/dqn_best.pt --episodes 100
python scripts/arena.py         # defaults race her_dqn_best vs dqn_best
python scripts/compare_live.py  # defaults compare her_dqn_best vs dqn_best
```

These need the episode snapshots, which are not committed, and only work after
a local `train.py` run:

- `scripts/compare_learning.py` and `record_video.py --all-stages` find no
  intermediate stages, so they show only the random policy and `dqn_best.pt`.
- The "Early DQN" and "Mid-training DQN" entries in `configs/experiment.yaml`
  point at `dqn_episode_250.pt` / `dqn_episode_1500.pt`.
- The `compare_live.py --checkpoint-b outputs/checkpoints/her_dqn_episode_500.pt`
  example in that script's docstring.

To ship a newly trained model, retrain (which overwrites these four files), then
commit them; to ship a different checkpoint, add a `!outputs/checkpoints/<name>.pt`
line to `.gitignore`.

---

## Camera controls

Every GUI script shares one camera rig. Bindings avoid letter keys, which
PyBullet's viewer already claims for its own toggles (`w` wireframe, `s`
shadows, `g` panels).

| Input | Action |
|---|---|
| **Arrow keys** | Pan horizontally, relative to where the camera is looking |
| **Shift / Ctrl** | Raise / lower |
| **PgUp / PgDn** | Zoom in / out |
| **Alt + arrows** | Change viewing angle (yaw / pitch) |
| **F1 / F2 / F3 / F4** | Mode: free / follow / orbit / cinematic |
| **F5** | Reset to the starting view |
| Left-drag, scroll | PyBullet's own mouse orbit and zoom still work |

Modes: **free** stays put; **follow** tracks the drone while you keep angle
control; **orbit** tracks the drone *and* circles it automatically (the
hands-off "show me the drone" mode); **cinematic** slowly orbits the city
centre for establishing shots. Panning manually drops out of a tracking mode so
the camera never fights you.

Set the initial mode from config or CLI:

```powershell
python scripts/demo.py --checkpoint outputs/checkpoints/dqn_best.pt --camera-mode orbit
python scripts/generate_city.py --seed 7 --gui --hold 0 --set render.camera_pitch=-89
```

---

## Recording clips

```powershell
# Every training stage (random -> early -> mid -> final), one clip each
python scripts/record_video.py --all-stages --gui --width 960 --height 600

# A highlight reel of successful deliveries only
python scripts/record_video.py --checkpoint outputs/checkpoints/dqn_best.pt `
  --episodes 20 --only-success --max-clips 3 --gui
```

Clips land in `outputs/videos/`.

**Use `--gui` for final clips.** It is counter-intuitive, but rendering through
a window is the *fast* path — hardware OpenGL costs ~84 ms/frame at 960×600
versus ~358 ms for the headless software rasteriser. Headless (the default) is
for recording unattended, e.g. alongside a training run, since it never steals
focus.

Frame capture dominates runtime either way, so `--every-n-steps 2` (with
`--fps` halved to match) roughly halves recording time at the same playback
speed.

**Overlays in clips are drawn by us, not by PyBullet.** PyBullet's
`addUserDebugText` / `addUserDebugLine` items (the HUD, the trajectory ribbon,
the sensor rays) are painted by the GUI's own render loop into the on-screen
window — they are *not* part of the scene `getCameraImage` rasterises. A naive
recording therefore shows the city and the drone but no episode information and
no flown path. `drone_rl.evaluation.overlay` composites them onto each captured
frame instead, projecting world points through the same view/projection
matrices used to render it, so the drawn trajectory lands exactly on the flown
path. Pass `--no-overlay` for clean scene-only footage.

So: the **live window** shows PyBullet's overlay; a **recorded clip** shows
ours. Both display the same information.

## Results

See [`RESULTS.md`](RESULTS.md) for the measured numbers, figures and the
training-progression comparison.

---

## Configuration

Three files under `configs/`, composed via a `defaults:` include list:

- **`environment.yaml`** — city geometry, drone dynamics, sensors, action set,
  reward weights, episode rules, render settings. The single source of truth
  for the simulation; every algorithm inherits it unchanged.
- **`dqn.yaml`** — DQN hyper-parameters and the training loop budget.
- **`experiment.yaml`** — the evaluation protocol and the fairness contract.

### Reward

All weights are in YAML; none are hard-coded. `info` exposes the full
decomposition every step (`progress_reward`, `proximity_penalty`,
`energy_cost`, `terminal_reward`, ...) so you can see *why* the agent got what
it got.

Progress shaping is **potential-based** — `w * (d_prev - d_now)` — which leaves
the optimal policy unchanged and cannot be farmed by oscillating (there is a
test for exactly that).

The obstacle-proximity weight is *sized against* the progress weight rather than
picked by feel. At full speed the drone closes 0.45 m per step, so progress pays
at most +0.45; if the proximity penalty does not exceed that before impact, then
flying into a building is net-positive on every single step and the only thing
opposing it is a discounted terminal penalty ~60 steps away. That is not
hypothetical — with an under-sized penalty this agent reached a **96% collision
rate by episode 500**, actively beelining into walls. `tests/test_rewards.py`
pins the corrected property.

### Procedural generalisation

City seeds are split into disjoint pools, validated for overlap at load time:

```yaml
city:
  train_seed_start: 0        ; train_seed_end: 799
  validation_seed_start: 800 ; validation_seed_end: 899
  test_seed_start: 900       ; test_seed_end: 999
```

Training draws only from `train`. Model selection uses `validation`. `test`
cities are touched only by the final evaluation, so the train-vs-unseen gap is
a genuine generalisation measurement.

### Reproducibility

Each run records seeds, the fully resolved config, library versions, device and
timestamp to `outputs/logs/<name>/experiment.json`, and every checkpoint embeds
its own config. The environment never reads global RNG state — episode sampling
uses an explicit `numpy.random.Generator` — so `reset(seed=s)` replays exactly,
regardless of what else the process is doing.

---

## Testing

```powershell
pytest -q                       # everything
pytest tests/test_dqn.py -q     # DQN correctness only
pytest -q -k "not pybullet"     # skip tests needing the renderer
```

Coverage worth knowing about:

- **DQN correctness** — terminal states are not bootstrapped; truncation *is*
  bootstrapped; the target network is frozen and syncs on schedule; and an
  end-to-end test where the agent must recover a known Q-value on a trivial MDP.
- **Analytic/PyBullet parity** — random rays through a generated city agree to
  within 2 cm.
- **Reward shaping** — potential-based progress telescopes to zero on a round
  trip; approaching a wall stops paying before impact.
- **Environment contract** — same seed replays identically; global RNG state is
  irrelevant; start and target never spawn inside a building; each episode ends
  in exactly one terminal category.

---

## Project layout

```
drone-rl/
├── configs/            environment.yaml, dqn.yaml, experiment.yaml
├── src/drone_rl/
│   ├── environment/    city, drone, geometry, sensors, spaces,
│   │                   rewards, drone_env, renderer, camera
│   ├── agents/         base, network, replay_buffer, exploration,
│   │                   dqn, random_agent, registry
│   ├── training/       trainer, callbacks
│   ├── evaluation/     evaluate, metrics, visualization, video
│   └── utils/          config, device, logging, reproducibility
├── scripts/            generate_city, random_agent, train, evaluate,
│                       demo, compare_learning, record_video, plot_results
├── tests/              city, geometry, drone, rewards, environment, dqn
└── outputs/            checkpoints, logs, figures, trajectories, videos, evaluation
```

---

## Research roadmap

**Implemented.** Procedural city generation, kinematic drone, ray sensors,
structured observations, discrete actions, Gymnasium environment, modular YAML
reward, random baseline, DQN from scratch, training with checkpoints and
TensorBoard, evaluation across disjoint seed pools, figures, 3D visualisation,
keyboard camera, video recording.

**Remaining for the research contribution.** Only the proposed algorithm. The
extension point is `drone_rl.agents.base.Agent` plus `registry.register_agent`:
implement `act`, `observe`, `update`, `state_dict`, register the class, and set
`experiment.algorithm` in YAML. The environment, reward, observation space,
training loop and evaluation protocol are untouched by that change, which is
what makes the comparison fair by construction.

Candidate directions from the brief: energy-aware DQN, latent-state DQN,
VAE+DQN, world-model-assisted DQN, curiosity-enhanced DQN, graph-based DQN.

Before claiming an improvement, run both algorithms under the pinned
`fairness:` block in `experiment.yaml` (same seeds, same budget, same
evaluation episodes) and report train *and* unseen-city numbers with more than
one training seed.
