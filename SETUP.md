# Setup on a new machine

Target: **Python 3.10 – 3.12**. Not 3.13+ — PyBullet does not build against it.

---

## Linux (easiest — PyBullet has wheels)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# Rebuild PyBullet so it uses numpy (see "Why" below) — optional but ~4x faster video
pip install --no-build-isolation --force-reinstall --no-cache-dir pybullet
```

## macOS

Same as Linux, but PyBullet compiles from source (a few minutes). Needs Xcode CLT:

```bash
xcode-select --install
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
pip install --no-build-isolation --force-reinstall --no-cache-dir pybullet
```

## Windows

**PyBullet has never published Windows wheels** — `pip install pybullet` always
compiles C++ and needs the MSVC toolchain. Budget ~15 min for the build tools
download and ~10 min for the PyBullet compile.

```powershell
winget install Python.Python.3.12
winget install Microsoft.VisualStudio.2022.BuildTools --override `
  "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# Must be --no-build-isolation, see below
pip install --no-build-isolation --force-reinstall --no-cache-dir pybullet
```

If `winget` is unavailable, install Python 3.12 from python.org and
"Build Tools for Visual Studio 2022" with the **Desktop development with C++**
workload from visualstudio.microsoft.com.

---

## Verify

```bash
python -c "import pybullet; print('numpy enabled:', pybullet.isNumpyEnabled())"   # want 1
pytest -q                                                                          # 128 passed
python scripts/generate_city.py --seeds 0 1 2 --check-determinism
python scripts/random_agent.py --episodes 20
python scripts/generate_city.py --seed 7 --gui --hold 0                            # opens 3D window
```

If all five pass, the install is good.

---

## Why `--no-build-isolation`

PyBullet only compiles its numpy fast path if numpy is importable **at build
time**. pip's isolated build environment has no numpy, so a plain
`pip install pybullet` silently produces a build where `getCameraImage` returns
a flat Python tuple of ~1M integers per frame. Converting that costs ~270 ms per
frame instead of ~84 ms, making video recording roughly 4× slower.

`isNumpyEnabled()` returning `0` means you got the slow build. Everything still
works — only recording is slower — so this is optional if you don't need clips.

`--no-build-isolation` requires `setuptools` and `wheel` already present in the
venv, which the commands above install first.

---

## GPU (optional)

CPU is the sensible default here: the network is a small MLP and the loop is
dominated by Python-side environment stepping. For a CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then `--device cuda` on any script. `--device auto` (the default) picks CUDA
only when it is actually available.

---

## No display / headless server

Training, evaluation, plotting and tests need **no** display and no PyBullet at
all. Only the 3D windows do. On a headless box just skip `--gui` and use
`scripts/record_video.py` (headless by default) for video.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No matching distribution found for pybullet` | Python 3.13+. Use 3.10–3.12. |
| `Microsoft Visual C++ 14.0 or greater is required` | Windows without MSVC. Install the Build Tools above. |
| `isNumpyEnabled()` returns `0` | Rebuilt without `--no-build-isolation`. Re-run that command. |
| `already holds a training run` | The concurrency guard. Stop the other run, or pass a different `--name`. |
| GUI window never opens | No display, or remote session. Use headless paths. |
| Import errors running `scripts/*.py` | `pip install -e .` was skipped (scripts also self-bootstrap `src/`, so this usually still works). |

---

## Then run the project

See [README.md](README.md) for the full workflow. Shortest path to something
visible:

```bash
python scripts/train.py --config configs/dqn.yaml --episodes 500 --name dqn_quick   # ~8 min
python scripts/demo.py --checkpoint outputs/checkpoints/dqn_best.pt --camera-mode orbit
```
