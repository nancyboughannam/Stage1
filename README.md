# Stage 1: Hard Safety Filtering for MEC Resource Allocation

This project is a compact Python prototype for the Stage 1 M-LBAC research setting.
It models a small MEC device whose queue, battery reserve, and thermal state are filtered by a conservative safety layer before the action is executed.

## Quick start

```bash
cd /Users/nancyboughannam/Desktop/Stage1
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy gymnasium stable-baselines3 torch pandas matplotlib pyyaml pytest
python -m pytest -q
python demo_stage1.py
```

## What the demo shows

- queue backlog remains bounded under a feasible action sequence
- battery reserve stays above the hard reserve threshold
- temperature stays under the thermal limit
- safety filter steps from nominal to corrected or safe fallback action when necessary

## Thermal model note

The default thermal coefficients in the repository are intentionally synthetic and uncalibrated. They are not hardware-validated parameters and are used only to exercise the model logic, safety-layer checks, and reporting semantics. Any real-world safety claims require a calibrated thermal model and measured device data.

## Main modules

- `mlbac_stage1/env.py`: Gymnasium environment wrapper
- `mlbac_stage1/constraints.py`: hard feasibility checks
- `mlbac_stage1/safety_filter.py`: action correction and fallback logic
- `mlbac_stage1/models.py`: resource and thermal transition model
- `tests/test_core.py`: unit tests for safety logic
- `tests/test_ppo_smoke.py`: environment smoke test
