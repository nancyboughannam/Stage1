# Stage 1: MEC Safety-Filter Simulator Diagnostics

This repository contains a small, deterministic-capable mobile edge computing simulator for inspecting one-step feasibility checks and safety-filter behavior. The action-scaling filter is a heuristic candidate search. It is not nearest-safe-action optimization, a recursive-feasibility method, or evidence of real-hardware safety.

## Setup and checks

```bash
cd /Users/nancyboughannam/Desktop/Stage1
. .venv/bin/activate
python -m pip install -e .
python -m pytest -q
```

The environment API checks can be repeated with:

```bash
python - <<'PY'
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check
from mlbac_stage1.env import make_env

gym_check(make_env(n_devices=2, horizon=3, deterministic=True))
sb3_check(make_env(n_devices=2, horizon=3, deterministic=True))
print("Gymnasium and Stable-Baselines3 checks passed")
PY
```

`tests/test_ppo_smoke.py` retains a short `learn()`/`predict()` integration smoke test only. It does not measure policy quality and is not a training result.

## Three deterministic diagnostics

Generate all scenarios with seed 0:

```bash
python diagnostics_stage1.py --scenario all --output diagnostics --seed 0
```

Run one scenario alone by replacing `all` with `normal`, `thermal`, or `battery`. The convenience normal-only command is:

```bash
python demo_stage1.py
```

Each scenario directory contains `state.png`, `actions.png`, `episode.csv`, `summary.json`, and `summary.txt`.

- `diagnostics/normal/`: an admissible one-step proposal passes unchanged; the physical transition and its state are observed.
- `diagnostics/thermal/`: a separately labelled synthetic, hardware-bounded thermal stress proposal is rejected for predicted temperature alone; the current heuristic finds a one-step verified correction. The stress coefficients are not calibrated device data.
- `diagnostics/battery/`: at `B_min + 0.5 J`, the one-step idle candidate would consume `1 J` of baseline energy and violate reserve. No candidate is verified, nothing is executed, the physical outcome is marked unobserved, and the episode ends with a negative controller-failure reward. This is not proof that the state is mathematically infeasible.

Plots begin at time zero and read `Q_max`, `B_min`, and `T_max` from each scenario's parameter configuration. Action plots show decoded proposal against executed action; the execution is blank when there was no verified action. CSV logs contain one row per device and attempted slot, numeric action/state fields, proposed and selected predictions, observed transition metrics, constraint flags/magnitudes, filter status, and episode-ending reason. Unobserved actual state/action fields are empty rather than fabricated.

## Units and model

- A full task is `0.2 decimal MB = 0.2 × 1,000,000 × 8 = 1,600,000 bits`; `d` is bits per full task. `c` is cycles per full task. Queue/admission/work values are task fractions.
- Battery capacity is `9 Wh × 3600 = 32,400 J`; the configured reserve is 10%, `B_min = 3,240 J`. Battery and energy are in joules. Power is in watts and slot duration is in seconds; default idle baseline energy is `1 W × 1 s = 1 J` per slot.
- Shared bandwidth is in bits/s, CPU frequencies are in cycles/s (Hz), transmission rate is bits/s, channel gain is dimensionless, and `N0` is noise spectral density in W/Hz.
- Observations use fixed reference scales: `Q/Q_max`, `B/B_cap`, `T/100 C`, `A/A_max`, each device's `g/(1e-5)`, shared bandwidth divided by `10 MHz`, edge CPU divided by `10 GHz`, and remaining slots divided by the episode horizon. Scaled values are not clipped at safety limits, so state violations remain visible. Raw physical values are in the CSV and environment info.
- Policy actions in `[-1, 1]` map affinely to nonnegative fractions of configured physical action maxima. The raw action, bounded policy action, decoded proposal, and filter-selected action are logged separately.

The synthetic endpoint thermal model is

`T_next = T_ambient + (T - T_ambient) exp(-delta/tau) + alpha h delta`,

where `h = 0.7 f/f_max + 0.3 p/p_max` is dimensionless, `delta` and `tau` are seconds, and `alpha` is degrees Celsius per second per normalized load. With constant normalized load, the discrete equilibrium is `T_ambient + alpha delta / (1 - exp(-delta/tau))`; default parameters yield about `26.88 C` at `25 C` ambient and maximum legal heating, so they do not challenge the `80 C` cap. The thermal diagnostic deliberately increases `alpha` as a synthetic stress setting. The implementation checks the start and predicted/recorded slot endpoint; it does not resolve continuous within-slot temperature transients. The plotted/recorded peak is `max(start, endpoint)`, not a continuous-time peak guarantee.

## Safety and episode semantics

The filter verifies candidates for one step only. A one-step verified idle candidate is not verified safe for the remaining horizon. If no candidate passes, the filter returns no executable action and labels the result `no_verified_candidate_found`; it does not claim mathematical infeasibility. The environment terminates the episode with `controller_failure_no_verified_continuation`, a negative penalty, outstanding work and elapsed time recorded, and actual post-state marked unobserved. An executed transition is checked separately against actual queue, battery, temperature, and recorded endpoint peak; detected actual violations end the episode.

No finite set of passing tests, zero violations in these demonstrations, or stopped simulation proves safety. The simulator uses synthetic/simplified task, queue, energy, radio, and thermal assumptions; its parameters require empirical calibration and validation before any hardware or deployment claim.
