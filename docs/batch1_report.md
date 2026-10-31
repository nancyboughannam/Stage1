# Batch 1 delivery report — 31 October 2026

Only the first new batch (configurable safety margins) is implemented. Based on
GitHub revision `d6859ef` (Add Batch C experiment logging and model-mismatch
diagnostics). No commits or pushes were made. The later energy mismatch,
comparison/metrics and final validation batches remain pending.

## Files changed

| File | Purpose |
| --- | --- |
| mlbac_stage1/config.py | Opt-in defaults, scalar/per-device validation, effective settings and inward thresholds |
| mlbac_stage1/constraints.py | Strict enabled-margin checks and separate ordinary/tightened audit fields |
| mlbac_stage1/optimization.py | Same margin thresholds in SLSQP, including starting-temperature condition |
| mlbac_stage1/safety_filter.py | Validate settings and report correct verification scope |
| mlbac_stage1/env.py | Snapshot effective arrays and retain the successful final gate's report |
| experiments_stage1.py | Optional scalar CLI flags and Python per-device settings |
| tests/test_safety_margins.py | 37 focused tests, including candidate/fallback/final-gate checks |
| README.md | Link and quick command |
| docs/model_uncertainty_and_margins.md | Equations, configuration, logs, scope and remaining batches |
| docs/batch1_report.md | Actual validation and delivery details |

The physical model, reward, reference policy and action/observation spaces were
not changed. Existing actual thermal override isolation is retained. No new
research PPO implementation or training run was added; the pre-existing tiny PPO
compatibility test ran as part of the full test suite.

## Conditions implemented

For enabled per-device margins mT (Celsius) and mB (joules):

```text
T_current <= T_max
T_predicted_next <= nextafter(T_max - mT, -infinity)
B_predicted_next >= nextafter(B_min + mB, +infinity)
```

No outward temperature/battery checker tolerance is applied to these enabled
conditions. The inward one-float adjustment is numerical, separate from the
configured model-error margins. Disabled mode retains the previous checks and
tolerances. Actual state checks use ordinary physical limits.

## Tests actually run

Initially `python -m pytest -q` could not start: pytest was absent. Installed
pytest, Gymnasium, SciPy/Stable-Baselines3 dependencies (including Torch).

| Command / source | Actual result |
| --- | --- |
| python -m pytest -q --ignore=tests/test_ppo_smoke.py, before edits | 118 passed in 10.34 s |
| python -m pytest -q, clean archive of d6859ef after dependencies installed | 119 passed in 9.53 s |
| python -m pytest -q, existing suite after initial margin edits | 119 passed in 9.49 s |
| python -m pytest -q tests/test_safety_margins.py, before runner integration test | 36 passed in 0.81 s |
| python -m pytest -q, final implementation including runner test | 156 passed in 10.58 s; 0 failures, 0 skips |

Early focused-test development had two failures because the assertions demanded
SLSQP correction specifically. The observed safe outcome was verified fallback.
The tests now accept either independently verified correction or fallback, as
required, and separately verify the corrected-candidate path without depending
on numerical solver convergence. Strict safety checks were not relaxed.

## Exploratory diagnostics actually run

```bash
python experiments_stage1.py --output /workspace/scratch/7d054eb94a5c/batch1_outputs --scenarios reference resource_pressure thermal_matched battery invalid_proposal --seeds 0 --devices 1 --horizon 3 --safety-margin-enabled --temperature-margin-c 2 --battery-margin-j 0.2
python experiments_stage1.py --output /workspace/scratch/7d054eb94a5c/batch1_outputs/shared_pressure_two_devices --scenarios resource_pressure --seeds 0 --devices 2 --horizon 3 --safety-margin-enabled --temperature-margin-c 2 --battery-margin-j 0.2
```

All runs used seed [0], load multiplier 1, a 1-second simulation slot, enabled
2 C / 0.2 J margins, and the existing scenario-labelled scripted policy. Full
initial conditions and all effective parameters are in each config snapshot.
These short matched-model checks are not the future paired mismatch study.

| Scenario | Devices | Executed / attempted | Observed outcome |
| --- | --- | --- | --- |
| Reference | 1 | 3 / 3 | 3 nominal accepted; horizon complete |
| Resource-pressure preset | 1 | 3 / 3 | 3 nominal accepted; one device does not overload shared resources |
| Thermal matched, synthetic coefficient | 1 | 3 / 3 | 2 verified idle fallbacks, 1 verified correction; horizon complete |
| Battery near reserve | 1 | 0 / 1 | No verified continuation; actual state and violation rate remain null |
| Invalid proposal | 1 | 3 / 3 | 3 verified idle fallbacks; horizon complete |
| Resource pressure, joint demand | 2 | 3 / 3 | 3 verified corrections; horizon complete |

No actual constraint violations were observed in the executed slots of these
runs. The unexecuted battery attempt is excluded. Queue conservation residuals
were zero except thermal matched (-5.55e-17 work units, floating-point roundoff).
These observations are not universal safety/error-bound evidence.

## Where to find the delivery

The download contains:

- `updated_files/`: only the ten changed/new files, with repository-relative paths.
- `batch1.patch`: an alternative Git patch including new files.
- `batch1_outputs/`: actual snapshots, steps.json, slots.csv, summaries, aggregates
  and state/service/action plots from the commands above.
- `APPLY_BATCH1.md`: application and local verification instructions.

Within each scenario, outputs are under `load_1/seed_0/`. The additional
2-device run is under `shared_pressure_two_devices/`.

## Software versions used

Python 3.12.14; NumPy 2.3.5; SciPy 1.17.0; Gymnasium 1.4.0;
pytest 9.2.0; Torch 2.15.0; Stable-Baselines3 2.9.2;
Matplotlib 3.10.8; pandas 2.2.3. Solver choices and timing may differ on another
platform. Versions and controller configuration are captured in the run snapshots
where supported by the existing runner; this report lists the additional packages.

## Remaining work and limitations

Strict enabled-margin boundary checks can reject a solver result and fall back
to idle. This preserves authorization semantics but can cost service. No
candidate found is not mathematical infeasibility. No one-step result proves
recovery, physical safety, recursive feasibility, stability, or hard real-time
execution. Margins and thermal stress parameters remain uncalibrated.

Batch 2 will add energy mismatch and realized-error coverage, including one
covered-error and one exceeded-error controlled test. Batch 3 will add the full
paired four-condition experiment design, expanded metrics and deadline reports.
Batch 4 will consolidate multi-seed evidence and documentation. No design input
is required to finish Batch 1, and no later batch has been implemented here.
