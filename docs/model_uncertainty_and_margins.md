# Model uncertainty and safety margins — Batches 1–2

Status: margin filtering, simulator-only energy mismatch and one-step prediction
error/coverage reporting are implemented and tested. Paired comparison
orchestration, expanded aggregates and deadline reporting remain for later batches. Historical Batch A–C documentation is unchanged.

## Purpose

Study whether explicit prediction buffers reduce simulated violations, and what
service and runtime they cost. The first two batches establish consistent margin checks and controlled mismatch
diagnostics; they do not establish the effectiveness of a universal error bound.

## Configuration and exact conditions

`make_default_params` accepts `safety_margin_enabled=False`,
`temperature_margin_c=0.0` and `battery_margin_j=0.0`. Each margin may be a scalar
for all devices or a one-dimensional array/list with exactly N entries. Values
must be finite and nonnegative even when disabled. The enable flag is boolean.
Invalid inputs raise ValueError at configuration/environment/filter construction
and at standalone constraint evaluation.

For device i, let mT be its Celsius margin and mB its joule margin. In enabled
mode the exact floating-point thresholds are:

```text
T_check = nextafter(T_max - mT, -infinity)
B_check = nextafter(B_min + mB, +infinity)
T_current <= T_max
T_predicted_next <= T_check
B_predicted_next >= B_check
```

`nextafter` moves one representable floating-point number inward. No outward
checker tolerance is added to these conditions. This is a numerical adjustment,
not an additional calibrated model-error margin. Enabled mode with zero buffers
therefore still differs from disabled mode at exact/numerically close boundaries.
If an extremely large battery margin produces an unrepresentable threshold,
no finite candidate can satisfy it; this is not an infeasibility certificate.

Disabled mode preserves the previous endpoint/start maximum thermal check and
battery checks, including their original numerical tolerances and the original
SLSQP residual arithmetic. Supplied margins are recorded but have effective value
zero. Other constraints and their tolerances remain unchanged in both modes.

The nominal check, SLSQP constraints, independent candidate checks, idle check,
and final environment gate use the same margin settings and thresholds. SLSQP
retains its existing conservative auxiliary transmission-time energy expression;
the independent checker recomputes energy from the physical action. Solver
success is not authorization. Strict boundary checks can reject near-boundary
solver outputs and select verified idle instead. No action is clipped after
verification, and the environment records its successful final-gate report.

## Prediction versus simulated transition

The filter predicts queues, battery consumption and temperature with the existing
nominal equations. The transition still uses the existing physics. The pre-existing
`actual_thermal_alpha` override stays in the environment's private simulator
configuration and is passed only to the actual transition, not the filter. The
experiment snapshot already records that simulator-only override separately.
Batch 2 adds `actual_energy_multiplier` (default 1.0), described below. It is
also stored separately from controller parameters.

Margins constrain predicted endpoints and the known starting temperature. They
are not a validated continuous-time thermal peak guarantee. Actual post-transition
checks remain against the original physical limits, not against the buffers.

## Audit fields and output

`steps.json` retains the proposal, each searched candidate check and selected or
fallback report. Valid reports now include:

- `safety_margins`: configured and effective per-device buffers and enable flag.
- `margin_checks`: predicted temperature/battery, ordinary and tightened slacks,
  exact inward thresholds, ordinary pass flags and margin pass flags.
- `violations.temperature_margin` and `violations.battery_margin`: separate from
  the pre-existing physical-limit violation fields and magnitudes.

Disabled margin pass flags are null (not evaluated). Existing ordinary thermal
pass flags refer to the existing modeled endpoint/start maximum. Tightened slacks
are before the one-float numerical adjustment; use the recorded thresholds and
pass flags for the final decision. Invalid inputs/already-unsafe states can be
rejected before prediction and retain the existing empty prediction report.

`config_snapshot.json` includes normalized `safety_margin_settings` inside
`controller_config`, including configured and effective arrays. `slots.csv`,
`summary.json`, `aggregate.json` and existing plots are retained. Detailed margin
arrays are in `steps.json`; expanded margin-specific CSV/aggregate metrics are
reserved for Batch 3. Existing `proposal_predicted_feasible` means feasible under
the selected mode; use ordinary pass flags to distinguish physical violations.

A failed/unexecuted attempt has no actual state or actual violation rate. It is
not a safe slot. Existing throughput uses completed work / executed simulated
seconds, rejection uses rejected / offered work over executed slots, and actual
violation rates use executed device-slots. Unexecuted offered arrivals are
reported separately. Zero denominators remain null. These summaries do not yet
provide every metric requested for the overall milestone.

## Commands

From the repository root, with project dependencies and pytest installed:

```bash
python -m pytest -q
python -m pytest -q tests/test_safety_margins.py
python experiments_stage1.py --output experiments_margins_batch1 --scenarios reference thermal_matched battery invalid_proposal --seeds 0 --devices 1 --horizon 3 --safety-margin-enabled --temperature-margin-c 2 --battery-margin-j 0.2
```

CLI margins are scalar; the Python configuration accepts per-device arrays.
The existing runner can run ten independent seeds in one margin condition:

```bash
python experiments_stage1.py --output experiments_margins_10seeds --scenarios reference thermal_matched --seeds 0 1 2 3 4 5 6 7 8 9 --devices 2 --horizon 30 --safety-margin-enabled --temperature-margin-c 2 --battery-margin-j 0.2
```

This ten-seed command is a future-run example, not a result reported here. The
four-condition paired mismatch study will be added later. Existing separate
arrival/channel streams are unchanged; paired exogenous seeds do not imply
identical trajectories once selected actions diverge. Policies are scripted,
not PPO benchmarks. Stress thermal coefficients and margins are illustrative,
not calibrated hardware measurements.

## Limits and next batches

One-step local SLSQP search is not a global nearest-action certificate. No verified
candidate means search/verification failure, not proof of mathematical
infeasibility. Verified idle is one-slot admissibility, not general recovery.
Stopping the simulation does not show physical recovery. There is no claim of
physical safety, recursive feasibility, long-term stability or real-time deadlines.

1. Batch 1: safety margins, gate consistency, audit fields and focused tests.
2. Batch 2 (implemented): simulator-only energy multiplier, prediction errors,
   margin coverage and controlled covered/exceeded-error tests; thermal separation retained.
3. Batch 3: paired four-condition experiments, boundary cases, complete service/
   safety metrics, per-episode aggregates and runtime deadline reporting.
4. Batch 4: final integrated validation, reproducibility evidence and documentation.

Open research work remains thermal/energy calibration, validated recovery or
shutdown, feasible-state/sufficient-condition analysis, and task-latency tracking.
PPO and PPO-Lagrangian as research methods remain outside this milestone. The
existing tiny PPO compatibility smoke test is retained in the full test suite.

Actual Batch 1 results and exact commands are in [batch1_report.md](batch1_report.md).


## Batch 2: actual-only energy mismatch

Set the optional simulator parameter before constructing the environment:

```python
params = make_default_params(2, safety_margin_enabled=True, battery_margin_j=0.5)
params["actual_energy_multiplier"] = 1.25
# Optional, independent thermal-model override:
params["actual_thermal_alpha"] = 0.3
env = make_env(params=params)
```

The multiplier is a finite, nonnegative scalar applied to every device. Arrays,
negative/nonfinite values and booleans are rejected. Zero is allowed as a synthetic
experiment. It is validated both at environment construction and by the direct
transition function. A supplied thermal override is validated as a finite,
nonnegative scalar at environment construction; None means no override.

For each device, with k the energy multiplier:

```text
E_local_actual = k * E_local_nominal
E_tx_actual = k * E_tx_nominal
E_baseline_actual = k * E_baseline_nominal
E_actual = E_local_actual + E_tx_actual + E_baseline_actual
B_actual_next = B_current - E_actual
```

The nominal energy equations remain unchanged. All existing transition energy
fields, including `energy`, `total_energy` and component fields, now represent
actual simulated energy. At k=1 they retain their previous values. Additional
transition fields record nominal energy and actual-minus-nominal energy error.
The checker now exposes a per-device `predicted_energy_j` array, avoiding loss of
precision from reconstructing small energy values by subtracting large batteries.

Both actual-only settings are removed from `env.params` before the filter is
constructed and are supplied only to the simulated transition. Controller
predictions and optimization do not use them. The recorded comparison uses the
final verified executed action, including a correction or idle fallback, rather
than the rejected original proposal. No action is changed after verification.

Energy mismatch does not change the thermal equation, queue/work conservation,
channel/timing assumptions or exogenous random traces. Thermal mismatch remains
an independent override. This deliberately does not claim physically calibrated
coupling between device energy and temperature. No hidden channel, CPU timing,
arrival or measurement errors were introduced. Actual reserve violations are
recorded and terminate the episode; state is not clipped to the reserve. This
retains the existing slot-end simulator semantics, without a physical shutdown
or within-slot energy exhaustion model.

### Prediction errors and coverage

Every environment `info` now contains `prediction_errors`. For executed slots,
this contains one value per device:

```text
eT = T_actual_next - T_predicted_next                  [Celsius]
eB = B_actual_next - B_predicted_next                  [joules]
eE = E_actual - E_predicted                           [joules]
adverse_T = max(0, eT)
adverse_B = max(0, -eB)
```

The one-sided coverage flags compare `adverse_T <= temperature_margin_c` and
`adverse_B <= battery_margin_j`. These correspond to the directions guarded by
the Batch 1 inequalities. Separate absolute-error coverage flags compare |eT|
and |eB| to their configured margins. A favorable error can pass the one-sided
check while failing the absolute-error check; the names distinguish them.

Coverage is reported against both configured and effective margins. Configured
margins can be studied retrospectively even when filtering is disabled; effective
margins are zero in that mode. Absolute-error coverage uses configured margins.
All comparisons use ordinary floating-point <= with no outward numerical
tolerance. Extremely close boundary outcomes can reflect floating-point rounding;
the buffers are never silently increased to mark an error covered.

For an unexecuted attempt all comparison values and coverage flags are null,
including the executed-action prediction fields. Configured margin settings
remain recorded; separate proposal/fallback prediction reports remain available.
This prevents a rejected fallback prediction from masquerading as an executed
transition. Observing coverage in finite simulations is not a universal bound.

### New outputs and configuration provenance

- `steps.json`: full per-device comparison in each record's
  `info.prediction_errors`, alongside the existing actions/reports.
- `prediction_errors.csv`: one row per attempted slot and device. Predictions,
  actual endpoints/energy, signed/adverse errors, coverage and margin settings.
  Missing outcomes are blank in CSV, not zero or False.
- `slots.csv`: adds joint `predicted_energy_j` and `energy_prediction_error_j`;
  existing `energy_j` is the actual consumed energy.
- `config_snapshot.json`: keeps `controller_config` and adds
  `simulator_only_config` with multiplier, thermal override and resolved actual
  thermal coefficient. Existing `simulator_only_actual_thermal_alpha` remains
  for compatibility in the experiment runner.
- `reset()` returns `effective_config` for the controller and a separate
  `simulator_only_config`. `env.simulator_config()` exposes the latter for logging.
  The older diagnostic runner also writes this separate config and error CSV.

### Batch 2 commands

```bash
python -m pytest -q
python -m pytest -q tests/test_model_uncertainty.py
python mismatch_diagnostics.py --output experiments_mismatch_batch2/controlled --seed 0
python experiments_stage1.py --output experiments_mismatch_batch2/runner --scenarios reference thermal_mismatch battery invalid_proposal --seeds 0 --devices 2 --horizon 3 --safety-margin-enabled --temperature-margin-c 2 --battery-margin-j 0.5 --actual-energy-multiplier 1.25
```

The optional runner flag `--actual-thermal-alpha VALUE` overrides the selected
scenario's actual thermal coefficient; omitting it preserves the scenario's
existing behavior. The energy multiplier defaults to 1.0. Controller thermal
parameters are not changed by either actual-only flag.

The new diagnostic command uses fixed scripted one-step actions. Energy cases
start at B_min + 1.75 J with a 0.5 J margin and idle nominal consumption of 1 J;
actual multipliers are 1, 1.25 and 2. The no-execution case starts at B_min + 0.5 J.
Thermal cases start at 79 C, use nominal alpha=6, a half-frequency heating input,
zero local work, and a 2 C margin. Actual alpha is 6, 8 or 20. These are deliberate
synthetic probes of the existing heating model, not workload/calibration evidence.
All seven use one device, horizon 1 and seed 0 in the reported run. Matching
seeds pair the exogenous traces, not necessarily the resulting state trajectories.

### Batch 2 validation actually performed

Base revision: `3ad14b3`. This execution environment initially lacked pytest;
the first full-suite attempt could not start. Installed pytest, Gymnasium and
Stable-Baselines3/Torch dependencies without changing project dependency files.
Before editing, the core baseline ran with
`python -m pytest -q --ignore=tests/test_ppo_smoke.py`: **155 passed in 18.37 s**.
After dependencies were ready, the untouched base revision was tested separately:
`python -m pytest -q`: **156 passed in 23.65 s**.
The final full suite: **195 passed in 21.91 s, zero failures or skips** (39 new
Batch 2 tests). The existing tiny PPO compatibility test is retained; no research
PPO method or experiment was added. `git diff --check` reported no whitespace issues.

The two diagnostic commands above were actually run with their output paths
replaced by `/workspace/scratch/7d054eb94a5c/batch2_checks/controlled` and
`/workspace/scratch/7d054eb94a5c/batch2_checks/runner`, respectively. Results:

| Controlled case | Signed error | Covered by configured margin? | Actual outcome |
| --- | --- | --- | --- |
| Energy matched | 0 J battery / energy | Yes | Ordinary limits pass |
| Energy covered | -0.25 J battery; +0.25 J energy | Yes, 0.5 J buffer | Ordinary limits pass |
| Energy exceeded | -1 J battery; +1 J energy | No | Reserve violated by 0.25 J |
| Thermal matched | 0 C | Yes | Ordinary limits pass |
| Thermal covered | About +0.7 C | Yes, 2 C buffer | Ordinary limits pass |
| Thermal exceeded | About +4.9 C | No | Thermal limit violated |
| No execution | null | null | Zero slots; no verified continuation |

All six executed controlled cases accepted the nominal action; the exceeded
cases terminated for actual simulated violation. The separate two-device,
three-slot runner check observed nominal acceptance for reference, two verified
corrections plus one verified fallback for thermal_mismatch, three verified idle
fallbacks for invalid_proposal, and zero executed slots for battery. The executed
runner slots had no actual violations and all queue conservation residuals were
zero. The battery attempt has null actual rates. These are exploratory development
checks; solver choices and timings can differ across machines. The same mismatch
need not cause a violation for every selected action, especially idle.

Software: Python 3.12.14; NumPy 2.3.5; SciPy 1.17.0; Gymnasium 1.4.0;
pytest 9.2.0; Torch 2.15.0; Stable-Baselines3 2.9.2; Matplotlib 3.10.8;
pandas 2.2.3. Each diagnostic snapshot includes effective configuration, initial
state, seed, Python version and core numerical package versions.

### Batch 2 changed files and remaining work

| File | Change |
| --- | --- |
| mlbac_stage1/model_uncertainty.py | Scalar validation and retrospective error/coverage calculations |
| mlbac_stage1/models.py | Actual-only component energy scaling and energy-error fields |
| mlbac_stage1/constraints.py | Per-device nominal energy in the checker report |
| mlbac_stage1/env.py | Hide actual settings, log separate provenance and executed-action errors |
| mlbac_stage1/experiment_logging.py | Per-device CSV rows and joint energy comparison fields |
| experiments_stage1.py | Optional mismatch settings, snapshots and error CSV |
| diagnostics_stage1.py | Add separate simulator config and error CSV to existing diagnostics |
| mismatch_diagnostics.py | Reproducible covered/exceeded one-step probes |
| tests/test_model_uncertainty.py | 39 regression, accounting, isolation, coverage and logging tests |
| .gitignore | Ignore the new generated-output folder |
| README.md | Batch 2 entry point and commands |
| docs/model_uncertainty_and_margins.md | Current model, equations, semantics, commands and evidence |

No commit/push was made. The delivery ZIP contains only these project files;
generated results are reproducible with the commands and are not attached.
Batch 3's full four-condition paired study, additional metrics, aggregation and
runtime deadlines remain unimplemented. Batch 4 will consolidate multi-seed
validation and final documentation. No physical safety, recursive feasibility,
long-term stability, recovery or real-time execution claim follows from Batch 2.
