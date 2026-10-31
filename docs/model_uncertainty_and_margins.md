# Model uncertainty and safety margins — Batch 1

Status: margin filtering implemented and tested. Energy mismatch, error coverage,
paired comparison orchestration, expanded aggregates and deadline reporting are
scheduled for later batches. Historical Batch A–C documentation is unchanged.

## Purpose

Study whether explicit prediction buffers reduce simulated violations, and what
service and runtime they cost. This first batch establishes consistent margin
checks; it does not establish the effectiveness of any chosen error bound.

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
No new energy multiplier is implemented in Batch 1.

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
2. Batch 2: simulator-only energy multiplier, prediction errors, margin coverage,
   controlled covered/exceeded-error tests; retain thermal-model separation.
3. Batch 3: paired four-condition experiments, boundary cases, complete service/
   safety metrics, per-episode aggregates and runtime deadline reporting.
4. Batch 4: final integrated validation, reproducibility evidence and documentation.

Open research work remains thermal/energy calibration, validated recovery or
shutdown, feasible-state/sufficient-condition analysis, and task-latency tracking.
PPO and PPO-Lagrangian as research methods remain outside this milestone. The
existing tiny PPO compatibility smoke test is retained in the full test suite.

Actual Batch 1 results and exact commands are in [batch1_report.md](batch1_report.md).
