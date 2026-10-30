# Batch C: auditable experiments and plots

This milestone adds `experiments_stage1.py` and `mlbac_stage1/experiment_logging.py`.
It does not change the filter, fallback, reward, physical dynamics, or PPO training.
The original diagnostics and reference demo commands continue to work.

## Start here

```bash
python -m pytest -q
python experiments_stage1.py --output experiments_batch_c --horizon 30 --seeds 0 1 2
```

The default is five devices and one edge server. Seven scenarios are run with each
of three seeds. A smaller inspection run is:

```bash
python experiments_stage1.py --output experiments_quick --horizon 5 --seeds 0
```

Each run writes to `SCENARIO/load_1/seed_0/` (and the other seeds):

- `states.png`: per-device Q/B/T, configured thresholds, and predicted endpoints
  marked with black crosses. Solid curves are simulated outcomes. Initial state
  is included. No unexecuted post-state is invented.
- `actions.png`: all seven physical action components. Dashed lines are decoded
  proposals; solid lines are executed actions. Missing execution is blank.
- `service_and_filter.png`: offered/admitted/completed/rejected work, proposed vs
  executed admissions, measured filter runtime, and cumulative consumed energy.
- `slots.csv`: one row per attempted **joint slot**, not per device. Statuses,
  solver runtime, correction size, work/energy, and actual per-constraint device
  counts and maximum magnitudes are recorded. Missing values are blank.
- `steps.json`: complete per-device states and raw/decoded/selected/executed
  actions, proposal and selected prediction reports (all constraint residuals),
  actual reports, candidate checks, solver attempts, and fallback reasons.
- `summary.json`: service totals, failure/intervention/fallback frequencies,
  state-constraint violation rates, maximum magnitudes, runtime percentiles,
  conservation residual, and provisional service assessment.
- `config_snapshot.json`: effective controller settings, numerical tolerances,
  separate simulator-only thermal override, seed, criteria, dependency versions.

The root `aggregate.json` reports means and approximate 95% Student-t intervals
across seeds. A single valid episode has **no interval**. The independent sampling
unit is an episode/seed, not a slot or device. Three seeds give exploratory evidence,
not a reliable publication benchmark. Intervals are not clipped to [0,1].

## Scenarios

| Scenario | Purpose |
|---|---|
| reference | Original modest scripted proposal; nominal acceptance expected |
| resource_pressure | Each device proposes 80% of shared BW and edge CPU; test joint correction |
| weak_channel | Observed channel range is reduced to [5e-8, 2e-7]; environmental perturbation |
| thermal_matched | Synthetic coefficient 60, initial T=79 C, proposed CPU=90%; model agrees with simulator |
| thermal_mismatch | Same controller, but simulator coefficient 90, hidden from filter; expose prediction/actual disagreement |
| battery | Start B=B_min+0.5 J; even idle uses 1 J; no verified execution expected |
| invalid_proposal | A NaN local-CPU proposal; skip optimization and independently verify idle |

All policies here are scripted, not trained PPO. The reference admission cap stays
at 0.5 work/device/slot, so high rejection may be caused by the proposal itself.
The thermal coefficients are uncalibrated synthetic stress settings. Weak channel
is observed before filtering; it is not an unmodelled within-slot channel error.
The thermal mismatch is a sensitivity demonstration, not robust safety validation.
No model-error safety margin is introduced. The zero-work thermal model remains
provisional. The suite also tests an injected solver timeout.

## Metric definitions and limits

- Throughput = completed local+offloaded work / executed simulated seconds.
  This can include work in the initial queue. It is not the fraction of new arrivals served.
- Rejection fraction = rejected arrivals / offered arrivals in **executed slots**.
  Arrivals at an unexecuted attempt are reported separately, never labelled rejected.
- With no execution, throughput, rejection fraction, actual violation rates and
  magnitudes are undefined (JSON null / CSV blank), not zero.
- Actual state-constraint violation rate = violating device-slots / executed
  device-slots. Queue, battery and thermal flags use the configured checker
  tolerances. Raw excess magnitude may be positive within acceptance tolerance.
- Shared bandwidth/CPU and timing are allocation/execution constraints; their
  predicted residuals are in `steps.json`. They are not inferred from state violations.
- Intervention and fallback frequencies use attempted **joint slots**. Invalid
  proposals are distinguished from normal optimization via `proposal_status`.
- The sum of all device queues must equal initial work + admissions - completions;
  `queue_balance_residual_work` checks this balance.
- Early-terminated runs have shorter exposure. Compare their completion flags and
  ending reasons alongside throughput, rates and energy, never averages alone.
- Correction distance and objective are normalized values from Batch B; they are
  not causal performance loss. Admission change compares the decoded proposal
  with execution, but an infeasible proposal is not a realizable baseline.
- Runtime is measured on the host and can vary. Simulated time does not include
  optimization delay. The solver time budget is cooperative, not a real-time bound.
- JSON nonfinite input/residual values become null; invalid-input status and
  feasibility reports preserve the distinction from a valid value.

## Provisional service criteria, fixed before running

Default requirements: full requested horizon, zero observed state-constraint
violations, rejection fraction <= 0.20, throughput >= 0.5 aggregate work units/s.
These are transparent development choices, **not advisor-approved service targets**.
Change them explicitly with `--max-rejection` and `--min-throughput`.
The full-horizon/zero-violation requirements cannot be disabled by those options.

Optional exploratory load sweep:

```bash
python experiments_stage1.py --scenarios reference resource_pressure --loads 0.5 1 1.5 --seeds 0 1 2 --horizon 30 --output load_exploration
```

Arrivals are uniform [0, 2*load), hence mean `load` work/device/slot; loads must be
in (0,2]. To avoid changing action normalization during this sweep, this runner
uses A_max=4 for **all** loads. This is recorded and differs from the original
reference demo's A_max=2; do not mix results as though configurations were identical.
Seeds pair exogenous random draws across scenarios (weak-channel values are scaled).
The suite does **not** label any load as a maximum safe load. Such a claim needs
agreed service targets, adequate seeds/horizon, meaningful policies/baselines,
and uncertainty/termination analysis.

## Next research work

Train and evaluate PPO with controlled seeds and matched workload traces; establish
comparison policies; calibrate thermal dynamics and error bounds; design recovery
beyond idle; then study the finite-horizon operating region. Latency is not reported
as measured task delay: the current aggregate queue has no task timestamps.

## Delivery verification

Built against pushed Batch B commit `1ea61dc` (all 13 Batch B files matched the
previously tested delivery). Full test suite: 119 passed, including PPO smoke.
The new tests cover missing execution, metric denominators, conservation, service
criteria, invalid proposals, injected timeout, JSON trace reconstruction, common
random traces, and hidden thermal mismatch.

A five-device run of all seven scenarios, 30 slots and seeds 0/1/2 was completed.
Reference, resource-pressure, weak-channel, matched-thermal and invalid-proposal
runs reached the horizon; mismatched thermal runs ended on an actual violation;
battery runs executed zero slots. All 455 executed joint actions were independently
rechecked from saved traces against predicted constraints. This is simulator
validation, not evidence of physical safety or PPO learning quality.
