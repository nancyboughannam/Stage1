# Batch A: verification foundation

Based on repository commit 896ab2c37674cad8d1d18d12ccd81431a5d5a010.

Replace mlbac_stage1/config.py, mlbac_stage1/models.py,
mlbac_stage1/constraints.py and tests/test_core.py. Add tests/test_batch_a.py
and this document. Copy files into the existing directories, not over the
whole directories. No new dependencies are required.

## Scope

The authoritative checker remains evaluate_constraints(state, action, params).
Existing valid-input report fields are retained. Additional fields include
input_status, checker_tolerances, constraint_residuals, and predicted_next_state.
Residuals are signed slack in the named physical units: negative means a
constraint is exceeded, subject to that constraint's numerical tolerance.
Missing resources with positive work produce infinite duration and negative
infinite timing slack. They cannot pass verification.

Malformed inputs return feasible=False, input_status=invalid_state or
invalid_proposal, and predicted_state=None. Valid physical states already
outside safety limits return already_unsafe with no prediction. No action is
modified by the checker. Physical state validation takes precedence over
proposal validation. Temperature below absolute zero is physically invalid;
negative Celsius temperatures above absolute zero are permitted.

Timing and energy equations are shared between the checker and simulator.
The simulator still permits physically executable actions that violate a
state safety limit, so actual violations can be measured. It does not use the
predicted-safety checker to suppress actual thermal model mismatch.

The checker uses named absolute tolerances (work units, joules, degrees C,
seconds, Hz, watts). Defaults preserve the prior 1e-9 thresholds. Nonnegative
action bounds and positive-resource requirements remain strict. These are
numerical choices, not physical error bounds. Model margins are zero for this
milestone; the thermal model remains provisional and the peak is endpoint-based.

The config helper action_metric_settings(params) derives scales and unit
weights from effective parameters, including scenario overrides. Its active
masks identify nonzero-range variables. Batch B must use those masks rather
than divide by a zero scale. There is no optimizer or solver tolerance setting
yet: unused solver controls and nonzero model-margin modes are deliberately
not exposed in Batch A.

## Integration boundary

The existing heuristic filter and environment are not rewritten in this batch.
The new input statuses are checker-level results. Routing malformed proposals
or already-unsafe states through filter/episode statuses is a Batch B/C task;
do not treat this batch as complete end-to-end malformed-input handling.
Idle is still only a verified one-slot backup. No recovery, infeasibility proof,
physical-safety guarantee or global nearest-action guarantee is added.

## Validation

Run from the repository root in the existing virtual environment:

    python -m pytest tests/test_core.py tests/test_regressions.py tests/test_reference_env.py tests/test_batch_a.py -q
    python diagnostics_stage1.py --scenario all --output diagnostics --seed 0
    python reference_demo.py --output reference_demo --seed 2026

The first command passed 68 tests during preparation. Gymnasium's environment
checker passed. PPO/SB3 training tests were not run in the preparation environment.
The old core test dictionaries now explicitly include A_max and N0, matching
the full environment configuration. New tests cover shared physics, mutation
avoidance, malformed proposals and states, unavailable execution resources,
joint resource limits, queue lower and upper limits, reserve exhaustion,
thermal mismatch, configured tolerances and zero-range scales.

Solver failure injection and projection-optimality tests belong to Batch B.
