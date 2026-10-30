# Batch B: nonlinear correction and verified idle

Based on commit a604339 (Batch A). No trained PPO is included.

## Formulation

Physical action order remains f, p, b, f_edge, l, o, U for each device.
The objective is one half of the weighted sum of squared changes divided by
configured physical action scales squared. The reference is the unchanged
finite decoded proposal, not the bounded solver starting point. All component
weights initially equal one and can be configured through projection_weights.
Action scales derive from effective hardware/resource/work limits; zero-range
components are fixed at zero and omitted from the objective and solver vector.
The environment's finite out-of-range raw policy values are still bounded by
its existing decoder; that preprocessing is recorded separately.

SLSQP searches locally in normalized variables. The problem is nonconvex.
No global nearest-action or infeasibility certificate is produced. A nominal
proposal already passing the checker is returned unchanged without solving.

Two auxiliary time fractions per device represent transmission time x and edge
execution time y. They are internal solver variables, not extra policy actions.
The search enforces rate*x >= d*o, f_edge*y >= c*o, and x+y <= delta.
Local timing uses f*delta >= c*l. These expressions avoid division by zero;
positive work with a missing execution resource still fails final verification.
The search's energy expression is kappa*c*l*f^2 + p*x + baseline*delta.
For nonnegative power, x upper-bounds the actual transmission duration, so this
energy is conservative. Any executable original action can choose the exact
transmission/edge times, giving the original energy constraint. The auxiliary
formulation thus represents the same physical feasible set in exact arithmetic,
apart from the explicitly documented inward shared-allocation target below.
The final checker computes actual required durations with the shared rate-floor
convention and recomputes the original physical energy model.

Residual normalization: work by max(Q_max,1 task-equivalent), arrivals by
max(A_max,1 task-equivalent), local cycles by max(f_max*delta,1 cycle),
transmitted bits by max(d*max(Q_max,1),1 bit), edge cycles by
max(c*max(Q_max,1),1 cycle), time by delta, battery budget by
max(abs(B-B_min),1 J), temperature by 100 C, shared resources by their capacity
(or 1 in their units if zero). Bounds on normalized actions and auxiliary times
are [0,1]. Battery upper bounds are implied by valid initial B<=B_cap and
nonnegative consumption; the final checker explicitly checks both bounds.
The endpoint temperature constraint is enforced by the search; initial-state
validation and the final peak checker additionally enforce the starting limit.

## Configuration and stopping

Defaults: solver_ftol=1e-10, solver_maxiter=200 per start,
solver_max_starts=2, solver_time_budget_s=2.0 for the combined search.
Starts are (1) bounded proposal with half-slot auxiliary times, and
(2) zero work/admission with small nonzero resources and quarter-slot times.
Every returned candidate, including one from a nonconverged run, is checked.
Among verified returned candidates, select the smallest original objective.
Neither starting-point clipping nor candidate selection overwrites the proposal.
Returned solver actions are not clipped or rounded after solving.

solver_allocation_guard=1e-10 asks the optimizer to stay this fraction inside
shared bandwidth/edge-CPU capacities to reduce roundoff at large Hz values.
It does not change the checker tolerances or provide a model-error margin.
It slightly restricts the solver search: 0.001 Hz bandwidth and 1 Hz edge CPU
at the reference capacities. It can be set to zero for sensitivity tests.

The time limit is COOPERATIVE: checked on Python objective/constraint evaluations,
callbacks, and after each solve. It cannot interrupt a native solver call while
that call is running. It is not a hard real-time guarantee. An over-budget solve's
result is discarded; earlier completed verified candidates can still be selected.
Timing includes all attempted solves and is available in the returned filter
result, SafetyFilter.last_result, and env.last_filter_result. Wall-clock timings
are intentionally not placed in Gymnasium step info, so repeated seeded API
checks can compare deterministic info. filter_demo writes them to CSV/JSON.

## Decision and failure semantics

state_status, proposal_status, solver_status, selected_solver_status,
filter status, and fallback_reason are separate. A malformed proposal can cause
fallback_verified while remaining labelled invalid_proposal. Its distance
objective is undefined and logged as null. Exceptions and time-budget exhaustion
are recorded rather than treated as infeasibility proofs.

Invalid states and already-unsafe states return no action from the filter and
never invoke backup. At the environment boundary, malformed physical states
raise ValueError before execution to avoid fabricating a valid Gym observation.
Already-unsafe finite physical states terminate the attempted step, preserve
physical time/state, and record already_unsafe plus the current violations.

Fallback construction delegates to the existing FallbackController. Idle keeps
queued work, rejects new admissions, does no processing and consumes baseline
energy for one slot. It is checked every time. It is not shutdown or a complete
recovery mechanism. Failure to verify idle yields no_verified_safe_continuation,
action=None and infeasibility_proven=False.

The environment has a last physical-unit verification gate. A false safe flag
cannot authorize execution. The verified action is executed unchanged. Actual
outcomes are still checked independently, including simulated model mismatch.

## Demonstrations and tests

    python -m pytest tests/test_core.py tests/test_regressions.py tests/test_reference_env.py tests/test_batch_a.py tests/test_batch_b.py -q
    python filter_demo.py --output filter_demo
    python diagnostics_stage1.py --scenario all --output diagnostics --seed 0
    python reference_demo.py --output reference_demo --seed 2026

filter_demo is a filter-only inspection, with no simulator execution. It writes
summary.txt, summary.csv and seven per-scenario JSON files containing original
and selected actions, config, objective, runtime, attempts, and verification.
Nonfinite diagnostic numbers serialize as null; explicit status/violation fields
preserve their meaning. It does not produce physical trajectory plots.

Expected cases: normal unchanged; 8+8 MHz corrected to approximately 5+5 MHz;
8+8 GHz corrected to approximately 5+5 GHz; bandwidth/timing corrected jointly;
synthetic thermal stress corrected; near-reserve battery no verified action;
invalid proposal recorded and replaced by independently verified idle.
Different solvers/platforms may take different iterations or produce different
local feasible corrections. Tests verify feasibility and known special-case
solutions, not exact nonlinear optimizer trajectories.

The reference demo still uses the same conservative scripted policy. Its
admission cap and reward are unchanged. Do not interpret its high rejection as
filter-imposed rejection or expect Batch B to train a better policy.

## Limits and next milestone

Current temperature model is uncalibrated and endpoint-based. Model-error margins
remain zero. A safe state can have no safe next action. No physical safety,
recursive feasibility, recovery, or global nearest-action guarantee is claimed.
Full per-slot CSV integration of the new optimizer diagnostics, richer plots,
service thresholds and systematic load/model-error sweeps remain Batch C work.
The PPO/SB3 smoke test was subsequently run after installing the dependencies;
see the numerical-state fix validation below.

## Preparation validation

103 selected tests passed on the final implementation. Gymnasium environment
check passed (only its optional render-mode warning). All three existing
diagnostics and the 300-slot reference demo ran successfully. The reference
summary exactly matched the committed seed-2026 summary. The thermal diagnostic
retained 0.05 task-equivalents of local work and ended at approximately 80 C
(the computed value exceeded 80 by approximately 2.45e-10 C, within the explicit
1e-9 C checker tolerance). This is numerical feasibility, not robust safety.

## Numerical-state consistency fix

The next-queue lower bound is now strictly nonnegative in predicted verification,
direct physical execution, and actual-outcome verification, consistent with
validate_state. Previously a queue of -5e-10 could pass a 1e-9 work tolerance and
then be rejected at the next environment step. Candidates producing such a
negative queue are now rejected; queues are not silently clipped and workload
conservation is preserved. Other configured tolerances are unchanged.

The full suite including PPO learn()/predict() now passes: 108 tests.
Four tests in tests/test_numerical_state.py cover tiny negative queues, exact zero,
actual-outcome detection and a rejected solver candidate followed by another step.
