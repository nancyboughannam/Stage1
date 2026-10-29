# Stage 1 Environment Specification

## Topology and Reference Scenario

The simulator represents $N$ battery-powered IoT devices sharing one edge server. Each device has a local queue, battery, temperature, arrival workload, and wireless channel power gain. The reference defaults are $N=5$, one edge server, a 1-second slot, and a 300-slot finite-horizon episode (300 simulated seconds). Device count, slot duration, horizon, and initial conditions are configurable through `make_default_params`; pass its result to `make_env(params=...)`. Existing one-device diagnostics explicitly use one-device parameter dictionaries.

| Reference value | Default |
| --- | ---: |
| Local CPU maximum | $10^9$ cycles/s/device |
| Battery capacity | 32,400 J/device = 9 Wh |
| Battery reserve | 3,240 J/device = 10% capacity |
| Temperature limit | 80 C |
| Queue capacity | 10 task-equivalents/device |
| Input size per task-equivalent | 1,600,000 bits = 0.2 decimal MB |
| CPU work per task-equivalent | 40,000,000 cycles |
| Ambient temperature | 25 C |
| Shared bandwidth capacity | $10^7$ Hz |
| Shared edge CPU capacity | $10^{10}$ cycles/s |
| Initial queue | 1 task-equivalent/device |
| Initial battery | 25,920 J/device = 80% capacity |
| Initial temperature | 30 C/device |

Hardware is identical across reference devices initially. Battery does not recharge during an episode. These are scenario definitions, not measurements from a particular device.

## State and Observation

At the beginning of slot $t$, each device's physical state is $(Q_t,B_t,T_t,A_t,g_t)$:

- $Q_t$: queued task-equivalents eligible for work this slot.
- $B_t$: battery energy in joules.
- $T_t$: device temperature in Celsius.
- $A_t$: divisible offered workload in task-equivalents for this slot.
- $g_t$: dimensionless channel power gain held constant during this slot.

The `float32` observation is grouped feature-major in this order: all $Q_t/Q_{max}$; all $B_t/B_{cap}$; all $T_t/(100\ C)$; all $A_t/A_{max}$; each device's $g_t/(10^{-5})$; shared bandwidth divided by $10^7$ Hz; shared edge CPU divided by $10^{10}$ cycles/s; remaining slots divided by the horizon. When a device's arrival bound is zero, its normalized arrival feature is zero. Its length is $5N+3$ (28 for five devices). State features are scaled but not clipped at safety limits. Raw physical states are included in environment info and demonstration CSVs. The observed $g_t$ is the same value used by the filter and transition for slot $t$.

`reset()` returns the actual seed, JSON-safe effective configuration, and raw initial physical state. The reference demonstration saves this configuration and seed in `config_snapshot.json`.

## Action Vector and Constraints

The policy action is a normalized vector in $[-1,1]$ with $7N$ values. It is flattened device-major; each device uses the existing order:

`[f, p, b, f_edge, l, o, U]`

- `f`: local CPU frequency, cycles/s.
- `p`: transmission power, watts.
- `b`: allocated bandwidth, Hz.
- `f_edge`: allocated edge CPU, cycles/s.
- `l`: queued work completed locally this slot, task-equivalents.
- `o`: queued work transmitted and completed at edge this slot, task-equivalents.
- `U`: newly offered work admitted, task-equivalents.

A normalized component $x\in[-1,1]$ maps to $(x+1)/2$ times its configured physical maximum and the optional global `action_scale`. `l` and `o` scale against queue capacity; `U` scales against `A_max`, which is set from the configured maximum arrival amount per slot. Raw policy values, decoded proposal, selected filter candidate, and executed action remain separately available in step info/logs.

The existing model/filter checks nonnegative work; $l_t+o_t\leq Q_t$; $0\leq U_t\leq A_t$; per-device hardware limits; $\sum_i b_i\leq 10^7$ Hz; and $\sum_i f_{edge,i}\leq10^{10}$ cycles/s. Positive local work requires positive CPU and must finish during the slot. Positive offload requires a positive radio path and edge CPU; transmission followed by edge execution must fit in the slot. Local CPU and the offload branch can run in parallel. There is no carry-over edge queue. The existing safety filter is a heuristic action-scaling search, not an optimization solver.

## Arrivals, Channels, and Random Seeds

Arrival quantities are continuous and divisible, not integer counts of indivisible jobs. For each device and slot, the reference draws an independent uniform sample from $[0,2)$ task-equivalents per slot. The mean is 1 task-equivalent/slot/device; only at the reference 1-second slot does that equal a mean 1 task-equivalent/s/device. Configurable arrivals are expressed per slot; changing slot duration does not silently preserve a per-second workload rate.

Each reference device independently draws a channel gain from the synthetic development distribution $[5\times10^{-6},2\times10^{-5})$. It is held for the slot and redrawn for the next slot. This distribution is not a calibrated propagation model. Deterministic mode retains constant, configurable arrivals and channel gains for diagnostics.

`reset(seed=...)` derives separate environment-owned random streams for arrivals and channels, then precomputes the episode traces. Thus action selection and filter iterations cannot alter the exogenous trace. The same configuration/seed reproduces the initial observation and traces; a different seed changes the stochastic traces. This enables future matched-trace comparisons.

## Slot Sequence and Conservation

At the start of slot $t$, the policy observes $Q_t,B_t,T_t,A_t,g_t$ and proposes an action. The filter evaluates it against the current state and channel. During the slot only work already in $Q_t$ is eligible; local/offloaded work executes, and the existing energy/thermal equations advance. At the end:

$$Q_{t+1}=Q_t-l_t-o_t+U_t$$

$$B_{t+1}=B_t-E_t$$

Unaccepted arrivals $A_t-U_t$ are logged as rejected work. Admitted $U_t$ becomes eligible only in slot $t+1$. Then the next $A_{t+1},g_{t+1}$ are read from the precomputed traces.

Example:

```text
Q_t = 4
A_t = 2
l_t = 1
o_t = 2
U_t = 1.5
Q_(t+1) = 4 - 1 - 2 + 1.5 = 2.5
Rejected work = A_t - U_t = 0.5
```

Remaining queue at episode end is reported as backlog, never counted as completed work.

## Episode and Failure Semantics

The horizon is a finite-horizon task terminal condition: the last successful slot returns `terminated=True, truncated=False`; it is not represented as a time-limit truncation. Remaining horizon is included in observations.

The outcomes are distinct:

- `finite_horizon_complete`: the last configured slot completed.
- `controller_failure_no_verified_continuation`: no one-step candidate was found. Nothing is executed; physical state, slot index, and elapsed physical time do not advance. Actual post-state is unobserved, and outstanding queue/arrival work plus a negative failure reward are recorded. This heuristic search failure is not proof of mathematical infeasibility, and stopping is not evidence of physical recovery.
- `actual_constraint_violation`: an action passed prediction and executed, then actual post-transition checks found a queue, battery-reserve, or endpoint-temperature violation.

Verification applies to one step only, not the remaining horizon.

## Provisional Models and Limits

Reference scenario values are distinct from retained provisional model values. The existing 1 W transmission-power maximum, $\kappa=10^{-26}$ CPU-energy coefficient, 1 W baseline power, $N_0=10^{-12}$ W/Hz noise parameter, and thermal coefficients ($\alpha=0.15$, $\tau=12$ s) remain provisional and uncalibrated. They are labelled in the effective configuration snapshot. Energy equations are unchanged: $E_{local}=\kappa f^3t_{local}$, $E_{tx}=pt_{tx}$, and $E=E_{local}+E_{tx}+P_{baseline}\Delta$; no recharge term is used.

The existing endpoint thermal equation is unchanged:

$$T_{t+1}=T_{ambient}+(T_t-T_{ambient})e^{-\Delta/\tau}+\alpha h\Delta,$$

where $h=0.7f/f_{max}+0.3p/p_{max}$ is dimensionless, $\Delta$ and $\tau$ are seconds, and $\alpha$ is C/s per normalized load. At constant maximum legal heating the discrete equilibrium is $T_{ambient}+\alpha\Delta/(1-e^{-\Delta/\tau})$, approximately 26.88 C at 25 C ambient under defaults. The default legal heating therefore does not challenge the 80 C cap. Peak is estimated as `max(start, endpoint)`; continuous within-slot transients are not resolved. None of these parameters is hardware-validated. The radio, queue, energy, and thermal models require empirical calibration. Passing tests or demonstrations does not establish formal guarantees or physical safety.

## Commands and Outputs

From the repository root with `.venv` active:

```bash
python -m pytest -q
python diagnostics_stage1.py --scenario all --output diagnostics --seed 0
python reference_demo.py --output reference_demo --seed 2026
```

Existing one-step normal/thermal/battery outputs remain under `diagnostics/`. The reference run writes `config_snapshot.json`, `episode.csv`, `summary.json`, `summary.txt`, `state_trajectories.png`, `workload_flows.png`, and `shared_resource_utilization.png` under `reference_demo/`.

The reference arrivals are a light development workload. At 1 GHz and 40 million cycles/task-equivalent, theoretical local CPU capacity is 25 task-equivalents/s before energy and thermal restrictions. The scripted reference policy is neither trained PPO nor optimized, and the run is not sufficient evidence of filter effectiveness or maximum safe load.

For seed 2026, the CSV reports 1,489.632965 offered, 657.148149 admitted, 439.707499 completed locally, 219.853749 completed by offloading, 832.484816 rejected, and 2.586901 task-equivalents remaining in queues. Completed work totals 659.561248. Offered work balances as admitted plus rejected, and queued work balances as initial queue plus admissions minus completions. All recorded device queues remain below the configured capacity of 10.

The scripted policy caps admissions at 0.5 task-equivalents/device/slot while the offered-arrival mean is 1.0, so about 56% of offers are rejected by the policy choice. For the same seed, all 300 actions are labeled `nominal_accepted`; the rejection is not caused by the safety filter. The policy also requests 99% of its chosen local/offload workload amount. That 1% slack is only to avoid float32 normalized-action decoding roundoff: exact `l+o=Q` proposals can exceed the queue by several tens of nanounits and activate the existing constraint. No constraint tolerance or safety limit is weakened.
