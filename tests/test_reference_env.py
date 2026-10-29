import csv
import json

import numpy as np
import pytest

from mlbac_stage1.config import make_default_params
from mlbac_stage1.constraints import evaluate_constraints
from mlbac_stage1.env import make_env
from mlbac_stage1.models import apply_transition, compute_rates
from reference_demo import run_reference_demo


def _policy(n_devices: int, *, f=0.0, p=0.0, b=0.0, f_edge=0.0, l=0.0, o=0.0, U=0.0) -> np.ndarray:
    ratios = np.tile(np.array([f, p, b, f_edge, l, o, U], dtype=float), (n_devices, 1))
    return (2.0 * ratios - 1.0).astype(np.float32).reshape(-1)


def _trace(seed: int, frequency_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    env = make_env(n_devices=2, horizon=5)
    env.reset(seed=seed, options={"state": {"Q": np.zeros(2)}})
    policy = _policy(2, f=frequency_ratio)
    arrivals, channels = [], []
    for _ in range(4):
        _, _, terminated, truncated, info = env.step(policy)
        assert not terminated and not truncated
        arrivals.append(info["pre_state"]["A"].copy())
        channels.append(info["pre_state"]["g"].copy())
    return np.asarray(arrivals), np.asarray(channels)


def test_reference_values_units_dimensions_and_initial_conditions():
    params = make_default_params()
    assert params["n_devices"] == 5
    assert params["n_edge_servers"] == 1
    assert params["horizon"] == 300
    assert params["delta"] == 1.0
    assert params["f_hw_max"][0] == 1e9
    assert params["B_cap"][0] == 32_400.0
    assert params["B_min"][0] == 3_240.0
    assert params["T_max"][0] == 80.0
    assert params["Q_max"][0] == 10.0
    assert params["d"][0] == 1_600_000.0
    assert params["c"][0] == 40e6
    assert params["ambient_temp"] == 25.0
    assert params["shared_bandwidth_hz"] == 10e6
    assert params["shared_edge_cpu_hz"] == 10e9

    env = make_env()
    obs, info = env.reset(seed=17)
    assert env.action_space.shape == (35,)
    assert env.observation_space.shape == (28,)
    np.testing.assert_allclose(env.state["Q"], np.ones(5))
    np.testing.assert_allclose(env.state["B"], np.full(5, 25_920.0))
    np.testing.assert_allclose(env.state["T"], np.full(5, 30.0))
    assert info["effective_config"]["horizon"] == 300
    assert info["seed"] == 17
    assert obs.shape == (28,)


def test_custom_one_device_initial_conditions_and_timing():
    params = make_default_params(
        1,
        deterministic=True,
        horizon=8,
        delta=0.5,
        initial_queue_work_units=2.0,
        initial_battery_fraction=0.5,
        initial_temperature_c=35.0,
    )
    env = make_env(params=params)
    env.reset(seed=5)
    assert env.horizon == 8
    assert env.action_space.shape == (7,)
    assert env.observation_space.shape == (8,)
    assert env.state["Q"][0] == pytest.approx(2.0)
    assert env.state["B"][0] == pytest.approx(16_200.0)
    assert env.state["T"][0] == pytest.approx(35.0)


def test_admission_action_scale_follows_configured_arrival_bound():
    params = make_default_params(
        1,
        deterministic=True,
        arrival_min_per_slot=0.5,
        arrival_max_per_slot=3.0,
        deterministic_arrival_per_slot=1.5,
    )
    env = make_env(params=params)
    env.reset(seed=2)
    assert env.state["A"][0] == pytest.approx(1.5)
    assert env._decode_action(np.ones(7))["U"][0] == pytest.approx(3.0)


def test_same_seed_reproduces_trace_independently_of_policy_actions():
    first = _trace(31, 0.0)
    second = _trace(31, 0.6)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    env_a = make_env(n_devices=2, horizon=4)
    env_b = make_env(n_devices=2, horizon=4)
    obs_a, _ = env_a.reset(seed=88)
    obs_b, _ = env_b.reset(seed=88)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_different_seeds_change_arrival_and_channel_traces_and_bounds_hold():
    arrivals_a, channels_a = _trace(31, 0.0)
    arrivals_b, channels_b = _trace(32, 0.0)
    assert not np.array_equal(arrivals_a, arrivals_b)
    assert not np.array_equal(channels_a, channels_b)
    assert np.all((arrivals_a >= 0.0) & (arrivals_a < 2.0))
    assert np.all((channels_a >= 5e-6) & (channels_a < 2e-5))


def test_current_slot_channel_is_observed_and_used_for_execution():
    env = make_env(n_devices=1, horizon=2)
    obs, _ = env.reset(seed=7, options={"state": {"Q": np.array([1.0]), "A": np.array([0.0])}})
    action = _policy(1, p=1.0, b=1.0, f_edge=1.0, o=0.01)
    _, _, _, _, info = env.step(action)
    gain_t = info["pre_state"]["g"]
    executed = info["executed_action"]
    expected_rate = compute_rates(gain_t, executed["b"], executed["p"], env.params["N0"])
    np.testing.assert_allclose(info["transition_metrics"]["rates"], expected_rate)
    assert obs[4] * env.params["observation_channel_gain_scale"] == pytest.approx(gain_t[0])
    next_obs, _, _, _, _ = env.step(_policy(1))
    assert next_obs[4] * env.params["observation_channel_gain_scale"] == pytest.approx(env.state["g"][0])


def test_new_admission_waits_until_next_slot_and_rejection_is_recorded():
    params = make_default_params(1, deterministic=True, deterministic_arrival_per_slot=2.0)
    env = make_env(params=params, horizon=2)
    env.reset(seed=1, options={"state": {"Q": np.array([0.0])}})
    _, _, _, _, info = env.step(_policy(1, U=0.75))
    assert info["pre_state"]["Q"][0] == pytest.approx(0.0)
    assert info["transition_metrics"]["completed_local_work"][0] == pytest.approx(0.0)
    assert info["transition_metrics"]["completed_offload_work"][0] == pytest.approx(0.0)
    assert info["transition_metrics"]["rejected_arrivals"][0] == pytest.approx(0.5)
    assert info["post_state"]["Q"][0] == pytest.approx(1.5)


def test_documented_queue_and_rejection_conservation_example():
    params = make_default_params(1, deterministic=True)
    state = {
        "Q": np.array([4.0]),
        "B": np.array([25_920.0]),
        "T": np.array([30.0]),
        "A": np.array([2.0]),
        "g": np.array([1e-5]),
    }
    action = {
        "f": np.array([40e6]),
        "p": np.array([1.0]),
        "b": np.array([10e6]),
        "f_edge": np.array([200e6]),
        "l": np.array([1.0]),
        "o": np.array([2.0]),
        "U": np.array([1.5]),
    }
    report = evaluate_constraints(state, action, params)
    assert report["feasible"]
    next_state = apply_transition(state, action, params)
    assert next_state["Q"][0] == pytest.approx(2.5)
    assert next_state["metrics"]["rejected_arrivals"][0] == pytest.approx(0.5)


def test_episode_horizon_elapsed_time_and_no_recharge():
    env = make_env(n_devices=1, horizon=3, deterministic=True)
    env.reset(seed=3, options={"state": {"Q": np.zeros(1)}})
    initial_battery = env.state["B"].copy()
    for slot in range(3):
        _, _, terminated, truncated, info = env.step(_policy(1))
        assert terminated is (slot == 2)
        assert not truncated
        assert info["elapsed_time_s"] == pytest.approx(slot + 1)
    np.testing.assert_allclose(env.state["B"], initial_battery - 3.0)


def test_failure_before_execution_does_not_advance_time():
    params = make_default_params(1, deterministic=True, horizon=4)
    params["B_min"][:] = params["B_cap"] * 0.8 - 0.5
    env = make_env(params=params)
    env.reset(seed=9, options={"state": {"Q": np.array([0.5]), "B": params["B_min"] + 0.5}})
    _, _, terminated, truncated, info = env.step(_policy(1))
    assert terminated and not truncated
    assert info["elapsed_time_s"] == pytest.approx(0.0)
    assert info["slot"] == 0
    assert env.timestep == 0
    assert info["executed_action"] is None
    assert info["actual_outcome"] == "unobserved"


def test_reference_demo_outputs_and_episode_conservation(tmp_path):
    output = run_reference_demo(tmp_path / "reference", seed=2026)
    with (output / "summary.json").open() as handle:
        summary = json.load(handle)
    with (output / "config_snapshot.json").open() as handle:
        snapshot = json.load(handle)
    with (output / "episode.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["attempted_slots"] == summary["executed_slots"] == 300
    assert summary["elapsed_simulated_seconds"] == pytest.approx(300.0)
    assert summary["episode_ending_reason"] == "finite_horizon_complete"
    assert summary["offered_work_units"] == pytest.approx(summary["admitted_work_units"] + summary["rejected_work_units"])
    assert summary["total_completed_work_units"] == pytest.approx(
        summary["completed_local_work_units"] + summary["completed_offload_work_units"]
    )
    assert summary["initial_queue_work_units"] == pytest.approx(5.0)
    assert summary["final_queue_work_units_total"] == pytest.approx(sum(summary["remaining_queue_work_units"]))
    assert summary["queue_conservation_residual_work_units"] == pytest.approx(0.0, abs=1e-10)
    assert summary["filter_status_counts"] == {"nominal_accepted": 300}
    assert summary["scripted_policy_queue_work_margin_fraction"] == pytest.approx(0.01)
    assert sum(summary["remaining_queue_work_units"]) == pytest.approx(
        5.0
        + summary["admitted_work_units"]
        - summary["completed_local_work_units"]
        - summary["completed_offload_work_units"]
    )
    assert len(rows) == 5 * 300
    assert summary["offered_work_units"] == pytest.approx(sum(float(row["A_t_offered_work_units"]) for row in rows))
    assert summary["admitted_work_units"] == pytest.approx(sum(float(row["admitted_U_work_units"]) for row in rows))
    assert summary["completed_local_work_units"] == pytest.approx(sum(float(row["completed_local_work_units"]) for row in rows))
    assert summary["completed_offload_work_units"] == pytest.approx(sum(float(row["completed_offload_work_units"]) for row in rows))
    assert summary["rejected_work_units"] == pytest.approx(sum(float(row["rejected_A_minus_U_work_units"]) for row in rows))
    assert snapshot["seed"] == 2026
    assert snapshot["effective_config"]["n_devices"] == 5
    assert snapshot["effective_config"]["horizon"] == 300
    assert "provisional_uncalibrated" in snapshot["effective_config"]["provisional_parameter_labels"].values()
    queue_caps = snapshot["effective_config"]["Q_max"]
    for row in rows:
        device = int(row["device"])
        q_expected = float(row["pre_Q_work_units"]) - float(row["executed_l"]) - float(row["executed_o"]) + float(row["executed_U"])
        assert float(row["actual_Q_next_work_units"]) == pytest.approx(q_expected)
        assert float(row["pre_Q_work_units"]) <= queue_caps[device] + 1e-9
        assert float(row["actual_Q_next_work_units"]) <= queue_caps[device] + 1e-9
    for device, final_queue in enumerate(summary["remaining_queue_work_units"]):
        device_rows = [row for row in rows if int(row["device"]) == device]
        device_balance = (
            snapshot["initial_state"]["Q"][device]
            + sum(float(row["admitted_U_work_units"]) for row in device_rows)
            - sum(float(row["completed_local_work_units"]) for row in device_rows)
            - sum(float(row["completed_offload_work_units"]) for row in device_rows)
        )
        assert final_queue == pytest.approx(device_balance)
    for name in ("state_trajectories.png", "workload_flows.png", "shared_resource_utilization.png"):
        assert (output / name).is_file()


@pytest.mark.parametrize('deterministic', [False, True])
def test_zero_arrival_bound_has_finite_observations(deterministic):
    params = make_default_params(
        2, horizon=3, deterministic=deterministic,
        arrival_max_per_slot=0.0, deterministic_arrival_per_slot=0.0,
    )
    env = make_env(params=params)
    with np.errstate(divide='raise', invalid='raise'):
        obs, _ = env.reset(seed=19)
        assert np.isfinite(obs).all()
        np.testing.assert_array_equal(obs[6:8], np.zeros(2))
        for _ in range(3):
            obs, _, _, _, _ = env.step(_policy(2))
            assert np.isfinite(obs).all()
            np.testing.assert_array_equal(obs[6:8], np.zeros(2))
    env.close()


@pytest.mark.parametrize('settings', [
    {'arrival_max_per_slot': 0.2},
    {'channel_gain_min': 1e-7, 'channel_gain_max': 1e-6},
])
def test_stochastic_bounds_ignore_unused_deterministic_defaults(settings):
    params = make_default_params(2, horizon=3, **settings)
    env = make_env(params=params)
    obs, _ = env.reset(seed=19)
    for _ in range(3):
        assert np.isfinite(obs).all()
        assert np.all(env.state['A'] >= params['arrival_min_per_slot'])
        assert np.all(env.state['A'] <= params['arrival_max_per_slot'])
        assert np.all(env.state['g'] >= params['channel_gain_min'])
        assert np.all(env.state['g'] <= params['channel_gain_max'])
        obs, _, _, _, _ = env.step(_policy(2))
    env.close()
    # Deterministic mode must still reject its out-of-range default value.
    with pytest.raises(ValueError, match='deterministic'):
        make_default_params(deterministic=True, **settings)


@pytest.mark.parametrize('settings', [
    {'arrival_min_per_slot': -0.1},
    {'arrival_min_per_slot': 3.0, 'arrival_max_per_slot': 2.0},
    {'channel_gain_min': -1e-7},
    {'channel_gain_min': 2e-5, 'channel_gain_max': 1e-5},
])
def test_invalid_stochastic_bounds_remain_rejected(settings):
    with pytest.raises(ValueError, match='bounds'):
        make_default_params(**settings)
