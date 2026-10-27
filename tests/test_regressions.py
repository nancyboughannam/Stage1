import csv
import json

import numpy as np
import pytest

from mlbac_stage1.config import make_default_params
from mlbac_stage1.constraints import evaluate_constraints, evaluate_post_transition
from mlbac_stage1.env import make_env
from mlbac_stage1.fallback import FallbackController
from mlbac_stage1.models import apply_transition
from diagnostics_stage1 import run_scenario


def _state(params, *, q=1.0, battery=None, temperature=30.0, arrival=0.3, gain=None):
    return {
        "Q": np.full(len(params["Q_max"]), q, dtype=float),
        "B": np.asarray(params["B_cap"] * 0.8 if battery is None else battery, dtype=float),
        "T": np.full(len(params["Q_max"]), temperature, dtype=float),
        "A": np.full(len(params["Q_max"]), arrival, dtype=float),
        "g": np.asarray(params["g_base"] if gain is None else gain, dtype=float),
    }


def _idle_action(n=1):
    return {key: np.zeros(n, dtype=float) for key in ("f", "p", "b", "f_edge", "l", "o", "U")}


def _local_action(n=1, *, frequency=0.2e9, work=0.05, admission=0.0):
    action = _idle_action(n)
    action.update(f=np.full(n, frequency), l=np.full(n, work), U=np.full(n, admission))
    return action


def test_positive_local_work_without_cpu_is_rejected():
    params = make_default_params(1, deterministic=True)
    state = _state(params)
    action = _local_action(frequency=0.0)
    assert not evaluate_constraints(state, action, params)["feasible"]


def test_positive_offload_without_transmission_or_edge_path_is_rejected():
    params = make_default_params(1, deterministic=True)
    state = _state(params)
    action = _idle_action()
    action["o"][:] = 0.1
    assert not evaluate_constraints(state, action, params)["feasible"]


def test_invalid_work_cannot_disappear_in_direct_transition():
    params = make_default_params(1, deterministic=True)
    with pytest.raises(ValueError, match="not physically executable"):
        apply_transition(_state(params), _local_action(frequency=0.0), params)
    offload = _idle_action()
    offload["o"][:] = 0.1
    with pytest.raises(ValueError, match="not physically executable"):
        apply_transition(_state(params), offload, params)


def test_queue_and_energy_conservation_include_idle_baseline():
    params = make_default_params(1, deterministic=True)
    state = _state(params, q=2.0, battery=np.array([10_000.0]), arrival=0.3)
    action = _local_action(work=0.1, admission=0.2)
    result = apply_transition(state, action, params)
    metrics = result["metrics"]
    assert result["Q"][0] == pytest.approx(2.1)
    assert metrics["completed_local_work"][0] == pytest.approx(0.1)
    assert result["B"][0] == pytest.approx(state["B"][0] - metrics["energy"][0])
    assert metrics["baseline_energy"][0] == pytest.approx(1.0)


def test_synthetic_predicted_thermal_violation_is_specific_and_hardware_admissible():
    params = make_default_params(1, deterministic=True)
    params["thermal_alpha"] = 60.0
    params["T_max"][:] = 80.0
    state = _state(params, temperature=79.0)
    action = _local_action(frequency=0.9e9, work=0.05)
    report = evaluate_constraints(state, action, params)
    assert not report["feasible"]
    assert report["violations"]["thermal_violation"] == [0]
    assert report["violations"]["resource_absence"] == []
    assert report["violations"]["timing"] == []
    assert action["f"][0] <= params["f_hw_max"][0]


def test_actual_thermal_outcome_is_checked_independently_of_prediction():
    params = make_default_params(1, deterministic=True)
    params["actual_thermal_alpha"] = 60.0
    state = _state(params, temperature=79.0)
    env = make_env(n_devices=1, horizon=3, params=params, deterministic=True)
    assert "actual_thermal_alpha" not in env.params
    env.reset(seed=0, options={"state": state})
    raw = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=float)
    _, _, terminated, _, info = env.step(raw)
    assert info["predicted_report"]["feasible"]
    assert not info["actual_report"]["feasible"]
    assert info["actual_report"]["flags"]["temperature_peak"] == [0]
    assert terminated


def test_no_verified_fallback_is_not_executed_or_reported_as_safe():
    params = make_default_params(1, deterministic=True)
    state = _state(params, q=0.7, battery=params["B_min"] + 0.5, arrival=0.0)
    env = make_env(n_devices=1, horizon=4, params=params, deterministic=True)
    env.reset(seed=0, options={"state": state})
    before = {key: value.copy() for key, value in env.state.items()}
    raw_idle = -np.ones(7, dtype=float)
    _, reward, terminated, truncated, info = env.step(raw_idle)
    assert info["filter_status"] == "no_verified_safe_continuation"
    assert info["failure_kind"] == "no_verified_candidate_found"
    assert info["infeasibility_proven"] is False
    assert info["executed_action"] is None
    assert info["fallback_mode"] == "idle_candidate"
    assert info["predicted_report"]["violations"]["magnitudes"]["battery_deficit"] == pytest.approx(0.5)
    assert info["actual_outcome"] == "unobserved"
    assert info["post_state"] is None
    assert info["outstanding_work"][0] == pytest.approx(0.7)
    assert info["total_outstanding_work"][0] == pytest.approx(0.7)
    assert info["elapsed_time_s"] == pytest.approx(0.0)
    assert reward == pytest.approx(-10.0) and terminated and not truncated
    assert env.state["B"][0] == pytest.approx(before["B"][0])
    assert env.last_info["episode_ending_reason"] == "controller_failure_no_verified_continuation"


def test_default_battery_reserve_is_ten_percent():
    params = make_default_params(1, deterministic=True)
    assert params["B_min"][0] == pytest.approx(3240.0)
    assert FallbackController(params).describe()["verification_scope"] == "one_step_only"
    assert not FallbackController(params).describe()["remaining_horizon_verified"]


def test_default_task_size_is_decimal_megabits_in_bits():
    params = make_default_params(1, deterministic=True)
    assert params["d"][0] == 0.2 * 1_000_000 * 8 == 1_600_000


def test_observation_scaling_preserves_capacity_and_out_of_limit_state():
    params_a = make_default_params(1, deterministic=True)
    params_b = make_default_params(1, deterministic=True)
    params_b["shared_bandwidth_hz"] = 20e6
    env_a = make_env(params=params_a, deterministic=True)
    env_b = make_env(params=params_b, deterministic=True)
    env_a.reset(options={"state": _state(params_a, q=22.0)})
    env_b.reset(options={"state": _state(params_b, q=22.0)})
    obs_a = env_a._observation()
    obs_b = env_b._observation()
    assert obs_a[-3] != obs_b[-3]
    assert obs_a[0] == pytest.approx(2.2)
    assert env_a.observation_space.contains(obs_a)


def test_observation_contains_each_devices_channel_gain():
    params = make_default_params(2, deterministic=True)
    params["g_base"] = np.array([2e-5, 0.5e-5])
    env = make_env(n_devices=2, params=params, deterministic=True)
    env.reset(options={"state": _state(params, gain=params["g_base"])})
    obs = env._observation()
    np.testing.assert_allclose(obs[8:10], [2.0, 0.5])


def test_action_decoding_maps_normalized_range_to_nonnegative_physical_ranges():
    env = make_env(n_devices=1, deterministic=True)
    env.reset(seed=0)
    low = env._decode_action(-np.ones(7, dtype=float))
    high = env._decode_action(np.ones(7, dtype=float))
    assert all(value[0] == 0.0 for value in low.values())
    assert high["f"][0] == pytest.approx(env.params["f_hw_max"][0])
    assert high["p"][0] == pytest.approx(env.params["p_hw_max"][0])
    assert high["b"][0] == pytest.approx(env.params["shared_bandwidth_hz"])
    assert high["f_edge"][0] == pytest.approx(env.params["shared_edge_cpu_hz"])


def test_raw_policy_preprocessing_is_not_labelled_nominal_acceptance():
    env = make_env(n_devices=1, horizon=2, deterministic=True)
    env.reset(seed=0)
    raw = -np.ones(7, dtype=float)
    raw[0] = 2.0
    _, _, _, _, info = env.step(raw)
    assert info["action_preprocessed"]
    assert info["filter_status"] == "preprocessed_candidate_verified"
    assert info["intervention"]


def test_diagnostic_csv_matches_recorded_state_and_marks_failure_unobserved(tmp_path):
    normal_dir = run_scenario("normal", tmp_path, seed=0)
    with (normal_dir / "config_snapshot.json").open() as handle:
        config_snapshot = json.load(handle)
    assert config_snapshot["seed"] == 0
    assert config_snapshot["scenario"] == "normal"
    with (normal_dir / "episode.csv").open(newline="") as handle:
        normal = next(csv.DictReader(handle))
    assert float(normal["pre_Q_work_units"]) == pytest.approx(1.0)
    assert float(normal["actual_Q_work_units"]) == pytest.approx(1.0)
    assert float(normal["executed_l"]) == pytest.approx(0.05)
    assert float(normal["elapsed_time_s_at_end"]) == pytest.approx(1.0)
    assert normal["filter_status"] == "nominal_accepted"
    assert normal["actual_outcome"] == "observed"
    for action_key in ("f", "p", "b", "f_edge", "l", "o", "U"):
        assert float(normal[f"decoded_nominal_{action_key}"]) == pytest.approx(float(normal[f"executed_{action_key}"]))

    battery_dir = run_scenario("battery", tmp_path, seed=0)
    with (battery_dir / "episode.csv").open(newline="") as handle:
        battery = next(csv.DictReader(handle))
    assert battery["executed_f"] == ""
    assert battery["actual_B_j"] == ""
    assert battery["actual_outcome"] == "unobserved"
    assert battery["failure_kind"] == "no_verified_candidate_found"
    assert float(battery["elapsed_time_s_at_end"]) == pytest.approx(0.0)


def test_deterministic_reproducibility_and_exact_horizon():
    def rollout():
        env = make_env(n_devices=1, horizon=3, deterministic=True)
        observations, rewards, endings = [], [], []
        obs, _ = env.reset(seed=12)
        observations.append(obs.copy())
        for _ in range(3):
            obs, reward, terminated, truncated, _ = env.step(-np.ones(7, dtype=float))
            observations.append(obs.copy())
            rewards.append(reward)
            endings.append((terminated, truncated))
        return observations, rewards, endings, env

    first = rollout()
    second = rollout()
    for left, right in zip(first[0], second[0]):
        np.testing.assert_array_equal(left, right)
    assert first[1] == second[1]
    assert len(first[1]) == 3
    assert first[2][-1] == (True, False)
    with pytest.raises(RuntimeError, match="after episode termination"):
        first[3].step(-np.ones(7, dtype=float))


def test_shared_resource_limits_are_checked_across_devices():
    params = make_default_params(2, deterministic=True)
    state = _state(params, q=1.0, arrival=0.0)
    action = _idle_action(2)
    action["b"][:] = 6e6
    action["f_edge"][:] = 6e9
    report = evaluate_constraints(state, action, params)
    assert not report["feasible"]
    assert report["violations"]["magnitudes"]["shared_bandwidth_excess_hz"] == pytest.approx(2e6)
    assert report["violations"]["magnitudes"]["shared_edge_cpu_excess_hz"] == pytest.approx(2e9)


def test_post_transition_checker_uses_actual_state_and_metrics():
    params = make_default_params(1, deterministic=True)
    post = {"Q": np.array([1.0]), "B": params["B_min"] - 0.5, "T": np.array([81.0])}
    actual = evaluate_post_transition(post, {"peak_temp": np.array([82.0])}, params)
    assert not actual["feasible"]
    assert actual["flags"]["battery_reserve"] == [0]
    assert actual["flags"]["temperature_endpoint"] == [0]
    assert actual["magnitudes"]["battery_reserve_deficit_j"] == pytest.approx(0.5)
