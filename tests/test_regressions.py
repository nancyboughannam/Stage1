import numpy as np

from mlbac_stage1.constraints import evaluate_constraints
from mlbac_stage1.env import make_env


def _base_params():
    return {
        "Q_max": np.array([10.0]),
        "B_min": np.array([0.9 * 9 * 3600]),
        "T_max": np.array([80.0]),
        "B_cap": np.array([9 * 3600]),
        "shared_bw": 10e6,
        "shared_edge_cpu": 10e9,
        "delta": 1.0,
        "d": np.array([0.2e6]),
        "c": np.array([40e6]),
        "f_hw_max": np.array([1e9]),
        "p_hw_max": np.array([1.0]),
        "kappa": np.array([1e-26]),
        "baseline_power": np.array([1.0]),
        "ambient_temp": 25.0,
        "thermal_alpha": 0.15,
        "thermal_tau": 12.0,
        "N0": 1e-12,
        "A_max": np.array([1.0]),
        "g_base": np.array([1e-5]),
        "deterministic": True,
    }


def test_zero_local_work_with_zero_cpu_is_infeasible():
    state = {
        "Q": np.array([1.0]),
        "B": np.array([8.5 * 3600]),
        "T": np.array([30.0]),
        "A": np.array([1.0]),
        "g": np.array([1e-5]),
    }
    action = {
        "f": np.array([0.0]),
        "p": np.array([0.1]),
        "b": np.array([1e6]),
        "f_edge": np.array([0.5e9]),
        "l": np.array([0.4]),
        "o": np.array([0.0]),
        "U": np.array([0.2]),
    }
    report = evaluate_constraints(state, action, _base_params())
    assert not report["feasible"]
    assert report["violations"]["resource_absence"]


def test_zero_offload_resources_are_infeasible():
    state = {
        "Q": np.array([1.0]),
        "B": np.array([8.5 * 3600]),
        "T": np.array([30.0]),
        "A": np.array([1.0]),
        "g": np.array([1e-5]),
    }
    action = {
        "f": np.array([0.2e9]),
        "p": np.array([0.1]),
        "b": np.array([0.0]),
        "f_edge": np.array([0.0]),
        "l": np.array([0.0]),
        "o": np.array([0.5]),
        "U": np.array([0.1]),
    }
    report = evaluate_constraints(state, action, _base_params())
    assert not report["feasible"]


def test_predicted_thermal_violation_is_rejected():
    state = {
        "Q": np.array([1.0]),
        "B": np.array([8.5 * 3600]),
        "T": np.array([79.0]),
        "A": np.array([1.0]),
        "g": np.array([1e-5]),
    }
    action = {
        "f": np.array([1e9]),
        "p": np.array([500.0]),
        "b": np.array([1e6]),
        "f_edge": np.array([1e8]),
        "l": np.array([0.1]),
        "o": np.array([0.1]),
        "U": np.array([0.2]),
    }
    report = evaluate_constraints(state, action, _base_params())
    assert not report["feasible"]
    assert report["violations"]["thermal_violation"]


def test_environment_is_deterministic_and_observation_is_normalized():
    env = make_env(n_devices=1, horizon=4, deterministic=True, action_scale=1.0)
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    assert np.allclose(obs1, obs2)
    assert obs1.shape == env.observation_space.shape
    assert np.all(obs1 >= env.observation_space.low)
    assert np.all(obs1 <= env.observation_space.high)


def test_horizon_termination_uses_documented_finite_horizon():
    env = make_env(n_devices=1, horizon=2, deterministic=True)
    env.reset(seed=0)
    env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float))
    _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float))
    assert terminated or truncated
