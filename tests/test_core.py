import numpy as np

from mlbac_stage1.constraints import evaluate_constraints
from mlbac_stage1.safety_filter import SafetyFilter


def test_safe_action_passes_through():
    state = {
        "Q": np.array([1.0]),
        "B": np.array([8.5 * 3600]),
        "T": np.array([30.0]),
        "A": np.array([1.0]),
        "g": np.array([1e-5]),
        "N0": 1e-12,
    }
    action = {
        "f": np.array([0.2e9]),
        "p": np.array([0.1]),
        "b": np.array([1e6]),
        "f_edge": np.array([0.5e9]),
        "l": np.array([0.5]),
        "o": np.array([0.2]),
        "U": np.array([0.4]),
    }
    params = {
        "Q_max": np.array([10.0]),
        "B_min": np.array([0.1 * 9 * 3600]),
        "T_max": np.array([80.0]),
        "B_cap": np.array([9 * 3600]),
        "shared_bandwidth_hz": 10e6,
        "shared_edge_cpu_hz": 10e9,
        "delta": 1.0,
        "d": np.array([1.6e6]),
        "c": np.array([40e6]),
        "f_hw_max": np.array([1e9]),
        "p_hw_max": np.array([1.0]),
        "kappa": np.array([1e-26]),
        "baseline_power": np.array([1.0]),
    }
    report = evaluate_constraints(state, action, params)
    assert report["feasible"]


def test_unsafe_action_is_corrected():
    state = {
        "Q": np.array([0.2]),
        "B": np.array([8.5 * 3600]),
        "T": np.array([78.0]),
        "A": np.array([1.0]),
        "g": np.array([1e-5]),
        "N0": 1e-12,
    }
    action = {
        "f": np.array([0.9e9]),
        "p": np.array([0.8]),
        "b": np.array([9e6]),
        "f_edge": np.array([10e9]),
        "l": np.array([1.0]),
        "o": np.array([1.0]),
        "U": np.array([1.0]),
    }
    params = {
        "Q_max": np.array([10.0]),
        "B_min": np.array([0.1 * 9 * 3600]),
        "T_max": np.array([80.0]),
        "B_cap": np.array([9 * 3600]),
        "shared_bandwidth_hz": 10e6,
        "shared_edge_cpu_hz": 10e9,
        "delta": 1.0,
        "d": np.array([1.6e6]),
        "c": np.array([40e6]),
        "f_hw_max": np.array([1e9]),
        "p_hw_max": np.array([1.0]),
        "kappa": np.array([1e-26]),
        "baseline_power": np.array([1.0]),
    }
    filter_obj = SafetyFilter(params)
    out = filter_obj.filter_action(state, action)
    assert out["status"] in {"nominal_accepted", "corrected_candidate_verified", "fallback_verified"}
    assert out["safe"] is True


def test_queue_and_energy_conservation_snapshot():
    state = {
        "Q": np.array([3.0]),
        "B": np.array([8.5 * 3600]),
        "T": np.array([35.0]),
        "A": np.array([1.0]),
        "g": np.array([2e-5]),
        "N0": 1e-12,
    }
    action = {
        "f": np.array([0.2e9]),
        "p": np.array([0.2]),
        "b": np.array([2e6]),
        "f_edge": np.array([0.5e9]),
        "l": np.array([1.0]),
        "o": np.array([0.5]),
        "U": np.array([0.5]),
    }
    params = {
        "Q_max": np.array([10.0]),
        "B_min": np.array([0.1 * 9 * 3600]),
        "T_max": np.array([80.0]),
        "B_cap": np.array([9 * 3600]),
        "shared_bandwidth_hz": 10e6,
        "shared_edge_cpu_hz": 10e9,
        "delta": 1.0,
        "d": np.array([1.6e6]),
        "c": np.array([40e6]),
        "f_hw_max": np.array([1e9]),
        "p_hw_max": np.array([1.0]),
        "kappa": np.array([1e-26]),
        "baseline_power": np.array([1.0]),
    }
    report = evaluate_constraints(state, action, params)
    assert report["feasible"]
    assert report["metrics"]["queue_slack"] >= 0.0
    assert report["metrics"]["battery_slack"] >= 0.0
