from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .config import checker_tolerances, safety_margin_thresholds, safety_verification_scope
from .models import (compute_rates, predict_next_temperature, thermal_peak_temperature,
                     validate_state, validate_arrays, ACTION_FIELDS, slot_quantities)


EPS = 1e-12


def _bad_indices(mask: np.ndarray) -> list[int]:
    return np.nonzero(np.asarray(mask, dtype=bool))[0].tolist()


def evaluate_constraints(state: Dict[str, np.ndarray], action: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether an action is feasible under the Stage 1 hard constraints.

    The check uses the pre-action state and the predicted next temperature from the
    proposed action; it does not re-evaluate the same action on the updated state.
    """
    tol = checker_tolerances(params)
    margins, temperature_threshold, battery_threshold = safety_margin_thresholds(params)
    state_validation = validate_state(state, params)
    action_errors = validate_arrays(action, ACTION_FIELDS, len(params["Q_max"]))
    if state_validation["status"] != "valid" or action_errors:
        status = state_validation["status"] if state_validation["status"] != "valid" else "invalid_proposal"
        return {"feasible": False, "input_status": status,
                "violations": {"input": state_validation["errors"] or action_errors},
                "metrics": {}, "predicted_state": None, "predicted_next_state": None,
                "constraint_residuals": {}, "checker_tolerances": tol}
    Q = np.asarray(state["Q"], dtype=float)
    B = np.asarray(state["B"], dtype=float)
    T = np.asarray(state["T"], dtype=float)
    A = np.asarray(state["A"], dtype=float)
    g = np.asarray(state["g"], dtype=float)
    N0 = float(params.get("N0", 1e-12))

    f = np.asarray(action.get("f", np.zeros_like(Q)), dtype=float)
    p = np.asarray(action.get("p", np.zeros_like(Q)), dtype=float)
    b = np.asarray(action.get("b", np.zeros_like(Q)), dtype=float)
    f_edge = np.asarray(action.get("f_edge", np.zeros_like(Q)), dtype=float)
    l = np.asarray(action.get("l", np.zeros_like(Q)), dtype=float)
    o = np.asarray(action.get("o", np.zeros_like(Q)), dtype=float)
    U = np.asarray(action.get("U", np.zeros_like(Q)), dtype=float)

    d = np.asarray(params["d"], dtype=float)
    c = np.asarray(params["c"], dtype=float)
    delta = float(params["delta"])
    Q_max = np.asarray(params["Q_max"], dtype=float)
    B_min = np.asarray(params["B_min"], dtype=float)
    T_max = np.asarray(params["T_max"], dtype=float)
    B_cap = np.asarray(params["B_cap"], dtype=float)
    f_hw_max = np.asarray(params["f_hw_max"], dtype=float)
    p_hw_max = np.asarray(params["p_hw_max"], dtype=float)

    queue_work_ok = (l + o <= Q + tol["work"])
    admit_ok = (U <= A + tol["work"]) & (U <= np.asarray(params["A_max"]) + tol["work"])
    nonneg_ok = np.all((l >= 0.0) & (o >= 0.0) & (U >= 0.0) & (f >= 0.0) & (p >= 0.0) & (b >= 0.0) & (f_edge >= 0.0))
    local_ok = f <= f_hw_max + tol["frequency_hz"]
    tx_ok = p <= p_hw_max + tol["power_w"]
    bw_ok = np.sum(b) <= params["shared_bandwidth_hz"] + tol["bandwidth_hz"]
    edge_ok = np.sum(f_edge) <= params["shared_edge_cpu_hz"] + tol["frequency_hz"]

    quantities = slot_quantities(state, action, params)
    rate, t_local, t_tx, t_edge = (quantities[k] for k in
        ("rate", "local_time", "tx_time", "edge_time"))
    timing_ok = (t_local <= delta + tol["time_s"]) & ((t_tx + t_edge) <= delta + tol["time_s"])

    # Explicitly reject positive work if the required processing resource is absent.
    positive_local = (l > 0.0) & (f <= 0.0)
    positive_tx = (o > 0.0) & ((rate <= 0.0) | (f_edge <= 0.0))
    resource_ok = ~(positive_local | positive_tx)

    kappa = np.asarray(params["kappa"], dtype=float)
    baseline = np.asarray(params.get("baseline_power", np.zeros_like(Q)), dtype=float)
    local_energy = quantities["local_energy"]
    tx_energy = quantities["tx_energy"]
    total_energy = quantities["total_energy"]
    battery_after = B - total_energy
    battery_ok = (battery_after >= B_min - tol["energy_j"])
    battery_cap_ok = (battery_after <= B_cap + tol["energy_j"])

    predicted_T = predict_next_temperature(T, action, params)
    predicted_peak_T = thermal_peak_temperature(T, action, params)
    temp_ok = (predicted_peak_T <= T_max + tol["temperature_c"])
    temperature_margin_ok = (T <= T_max) & (predicted_T <= temperature_threshold)
    battery_margin_ok = battery_after >= battery_threshold
    selected_temp_ok = temperature_margin_ok if margins["enabled"] else temp_ok
    selected_battery_ok = battery_margin_ok if margins["enabled"] else battery_ok
    next_queue = Q - l - o + U
    queue_cap_ok = (next_queue <= Q_max + tol["work"])
    # Physical nonnegativity must agree with validate_state on the next step.
    # A numerical tolerance must not authorize a negative stored queue.
    queue_lower_ok = next_queue >= 0.0
    work_bounds_ok = (l <= Q_max + tol["work"]) & (o <= Q_max + tol["work"])

    feasible = bool(
        np.all(queue_work_ok)
        and np.all(admit_ok)
        and nonneg_ok
        and np.all(local_ok)
        and np.all(tx_ok)
        and bw_ok
        and edge_ok
        and np.all(timing_ok)
        and np.all(resource_ok)
        and np.all(selected_battery_ok)
        and np.all(battery_cap_ok)
        and np.all(selected_temp_ok)
        and np.all(queue_cap_ok)
        and np.all(queue_lower_ok)
        and np.all(work_bounds_ok)
    )

    metrics = {
        "queue_slack": float(np.min(Q_max - next_queue)) if next_queue.size else 0.0,
        "battery_slack": float(np.min(battery_after - B_min)) if battery_after.size else 0.0,
        "temp_slack": float(np.min(T_max - predicted_peak_T)) if predicted_peak_T.size else 0.0,
        "local_time_max": float(np.max(t_local)) if t_local.size else 0.0,
        "tx_plus_edge_max": float(np.max(t_tx + t_edge)) if (t_tx + t_edge).size else 0.0,
        "rate": float(np.max(rate)) if rate.size else 0.0,
        "energy_total": float(np.sum(total_energy)),
    }
    report = {
        "feasible": feasible,
        "input_status": "valid",
        "verification_scope": safety_verification_scope(params),
        "checker_tolerances": tol,
        "safety_margins": margins,
        "margin_checks": {
            "predicted_temperature_c": predicted_T.copy(),
            "predicted_battery_j": battery_after.copy(),
            "ordinary_temperature_endpoint_slack_c": T_max - predicted_T,
            "ordinary_battery_slack_j": battery_after - B_min,
            "tightened_temperature_slack_c": T_max - predicted_T - margins["effective_temperature_margin_c"],
            "tightened_battery_slack_j": battery_after - B_min - margins["effective_battery_margin_j"],
            "temperature_threshold_c": temperature_threshold,
            "battery_threshold_j": battery_threshold,
            "ordinary_temperature_pass": temp_ok,
            "ordinary_battery_pass": battery_ok,
            "current_temperature_hard_limit_pass": T <= T_max,
            "margin_temperature_pass": temperature_margin_ok if margins["enabled"] else None,
            "margin_battery_pass": battery_margin_ok if margins["enabled"] else None,
        },
        "constraint_residuals": {
            "nonnegative_action": {k: np.asarray(action[k]).copy() for k in ACTION_FIELDS},
            "local_cpu_hz": f_hw_max - f, "transmit_power_w": p_hw_max - p,
            "shared_bandwidth_hz": float(params["shared_bandwidth_hz"] - np.sum(b)),
            "shared_edge_cpu_hz": float(params["shared_edge_cpu_hz"] - np.sum(f_edge)),
            "available_work": Q - l - o, "available_arrivals": A - U,
            "admission_bound": np.asarray(params["A_max"]) - U,
            "local_work_bound": Q_max - l, "offload_work_bound": Q_max - o,
            "local_time_s": delta - t_local, "offload_time_s": delta - t_tx - t_edge,
            "queue_lower": next_queue, "queue_upper": Q_max - next_queue,
            "battery_reserve_j": battery_after - B_min,
            "battery_capacity_j": B_cap - battery_after,
            "temperature_peak_c": T_max - predicted_peak_T,
        },
        "violations": {
            "queue": _bad_indices(~queue_work_ok),
            "queue_lower": _bad_indices(~queue_lower_ok),
            "work_bounds": _bad_indices(~work_bounds_ok),
            "local_cpu": _bad_indices(~local_ok),
            "transmit_power": _bad_indices(~tx_ok),
            "negative_action": {k: _bad_indices(np.asarray(action[k]) < 0) for k in ACTION_FIELDS},
            "admission": _bad_indices(~admit_ok),
            "timing": _bad_indices(~timing_ok),
            "battery": _bad_indices(~battery_ok),
            "temperature": _bad_indices(~temp_ok),
            "shared_bandwidth_violation": float(max(0.0, np.sum(b) - params["shared_bandwidth_hz"])),
            "shared_edge_cpu_violation": float(max(0.0, np.sum(f_edge) - params["shared_edge_cpu_hz"])),
            "queue_overflow": _bad_indices(~queue_cap_ok),
            "battery_overflow": _bad_indices(~battery_cap_ok),
            "thermal_violation": _bad_indices(~temp_ok),
            "temperature_margin": _bad_indices(~temperature_margin_ok) if margins["enabled"] else [],
            "battery_margin": _bad_indices(~battery_margin_ok) if margins["enabled"] else [],
            "resource_absence": _bad_indices(~resource_ok),
            "work_without_frequency": _bad_indices(positive_local),
            "work_without_transmission": _bad_indices(positive_tx),
            "magnitudes": {
                "queue_excess": float(np.max(np.maximum(0.0, next_queue - Q_max))) if next_queue.size else 0.0,
                "queue_work_excess": float(np.max(np.maximum(0.0, l + o - Q))) if Q.size else 0.0,
                "admission_excess": float(np.max(np.maximum(0.0, U - A))) if A.size else 0.0,
                "battery_deficit": float(np.max(np.maximum(0.0, B_min - battery_after))) if battery_after.size else 0.0,
                "battery_overflow": float(np.max(np.maximum(0.0, battery_after - B_cap))) if battery_after.size else 0.0,
                "temperature_excess": float(np.max(np.maximum(0.0, predicted_T - T_max))) if predicted_T.size else 0.0,
                "temperature_peak_excess": float(np.max(np.maximum(0.0, predicted_peak_T - T_max))) if predicted_peak_T.size else 0.0,
                "local_cpu_excess_hz": float(np.max(np.maximum(0.0, f - f_hw_max))) if f.size else 0.0,
                "transmit_power_excess_w": float(np.max(np.maximum(0.0, p - p_hw_max))) if p.size else 0.0,
                "shared_bandwidth_excess_hz": float(max(0.0, np.sum(b) - params["shared_bandwidth_hz"])) if np.size(b) else 0.0,
                "shared_edge_cpu_excess_hz": float(max(0.0, np.sum(f_edge) - params["shared_edge_cpu_hz"])) if np.size(f_edge) else 0.0,
                "timing_excess": float(np.max(np.maximum(0.0, np.maximum(t_local, t_tx + t_edge) - delta))) if Q.size else 0.0,
                "local_work_without_cpu": float(np.sum(positive_local)),
                "offload_work_without_path": float(np.sum(positive_tx)),
            },
        },
        "metrics": metrics,
        "predicted_energy_j": total_energy.copy(),
        "predicted_temperature": predicted_T,
        "predicted_peak_temperature": predicted_peak_T,
        "predicted_state": {"Q": next_queue, "B": battery_after, "T": predicted_T},
    }
    report["predicted_next_state"] = report["predicted_state"]
    return report


def evaluate_post_transition(
    post_state: Dict[str, np.ndarray],
    transition_metrics: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify recorded actual state and transition peak without rerunning prediction."""
    tol = checker_tolerances(params)
    queue = np.asarray(post_state["Q"], dtype=float)
    battery = np.asarray(post_state["B"], dtype=float)
    temperature = np.asarray(post_state["T"], dtype=float)
    peak_temperature = np.asarray(transition_metrics.get("peak_temp", temperature), dtype=float)
    queue_max = np.asarray(params["Q_max"], dtype=float)
    battery_min = np.asarray(params["B_min"], dtype=float)
    temperature_max = np.asarray(params["T_max"], dtype=float)
    queue_flag = queue > queue_max + tol["work"]
    battery_flag = battery < battery_min - tol["energy_j"]
    temperature_flag = temperature > temperature_max + tol["temperature_c"]
    peak_flag = peak_temperature > temperature_max + tol["temperature_c"]
    flags = {
        "queue_capacity": _bad_indices(queue_flag),
        "queue_lower": _bad_indices(queue < 0.0),
        "battery_capacity": _bad_indices(battery > np.asarray(params["B_cap"]) + tol["energy_j"]),
        "nonfinite_state": _bad_indices(~(np.isfinite(queue) & np.isfinite(battery) & np.isfinite(temperature) & np.isfinite(peak_temperature))),
        "battery_reserve": _bad_indices(battery_flag),
        "temperature_endpoint": _bad_indices(temperature_flag),
        "temperature_peak": _bad_indices(peak_flag),
    }
    magnitudes = {
        "queue_excess_work_units": float(np.max(np.maximum(0.0, queue - queue_max))) if queue.size else 0.0,
        "battery_reserve_deficit_j": float(np.max(np.maximum(0.0, battery_min - battery))) if battery.size else 0.0,
        "temperature_endpoint_excess_c": float(np.max(np.maximum(0.0, temperature - temperature_max))) if temperature.size else 0.0,
        "temperature_peak_excess_c": float(np.max(np.maximum(0.0, peak_temperature - temperature_max))) if peak_temperature.size else 0.0,
    }
    return {
        "feasible": not any(flags.values()),
        "flags": flags,
        "magnitudes": magnitudes,
        "actual_state": {"Q": queue.copy(), "B": battery.copy(), "T": temperature.copy()},
        "actual_peak_temperature": peak_temperature.copy(),
    }
