from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .models import compute_rates, predict_next_temperature


EPS = 1e-12


def _bad_indices(mask: np.ndarray) -> list[int]:
    return np.nonzero(np.asarray(mask, dtype=bool))[0].tolist()


def evaluate_constraints(state: Dict[str, np.ndarray], action: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether an action is feasible under the Stage 1 hard constraints.

    The check uses the pre-action state and the predicted next temperature from the
    proposed action; it does not re-evaluate the same action on the updated state.
    """
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

    queue_work_ok = (l + o <= Q + 1e-9)
    admit_ok = (U <= A + 1e-9)
    nonneg_ok = np.all((l >= 0.0) & (o >= 0.0) & (U >= 0.0) & (f >= 0.0) & (p >= 0.0) & (b >= 0.0) & (f_edge >= 0.0))
    local_ok = f <= f_hw_max + 1e-9
    tx_ok = p <= p_hw_max + 1e-9
    bw_ok = np.sum(b) <= params["shared_bw"] + 1e-9
    edge_ok = np.sum(f_edge) <= params["shared_edge_cpu"] + 1e-9

    rate = compute_rates(g, b, p, N0)
    t_local = np.divide(c * l, np.maximum(f, EPS), out=np.zeros_like(Q), where=f > 0)
    t_tx = np.divide(d * o, np.maximum(rate, EPS), out=np.zeros_like(Q), where=(rate > 0.0) & (o > 0.0))
    t_edge = np.divide(c * o, np.maximum(f_edge, EPS), out=np.zeros_like(Q), where=(f_edge > 0.0) & (o > 0.0))
    timing_ok = (t_local <= delta + 1e-9) & ((t_tx + t_edge) <= delta + 1e-9)

    # Explicitly reject positive work if the required processing resource is absent.
    positive_local = (l > 0.0) & (f <= 0.0)
    positive_tx = (o > 0.0) & ((rate <= 0.0) | (f_edge <= 0.0))
    resource_ok = ~(positive_local | positive_tx)

    kappa = np.asarray(params["kappa"], dtype=float)
    baseline = np.asarray(params.get("baseline_power", np.zeros_like(Q)), dtype=float)
    local_energy = np.where(f > 0.0, kappa * np.power(f, 3.0) * t_local, 0.0)
    tx_energy = np.where(rate > 0.0, p * t_tx, 0.0)
    total_energy = local_energy + tx_energy + baseline * delta
    battery_after = B - total_energy
    battery_ok = (battery_after >= B_min - 1e-9)
    battery_cap_ok = (battery_after <= B_cap + 1e-9)

    predicted_T = predict_next_temperature(T, action, params)
    temp_ok = (predicted_T <= T_max + 1e-9)
    next_queue = Q - l - o + U
    queue_cap_ok = (next_queue <= Q_max + 1e-9)

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
        and np.all(battery_ok)
        and np.all(battery_cap_ok)
        and np.all(temp_ok)
        and np.all(queue_cap_ok)
    )

    metrics = {
        "queue_slack": float(np.min(Q_max - next_queue)) if next_queue.size else 0.0,
        "battery_slack": float(np.min(battery_after - B_min)) if battery_after.size else 0.0,
        "temp_slack": float(np.min(T_max - predicted_T)) if predicted_T.size else 0.0,
        "local_time_max": float(np.max(t_local)) if t_local.size else 0.0,
        "tx_plus_edge_max": float(np.max(t_tx + t_edge)) if (t_tx + t_edge).size else 0.0,
        "rate": float(np.max(rate)) if rate.size else 0.0,
        "energy_total": float(np.sum(total_energy)),
    }
    report = {
        "feasible": feasible,
        "violations": {
            "queue": _bad_indices(~queue_work_ok),
            "admission": _bad_indices(~admit_ok),
            "timing": _bad_indices(~timing_ok),
            "battery": _bad_indices(~battery_ok),
            "temperature": _bad_indices(~temp_ok),
            "shared_bw_violation": float(max(0.0, np.sum(b) - params["shared_bw"])),
            "shared_edge_violation": float(max(0.0, np.sum(f_edge) - params["shared_edge_cpu"])),
            "queue_overflow": _bad_indices(~queue_cap_ok),
            "battery_overflow": _bad_indices(~battery_cap_ok),
            "thermal_violation": _bad_indices(~temp_ok),
            "resource_absence": _bad_indices(~resource_ok),
            "work_without_frequency": _bad_indices(positive_local),
            "work_without_transmission": _bad_indices(positive_tx),
            "magnitudes": {
                "queue_excess": float(np.max(np.maximum(0.0, next_queue - Q_max))) if next_queue.size else 0.0,
                "battery_deficit": float(np.max(np.maximum(0.0, B_min - battery_after))) if battery_after.size else 0.0,
                "temperature_excess": float(np.max(np.maximum(0.0, predicted_T - T_max))) if predicted_T.size else 0.0,
                "shared_bw_excess": float(max(0.0, np.sum(b) - params["shared_bw"])) if np.size(b) else 0.0,
                "shared_edge_excess": float(max(0.0, np.sum(f_edge) - params["shared_edge_cpu"])) if np.size(f_edge) else 0.0,
                "timing_excess": float(np.max(np.maximum(0.0, np.maximum(t_local, t_tx + t_edge) - delta))) if Q.size else 0.0,
            },
        },
        "metrics": metrics,
        "predicted_temperature": predicted_T,
    }
    return report
