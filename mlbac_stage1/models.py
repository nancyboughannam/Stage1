from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .config import checker_tolerances


EPS = 1e-12


@dataclass(frozen=True)
class DeviceSpec:
    data_bits: float
    cpu_cycles: float
    local_cpu_max: float
    power_max: float
    baseline_power: float
    kappa: float


ACTION_FIELDS = ("f", "p", "b", "f_edge", "l", "o", "U")
STATE_FIELDS = ("Q", "B", "T", "A", "g")


def validate_arrays(values, fields, n):
    """Reject missing, nonnumeric, nonfinite or wrongly shaped fields; never broadcast."""
    errors = {}
    if not isinstance(values, dict):
        return {"input": "expected a mapping"}
    for key in fields:
        try:
            array = np.asarray(values[key], dtype=float)
            if array.shape != (n,):
                errors[key] = "expected shape (N,)"
            elif not np.isfinite(array).all():
                errors[key] = "nonfinite value"
        except (KeyError, TypeError, ValueError, OverflowError):
            errors[key] = "missing or nonnumeric field"
    return errors


def validate_state(state, params):
    n = len(params["Q_max"])
    errors = validate_arrays(state, STATE_FIELDS, n)
    if errors:
        return {"status": "invalid_state", "errors": errors}
    for key in ("Q", "B", "A", "g"):
        if np.any(np.asarray(state[key]) < 0):
            errors[key] = "negative physical value"
    if np.any(np.asarray(state["B"]) > params["B_cap"]):
        errors["B"] = "above battery capacity"
    if np.any(np.asarray(state["T"]) < -273.15):
        errors["T"] = "below absolute zero"
    if errors:
        return {"status": "invalid_state", "errors": errors}
    tol = checker_tolerances(params)
    unsafe = {}
    for key, mask in {
        "queue": np.asarray(state["Q"]) > params["Q_max"] + tol["work"],
        "battery": np.asarray(state["B"]) < params["B_min"] - tol["energy_j"],
        "temperature": np.asarray(state["T"]) > params["T_max"] + tol["temperature_c"],
    }.items():
        if np.any(mask):
            unsafe[key] = np.flatnonzero(mask).tolist()
    return {"status": "already_unsafe" if unsafe else "valid", "errors": unsafe}


def slot_quantities(state, action, params):
    """Shared physical equations. Call only after shape/finiteness validation.

    Missing resources yield infinite durations for positive work, never free service.
    """
    f, p, b, edge, l, o, _ = (np.asarray(action[k], dtype=float) for k in ACTION_FIELDS)
    rate = compute_rates(state["g"], b, p, float(params.get("N0", 1e-12)))
    def duration(work, amount, resource):
        out = np.where(work > 0, np.inf, 0.0)
        return np.divide(amount * work, resource, out=out, where=(work > 0) & (resource > 0))
    local = duration(l, np.asarray(params["c"]), f)
    tx = duration(o, np.asarray(params["d"]), rate)
    remote = duration(o, np.asarray(params["c"]), edge)
    # c*l*f^2 is algebraically equivalent to f^3*t for executable local work.
    e_local = np.asarray(params["kappa"]) * np.asarray(params["c"]) * l * f**2
    e_tx = np.zeros_like(tx)
    np.multiply(p, tx, out=e_tx, where=np.isfinite(tx))
    e_tx[~np.isfinite(tx)] = np.inf
    base = np.asarray(params["baseline_power"]) * float(params["delta"])
    return {"rate": rate, "local_time": local, "tx_time": tx, "edge_time": remote,
            "local_energy": e_local, "tx_energy": e_tx, "baseline_energy": base,
            "total_energy": e_local + e_tx + base}


def safe_log2_ratio(num: float, den: float) -> float:
    return float(np.log2(1.0 + max(num, 0.0) / max(den, EPS))) if num > 0.0 and den > 0.0 else 0.0


def compute_rates(g: np.ndarray, b: np.ndarray, p: np.ndarray, N0: float) -> np.ndarray:
    """Compute communication rate using a numerically safe, physically consistent model."""
    b_safe = np.maximum(np.asarray(b, dtype=float), 0.0)
    p_safe = np.maximum(np.asarray(p, dtype=float), 0.0)
    g_safe = np.maximum(np.asarray(g, dtype=float), 0.0)
    denom = np.maximum(N0 * b_safe, EPS)
    ratio = np.maximum(g_safe * p_safe / denom, 0.0)
    return b_safe * np.log2(1.0 + ratio)


def thermal_step(
    T: np.ndarray,
    T_amb: float,
    heating: np.ndarray,
    alpha: float,
    tau: float,
    dt: float,
) -> np.ndarray:
    """Compute the explicit first-order thermal update for a single slot."""
    T = np.asarray(T, dtype=float)
    heating = np.asarray(heating, dtype=float)
    return T_amb + (T - T_amb) * np.exp(-dt / max(tau, EPS)) + heating * alpha * dt


def compute_thermal_heating(action: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
    f = np.maximum(0.0, np.asarray(action.get("f", np.zeros_like(action.get("l", []))), dtype=float))
    p = np.maximum(0.0, np.asarray(action.get("p", np.zeros_like(action.get("l", []))), dtype=float))
    f_scale = np.asarray(params.get("f_hw_max", np.ones_like(f)), dtype=float)
    p_scale = np.asarray(params.get("p_hw_max", np.ones_like(p)), dtype=float)
    heating = 0.7 * f / np.maximum(f_scale, EPS)
    heating += 0.3 * p / np.maximum(p_scale, EPS)
    return heating


def predict_next_temperature(T: np.ndarray, action: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
    heating = compute_thermal_heating(action, params)
    return thermal_step(
        T,
        float(params.get("ambient_temp", 25.0)),
        heating=heating,
        alpha=float(params.get("thermal_alpha", 0.15)),
        tau=float(params.get("thermal_tau", 12.0)),
        dt=float(params["delta"]),
    )


def thermal_peak_temperature(T: np.ndarray, action: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
    """Endpoint-based peak estimate; not a validated within-slot physical bound."""
    start = np.asarray(T, dtype=float)
    end = predict_next_temperature(start, action, params)
    return np.maximum(start, end)


def normalize_action(action: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out = {key: np.asarray(action[key], dtype=float).copy() for key in ["f", "p", "b", "f_edge", "l", "o", "U"]}
    out["f"] = np.clip(out["f"], 0.0, np.asarray(params["f_hw_max"], dtype=float))
    out["p"] = np.clip(out["p"], 0.0, np.asarray(params["p_hw_max"], dtype=float))
    out["b"] = np.clip(out["b"], 0.0, float(params["shared_bandwidth_hz"]))
    out["f_edge"] = np.clip(out["f_edge"], 0.0, float(params["shared_edge_cpu_hz"]))
    out["l"] = np.clip(out["l"], 0.0, np.asarray(params["Q_max"], dtype=float))
    out["o"] = np.clip(out["o"], 0.0, np.asarray(params["Q_max"], dtype=float))
    out["U"] = np.clip(out["U"], 0.0, np.asarray(params["A_max"], dtype=float))
    return out


def transition_metrics(
    state: Dict[str, np.ndarray],
    action: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute actual slot metrics after validating physical executability."""
    state_report = validate_state(state, params)
    if state_report["status"] == "invalid_state":
        raise ValueError("Invalid physical state")
    if validate_arrays(action, ACTION_FIELDS, len(params["Q_max"])):
        raise ValueError("Action is not physically executable in this slot")
    tol = checker_tolerances(params)
    Q = np.asarray(state["Q"], dtype=float)
    B = np.asarray(state["B"], dtype=float)
    T = np.asarray(state["T"], dtype=float)
    g = np.asarray(state["g"], dtype=float)
    N0 = float(params.get("N0", 1e-12))

    f = np.asarray(action["f"], dtype=float)
    p = np.asarray(action["p"], dtype=float)
    b = np.asarray(action["b"], dtype=float)
    f_edge = np.asarray(action["f_edge"], dtype=float)
    l = np.asarray(action["l"], dtype=float)
    o = np.asarray(action["o"], dtype=float)
    U = np.asarray(action["U"], dtype=float)

    delta = float(params["delta"])
    d = np.asarray(params["d"], dtype=float)
    c = np.asarray(params["c"], dtype=float)
    kappa = np.asarray(params["kappa"], dtype=float)
    baseline = np.asarray(params.get("baseline_power", np.zeros_like(Q)), dtype=float)

    quantities = slot_quantities(state, action, params)
    local_time, tx_time, edge_time, tx_rate = (quantities[k] for k in
        ("local_time", "tx_time", "edge_time", "rate"))

    local_invalid = (l > 0.0) & ((f <= 0.0) | (local_time > delta + tol["time_s"]))
    offload_invalid = (o > 0.0) & (
        (tx_rate <= 0.0) | (f_edge <= 0.0) | (tx_time + edge_time > delta + tol["time_s"])
    )
    action_arrays = [f, p, b, f_edge, l, o, U]
    action_invalid = (
        any(np.any(~np.isfinite(values)) for values in action_arrays)
        or any(np.any(values < 0.0) for values in action_arrays)
        or np.any(l + o > Q + tol["work"])
        or np.any(Q - l - o + U < 0.0)
        or np.any(U > np.asarray(state["A"], dtype=float) + tol["work"])
        or np.any(f > np.asarray(params["f_hw_max"], dtype=float) + tol["frequency_hz"])
        or np.any(p > np.asarray(params["p_hw_max"], dtype=float) + tol["power_w"])
        or np.any(l > np.asarray(params["Q_max"], dtype=float) + tol["work"])
        or np.any(o > np.asarray(params["Q_max"], dtype=float) + tol["work"])
        or np.any(U > np.asarray(params["A_max"], dtype=float) + tol["work"])
        or np.sum(b) > float(params["shared_bandwidth_hz"]) + tol["bandwidth_hz"]
        or np.sum(f_edge) > float(params["shared_edge_cpu_hz"]) + tol["frequency_hz"]
        or np.any(local_invalid)
        or np.any(offload_invalid)
    )
    if action_invalid:
        raise ValueError("Action is not physically executable in this slot")

    local_energy = quantities["local_energy"]
    tx_energy = quantities["tx_energy"]
    total_energy = quantities["total_energy"]
    next_B = B - total_energy
    next_Q = Q - l - o + U

    actual_params = dict(params)
    actual_params["thermal_alpha"] = float(params.get("actual_thermal_alpha", params.get("thermal_alpha", 0.15)))
    T_next = predict_next_temperature(T, action, actual_params)
    info = {
        "local_time": local_time,
        "tx_time": tx_time,
        "edge_time": edge_time,
        "rates": tx_rate,
        "energy": total_energy,
        "peak_temp": thermal_peak_temperature(T, action, actual_params),
        "queue_next": next_Q,
        "battery_next": next_B,
        "temperature_next": T_next,
        "total_energy": total_energy,
        "local_energy": local_energy,
        "tx_energy": tx_energy,
        "baseline_energy": baseline * delta,
        "completed_local_work": l.copy(),
        "completed_offload_work": o.copy(),
        "rejected_arrivals": np.maximum(0.0, np.asarray(state["A"], dtype=float) - U),
        "shared_bandwidth_hz_used": np.sum(b),
        "shared_edge_cpu_hz_used": np.sum(f_edge),
    }
    return info


def apply_transition(
    state: Dict[str, np.ndarray],
    action: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the simulator transition model for the given action."""
    info = transition_metrics(state, action, params)
    Q = np.asarray(state["Q"], dtype=float)
    A = np.asarray(state["A"], dtype=float)
    g = np.asarray(state["g"], dtype=float)
    next_state = {
        "Q": info["queue_next"],
        "B": info["battery_next"],
        "T": info["temperature_next"],
        "g": g,
        "A": A,
        "metrics": info,
    }
    return next_state


def queue_capacity_margin(Q: np.ndarray, Q_max: np.ndarray) -> np.ndarray:
    return np.asarray(Q_max, dtype=float) - np.asarray(Q, dtype=float)


def battery_reserve_margin(B: np.ndarray, B_min: np.ndarray) -> np.ndarray:
    return np.asarray(B, dtype=float) - np.asarray(B_min, dtype=float)


def temperature_margin(T: np.ndarray, T_max: np.ndarray) -> np.ndarray:
    return np.asarray(T_max, dtype=float) - np.asarray(T, dtype=float)


def service_acceptability(throughput: float, rejection: float, latency: float, energy: float) -> bool:
    return np.isfinite(throughput) and np.isfinite(rejection) and np.isfinite(latency) and np.isfinite(energy)
