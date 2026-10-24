from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class DeviceSpec:
    data_bits: float
    cpu_cycles: float
    local_cpu_max: float
    power_max: float
    baseline_power: float
    kappa: float


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


def normalize_action(action: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out = {key: np.asarray(action[key], dtype=float).copy() for key in ["f", "p", "b", "f_edge", "l", "o", "U"]}
    out["f"] = np.clip(out["f"], 0.0, np.asarray(params["f_hw_max"], dtype=float))
    out["p"] = np.clip(out["p"], 0.0, np.asarray(params["p_hw_max"], dtype=float))
    out["b"] = np.clip(out["b"], 0.0, float(params["shared_bw"]))
    out["f_edge"] = np.clip(out["f_edge"], 0.0, float(params["shared_edge_cpu"]))
    out["l"] = np.clip(out["l"], 0.0, np.asarray(params["Q_max"], dtype=float))
    out["o"] = np.clip(out["o"], 0.0, np.asarray(params["Q_max"], dtype=float))
    out["U"] = np.clip(out["U"], 0.0, np.asarray(params["A_max"], dtype=float))
    return out


def transition_metrics(
    state: Dict[str, np.ndarray],
    action: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Shared simulator interface used for both transition execution and constraint checks."""
    Q = np.asarray(state["Q"], dtype=float)
    B = np.asarray(state["B"], dtype=float)
    T = np.asarray(state["T"], dtype=float)
    g = np.asarray(state["g"], dtype=float)
    N0 = float(params.get("N0", 1e-12))

    f = np.maximum(0.0, np.asarray(action["f"], dtype=float))
    p = np.maximum(0.0, np.asarray(action["p"], dtype=float))
    b = np.maximum(0.0, np.asarray(action["b"], dtype=float))
    f_edge = np.maximum(0.0, np.asarray(action["f_edge"], dtype=float))
    l = np.maximum(0.0, np.asarray(action["l"], dtype=float))
    o = np.maximum(0.0, np.asarray(action["o"], dtype=float))
    U = np.maximum(0.0, np.asarray(action["U"], dtype=float))

    delta = float(params["delta"])
    d = np.asarray(params["d"], dtype=float)
    c = np.asarray(params["c"], dtype=float)
    kappa = np.asarray(params["kappa"], dtype=float)
    baseline = np.asarray(params.get("baseline_power", np.zeros_like(Q)), dtype=float)

    local_time = np.where(l > 0.0, c * l / np.maximum(f, EPS), 0.0)
    tx_rate = compute_rates(g, b, p, N0)
    tx_time = np.where(o > 0.0, d * o / np.maximum(tx_rate, EPS), 0.0)
    edge_time = np.where(o > 0.0, c * o / np.maximum(f_edge, EPS), 0.0)

    local_energy = np.where(f > 0.0, kappa * np.power(f, 3.0) * local_time, 0.0)
    tx_energy = np.where(tx_rate > 0.0, p * tx_time, 0.0)
    total_energy = local_energy + tx_energy + baseline * delta
    next_B = B - total_energy
    next_Q = Q - l - o + U

    T_next = predict_next_temperature(T, action, params)
    info = {
        "local_time": local_time,
        "tx_time": tx_time,
        "edge_time": edge_time,
        "rates": tx_rate,
        "energy": total_energy,
        "peak_temp": np.maximum(T, T_next),
        "queue_next": next_Q,
        "battery_next": next_B,
        "temperature_next": T_next,
        "total_energy": total_energy,
        "shared_bw_used": np.sum(b),
        "shared_edge_cpu_used": np.sum(f_edge),
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
