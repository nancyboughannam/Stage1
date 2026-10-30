from __future__ import annotations

from typing import Any, Dict

import numpy as np


# Absolute numerical tolerances in the indicated physical units; not error bounds.
CHECKER_TOLERANCES = {
    "work": 1e-9, "energy_j": 1e-9, "temperature_c": 1e-9,
    "time_s": 1e-9, "frequency_hz": 1e-9, "bandwidth_hz": 1e-9,
    "power_w": 1e-9,
}


def checker_tolerances(params):
    values = dict(CHECKER_TOLERANCES)
    supplied = params.get("checker_tolerances", {})
    if set(supplied) - set(values):
        raise ValueError("Unknown checker tolerance")
    values.update(supplied)
    if any(not np.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Checker tolerances must be finite and nonnegative")
    return values


def make_default_params(
    n_devices: int = 5,
    *,
    deterministic: bool = False,
    horizon: int = 300,
    delta: float = 1.0,
    arrival_min_per_slot: float = 0.0,
    arrival_max_per_slot: float = 2.0,
    deterministic_arrival_per_slot: float = 0.3,
    channel_gain_min: float = 5e-6,
    channel_gain_max: float = 2e-5,
    deterministic_channel_gain: float = 1e-5,
    initial_queue_work_units: float = 1.0,
    initial_battery_fraction: float = 0.8,
    initial_temperature_c: float = 30.0,
) -> Dict[str, Any]:
    numeric = (delta, arrival_min_per_slot, arrival_max_per_slot,
               deterministic_arrival_per_slot, channel_gain_min, channel_gain_max,
               deterministic_channel_gain, initial_queue_work_units,
               initial_battery_fraction, initial_temperature_c)
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("Configuration values must be finite")
    if not isinstance(n_devices, (int, np.integer)) or not isinstance(horizon, (int, np.integer)):
        raise ValueError("device count and horizon must be integers")
    if n_devices < 1 or horizon < 1 or delta <= 0.0:
        raise ValueError("device count and horizon must be positive; slot duration must be positive")
    if arrival_min_per_slot < 0.0 or arrival_max_per_slot < arrival_min_per_slot:
        raise ValueError("arrival bounds must be nonnegative and ordered")
    if deterministic and not arrival_min_per_slot <= deterministic_arrival_per_slot <= arrival_max_per_slot:
        raise ValueError("deterministic arrival must be within the configured arrival bounds")
    if channel_gain_min < 0.0 or channel_gain_max < channel_gain_min:
        raise ValueError("channel gain bounds must be nonnegative and ordered")
    if deterministic and not channel_gain_min <= deterministic_channel_gain <= channel_gain_max:
        raise ValueError("deterministic channel gain must be within the configured bounds")
    if initial_queue_work_units < 0.0 or not 0.0 <= initial_battery_fraction <= 1.0:
        raise ValueError("initial queue must be nonnegative and battery fraction must be in [0, 1]")
    q_max = np.full(n_devices, 10.0, dtype=float)
    b_cap = np.full(n_devices, 9.0 * 3600.0, dtype=float)
    b_min = np.full(n_devices, 0.10 * 9.0 * 3600.0, dtype=float)
    t_max = np.full(n_devices, 80.0, dtype=float)
    a_max = np.full(n_devices, arrival_max_per_slot, dtype=float)
    d = np.full(n_devices, 0.2 * 1_000_000 * 8, dtype=float)
    c = np.full(n_devices, 40e6, dtype=float)
    f_hw_max = np.full(n_devices, 1.0e9, dtype=float)
    p_hw_max = np.full(n_devices, 1.0, dtype=float)
    kappa = np.full(n_devices, 1e-26, dtype=float)
    baseline_power = np.full(n_devices, 1.0, dtype=float)
    g_base = np.full(n_devices, deterministic_channel_gain, dtype=float)
    params = {
        "checker_tolerances": dict(CHECKER_TOLERANCES),
        "verification_scope": "one_step_predicted_zero_model_margin",
        "n_devices": n_devices,
        "n_edge_servers": 1,
        "horizon": horizon,
        "deterministic": bool(deterministic),
        "Q_max": q_max,
        "B_cap": b_cap,
        "B_min": b_min,
        "T_max": t_max,
        "A_max": a_max,
        "shared_bandwidth_hz": 10e6,
        "shared_edge_cpu_hz": 10e9,
        "delta": delta,
        "d": d,
        "c": c,
        "f_hw_max": f_hw_max,
        "p_hw_max": p_hw_max,
        "kappa": kappa,
        "baseline_power": baseline_power,
        "ambient_temp": 25.0,
        "thermal_alpha": 0.15,
        "thermal_tau": 12.0,
        "N0": 1e-12,
        "g_base": g_base,
        "arrival_min_per_slot": arrival_min_per_slot,
        "arrival_max_per_slot": arrival_max_per_slot,
        "deterministic_arrival_per_slot": deterministic_arrival_per_slot,
        "channel_gain_min": channel_gain_min,
        "channel_gain_max": channel_gain_max,
        "deterministic_channel_gain": deterministic_channel_gain,
        "channel_distribution_label": "synthetic_uniform_development_not_calibrated_propagation",
        "initial_queue_work_units": initial_queue_work_units,
        "initial_battery_fraction": initial_battery_fraction,
        "initial_temperature_c": initial_temperature_c,
        "observation_temperature_scale_c": 100.0,
        "observation_channel_gain_scale": 1e-5,
        "observation_bandwidth_scale_hz": 10e6,
        "observation_edge_cpu_scale_hz": 10e9,
        "controller_failure_penalty": 10.0,
        "actual_violation_penalty": 10.0,
        "task_size_description": "0.2 decimal MB = 1,600,000 bits",
        "provisional_parameter_labels": {
            "p_hw_max": "provisional_uncalibrated",
            "kappa": "provisional_uncalibrated",
            "baseline_power": "provisional_uncalibrated",
            "N0": "provisional_uncalibrated",
            "thermal_alpha": "provisional_uncalibrated",
            "thermal_tau": "provisional_uncalibrated",
        },
    }
    return params


def make_debug_params(n_devices: int = 1) -> Dict[str, Any]:
    params = make_default_params(n_devices, deterministic=True, horizon=10)
    params["A_max"] = np.full(n_devices, 2.0, dtype=float)
    params["g_base"] = np.full(n_devices, 2e-5, dtype=float)
    params["deterministic_channel_gain"] = 2e-5
    return params


def action_metric_settings(params):
    """Derive scales from effective configuration (including scenario overrides).

    Zero-range components are fixed at zero and excluded using active masks.
    No optimizer is enabled by these settings in Batch A.
    """
    n = len(params["Q_max"])
    scales = {
        "f": np.asarray(params["f_hw_max"], dtype=float).copy(),
        "p": np.asarray(params["p_hw_max"], dtype=float).copy(),
        "b": np.full(n, params["shared_bandwidth_hz"], dtype=float),
        "f_edge": np.full(n, params["shared_edge_cpu_hz"], dtype=float),
        "l": np.asarray(params["Q_max"], dtype=float).copy(),
        "o": np.asarray(params["Q_max"], dtype=float).copy(),
        "U": np.asarray(params["A_max"], dtype=float).copy(),
    }
    if any(v.shape != (n,) or not np.isfinite(v).all() or np.any(v < 0) for v in scales.values()):
        raise ValueError("Action scales must be finite, nonnegative arrays of length N")
    return {"scales": scales, "weights": {k: np.ones(n) for k in scales},
            "active": {k: v > 0 for k, v in scales.items()}}
