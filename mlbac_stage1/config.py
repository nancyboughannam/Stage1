from __future__ import annotations

from typing import Any, Dict

import numpy as np


def make_default_params(n_devices: int = 1, *, deterministic: bool = False) -> Dict[str, Any]:
    q_max = np.full(n_devices, 10.0, dtype=float)
    b_cap = np.full(n_devices, 9.0 * 3600.0, dtype=float)
    b_min = np.full(n_devices, 0.10 * 9.0 * 3600.0, dtype=float)
    t_max = np.full(n_devices, 80.0, dtype=float)
    a_max = np.full(n_devices, 1.0, dtype=float)
    d = np.full(n_devices, 0.2e6, dtype=float)
    c = np.full(n_devices, 40e6, dtype=float)
    f_hw_max = np.full(n_devices, 1.0e9, dtype=float)
    p_hw_max = np.full(n_devices, 1.0, dtype=float)
    kappa = np.full(n_devices, 1e-26, dtype=float)
    baseline_power = np.full(n_devices, 1.0, dtype=float)
    if deterministic:
        g_base = np.full(n_devices, 1e-5, dtype=float)
    else:
        rng = np.random.default_rng(42)
        g_base = rng.uniform(5e-6, 2e-5, size=n_devices)
    params = {
        "n_devices": n_devices,
        "Q_max": q_max,
        "B_cap": b_cap,
        "B_min": b_min,
        "T_max": t_max,
        "A_max": a_max,
        "shared_bw": 10e6,
        "shared_edge_cpu": 10e9,
        "delta": 1.0,
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
        "g_seed": 42,
        "g_base": g_base,
        "queue_capacity_margin": 0.05,
        "temperature_margin": 0.05,
        "energy_margin": 0.05,
    }
    return params


def make_debug_params(n_devices: int = 1) -> Dict[str, Any]:
    params = make_default_params(n_devices, deterministic=True)
    params["A_max"] = np.full(n_devices, 2.0, dtype=float)
    params["g_base"] = np.full(n_devices, 2e-5, dtype=float)
    params["shared_bw"] = 10e6
    params["shared_edge_cpu"] = 10e9
    return params
