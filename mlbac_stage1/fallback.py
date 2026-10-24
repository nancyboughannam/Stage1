from __future__ import annotations

from typing import Any, Dict

import numpy as np


class FallbackController:
    """One-slot idle action; baseline energy consumption continues."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def action_for(self, state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        n = len(state["Q"])
        return {
            "f": np.zeros(n),
            "p": np.zeros(n),
            "b": np.zeros(n),
            "f_edge": np.zeros(n),
            "l": np.zeros(n),
            "o": np.zeros(n),
            "U": np.zeros(n),
        }

    def describe(self) -> Dict[str, Any]:
        baseline = np.asarray(self.params.get("baseline_power", []), dtype=float)
        delta = float(self.params.get("delta", 1.0))
        return {
            "mode": "idle",
            "entry_energy_j": 0.0,
            "verification_scope": "one_step_only",
            "ongoing_power_w": baseline.copy(),
            "one_step_baseline_energy_j": baseline * delta,
            "queue_retention": "retained",
            "remaining_horizon_verified": False,
        }
