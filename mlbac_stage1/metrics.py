from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class RunMetrics:
    throughput: float
    rejection_fraction: float
    latency: float
    energy: float
    completion_rate: float
    violation_rate: float
    backlog: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "throughput": float(self.throughput),
            "rejection_fraction": float(self.rejection_fraction),
            "latency": float(self.latency),
            "energy": float(self.energy),
            "completion_rate": float(self.completion_rate),
            "violation_rate": float(self.violation_rate),
            "backlog": float(self.backlog),
        }


def summarize_run(history: List[Dict[str, Any]]) -> RunMetrics:
    if not history:
        return RunMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    throughput = float(np.mean([h.get("throughput", 0.0) for h in history]))
    rejection_fraction = float(np.mean([h.get("rejection_fraction", 0.0) for h in history]))
    latency = float(np.mean([h.get("latency", 0.0) for h in history]))
    energy = float(np.mean([h.get("energy", 0.0) for h in history]))
    completion_rate = float(np.mean([h.get("completion_rate", 0.0) for h in history]))
    violation_rate = float(np.mean([h.get("violation_rate", 0.0) for h in history]))
    backlog = float(np.mean([h.get("backlog", 0.0) for h in history]))
    return RunMetrics(
        throughput=throughput,
        rejection_fraction=rejection_fraction,
        latency=latency,
        energy=energy,
        completion_rate=completion_rate,
        violation_rate=violation_rate,
        backlog=backlog,
    )
