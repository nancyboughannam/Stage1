from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .constraints import evaluate_constraints


class SafetyFilter:
    """A heuristic action-scaling search that returns one-step model-verified candidates."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.params.setdefault("A_max", np.full(len(self.params.get("Q_max", np.array([1.0]))), 1.0, dtype=float))

    def _normalize_action(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {k: np.asarray(v, dtype=float).copy() for k, v in action.items()}
        out["f"] = np.clip(out["f"], 0.0, np.asarray(self.params["f_hw_max"], dtype=float))
        out["p"] = np.clip(out["p"], 0.0, np.asarray(self.params["p_hw_max"], dtype=float))
        out["b"] = np.clip(out["b"], 0.0, float(self.params["shared_bandwidth_hz"]))
        out["f_edge"] = np.clip(out["f_edge"], 0.0, float(self.params["shared_edge_cpu_hz"]))
        out["l"] = np.clip(out["l"], 0.0, np.asarray(self.params["Q_max"], dtype=float))
        out["o"] = np.clip(out["o"], 0.0, np.asarray(self.params["Q_max"], dtype=float))
        out["U"] = np.clip(out["U"], 0.0, np.asarray(self.params["A_max"], dtype=float))
        return out

    def _candidate_from_nominal(self, state: Dict[str, np.ndarray], action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cand = {k: np.asarray(v, dtype=float).copy() for k, v in action.items()}
        q = np.asarray(state["Q"], dtype=float)
        b = np.asarray(state["B"], dtype=float)
        t = np.asarray(state["T"], dtype=float)
        a = np.asarray(state["A"], dtype=float)
        d = np.asarray(self.params["d"], dtype=float)
        c = np.asarray(self.params["c"], dtype=float)
        n = len(q)

        scales = [
            np.asarray(self.params["f_hw_max"], dtype=float),
            np.asarray(self.params["p_hw_max"], dtype=float),
            np.full(n, float(self.params["shared_bandwidth_hz"]), dtype=float),
            np.full(n, float(self.params["shared_edge_cpu_hz"]), dtype=float),
            np.asarray(self.params["Q_max"], dtype=float),
            np.asarray(self.params["Q_max"], dtype=float),
            np.asarray(self.params["A_max"], dtype=float),
        ]
        for idx in range(n):
            for key, scale in zip(["f", "p", "b", "f_edge", "l", "o", "U"], scales):
                if key in cand and np.isscalar(cand[key]):
                    cand[key] = np.asarray(cand[key], dtype=float) * np.ones(n)
                v = np.asarray(cand[key], dtype=float)
                cand[key] = np.clip(v, 0.0, scale)

        factor = 1.0
        for _ in range(30):
            report = evaluate_constraints(state, cand, self.params)
            if report["feasible"]:
                return cand
            factor *= 0.8
            cand["U"] = factor * np.minimum(cand["U"], a)
            cand["l"] = factor * np.minimum(cand["l"], q)
            cand["o"] = factor * np.minimum(cand["o"], q)
            cand["f"] = factor * np.minimum(cand["f"], np.asarray(self.params["f_hw_max"], dtype=float))
            cand["p"] = factor * np.minimum(cand["p"], np.asarray(self.params["p_hw_max"], dtype=float))
            cand["b"] = factor * np.minimum(cand["b"], float(self.params["shared_bandwidth_hz"]))
            cand["f_edge"] = factor * np.minimum(cand["f_edge"], float(self.params["shared_edge_cpu_hz"]))
        return cand

    def filter_action(self, state: Dict[str, np.ndarray], action: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Return a one-step verified candidate or an explicit verification failure."""
        proposal_report = evaluate_constraints(state, action, self.params)
        nominal = self._normalize_action(action)
        report = evaluate_constraints(state, nominal, self.params)
        if report["feasible"]:
            changed = any(not np.array_equal(np.asarray(action[k]), nominal[k]) for k in nominal)
            status = "preprocessed_candidate_verified" if changed else "nominal_accepted"
            return {"action": nominal, "safe": True, "status": status, "verification_scope": "one_step", "report": report, "proposal_report": proposal_report, "infeasibility_proven": False}

        corrected = self._candidate_from_nominal(state, nominal)
        corrected_report = evaluate_constraints(state, corrected, self.params)
        if corrected_report["feasible"]:
            return {"action": corrected, "safe": True, "status": "corrected_candidate_verified", "verification_scope": "one_step", "report": corrected_report, "proposal_report": proposal_report, "infeasibility_proven": False}

        shutdown = {
            "f": np.zeros_like(np.asarray(state["Q"], dtype=float)),
            "p": np.zeros_like(np.asarray(state["Q"], dtype=float)),
            "b": np.zeros_like(np.asarray(state["Q"], dtype=float)),
            "f_edge": np.zeros_like(np.asarray(state["Q"], dtype=float)),
            "l": np.zeros_like(np.asarray(state["Q"], dtype=float)),
            "o": np.zeros_like(np.asarray(state["Q"], dtype=float)),
            "U": np.zeros_like(np.asarray(state["Q"], dtype=float)),
        }
        fallback_report = evaluate_constraints(state, shutdown, self.params)
        if fallback_report["feasible"]:
            return {"action": shutdown, "safe": True, "status": "fallback_verified", "verification_scope": "one_step", "report": fallback_report, "proposal_report": proposal_report, "infeasibility_proven": False}
        return {
            "action": None,
            "safe": False,
            "status": "no_verified_safe_continuation",
            "verification_scope": "one_step",
            "failure_kind": "no_verified_candidate_found",
            "infeasibility_proven": False,
            "report": fallback_report,
            "proposal_report": proposal_report,
        }
