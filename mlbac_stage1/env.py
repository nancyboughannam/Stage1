from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from .config import make_default_params
from .constraints import evaluate_constraints, evaluate_post_transition
from .models import apply_transition
from .safety_filter import SafetyFilter


class MBACEnv(gym.Env):
    """MEC allocation simulator with one-step candidate verification."""

    def __init__(
        self,
        n_devices: int = 1,
        horizon: int = 10,
        *,
        params: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        max_steps: Optional[int] = None,
        action_scale: float = 1.0,
    ):
        self.n_devices = n_devices
        self.params = make_default_params(n_devices, deterministic=deterministic) if params is None else dict(params)
        self._actual_thermal_alpha = self.params.pop("actual_thermal_alpha", None)
        self.params["deterministic"] = bool(self.params.get("deterministic", deterministic))
        self.params.setdefault("A_max", np.full(n_devices, 1.0, dtype=float))
        self.params.setdefault("B_min", np.full(n_devices, 0.1 * 9.0 * 3600.0, dtype=float))
        self.params.setdefault("B_cap", np.full(n_devices, 9.0 * 3600.0, dtype=float))
        self.params.setdefault("Q_max", np.full(n_devices, 10.0, dtype=float))
        self.params.setdefault("T_max", np.full(n_devices, 80.0, dtype=float))
        self.horizon = horizon if max_steps is None else max_steps
        if self.horizon < 1:
            raise ValueError("horizon must be at least one slot")
        self.action_scale = action_scale
        self.timestep = 0
        self.current_step = 0
        self.nominal_action = None
        self.executed_action = None

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7 * n_devices,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full(5 * n_devices + 3, -np.finfo(np.float32).max, dtype=np.float32),
            high=np.full(5 * n_devices + 3, np.finfo(np.float32).max, dtype=np.float32),
            dtype=np.float32,
        )

        self.filter = SafetyFilter(self.params)
        self.rng = np.random.default_rng(0)
        self.state = self._initial_state()
        self.last_info: Dict[str, Any] = {}
        self._episode_done = False

    def _initial_state(self) -> Dict[str, np.ndarray]:
        q = np.full(self.n_devices, 1.0, dtype=float)
        b = np.full(self.n_devices, 0.8 * np.asarray(self.params["B_cap"], dtype=float), dtype=float)
        t = np.full(self.n_devices, 30.0, dtype=float)
        a = np.full(self.n_devices, 0.2, dtype=float)
        g = np.asarray(self.params.get("g_base", np.full(self.n_devices, 1e-5, dtype=float)), dtype=float)
        return {"Q": q, "B": b, "T": t, "A": a, "g": g}

    def _sample_arrivals(self) -> np.ndarray:
        if self.params.get("deterministic", False):
            return np.full(self.n_devices, 0.3, dtype=float)
        return self.rng.uniform(0.0, 1.0, size=self.n_devices)

    def _decode_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        raw = np.asarray(action, dtype=float).reshape(self.n_devices, 7)
        bounded = np.clip(raw, -1.0, 1.0)
        ratios = (bounded + 1.0) * 0.5 * float(self.action_scale)
        dec = {
            "f": ratios[:, 0] * np.asarray(self.params["f_hw_max"], dtype=float),
            "p": ratios[:, 1] * np.asarray(self.params["p_hw_max"], dtype=float),
            "b": ratios[:, 2] * float(self.params["shared_bw"]),
            "f_edge": ratios[:, 3] * float(self.params["shared_edge_cpu"]),
            "l": ratios[:, 4] * np.asarray(self.params["Q_max"], dtype=float),
            "o": ratios[:, 5] * np.asarray(self.params["Q_max"], dtype=float),
            "U": ratios[:, 6] * np.asarray(self.params["A_max"], dtype=float),
        }
        return dec

    def _observation(self) -> np.ndarray:
        q = np.asarray(self.state["Q"], dtype=float) / np.asarray(self.params["Q_max"], dtype=float)
        b = np.asarray(self.state["B"], dtype=float) / np.asarray(self.params["B_cap"], dtype=float)
        t = np.asarray(self.state["T"], dtype=float) / float(self.params.get("observation_temperature_scale_c", 100.0))
        a = np.asarray(self.state["A"], dtype=float) / np.asarray(self.params["A_max"], dtype=float)
        g = np.asarray(self.state["g"], dtype=float) / float(self.params.get("observation_channel_gain_scale", 1e-5))
        remaining = max(0, self.horizon - self.timestep) / self.horizon
        globals_ = np.array(
            [
                float(self.params["shared_bw"]) / float(self.params.get("observation_bandwidth_scale_bps", 10e6)),
                float(self.params["shared_edge_cpu"]) / float(self.params.get("observation_edge_cpu_scale_hz", 10e9)),
                remaining,
            ],
            dtype=float,
        )
        return np.concatenate([q, b, t, a, g, globals_]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.timestep = 0
        self.current_step = 0
        self.nominal_action = None
        self.executed_action = None
        self.state = self._initial_state()
        self._episode_done = False
        self.state["A"] = np.clip(self._sample_arrivals(), 0.0, np.asarray(self.params["A_max"], dtype=float))
        options = options or {}
        supplied_state = options.get("state")
        if supplied_state is not None:
            for key in ("Q", "B", "T", "A", "g"):
                if key in supplied_state:
                    value = np.asarray(supplied_state[key], dtype=float)
                    if value.shape != (self.n_devices,):
                        raise ValueError(f"reset state {key} must have shape ({self.n_devices},)")
                    self.state[key] = value.copy()
        self.last_info = {}
        return self._observation(), {"physical_state": self._copy_state(self.state)}

    @staticmethod
    def _copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
        return {k: np.asarray(v, dtype=float).copy() for k, v in state.items() if k in {"Q", "B", "T", "A", "g"}}

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("step() called after episode termination; call reset() first")
        raw_action = np.asarray(action, dtype=float).copy()
        bounded_action = np.clip(raw_action, -1.0, 1.0)
        self.nominal_action = self._decode_action(raw_action)
        pre_state = self._copy_state(self.state)
        elapsed_time = self.timestep * float(self.params["delta"])
        filtered = self.filter.filter_action(self.state, self.nominal_action)
        action_preprocessed = not np.array_equal(raw_action, bounded_action)
        if action_preprocessed and filtered["safe"] and filtered["status"] == "nominal_accepted":
            filtered["status"] = "preprocessed_candidate_verified"
        chosen = filtered["action"]
        intervention = filtered["status"] != "nominal_accepted"
        base_info = {
            "raw_policy_action": raw_action,
            "bounded_policy_action": bounded_action,
            "decoded_nominal_action": {k: v.copy() for k, v in self.nominal_action.items()},
            "proposed_action": {k: v.copy() for k, v in self.nominal_action.items()},
            "pre_state": pre_state,
            "filter_status": filtered["status"],
            "intervention": intervention,
            "action_preprocessed": action_preprocessed,
            "verification_status": "verified_one_step" if filtered["safe"] else "no_verified_candidate_found",
            "verification_scope": filtered.get("verification_scope", "one_step"),
            "fallback_mode": "idle" if filtered["status"] == "fallback_verified" else ("idle_candidate" if filtered["status"] == "no_verified_safe_continuation" else None),
            "predicted_report": filtered["report"],
            "proposal_predicted_report": filtered["proposal_report"],
            "infeasibility_proven": filtered["infeasibility_proven"],
            "predicted_state": filtered["report"].get("predicted_state"),
            "elapsed_time_s": elapsed_time,
            "elapsed_time_s_start": elapsed_time,
            "outstanding_work": np.asarray(self.state["Q"], dtype=float).copy(),
            "outstanding_queue_work": np.asarray(self.state["Q"], dtype=float).copy(),
            "outstanding_arrival_work": np.asarray(self.state["A"], dtype=float).copy(),
            "total_outstanding_work": np.asarray(self.state["Q"], dtype=float) + np.asarray(self.state["A"], dtype=float),
        }
        if not filtered["safe"]:
            self.executed_action = None
            self.timestep += 1
            self.current_step = self.timestep
            penalty = float(self.params.get("controller_failure_penalty", 10.0))
            terminated = True
            truncated = False
            info = {
                **base_info,
                "executed_action": None,
                "actual_outcome": "unobserved",
                "post_state": None,
                "transition_metrics": None,
                "actual_report": None,
                "controller_failure": True,
                "failure_kind": filtered.get("failure_kind", "no_verified_candidate_found"),
                "failure_reason": "no candidate passed the one-step predicted constraints",
                "episode_ending_reason": "controller_failure_no_verified_continuation",
                "reward": -penalty,
                "feasible": False,
                "terminated": terminated,
                "truncated": truncated,
            }
            self.last_info = info
            self._episode_done = True
            return self._observation(), -penalty, terminated, truncated, info

        self.executed_action = chosen
        transition_params = self.params
        if self._actual_thermal_alpha is not None:
            transition_params = {**self.params, "actual_thermal_alpha": self._actual_thermal_alpha}
        next_state = apply_transition(self.state, chosen, transition_params)
        next_state["A"] = np.clip(self._sample_arrivals(), 0.0, np.asarray(self.params["A_max"], dtype=float))
        actual_report = evaluate_post_transition(next_state, next_state["metrics"], self.params)
        self.state = next_state
        terminated = self.timestep + 1 >= self.horizon
        truncated = False
        reward = float(np.sum(chosen["U"]) - 0.01 * np.sum(np.maximum(0.0, self.state["Q"])))
        actual_violation = not actual_report["feasible"]
        if actual_violation:
            reward -= float(self.params.get("actual_violation_penalty", 10.0))
            terminated = True
        self.timestep += 1
        self.current_step = self.timestep
        ending_reason = "actual_constraint_violation" if actual_violation else ("finite_horizon_complete" if terminated else None)
        info = {
            **base_info,
            "elapsed_time_s": self.timestep * float(self.params["delta"]),
            "executed_action": {k: np.asarray(v, dtype=float).copy() for k, v in chosen.items()},
            "actual_outcome": "observed",
            "post_state": self._copy_state(self.state),
            "transition_metrics": self.state["metrics"],
            "actual_report": actual_report,
            "controller_failure": False,
            "failure_kind": None,
            "failure_reason": None,
            "episode_ending_reason": ending_reason,
            "reward": reward,
            "feasible": actual_report["feasible"],
            "terminated": terminated,
            "truncated": truncated,
        }
        self.last_info = info
        self._episode_done = bool(terminated or truncated)
        return self._observation(), reward, terminated, truncated, info


def make_env(**kwargs):
    return MBACEnv(**kwargs)


__all__ = ["MBACEnv", "make_env"]
