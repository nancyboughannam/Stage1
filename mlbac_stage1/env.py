from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from .config import make_default_params, safety_margin_settings, safety_verification_scope
from .constraints import evaluate_constraints, evaluate_post_transition
from .models import apply_transition, validate_state
from .safety_filter import SafetyFilter
from .model_uncertainty import nonnegative_scalar, prediction_error_report


ACTION_COMPONENTS = ("f", "p", "b", "f_edge", "l", "o", "U")


class MBACEnv(gym.Env):
    """MEC allocation simulator with one-step candidate verification."""

    def __init__(
        self,
        n_devices: Optional[int] = None,
        horizon: Optional[int] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        deterministic: Optional[bool] = None,
        max_steps: Optional[int] = None,
        action_scale: float = 1.0,
    ):
        if params is None:
            resolved_devices = 5 if n_devices is None else n_devices
            default_horizon = 300 if horizon is None and max_steps is None else (horizon if horizon is not None else max_steps)
            self.params = make_default_params(resolved_devices, deterministic=bool(deterministic), horizon=default_horizon)
        else:
            supplied = dict(params)
            resolved_devices = int(supplied.get("n_devices", n_devices if n_devices is not None else len(supplied.get("Q_max", [])) or 5))
            default_horizon = int(supplied.get("horizon", horizon if horizon is not None else max_steps or 300))
            self.params = make_default_params(
                resolved_devices,
                deterministic=bool(supplied.get("deterministic", False) if deterministic is None else deterministic),
                horizon=default_horizon,
                delta=float(supplied.get("delta", 1.0)),
            )
            self.params.update(supplied)
        self.n_devices = resolved_devices if n_devices is None else n_devices
        if self.n_devices != resolved_devices:
            raise ValueError("n_devices must match the supplied parameter arrays")
        self._actual_thermal_alpha = self.params.pop("actual_thermal_alpha", None)
        if self._actual_thermal_alpha is not None:
            self._actual_thermal_alpha = nonnegative_scalar(self._actual_thermal_alpha, "actual_thermal_alpha")
        self._actual_energy_multiplier = nonnegative_scalar(
            self.params.pop("actual_energy_multiplier", 1.0), "actual_energy_multiplier")
        if deterministic is not None:
            self.params["deterministic"] = bool(deterministic)
        self.horizon = int(self.params.get("horizon", 300)) if horizon is None else int(horizon)
        if max_steps is not None:
            self.horizon = int(max_steps)
        self.params["n_devices"] = self.n_devices
        self.params["n_edge_servers"] = 1
        self.params["horizon"] = self.horizon
        if self.horizon < 1:
            raise ValueError("horizon must be at least one slot")
        if np.any(np.asarray(self.params["A_max"], dtype=float) + 1e-9 < float(self.params["arrival_max_per_slot"])):
            raise ValueError("A_max must cover the configured maximum arrival per slot")
        self.action_scale = action_scale
        self.timestep = 0
        self.current_step = 0
        self.nominal_action = None
        self.executed_action = None

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7 * self.n_devices,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full(5 * self.n_devices + 3, -np.finfo(np.float32).max, dtype=np.float32),
            high=np.full(5 * self.n_devices + 3, np.finfo(np.float32).max, dtype=np.float32),
            dtype=np.float32,
        )

        self.filter = SafetyFilter(self.params)
        self._arrival_rng: Optional[np.random.Generator] = None
        self._channel_rng: Optional[np.random.Generator] = None
        self._arrival_trace = np.empty((0, self.n_devices), dtype=float)
        self._channel_trace = np.empty((0, self.n_devices), dtype=float)
        self._last_seed: Optional[int] = None
        self.state = self._initial_state()
        self.last_info: Dict[str, Any] = {}
        self._episode_done = False

    def _initial_state(self) -> Dict[str, np.ndarray]:
        q = np.full(self.n_devices, float(self.params["initial_queue_work_units"]), dtype=float)
        b = np.asarray(self.params["B_cap"], dtype=float) * float(self.params["initial_battery_fraction"])
        t = np.full(self.n_devices, float(self.params["initial_temperature_c"]), dtype=float)
        a = np.full(self.n_devices, float(self.params["deterministic_arrival_per_slot"]), dtype=float)
        g = np.asarray(self.params.get("g_base", np.full(self.n_devices, 1e-5, dtype=float)), dtype=float)
        return {"Q": q, "B": b, "T": t, "A": a, "g": g}

    def _make_exogenous_traces(self) -> None:
        if self.params.get("deterministic", False):
            self._arrival_trace = np.full((self.horizon + 1, self.n_devices), float(self.params["deterministic_arrival_per_slot"]))
            self._channel_trace = np.full((self.horizon + 1, self.n_devices), float(self.params["deterministic_channel_gain"]))
            return
        if self._arrival_rng is None or self._channel_rng is None:
            raise RuntimeError("reset() must initialize the exogenous random streams first")
        self._arrival_trace = self._arrival_rng.uniform(
            float(self.params["arrival_min_per_slot"]),
            float(self.params["arrival_max_per_slot"]),
            size=(self.horizon + 1, self.n_devices),
        )
        self._channel_trace = self._channel_rng.uniform(
            float(self.params["channel_gain_min"]),
            float(self.params["channel_gain_max"]),
            size=(self.horizon + 1, self.n_devices),
        )

    def _decode_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        raw = np.asarray(action, dtype=float).reshape(self.n_devices, 7)
        bounded = np.clip(raw, -1.0, 1.0)
        ratios = (bounded + 1.0) * 0.5 * float(self.action_scale)
        dec = {
            "f": ratios[:, 0] * np.asarray(self.params["f_hw_max"], dtype=float),
            "p": ratios[:, 1] * np.asarray(self.params["p_hw_max"], dtype=float),
            "b": ratios[:, 2] * float(self.params["shared_bandwidth_hz"]),
            "f_edge": ratios[:, 3] * float(self.params["shared_edge_cpu_hz"]),
            "l": ratios[:, 4] * np.asarray(self.params["Q_max"], dtype=float),
            "o": ratios[:, 5] * np.asarray(self.params["Q_max"], dtype=float),
            "U": ratios[:, 6] * np.asarray(self.params["A_max"], dtype=float),
        }
        return dec

    def _observation(self) -> np.ndarray:
        q = np.asarray(self.state["Q"], dtype=float) / np.asarray(self.params["Q_max"], dtype=float)
        b = np.asarray(self.state["B"], dtype=float) / np.asarray(self.params["B_cap"], dtype=float)
        t = np.asarray(self.state["T"], dtype=float) / float(self.params["observation_temperature_scale_c"])
        arrivals = np.asarray(self.state["A"], dtype=float)
        arrival_max = np.asarray(self.params["A_max"], dtype=float)
        # A zero arrival bound means no arriving work, represented by zero.
        a = np.divide(arrivals, arrival_max, out=np.zeros_like(arrivals), where=arrival_max > 0.0)
        g = np.asarray(self.state["g"], dtype=float) / float(self.params["observation_channel_gain_scale"])
        remaining = max(0, self.horizon - self.timestep) / self.horizon
        globals_ = np.array(
            [
                float(self.params["shared_bandwidth_hz"]) / float(self.params["observation_bandwidth_scale_hz"]),
                float(self.params["shared_edge_cpu_hz"]) / float(self.params["observation_edge_cpu_scale_hz"]),
                remaining,
            ],
            dtype=float,
        )
        return np.concatenate([q, b, t, a, g, globals_]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._last_seed = int(self.np_random.integers(0, np.iinfo(np.int64).max)) if seed is None else int(seed)
        arrival_seed, channel_seed = np.random.SeedSequence(self._last_seed).spawn(2)
        self._arrival_rng = np.random.default_rng(arrival_seed)
        self._channel_rng = np.random.default_rng(channel_seed)
        self._make_exogenous_traces()
        self.timestep = 0
        self.current_step = 0
        self.nominal_action = None
        self.executed_action = None
        self.state = self._initial_state()
        self._episode_done = False
        self.state["A"] = self._arrival_trace[0].copy()
        self.state["g"] = self._channel_trace[0].copy()
        options = options or {}
        supplied_state = options.get("state")
        if supplied_state is not None:
            for key in ("Q", "B", "T", "A", "g"):
                if key in supplied_state:
                    value = np.asarray(supplied_state[key], dtype=float)
                    if value.shape != (self.n_devices,):
                        raise ValueError(f"reset state {key} must have shape ({self.n_devices},)")
                    self.state[key] = value.copy()
        if validate_state(self.state, self.params)['status'] == 'invalid_state':
            raise ValueError("reset received invalid physical state")
        self.last_info = {}
        return self._observation(), {
            "seed": self._last_seed,
            "effective_config": self.effective_config(),
            "simulator_only_config": self.simulator_config(),
            "physical_state": self._copy_state(self.state),
        }

    @staticmethod
    def _copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
        return {k: np.asarray(v, dtype=float).copy() for k, v in state.items() if k in {"Q", "B", "T", "A", "g"}}

    def effective_config(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self.params.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, np.generic):
                result[key] = value.item()
            elif isinstance(value, dict):
                result[key] = dict(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                result[key] = value
        result.update({"n_devices": self.n_devices, "n_edge_servers": 1, "horizon": self.horizon})
        margins = safety_margin_settings(self.params)
        result["safety_margin_settings"] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in margins.items()}
        result["verification_scope"] = safety_verification_scope(self.params)
        return result

    def simulator_config(self):
        """Provenance only; not passed to the policy/filter prediction model."""
        return {
            "actual_energy_multiplier": self._actual_energy_multiplier,
            "actual_thermal_alpha_override": self._actual_thermal_alpha,
            "effective_actual_thermal_alpha": (float(self.params["thermal_alpha"])
                if self._actual_thermal_alpha is None else self._actual_thermal_alpha),
        }

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("step() called after episode termination; call reset() first")
        if validate_state(self.state, self.params)['status'] == 'invalid_state':
            raise ValueError("Invalid physical state: no action executed")
        try:
            raw_action = np.asarray(action, dtype=float).copy()
            raw_valid = raw_action.shape == self.action_space.shape and np.isfinite(raw_action).all()
        except (ValueError, TypeError, OverflowError):
            raw_action = action
            raw_valid = False
        bounded_action = np.clip(raw_action, -1.0, 1.0) if raw_valid else None
        self.nominal_action = self._decode_action(raw_action) if raw_valid else {}
        pre_state = self._copy_state(self.state)
        elapsed_time = self.timestep * float(self.params["delta"])
        filtered = self.filter.filter_action(self.state, self.nominal_action)
        # Last gate: execute exactly the physical action checked here, without clipping.
        if filtered["safe"]:
            final_report = evaluate_constraints(self.state, filtered["action"], self.params)
            if final_report["feasible"]:
                filtered = {**filtered, "report": final_report}
            if not final_report["feasible"]:
                filtered = {**filtered, "safe": False, "action": None,
                            "status": "final_verification_failed", "report": final_report,
                            "failure_kind": "final_verification_failed"}
        self.last_filter_result = filtered
        action_preprocessed = bool(raw_valid and not np.array_equal(raw_action, bounded_action))
        if action_preprocessed and filtered["safe"] and filtered["status"] == "nominal_accepted":
            filtered["status"] = "preprocessed_candidate_verified"
        chosen = filtered["action"]
        intervention = filtered["status"] != "nominal_accepted"
        base_info = {
            "proposal_status": filtered.get("proposal_status"),
            "state_status": filtered.get("state_status"),
            "solver_status": filtered.get("solver_status"),
            "selected_solver_status": filtered.get("selected_solver_status"),
            "solver_attempts": filtered.get("solver_attempts", []),
            "fallback_reason": filtered.get("fallback_reason"),
            "fallback_attempted": filtered.get("fallback_attempted", False),
            "objective_value": filtered.get("objective_value"),
            "correction_norm": filtered.get("correction_norm"),
            "safety_claim": filtered.get("safety_claim"),
            "seed": self._last_seed,
            "slot": self.timestep,
            "raw_policy_action": raw_action,
            "bounded_policy_action": bounded_action,
            "decoded_nominal_action": {k: v.copy() for k, v in self.nominal_action.items()},
            "proposed_action": {k: v.copy() for k, v in self.nominal_action.items()},
            "selected_candidate_action": None if chosen is None else {k: np.asarray(v, dtype=float).copy() for k, v in chosen.items()},
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
                "prediction_errors": prediction_error_report(None, None, None, self.params),
                "controller_failure": True,
                "failure_kind": filtered.get("failure_kind", "no_verified_candidate_found"),
                "failure_reason": filtered.get("failure_kind", "no candidate passed the one-step predicted constraints"),
                "episode_ending_reason": ("controller_failure_no_verified_continuation"
                    if filtered["status"] == "no_verified_safe_continuation" else filtered["status"]),
                "reward": -penalty,
                "feasible": False,
                "terminated": terminated,
                "truncated": truncated,
            }
            self.last_info = info
            self._episode_done = True
            return self._observation(), -penalty, terminated, truncated, info

        self.executed_action = chosen
        transition_params = {**self.params, "actual_energy_multiplier": self._actual_energy_multiplier}
        if self._actual_thermal_alpha is not None:
            transition_params["actual_thermal_alpha"] = self._actual_thermal_alpha
        next_state = apply_transition(self.state, chosen, transition_params)
        next_index = self.timestep + 1
        next_state["A"] = self._arrival_trace[next_index].copy()
        next_state["g"] = self._channel_trace[next_index].copy()
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
            "prediction_errors": prediction_error_report(filtered["report"], self.state, self.state["metrics"], self.params),
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
