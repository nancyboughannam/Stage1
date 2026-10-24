from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .config import make_default_params
from .constraints import evaluate_constraints
from .controllers import ManualController, RandomController
from .fallback import FallbackController
from .models import apply_transition
from .safety_filter import SafetyFilter


class MBACEnv(gym.Env):
    """Stage 1 MEC resource-allocation environment with hard safety filtering."""

    def __init__(
        self,
        n_devices: int = 1,
        horizon: int = 10,
        *,
        params: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        use_ppo: bool = False,
        max_steps: Optional[int] = None,
        action_scale: float = 1.0,
    ):
        self.n_devices = n_devices
        self.params = make_default_params(n_devices, deterministic=deterministic) if params is None else params
        self.params.setdefault("A_max", np.full(n_devices, 1.0, dtype=float))
        self.params.setdefault("B_min", np.full(n_devices, 0.1 * 9.0 * 3600.0, dtype=float))
        self.params.setdefault("B_cap", np.full(n_devices, 9.0 * 3600.0, dtype=float))
        self.params.setdefault("Q_max", np.full(n_devices, 10.0, dtype=float))
        self.params.setdefault("T_max", np.full(n_devices, 80.0, dtype=float))
        self.horizon = horizon if max_steps is None else max_steps
        self.use_ppo = use_ppo
        self.action_scale = action_scale
        self.timestep = 0
        self.current_step = 0
        self.nominal_action = None
        self.executed_action = None

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7 * n_devices,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1e6, high=1e6, shape=(4 * n_devices + 7,), dtype=np.float32)

        self.filter = SafetyFilter(self.params)
        self.fallback = FallbackController(self.params)
        self.manual = ManualController(self.params)
        self.random = RandomController(self.params, seed=0)

        self.rng = np.random.default_rng(0)
        self.state = self._initial_state()

    def _initial_state(self) -> Dict[str, np.ndarray]:
        q = np.full(self.n_devices, 1.0, dtype=float)
        b = np.full(self.n_devices, 0.8 * self.params["B_cap"], dtype=float)
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
        q = np.asarray(self.state["Q"], dtype=float)
        a = np.asarray(self.state["A"], dtype=float)
        scale = float(self.action_scale)
        dec = {
            "f": raw[:, 0] * scale * np.asarray(self.params["f_hw_max"], dtype=float),
            "p": raw[:, 1] * scale * np.asarray(self.params["p_hw_max"], dtype=float),
            "b": raw[:, 2] * scale * float(self.params["shared_bw"]),
            "f_edge": raw[:, 3] * scale * float(self.params["shared_edge_cpu"]),
            "l": raw[:, 4] * scale * np.maximum(q, 0.0),
            "o": raw[:, 5] * scale * np.maximum(q, 0.0),
            "U": raw[:, 6] * scale * np.maximum(a, 0.0),
        }
        return dec

    def _observation(self) -> np.ndarray:
        q = np.asarray(self.state["Q"], dtype=float)
        b = np.asarray(self.state["B"], dtype=float)
        t = np.asarray(self.state["T"], dtype=float)
        a = np.asarray(self.state["A"], dtype=float)
        globals_ = np.array(
            [
                np.max(self.params["Q_max"]),
                np.max(self.params["B_cap"]),
                np.max(self.params["T_max"]),
                float(self.params["shared_bw"]),
                float(self.params["shared_edge_cpu"]),
                float(self.timestep),
                float(np.mean(a)),
            ],
            dtype=float,
        )
        obs = np.concatenate([q, b, t, a, globals_])
        return obs.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.timestep = 0
        self.state = self._initial_state()
        self.state["A"] = self._sample_arrivals()
        if self.use_ppo:
            self.state["A"] = np.clip(self.state["A"], 0.0, np.asarray(self.params["A_max"], dtype=float))
        return self._observation(), {}

    def step(self, action):
        nominal = self._decode_action(np.asarray(action, dtype=float))
        self.nominal_action = nominal
        filtered = self.filter.filter_action(self.state, nominal)
        chosen = filtered["action"]
        self.executed_action = chosen
        next_state = apply_transition(self.state, chosen, self.params)
        next_state["A"] = self._sample_arrivals()
        self.state = next_state
        done = self.timestep >= self.horizon
        reward = float(np.sum(chosen["U"]) - 0.01 * np.sum(np.maximum(0.0, self.state["Q"])))
        info = {
            "nominal_action": nominal,
            "executed_action": chosen,
            "feasible": evaluate_constraints(self.state, chosen, self.params)["feasible"],
            "filter_status": filtered["status"],
            "reward": reward,
        }
        self.timestep += 1
        return self._observation(), reward, done, False, info


def make_env(**kwargs):
    return MBACEnv(**kwargs)


__all__ = ["MBACEnv", "make_env"]
