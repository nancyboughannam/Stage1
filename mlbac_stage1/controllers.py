from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from stable_baselines3 import PPO


class ManualController:
    """Simple deterministic controller for debugging."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def act(self, state: Dict[str, np.ndarray], *, safe: bool = True) -> Dict[str, np.ndarray]:
        n = len(state["Q"])
        q = np.asarray(state["Q"], dtype=float)
        a = np.asarray(state["A"], dtype=float)
        base = np.minimum(np.maximum(q, 0.0), a + 0.5)
        action = {
            "f": np.full(n, 0.2e9, dtype=float),
            "p": np.full(n, 0.1, dtype=float),
            "b": np.full(n, 1e6, dtype=float),
            "f_edge": np.full(n, 0.5e9, dtype=float),
            "l": np.minimum(base, 0.6 * q),
            "o": np.minimum(np.maximum(0.0, q - base), 0.5 * q),
            "U": np.minimum(np.maximum(0.0, a), 0.5 * a),
        }
        return action


class RandomController:
    """Random but bounded action generator used for stress tests."""

    def __init__(self, params: Dict[str, Any], seed: Optional[int] = None):
        self.params = params
        self.rng = np.random.default_rng(seed)

    def act(self, state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        n = len(state["Q"])
        q = np.asarray(state["Q"], dtype=float)
        a = np.asarray(state["A"], dtype=float)
        action = {
            "f": self.rng.uniform(0.0, 0.5e9, size=n),
            "p": self.rng.uniform(0.0, 0.3, size=n),
            "b": self.rng.uniform(0.0, 1e6, size=n),
            "f_edge": self.rng.uniform(0.0, 0.5e9, size=n),
            "l": self.rng.uniform(0.0, np.maximum(0.5, q), size=n),
            "o": self.rng.uniform(0.0, np.maximum(0.5, q), size=n),
            "U": self.rng.uniform(0.0, np.maximum(0.0, a), size=n),
        }
        return action


class PPOController:
    """Lightweight wrapper for PPO training experiments. Real training is a separate script."""

    def __init__(self, env, *, model_path: Optional[str] = None, **kwargs):
        self.env = env
        self.model = None
        if model_path is not None:
            self.model = PPO.load(model_path)

    def train(self, total_timesteps: int = 64, seed: int = 0):
        self.model = PPO("MlpPolicy", self.env, seed=seed, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)
        return self.model
