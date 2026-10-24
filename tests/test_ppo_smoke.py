import numpy as np

from mlbac_stage1.env import make_env


def test_ppo_smoke_training_runs():
    env = make_env(
        n_devices=1,
        horizon=5,
        deterministic=True,
        use_ppo=True,
        max_steps=8,
        action_scale=1.0,
    )
    obs, _ = env.reset(seed=7)
    for _ in range(6):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=7)
    assert obs.shape == (7 * 1 + 4,)
    assert np.isfinite(reward)
