import numpy as np
from stable_baselines3 import PPO

from mlbac_stage1.env import make_env


def test_ppo_smoke_training_runs():
    env = make_env(
        n_devices=1,
        horizon=4,
        deterministic=True,
        use_ppo=True,
        action_scale=1.0,
    )
    obs, _ = env.reset(seed=7)
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, seed=0, verbose=0)
    model.learn(total_timesteps=32)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (7,)
    assert np.isfinite(np.asarray(action, dtype=float)).all()
