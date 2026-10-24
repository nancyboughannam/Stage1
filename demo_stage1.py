from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbac_stage1.env import make_env


def run_demo(output_dir: str = "plots") -> Path:
    env = make_env(horizon=20, deterministic=True)
    obs, info = env.reset(seed=0)

    queue_history = []
    battery_history = []
    temp_history = []
    reward_history = []
    statuses = []

    for _ in range(20):
        nominal = np.array([0.4, 0.35, 0.25, 0.15, 0.3, 0.2, 0.25], dtype=float)
        obs, reward, terminated, truncated, info = env.step(nominal)
        queue_history.append(float(env.state["Q"][0]))
        battery_history.append(float(env.state["B"][0]))
        temp_history.append(float(env.state["T"][0]))
        reward_history.append(float(reward))
        statuses.append(info["filter_status"])
        if terminated or truncated:
            break

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(queue_history, marker="o")
    axes[0, 0].set_title("Queue backlog")
    axes[0, 0].set_ylabel("Q")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(battery_history, marker="o", color="tab:green")
    axes[0, 1].set_title("Battery reserve")
    axes[0, 1].set_ylabel("B")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(temp_history, marker="o", color="tab:orange")
    axes[1, 0].set_title("Temperature")
    axes[1, 0].set_ylabel("T [C]")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(reward_history, marker="o", color="tab:purple")
    axes[1, 1].set_title("Reward")
    axes[1, 1].set_ylabel("reward")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle("Stage 1 MEC safety-filter demo")
    fig.tight_layout()
    out = output_path / "stage1_demo.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print(f"Saved: {out}")
    print(f"Final queue={queue_history[-1]:.3f}, battery={battery_history[-1]:.3f}, temp={temp_history[-1]:.3f}")
    print(f"Statuses: {statuses}")
    return out


if __name__ == "__main__":
    run_demo()
