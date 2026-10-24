from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbac_stage1.env import make_env


def run_demo(output_dir: str = "plots") -> Path:
    env = make_env(horizon=20, deterministic=True)
    obs, _ = env.reset(seed=0)

    queue_history = []
    battery_history = []
    temp_history = []
    reward_history = []
    statuses = []
    csv_rows = []

    for step_idx in range(20):
        proposed = np.array([0.4, 0.35, 0.25, 0.15, 0.3, 0.2, 0.25], dtype=float)
        pre_state = {k: np.asarray(v, dtype=float).copy() for k, v in env.state.items() if k in {"Q", "B", "T", "A", "g"}}
        obs, reward, terminated, truncated, info = env.step(proposed)
        queue_history.append(float(env.state["Q"][0]))
        battery_history.append(float(env.state["B"][0]))
        temp_history.append(float(env.state["T"][0]))
        reward_history.append(float(reward))
        statuses.append(info["filter_status"])
        csv_rows.append(
            {
                "step": step_idx,
                "pre_state": np.array2string(np.asarray(pre_state["Q"]), separator=","),
                "proposed_action": np.array2string(np.asarray(proposed), separator=","),
                "executed_action": np.array2string(np.asarray(info["executed_action"]["U"]), separator=","),
                "post_state": np.array2string(np.asarray(env.state["Q"]), separator=","),
                "status": info["filter_status"],
                "feasible": bool(info["feasible"]),
            }
        )
        if terminated or truncated:
            break

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "stage1_demo_log.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "pre_state", "proposed_action", "executed_action", "post_state", "status", "feasible"])
        writer.writeheader()
        writer.writerows(csv_rows)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(queue_history, marker="o")
    axes[0, 0].axhline(10.0, color="r", linestyle="--", alpha=0.8, label="Q_max")
    axes[0, 0].set_title("Queue backlog")
    axes[0, 0].set_ylabel("Q")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(battery_history, marker="o", color="tab:green")
    axes[0, 1].axhline(0.9 * 9 * 3600, color="r", linestyle="--", alpha=0.8, label="reserve")
    axes[0, 1].set_title("Battery reserve")
    axes[0, 1].set_ylabel("B")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(temp_history, marker="o", color="tab:orange")
    axes[1, 0].axhline(80.0, color="r", linestyle="--", alpha=0.8, label="T_max")
    axes[1, 0].set_title("Temperature")
    axes[1, 0].set_ylabel("T [C]")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

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
    print(f"Saved: {csv_path}")
    print(f"Final queue={queue_history[-1]:.3f}, battery={battery_history[-1]:.3f}, temp={temp_history[-1]:.3f}")
    print(f"Statuses: {statuses}")
    return out


if __name__ == "__main__":
    run_demo()
