from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbac_stage1.env import ACTION_COMPONENTS, make_env


POLICY_NAME = "scripted_nominal_resource_aware_not_trained_or_optimized"
ACTION_LABELS = ("local CPU", "tx power", "bandwidth", "edge CPU", "local work", "offload work", "admission")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fraction(value: float, maximum: float) -> float:
    return float(np.clip(value / maximum, 0.0, 1.0))


def scripted_nominal_action(env) -> np.ndarray:
    """Propose modest queued-work processing and admission; the filter remains in control."""
    params = env.params
    fractions = np.zeros((env.n_devices, len(ACTION_COMPONENTS)), dtype=float)
    for device in range(env.n_devices):
        queue = max(0.0, float(env.state["Q"][device]))
        arrival = max(0.0, float(env.state["A"][device]))
        # Preserve slack against float32 policy-action decoding roundoff at l+o <= Q.
        work = 0.99 * min(queue, 0.5)
        local_work = (2.0 / 3.0) * work
        offload_work = work - local_work
        admitted = max(0.0, min(arrival, 0.5) - 1e-6)
        fractions[device, 0] = 0.2 if local_work > 0 else 0.0
        fractions[device, 1] = 0.1 if offload_work > 0 else 0.0
        fractions[device, 2] = _fraction(1.5e6, float(params["shared_bandwidth_hz"])) if offload_work > 0 else 0.0
        fractions[device, 3] = _fraction(0.1e9, float(params["shared_edge_cpu_hz"])) if offload_work > 0 else 0.0
        fractions[device, 4] = _fraction(local_work, float(params["Q_max"][device]))
        fractions[device, 5] = _fraction(offload_work, float(params["Q_max"][device]))
        fractions[device, 6] = _fraction(admitted, float(params["A_max"][device]))
    return (2.0 * fractions - 1.0).astype(np.float32).reshape(-1)


def _scalar(values: Any, device: int) -> float | str:
    if values is None:
        return ""
    values = np.asarray(values)
    return float(values[device]) if values.ndim else float(values)


def _flag(report: Dict[str, Any] | None, name: str, device: int) -> bool | str:
    if report is None:
        return ""
    value = report.get("violations", {}).get(name, [])
    return device in value if isinstance(value, list) else bool(value)


def _run_episode(env, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    params = env.params
    rows: list[dict[str, Any]] = []
    initial = env._copy_state(env.state)
    state_times = [0.0]
    state_history = {key: [initial[key].copy()] for key in ("Q", "B", "T")}
    slot_times: list[float] = []
    flows = {key: [] for key in ("offered", "admitted", "local", "offload", "rejected")}
    utilization = {"bandwidth": [], "edge_cpu": []}
    attempted = 0
    executed_count = 0
    reward_sum = 0.0
    ending_reason = None
    outcome_kind = "early_stop"
    filter_status_counts: dict[str, int] = {}

    for slot in range(env.horizon):
        proposal_raw = scripted_nominal_action(env)
        _, reward, terminated, truncated, info = env.step(proposal_raw)
        attempted += 1
        reward_sum += float(reward)
        ending_reason = info["episode_ending_reason"]
        filter_status_counts[info["filter_status"]] = filter_status_counts.get(info["filter_status"], 0) + 1
        outcome_kind = "terminated" if terminated else ("truncated" if truncated else outcome_kind)
        pre = info["pre_state"]
        decoded = info["decoded_nominal_action"]
        candidate = info["selected_candidate_action"]
        executed = info["executed_action"]
        proposal_report = info["proposal_predicted_report"]
        selected_report = info["predicted_report"]
        actual_report = info["actual_report"]
        metrics = info["transition_metrics"]
        post = info["post_state"]
        predicted = selected_report.get("predicted_state")
        time_start = float(info["elapsed_time_s_start"])
        slot_times.append(time_start)
        flows["offered"].append(float(np.sum(pre["A"])))

        if executed is None:
            for key in ("admitted", "local", "offload", "rejected"):
                flows[key].append(np.nan)
            utilization["bandwidth"].append(np.nan)
            utilization["edge_cpu"].append(np.nan)
        else:
            executed_count += 1
            flows["admitted"].append(float(np.sum(executed["U"])))
            flows["local"].append(float(np.sum(metrics["completed_local_work"])))
            flows["offload"].append(float(np.sum(metrics["completed_offload_work"])))
            flows["rejected"].append(float(np.sum(metrics["rejected_arrivals"])))
            utilization["bandwidth"].append(float(metrics["shared_bandwidth_hz_used"]) / params["shared_bandwidth_hz"])
            utilization["edge_cpu"].append(float(metrics["shared_edge_cpu_hz_used"]) / params["shared_edge_cpu_hz"])

        for device in range(env.n_devices):
            row: Dict[str, Any] = {
                "seed": seed,
                "scenario": "reference_multi_device_development",
                "policy": POLICY_NAME,
                "device": device,
                "slot": slot,
                "time_start_s": time_start,
                "time_end_s": float(info["elapsed_time_s"]),
                "pre_Q_work_units": float(pre["Q"][device]),
                "pre_B_j": float(pre["B"][device]),
                "pre_T_c": float(pre["T"][device]),
                "A_t_offered_work_units": float(pre["A"][device]),
                "g_t_channel_gain": float(pre["g"][device]),
                "selected_predicted_Q_next": _scalar(predicted.get("Q") if predicted else None, device),
                "selected_predicted_B_next_j": _scalar(predicted.get("B") if predicted else None, device),
                "selected_predicted_T_next_c": _scalar(predicted.get("T") if predicted else None, device),
                "actual_Q_next_work_units": _scalar(post.get("Q") if post else None, device),
                "actual_B_next_j": _scalar(post.get("B") if post else None, device),
                "actual_T_next_c": _scalar(post.get("T") if post else None, device),
                "A_next_offered_work_units": _scalar(post.get("A") if post else None, device),
                "g_next_channel_gain": _scalar(post.get("g") if post else None, device),
                "actual_peak_T_c": _scalar(metrics.get("peak_temp") if metrics else None, device),
                "energy_consumed_j": _scalar(metrics.get("energy") if metrics else None, device),
                "completed_local_work_units": _scalar(metrics.get("completed_local_work") if metrics else None, device),
                "completed_offload_work_units": _scalar(metrics.get("completed_offload_work") if metrics else None, device),
                "admitted_U_work_units": _scalar(executed.get("U") if executed else None, device),
                "rejected_A_minus_U_work_units": _scalar(metrics.get("rejected_arrivals") if metrics else None, device),
                "filter_status": info["filter_status"],
                "intervention": info["intervention"],
                "action_preprocessed": info["action_preprocessed"],
                "verification_status": info["verification_status"],
                "verification_scope": info["verification_scope"],
                "fallback_mode": info["fallback_mode"] or "",
                "proposal_predicted_feasible": proposal_report["feasible"],
                "selected_predicted_feasible": selected_report["feasible"],
                "actual_feasible": actual_report["feasible"] if actual_report else "",
                "controller_failure": info["controller_failure"],
                "failure_kind": info["failure_kind"] or "",
                "failure_reason": info["failure_reason"] or "",
                "actual_outcome": info["actual_outcome"],
                "episode_ending_reason": info["episode_ending_reason"] or "",
                "elapsed_time_s_at_end": info["elapsed_time_s"],
                "outstanding_queue_work_units": float(info["outstanding_queue_work"][device]) if info["controller_failure"] else "",
                "infeasibility_proven": info["infeasibility_proven"],
                "proposal_temperature_violation": _flag(proposal_report, "thermal_violation", device),
                "selected_battery_violation": _flag(selected_report, "battery", device),
                "actual_temperature_violation": device in actual_report["flags"]["temperature_peak"] if actual_report else "",
                "bandwidth_utilization_fraction": "" if executed is None else float(metrics["shared_bandwidth_hz_used"]) / params["shared_bandwidth_hz"],
                "edge_cpu_utilization_fraction": "" if executed is None else float(metrics["shared_edge_cpu_hz_used"]) / params["shared_edge_cpu_hz"],
            }
            for component, key in enumerate(ACTION_COMPONENTS):
                row[f"raw_policy_{key}"] = float(info["raw_policy_action"].reshape(env.n_devices, 7)[device, component])
                row[f"decoded_proposal_{key}"] = _scalar(decoded[key], device)
                row[f"selected_candidate_{key}"] = _scalar(candidate.get(key) if candidate else None, device)
                row[f"executed_{key}"] = _scalar(executed.get(key) if executed else None, device)
            rows.append(row)

        if post is not None:
            for key in state_history:
                state_history[key].append(np.asarray(post[key], dtype=float).copy())
            state_times.append(float(info["elapsed_time_s"]))
        if terminated or truncated:
            break

    final_state = env._copy_state(env.state)
    total_completed = float(np.nansum(flows["local"]) + np.nansum(flows["offload"]))
    initial_queue_total = float(np.sum(initial["Q"]))
    final_queue_total = float(np.sum(final_state["Q"]))
    queue_balance_final = initial_queue_total + float(np.nansum(flows["admitted"])) - total_completed
    summary = {
        "seed": seed,
        "policy": POLICY_NAME,
        "attempted_slots": attempted,
        "executed_slots": executed_count,
        "elapsed_simulated_seconds": float(env.timestep * env.params["delta"]),
        "termination_kind": outcome_kind,
        "episode_ending_reason": ending_reason,
        "diagnostic_reward_sum": reward_sum,
        "offered_work_units": float(np.nansum(flows["offered"])),
        "admitted_work_units": float(np.nansum(flows["admitted"])),
        "completed_local_work_units": float(np.nansum(flows["local"])),
        "completed_offload_work_units": float(np.nansum(flows["offload"])),
        "total_completed_work_units": total_completed,
        "rejected_work_units": float(np.nansum(flows["rejected"])),
        "initial_queue_work_units": initial_queue_total,
        "final_queue_work_units_total": final_queue_total,
        "queue_conservation_residual_work_units": final_queue_total - queue_balance_final,
        "queue_capacity_work_units_per_device": np.asarray(params["Q_max"], dtype=float).tolist(),
        "maximum_recorded_queue_work_units_per_device": np.max(np.asarray(state_history["Q"]), axis=0).tolist(),
        "filter_status_counts": filter_status_counts,
        "scripted_policy_queue_work_margin_fraction": 0.01,
        "scripted_policy_admission_cap_work_units_per_device_slot": 0.5,
        "rejection_fraction_of_offered": float(np.nansum(flows["rejected"]) / np.nansum(flows["offered"])) if np.nansum(flows["offered"]) else 0.0,
        "remaining_queue_work_units": np.asarray(final_state["Q"], dtype=float).tolist(),
        "final_battery_j": np.asarray(final_state["B"], dtype=float).tolist(),
        "final_temperature_c": np.asarray(final_state["T"], dtype=float).tolist(),
        "notes": [
            "Scripted nominal policy; not trained PPO and not optimized.",
            "Light development workload; not evidence of filter effectiveness or maximum safe load.",
            "Remaining queue is backlog, not completed workload.",
        ],
    }
    history = {
        "state_times": state_times,
        "state": state_history,
        "slot_times": slot_times,
        "flows": flows,
        "utilization": utilization,
        "early_stop": f"EARLY STOP: {ending_reason} after {attempted}/{env.horizon} slots" if ending_reason and ending_reason != "finite_horizon_complete" else "",
    }
    return rows, summary, history, initial


def _write_plots(output: Path, env, history: Dict[str, Any]) -> None:
    params = env.params
    state = history["state"]
    state_time = history["state_times"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, key, label, limit_key in (
        (axes[0], "Q", "Queue (task-equivalents)", "Q_max"),
        (axes[1], "B", "Battery (J)", "B_min"),
        (axes[2], "T", "Temperature (C)", "T_max"),
    ):
        values = np.asarray(state[key])
        for device in range(env.n_devices):
            axis.plot(state_time, values[:, device], label=f"device {device}")
            limit = float(params[limit_key][device])
            axis.axhline(limit, color="black", linestyle="--", alpha=0.45, label=f"{limit_key}={limit:g}" if device == 0 else None)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Elapsed simulated time (s)")
    title = "Reference state trajectories and configured limits"
    if history["early_stop"]:
        title += f" | {history['early_stop']}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / "state_trajectories.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    for key, label in (("offered", "offered A_t"), ("admitted", "admitted U_t"), ("local", "completed locally"), ("offload", "completed by offload"), ("rejected", "rejected A_t-U_t")):
        axis.plot(history["slot_times"], history["flows"][key], label=label)
    axis.set_xlabel("Slot start time (s)")
    axis.set_ylabel("Task-equivalents per slot, all devices")
    title = "Offered, admitted, completed, and rejected workload"
    if history["early_stop"]:
        title += f" | {history['early_stop']}"
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "workload_flows.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(history["slot_times"], history["utilization"]["bandwidth"], label="shared bandwidth utilization")
    axis.plot(history["slot_times"], history["utilization"]["edge_cpu"], label="shared edge CPU utilization")
    axis.set_xlabel("Slot start time (s)")
    axis.set_ylabel("Fraction of configured shared capacity")
    axis.set_ylim(bottom=0.0)
    title = "Shared resource utilization"
    if history["early_stop"]:
        title += f" | {history['early_stop']}"
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "shared_resource_utilization.png", dpi=170)
    plt.close(fig)


def run_reference_demo(output_dir: str | Path = "reference_demo", seed: int = 2026) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = make_env()
    _, reset_info = env.reset(seed=seed)
    rows, summary, history, initial = _run_episode(env, seed)
    snapshot = {
        "seed": seed,
        "effective_config": reset_info["effective_config"],
        "initial_state": initial,
        "policy": POLICY_NAME,
        "policy_description": "Each device proposes up to 0.5 queued task-equivalents split 2:1 local/offload and admits up to 0.5 task-equivalents per slot. The existing safety filter is applied.",
    }
    (output / "config_snapshot.json").write_text(json.dumps(_json_safe(snapshot), indent=2, sort_keys=True) + "\n")
    with (output / "episode.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_plots(output, env, history)
    (output / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    (output / "summary.txt").write_text(
        f"seed={seed}; slots={summary['executed_slots']}/{env.horizon}; elapsed={summary['elapsed_simulated_seconds']:.1f} s; "
        f"ending={summary['episode_ending_reason'] or 'unknown'}; remaining queue={summary['remaining_queue_work_units']} task-equivalents.\n"
        "Scripted nominal policy only; not trained PPO/optimized. Light development run; not a safety or maximum-load result.\n"
    )
    env.close()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the five-device reference development scenario.")
    parser.add_argument("--output", default="reference_demo")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(f"Reference demonstration outputs: {run_reference_demo(args.output, args.seed)}")
