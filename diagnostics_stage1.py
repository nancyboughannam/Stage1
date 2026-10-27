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

from mlbac_stage1.config import make_default_params
from mlbac_stage1.env import make_env


ACTION_KEYS = ("f", "p", "b", "f_edge", "l", "o", "U")
ACTION_LABELS = ("local CPU", "tx power", "bandwidth", "edge CPU", "local work", "offload work", "admission")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _raw_for_ratios(ratios: list[float]) -> np.ndarray:
    return 2.0 * np.asarray(ratios, dtype=float) - 1.0


def _scenario(name: str) -> tuple[Dict[str, Any], Dict[str, np.ndarray], np.ndarray, str]:
    params = make_default_params(1, deterministic=True)
    if name == "normal":
        state = {
            "Q": np.array([1.0]),
            "B": params["B_cap"] * 0.8,
            "T": np.array([30.0]),
            "A": np.array([0.3]),
            "g": params["g_base"].copy(),
        }
        raw = _raw_for_ratios([0.2, 0.1, 0.1, 0.05, 0.005, 0.0, 0.025])
        description = "A hardware-admissible local-work proposal expected to pass unchanged for one step."
    elif name == "thermal":
        params["thermal_alpha"] = 60.0
        state = {
            "Q": np.array([1.0]),
            "B": params["B_cap"] * 0.8,
            "T": np.array([79.0]),
            "A": np.array([0.0]),
            "g": params["g_base"].copy(),
        }
        raw = _raw_for_ratios([0.9, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0])
        description = "Synthetic thermal stress; the nominal action is hardware-admissible and otherwise feasible but exceeds predicted T_max."
    elif name == "battery":
        state = {
            "Q": np.array([0.7]),
            "B": params["B_min"] + 0.5,
            "T": np.array([30.0]),
            "A": np.array([0.0]),
            "g": params["g_base"].copy(),
        }
        raw = _raw_for_ratios([0.0] * len(ACTION_KEYS))
        description = "Near-reserve idle candidate with 0.5 J headroom and 1 J baseline consumption per one-second slot."
    else:
        raise ValueError(name)
    return params, state, raw, description


def _value(values: Any, index: int) -> Any:
    if values is None:
        return ""
    array = np.asarray(values)
    return float(array[index]) if array.ndim else float(array)


def _get_violation(report: Dict[str, Any] | None, name: str, index: int) -> bool | str:
    if report is None:
        return ""
    values = report.get("violations", {}).get(name, [])
    if isinstance(values, list):
        return index in values
    return bool(values)


def _write_csv(path: Path, scenario: str, seed: int, info: Dict[str, Any], params: Dict[str, Any]) -> None:
    pre = info["pre_state"]
    nominal = info["decoded_nominal_action"]
    executed = info["executed_action"]
    prediction = info.get("predicted_state") or {}
    proposed_report = info.get("proposal_predicted_report")
    selected_report = info.get("predicted_report")
    actual_report = info.get("actual_report")
    actual = info.get("post_state")
    metrics = info.get("transition_metrics")
    rows = []
    for device in range(len(pre["Q"])):
        row: Dict[str, Any] = {
            "scenario": scenario,
            "seed": seed,
            "device": device,
            "slot": 0,
            "time_s": info["elapsed_time_s_start"],
            "pre_Q_work_units": _value(pre["Q"], device),
            "pre_B_j": _value(pre["B"], device),
            "pre_T_c": _value(pre["T"], device),
            "pre_g_channel_gain": _value(pre["g"], device),
            "pre_A_work_units": _value(pre["A"], device),
            "selected_predicted_next_Q_work_units": _value(prediction.get("Q"), device),
            "selected_predicted_next_B_j": _value(prediction.get("B"), device),
            "selected_predicted_next_T_c": _value(prediction.get("T"), device),
            "proposal_predicted_next_Q_work_units": _value(proposed_report.get("predicted_state", {}).get("Q") if proposed_report else None, device),
            "proposal_predicted_next_B_j": _value(proposed_report.get("predicted_state", {}).get("B") if proposed_report else None, device),
            "proposal_predicted_next_T_c": _value(proposed_report.get("predicted_state", {}).get("T") if proposed_report else None, device),
            "selected_predicted_peak_T_c": _value(selected_report.get("predicted_peak_temperature") if selected_report else None, device),
            "proposal_predicted_peak_T_c": _value(proposed_report.get("predicted_peak_temperature") if proposed_report else None, device),
            "actual_Q_work_units": _value(actual.get("Q") if actual else None, device),
            "actual_B_j": _value(actual.get("B") if actual else None, device),
            "actual_T_c": _value(actual.get("T") if actual else None, device),
            "actual_A_work_units": _value(actual.get("A") if actual else None, device),
            "actual_g_channel_gain": _value(actual.get("g") if actual else None, device),
            "actual_peak_T_c": _value(metrics.get("peak_temp") if metrics else None, device),
            "energy_consumed_j": _value(metrics.get("energy") if metrics else None, device),
            "completed_local_work_units": _value(metrics.get("completed_local_work") if metrics else None, device),
            "completed_offload_work_units": _value(metrics.get("completed_offload_work") if metrics else None, device),
            "rejected_arrival_work_units": _value(metrics.get("rejected_arrivals") if metrics else None, device),
            "filter_status": info["filter_status"],
            "intervention": info["intervention"],
            "action_preprocessed": info["action_preprocessed"],
            "fallback_mode": info["fallback_mode"] or "",
            "verification_status": info["verification_status"],
            "verification_scope": info["verification_scope"],
            "proposal_predicted_feasible": proposed_report["feasible"],
            "selected_predicted_feasible": selected_report["feasible"],
            "actual_feasible": actual_report["feasible"] if actual_report else "",
            "failure_kind": info["failure_kind"] or "",
            "failure_reason": info["failure_reason"] or "",
            "actual_outcome": info["actual_outcome"],
            "episode_ending_reason": info["episode_ending_reason"] or "",
            "elapsed_time_s_at_end": info["elapsed_time_s"],
            "outstanding_work_units_at_failure": float(info["outstanding_work"][device]) if info["controller_failure"] else "",
            "outstanding_arrival_work_units_at_failure": float(info["outstanding_arrival_work"][device]) if info["controller_failure"] else "",
            "total_outstanding_work_units_at_failure": float(info["total_outstanding_work"][device]) if info["controller_failure"] else "",
            "infeasibility_proven": info["infeasibility_proven"],
        }
        for component, key in enumerate(ACTION_KEYS):
            row[f"raw_policy_{key}"] = float(info["raw_policy_action"].reshape(-1, 7)[device, component])
            row[f"decoded_nominal_{key}"] = _value(nominal[key], device)
            row[f"executed_{key}"] = _value(executed.get(key) if executed else None, device)
        row.update(
            {
                "pred_queue_work_violation": _get_violation(proposed_report, "queue", device),
                "pred_queue_violation": _get_violation(proposed_report, "queue_overflow", device),
                "pred_admission_violation": _get_violation(proposed_report, "admission", device),
                "pred_timing_violation": _get_violation(proposed_report, "timing", device),
                "pred_battery_violation": _get_violation(proposed_report, "battery", device),
                "pred_battery_capacity_violation": _get_violation(proposed_report, "battery_overflow", device),
                "pred_temperature_violation": _get_violation(proposed_report, "thermal_violation", device),
                "pred_local_cpu_violation": proposed_report["violations"]["magnitudes"]["local_cpu_excess_hz"] > 0.0,
                "pred_tx_power_violation": proposed_report["violations"]["magnitudes"]["transmit_power_excess_w"] > 0.0,
                "pred_resource_absence": _get_violation(proposed_report, "resource_absence", device),
                "pred_local_work_without_cpu": _get_violation(proposed_report, "work_without_frequency", device),
                "pred_offload_without_path": _get_violation(proposed_report, "work_without_transmission", device),
                "pred_shared_bandwidth_violation": proposed_report["violations"]["shared_bandwidth_violation"] > 0.0,
                "pred_shared_edge_cpu_violation": proposed_report["violations"]["shared_edge_cpu_violation"] > 0.0,
                "pred_queue_excess_work_units": proposed_report["violations"]["magnitudes"]["queue_excess"],
                "pred_queue_work_excess_units": proposed_report["violations"]["magnitudes"]["queue_work_excess"],
                "pred_admission_excess_units": proposed_report["violations"]["magnitudes"]["admission_excess"],
                "pred_battery_deficit_j": proposed_report["violations"]["magnitudes"]["battery_deficit"],
                "pred_battery_overflow_j": proposed_report["violations"]["magnitudes"].get("battery_overflow", 0.0),
                "pred_temperature_excess_c": proposed_report["violations"]["magnitudes"]["temperature_peak_excess"],
                "pred_local_cpu_excess_hz": proposed_report["violations"]["magnitudes"]["local_cpu_excess_hz"],
                "pred_tx_power_excess_w": proposed_report["violations"]["magnitudes"]["transmit_power_excess_w"],
                "pred_shared_bandwidth_excess_hz": proposed_report["violations"]["magnitudes"]["shared_bandwidth_excess_hz"],
                "pred_shared_edge_cpu_excess_hz": proposed_report["violations"]["magnitudes"]["shared_edge_cpu_excess_hz"],
                "pred_timing_excess_s": proposed_report["violations"]["magnitudes"]["timing_excess"],
                "actual_queue_violation": actual_report["flags"]["queue_capacity"].__contains__(device) if actual_report else "",
                "actual_battery_violation": actual_report["flags"]["battery_reserve"].__contains__(device) if actual_report else "",
                "actual_temperature_endpoint_violation": actual_report["flags"]["temperature_endpoint"].__contains__(device) if actual_report else "",
                "actual_temperature_peak_violation": actual_report["flags"]["temperature_peak"].__contains__(device) if actual_report else "",
                "actual_queue_excess_work_units": actual_report["magnitudes"]["queue_excess_work_units"] if actual_report else "",
                "actual_battery_deficit_j": actual_report["magnitudes"]["battery_reserve_deficit_j"] if actual_report else "",
                "actual_temperature_endpoint_excess_c": actual_report["magnitudes"]["temperature_endpoint_excess_c"] if actual_report else "",
                "actual_temperature_peak_excess_c": actual_report["magnitudes"]["temperature_peak_excess_c"] if actual_report else "",
            }
        )
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(directory: Path, scenario: str, state: Dict[str, np.ndarray], params: Dict[str, Any], info: Dict[str, Any]) -> None:
    post = info.get("post_state")
    delta = float(params["delta"])
    times = [0.0, delta] if post is not None else [0.0]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    specs = (("Q", "queue (work units)", "Q_max"), ("B", "battery (J)", "B_min"), ("T", "temperature (C)", "T_max"))
    for axis, (key, label, bound_key) in zip(axes, specs):
        initial = np.asarray(state[key], dtype=float)
        values = [initial] + ([np.asarray(post[key], dtype=float)] if post is not None else [])
        for device in range(initial.size):
            axis.plot(times, [float(value[device]) for value in values], marker="o", label=f"device {device}")
            threshold = float(np.asarray(params[bound_key])[device])
            axis.axhline(threshold, color="red", linestyle="--", alpha=0.65, label=f"{bound_key}={threshold:g}" if device == 0 else None)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("time (s; initial state at t=0)")
    axes[-1].set_xlim(left=0.0, right=max(delta, 1e-6))
    fig.suptitle(f"{scenario}: observed state and configured thresholds")
    fig.tight_layout()
    fig.savefig(directory / "state.png", dpi=170)
    plt.close(fig)

    nominal = info["decoded_nominal_action"]
    executed = info["executed_action"]
    scales = {
        "f": np.asarray(params["f_hw_max"]),
        "p": np.asarray(params["p_hw_max"]),
        "b": np.full(len(state["Q"]), params["shared_bandwidth_hz"]),
        "f_edge": np.full(len(state["Q"]), params["shared_edge_cpu_hz"]),
        "l": np.asarray(params["Q_max"]),
        "o": np.asarray(params["Q_max"]),
        "U": np.asarray(params["A_max"]),
    }
    figure, axis = plt.subplots(figsize=(10, 4))
    x = np.arange(len(ACTION_KEYS))
    width = 0.36
    proposed_ratio = [float(np.asarray(nominal[key])[0] / scales[key][0]) for key in ACTION_KEYS]
    executed_ratio = [float(np.asarray(executed[key])[0] / scales[key][0]) if executed else np.nan for key in ACTION_KEYS]
    axis.bar(x - width / 2, proposed_ratio, width, label="decoded proposal")
    axis.bar(x + width / 2, executed_ratio, width, label="executed; blank if unobserved")
    axis.set_xticks(x, ACTION_LABELS, rotation=20, ha="right")
    axis.set_ylabel("fraction of configured action scale")
    axis.set_ylim(bottom=0)
    axis.set_title(f"{scenario}: proposed vs executed action")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    if info["intervention"]:
        axis.text(0.99, 0.95, f"intervention: {info['filter_status']}", transform=axis.transAxes, ha="right", va="top")
    figure.tight_layout()
    figure.savefig(directory / "actions.png", dpi=170)
    plt.close(figure)


def run_scenario(name: str, output_root: Path, seed: int = 0) -> Path:
    params, initial_state, raw_action, description = _scenario(name)
    env = make_env(n_devices=1, horizon=1, params=params, deterministic=True)
    _, reset_info = env.reset(seed=seed, options={"state": initial_state})
    _, reward, terminated, truncated, info = env.step(raw_action)
    directory = output_root / name
    directory.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "scenario": name,
        "seed": seed,
        "effective_config": reset_info["effective_config"],
        "initial_state": {key: np.asarray(value, dtype=float).tolist() for key, value in initial_state.items()},
        "description": description,
    }
    (directory / "config_snapshot.json").write_text(json.dumps(_json_safe(config_snapshot), indent=2, sort_keys=True) + "\n")
    _write_csv(directory / "episode.csv", name, seed, info, params)
    _write_plots(directory, name, initial_state, params, info)
    executed = info["executed_action"]
    summary = {
        "scenario": name,
        "seed": seed,
        "description": description,
        "slots_attempted": 1,
        "slots_executed": int(executed is not None),
        "filter_status": info["filter_status"],
        "one_step_verification": info["verification_status"],
        "verification_scope": info["verification_scope"],
        "intervention": info["intervention"],
        "controller_failure": info["controller_failure"],
        "failure_kind": info["failure_kind"],
        "infeasibility_proven": info["infeasibility_proven"],
        "proposal_predicted_feasible": info["proposal_predicted_report"]["feasible"],
        "selected_candidate_predicted_feasible": info["predicted_report"]["feasible"],
        "actual_outcome": info["actual_outcome"],
        "actual_feasible": info["actual_report"]["feasible"] if info["actual_report"] else None,
        "ending_reason": info["episode_ending_reason"],
        "terminated": terminated,
        "truncated": truncated,
        "reward": reward,
        "elapsed_time_s": info["elapsed_time_s"],
        "outstanding_queue_work_units_at_controller_failure": float(np.sum(info["outstanding_queue_work"])) if info["controller_failure"] else None,
        "outstanding_arrival_work_units_at_controller_failure": float(np.sum(info["outstanding_arrival_work"])) if info["controller_failure"] else None,
        "total_outstanding_work_units_at_controller_failure": float(np.sum(info["total_outstanding_work"])) if info["controller_failure"] else None,
        "thresholds": {
            "Q_max_work_units": np.asarray(params["Q_max"], dtype=float).tolist(),
            "B_min_j": np.asarray(params["B_min"], dtype=float).tolist(),
            "T_max_c": np.asarray(params["T_max"], dtype=float).tolist(),
        },
        "initial_state": {key: np.asarray(initial_state[key], dtype=float).tolist() for key in ("Q", "B", "T", "A", "g")},
        "predicted_state": info["predicted_state"],
        "actual_state": info["post_state"],
        "actual_peak_temperature_c": info["transition_metrics"]["peak_temp"].tolist() if info["transition_metrics"] else None,
        "predicted_violation_magnitudes": info["proposal_predicted_report"]["violations"]["magnitudes"],
        "actual_violation_magnitudes": info["actual_report"]["magnitudes"] if info["actual_report"] else None,
        "thermal_model_note": "Peak is max(start, endpoint) for the first-order endpoint model; continuous within-slot transients are not resolved.",
    }
    if name == "thermal":
        alpha = float(params["thermal_alpha"])
        decay = float(np.exp(-params["delta"] / params["thermal_tau"]))
        summary["thermal_stress_parameters"] = {
            "synthetic_uncalibrated": True,
            "thermal_alpha_c_per_s_per_normalized_load": alpha,
            "tau_s": float(params["thermal_tau"]),
            "ambient_c": float(params["ambient_temp"]),
            "maximum_normalized_heating": 1.0,
            "discrete_equilibrium_c_at_max_heating": float(params["ambient_temp"] + alpha * params["delta"] / (1.0 - decay)),
        }
    (directory / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    (directory / "summary.txt").write_text(
        f"{name}: status={info['filter_status']}; proposal_feasible={summary['proposal_predicted_feasible']}; "
        f"executed={executed is not None}; actual_outcome={info['actual_outcome']}; "
        f"reason={info['episode_ending_reason'] or 'one-step diagnostic complete'}\n"
    )
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Stage 1 safety diagnostics.")
    parser.add_argument("--scenario", choices=("normal", "thermal", "battery", "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("diagnostics"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    names = ("normal", "thermal", "battery") if args.scenario == "all" else (args.scenario,)
    for name in names:
        directory = run_scenario(name, args.output, args.seed)
        print(f"{name}: {directory}/summary.txt, {directory}/state.png, {directory}/actions.png, {directory}/episode.csv")


if __name__ == "__main__":
    main()