"""Batch 2 controlled one-step checks; synthetic values, not calibrated bounds."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import numpy as np

from mlbac_stage1.config import make_default_params
from mlbac_stage1.env import make_env
from mlbac_stage1.experiment_logging import write_json, write_csv, slot_record, device_prediction_records

CASES = ('energy_matched', 'energy_covered', 'energy_exceeded',
         'thermal_matched', 'thermal_covered', 'thermal_exceeded', 'no_execution')


def run(output, seed=0):
    """Selected buffers intentionally cover one error and fail to cover another."""
    root = Path(output)
    summaries = []
    for name in CASES:
        p = make_default_params(1, horizon=1, safety_margin_enabled=True)
        raw = np.full(7, -1.)
        options = None
        if name.startswith('energy_') or name == 'no_execution':
            p['battery_margin_j'] = .5
            p['actual_energy_multiplier'] = {'energy_matched': 1., 'energy_covered': 1.25,
                                             'energy_exceeded': 2., 'no_execution': 1.25}[name]
            options = {'state': {'B': p['B_min'] + (0.5 if name == 'no_execution' else 1.75)}}
        else:
            p.update(thermal_alpha=6., initial_temperature_c=79., temperature_margin_c=2.)
            p['actual_thermal_alpha'] = {'thermal_matched': 6., 'thermal_covered': 8.,
                                        'thermal_exceeded': 20.}[name]
            raw[0] = 0.  # fixed half-frequency heating input, no completed local work
        env = make_env(params=p)
        try:
            _, reset = env.reset(seed=seed, options=options)
            initial = env._copy_state(env.state)
            info = env.step(raw)[4]
            result = env.last_filter_result
            destination = root/name
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination/'config_snapshot.json', {
                'controller_config': reset['effective_config'],
                'simulator_only_config': reset['simulator_only_config'],
                'initial_state': initial, 'seed': seed, 'scenario': name,
                'policy': 'fixed_scripted_one_step_diagnostic_not_PPO',
                'python': platform.python_version(),
                'versions': {key: importlib.metadata.version(key) for key in ('numpy', 'scipy', 'gymnasium')},
                'scope': 'Illustrative thermal coefficients and margins; no hardware calibration or universal error bound.'})
            write_json(destination/'steps.json', [{'info': info, 'filter_result': result}])
            write_csv(destination/'slots.csv', [slot_record(info, result)])
            write_csv(destination/'prediction_errors.csv', device_prediction_records(info))
            errors = info['prediction_errors']
            summary = {
                'scenario': name, 'seed': seed, 'filter_status': info['filter_status'],
                'executed_slots': int(info['executed_action'] is not None), 'attempted_slots': 1,
                'elapsed_simulated_s': info['elapsed_time_s'], 'ending_reason': info['episode_ending_reason'],
                'actual_feasible': info['actual_report']['feasible'] if info['actual_report'] else None,
                'temperature_error_c': errors['temperature_error_c'],
                'battery_error_j': errors['battery_error_j'], 'energy_error_j': errors['energy_error_j'],
                'thermal_error_covered': errors['temperature_adverse_error_within_configured_margin'],
                'battery_error_covered': errors['battery_adverse_error_within_configured_margin'],
            }
            write_json(destination/'summary.json', summary)
            summaries.append(summary)
            print(f"{name}: {info['filter_status']}; actual_feasible={summary['actual_feasible']}", flush=True)
        finally:
            env.close()
    write_json(root/'diagnostic_summary.json', summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='experiments_mismatch_batch2')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    run(args.output, args.seed)


if __name__ == '__main__':
    main()
