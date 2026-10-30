"""Auditable episode logs; missing execution is never counted as a safe slot."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def slot_record(info, result):
    """One joint row per attempted slot (not duplicated per device)."""
    executed = info['executed_action'] is not None
    metrics = info['transition_metrics']
    actual = info['actual_report']
    proposal = info['decoded_nominal_action']
    row = {k: info.get(k) for k in (
        'slot', 'elapsed_time_s_start', 'elapsed_time_s', 'filter_status',
        'proposal_status', 'solver_status', 'selected_solver_status',
        'fallback_reason', 'fallback_attempted', 'intervention',
        'action_preprocessed', 'objective_value', 'correction_norm',
        'controller_failure', 'episode_ending_reason', 'reward')}
    row.update(executed=executed,
               proposal_predicted_feasible=info['proposal_predicted_report']['feasible'],
               selected_predicted_feasible=(info['predicted_report']['feasible']
                   if info['selected_candidate_action'] is not None else None),
               actual_feasible=actual['feasible'] if actual else None,
               filter_elapsed_ms=1000 * result['filter_elapsed_s'],
               solver_elapsed_ms=1000 * result['solver_elapsed_s'],
               offered_work=float(np.sum(info['pre_state']['A'])),
               admitted_work=float(np.sum(info['executed_action']['U'])) if executed else None,
               completed_work=float(np.sum(metrics['completed_local_work']) +
                                    np.sum(metrics['completed_offload_work'])) if executed else None,
               rejected_work=float(np.sum(metrics['rejected_arrivals'])) if executed else None,
               energy_j=float(np.sum(metrics['energy'])) if executed else None,
               admission_change_work=(float(np.sum(np.asarray(info['executed_action']['U']) - np.asarray(proposal['U'])))
                   if executed and 'U' in proposal else None))
    for key in ('queue_capacity', 'queue_lower', 'battery_capacity', 'nonfinite_state',
                'battery_reserve', 'temperature_endpoint', 'temperature_peak'):
        row['actual_' + key + '_violating_devices'] = None
    for key in ('queue_excess_work_units', 'battery_reserve_deficit_j',
                'temperature_endpoint_excess_c', 'temperature_peak_excess_c'):
        row['actual_' + key] = None
    if actual:
        for key, indices in actual['flags'].items():
            row['actual_' + key + '_violating_devices'] = len(indices)
        for key, value in actual['magnitudes'].items():
            row['actual_' + key] = value
    return row


def summarize(rows, records, config, initial, horizon, max_rejection, min_throughput):
    observed = [r for r in rows if r['executed']]
    def total(key):
        return float(sum(r[key] for r in observed))
    elapsed = len(observed) * config['delta']
    offered = sum(r['offered_work'] for r in observed)
    rejection = total('rejected_work') / offered if offered else None
    throughput = total('completed_work') / elapsed if elapsed else None
    last_post = next((r['info']['post_state'] for r in reversed(records)
                      if r['info']['post_state'] is not None), initial)
    balance = (float(np.sum(last_post['Q']) - np.sum(initial['Q']))
               - total('admitted_work') + total('completed_work'))
    actual_rates = {}
    magnitudes = {}
    for record in records:
        report = record['info']['actual_report']
        if report:
            for key, indices in report['flags'].items():
                actual_rates[key] = actual_rates.get(key, 0) + len(indices)
            for key, value in report['magnitudes'].items():
                magnitudes[key] = max(magnitudes.get(key, 0), value)
    denominator = len(observed) * config['n_devices']
    actual_rates = {k: v / denominator for k, v in actual_rates.items()}
    full = len(observed) == horizon
    no_violation = bool(observed) and all(r['actual_feasible'] for r in observed)
    acceptable = (full and no_violation and rejection is not None and
                  rejection <= max_rejection and throughput >= min_throughput)
    return {
        'attempted_slots': len(rows), 'executed_slots': len(observed),
        'elapsed_simulated_s': elapsed, 'full_horizon_completed': full,
        'ending_reason': rows[-1]['episode_ending_reason'],
        'filter_status_counts': dict(Counter(r['filter_status'] for r in rows)),
        'intervention_fraction_per_attempt': sum(r['intervention'] for r in rows)/len(rows),
        'fallback_attempt_fraction': sum(r['fallback_attempted'] for r in rows)/len(rows),
        'verified_fallback_fraction': sum(r['filter_status']=='fallback_verified' for r in rows)/len(rows),
        'controller_failure_count': sum(r['controller_failure'] for r in rows),
        'offered_work_executed_slots': offered,
        'offered_work_unexecuted_attempts': sum(r['offered_work'] for r in rows if not r['executed']),
        'admitted_work': total('admitted_work'), 'completed_work': total('completed_work'),
        'rejected_work': total('rejected_work'), 'energy_j': total('energy_j'),
        'rejection_fraction_executed_slots': rejection,
        'throughput_work_per_simulated_s': throughput,
        'queue_balance_residual_work': balance,
        'actual_constraint_device_slot_denominator': denominator,
        'actual_violation_rates_per_device_slot': actual_rates if denominator else None,
        'maximum_actual_violation_magnitudes': magnitudes if denominator else None,
        'filter_runtime_ms_p50': float(np.percentile([r['filter_elapsed_ms'] for r in rows],50)),
        'filter_runtime_ms_p95': float(np.percentile([r['filter_elapsed_ms'] for r in rows],95)),
        'filter_runtime_ms_max': max(r['filter_elapsed_ms'] for r in rows),
        'mean_correction_norm': (float(np.mean([r['correction_norm'] for r in rows if r['correction_norm'] is not None]))
                                 if any(r['correction_norm'] is not None for r in rows) else None),
        'service_acceptable_under_provisional_criteria': bool(acceptable),
        'criteria': {'max_rejection_fraction':max_rejection, 'min_throughput_work_per_s':min_throughput,
                     'requires_full_horizon_and_zero_observed_violations':True},
        'scope':'Finite-horizon simulator evidence; not physical safety, stability, or maximum safe load.'}


def aggregate(summaries):
    """Independent seeds are the sampling units; never treat devices/slots as replicates."""
    from scipy.stats import t
    result = {}
    for key in ('throughput_work_per_simulated_s', 'rejection_fraction_executed_slots',
                'energy_j', 'intervention_fraction_per_attempt', 'filter_runtime_ms_p95'):
        values = [s[key] for s in summaries if s[key] is not None]
        mean = float(np.mean(values)) if values else None
        interval = None
        if len(values) > 1:
            half = float(t.ppf(.975, len(values)-1) * np.std(values, ddof=1) / np.sqrt(len(values)))
            interval = [mean-half, mean+half]
        result[key] = {'mean':mean, 'n_valid_episodes':len(values), 'approximate_95pct_t_interval':interval}
    return {'metrics':result, 'n_episodes':len(summaries),
            'acceptable_episode_count':sum(s['service_acceptable_under_provisional_criteria'] for s in summaries),
            'note':'Approximate t intervals across seeds; small samples and early stops limit interpretation. Undefined metrics are excluded and counts reported.'}
