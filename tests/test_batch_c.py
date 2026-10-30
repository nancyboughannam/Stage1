"""Validate metric denominators, missing outcomes, provenance, and perturbations."""
import csv
import json
import numpy as np
import pytest
from experiments_stage1 import run_episode, build_env, SCENARIOS
from mlbac_stage1.experiment_logging import aggregate, slot_record


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_logged_scenarios_and_conservation(tmp_path, scenario):
    s = run_episode(tmp_path, scenario, 0, devices=2, horizon=3, make_plots=False)
    steps = json.loads((tmp_path/'steps.json').read_text())
    rows = list(csv.DictReader((tmp_path/'slots.csv').open()))
    assert len(rows) == len(steps) == s['attempted_slots']
    for stored, record in zip(rows, steps):
        reconstructed = slot_record(record['info'], record['filter_result'])
        assert reconstructed['filter_status'] == stored['filter_status']
        if reconstructed['admitted_work'] is not None:
            assert reconstructed['admitted_work'] == float(stored['admitted_work'])
    assert abs(s['queue_balance_residual_work']) < 1e-7
    assert s['actual_constraint_device_slot_denominator'] == 2*s['executed_slots']
    for record in steps:
        info = record['info']
        if info['executed_action'] is not None:
            assert info['predicted_report']['feasible']
            assert info['executed_action'] == info['selected_candidate_action']
        else:
            assert info['post_state'] is None and info['actual_report'] is None
    if scenario == 'battery':
        assert s['executed_slots']==0 and s['elapsed_simulated_s']==0
        assert s['actual_violation_rates_per_device_slot'] is None
        assert s['throughput_work_per_simulated_s'] is None
        assert rows[0]['actual_feasible']==rows[0]['rejected_work']==''
        assert rows[0]['actual_temperature_peak_violating_devices']==''
        assert not s['service_acceptable_under_provisional_criteria']
    elif scenario == 'thermal_mismatch':
        assert s['ending_reason']=='actual_constraint_violation'
        assert steps[0]['info']['predicted_report']['feasible']
        assert not steps[0]['info']['actual_report']['feasible']
        assert s['maximum_actual_violation_magnitudes']['temperature_peak_excess_c'] > 0
    elif scenario == 'invalid_proposal':
        assert all(r['proposal_status']=='invalid_proposal' for r in rows)
        assert s['verified_fallback_fraction']==1
        assert not s['service_acceptable_under_provisional_criteria']
    else:
        assert s['full_horizon_completed']
        assert all(v==0 for v in s['actual_violation_rates_per_device_slot'].values())


def test_simulator_mismatch_hidden_and_common_random_traces():
    a,_=build_env('thermal_matched',2,3,9,1.)
    b,_=build_env('thermal_mismatch',2,3,9,1.)
    assert a.params['thermal_alpha']==b.params['thermal_alpha']==60.
    assert b._actual_thermal_alpha==90.
    assert 'actual_thermal_alpha' not in b.filter.params
    np.testing.assert_array_equal(a._arrival_trace,b._arrival_trace)
    np.testing.assert_array_equal(a._channel_trace,b._channel_trace)
    a.close(); b.close()


def test_timeout_logged_without_fake_violation(tmp_path, monkeypatch):
    def timed_out(*args):
        return {'status':'timeout','elapsed_s':.01,'attempts':[], 'candidates':[]}
    monkeypatch.setattr('mlbac_stage1.safety_filter.search_candidates',timed_out)
    s=run_episode(tmp_path,'resource_pressure',0,devices=2,horizon=2,make_plots=False)
    rows=list(csv.DictReader((tmp_path/'slots.csv').open()))
    assert all(r['solver_status']=='timeout' for r in rows)
    assert s['verified_fallback_fraction']==1
    assert all(v==0 for v in s['actual_violation_rates_per_device_slot'].values())


def test_service_failure_despite_small_queues_and_no_violations(tmp_path):
    s=run_episode(tmp_path,'reference',0,devices=2,horizon=3,
                  max_rejection=0.,min_throughput=100.,make_plots=False)
    assert s['full_horizon_completed']
    assert not s['service_acceptable_under_provisional_criteria']


def test_single_seed_has_no_confidence_interval_and_missing_metrics_excluded(tmp_path):
    a=run_episode(tmp_path/'a','reference',0,devices=2,horizon=2,make_plots=False)
    b=run_episode(tmp_path/'b','battery',0,devices=2,horizon=2,make_plots=False)
    summary=aggregate([a,b])
    metric=summary['metrics']['throughput_work_per_simulated_s']
    assert metric['n_valid_episodes']==1
    assert metric['mean']==a['throughput_work_per_simulated_s']
    assert metric['approximate_95pct_t_interval'] is None
