"""Batch 2: hidden simulator mismatch and honest one-step error observations."""
from copy import deepcopy
import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from mlbac_stage1.config import make_default_params
from mlbac_stage1.constraints import evaluate_constraints
from mlbac_stage1.env import make_env
from mlbac_stage1.models import ACTION_FIELDS, slot_quantities, transition_metrics
from mlbac_stage1.model_uncertainty import COMPARISON_FIELDS
from mlbac_stage1.experiment_logging import device_prediction_records, slot_record, write_json
from experiments_stage1 import run_episode, build_env


def physical_case():
    p = make_default_params(2)
    s = {k: np.full(2, v) for k, v in dict(Q=1., B=25920., T=30., A=1., g=1e-5).items()}
    a = {k: np.full(2, v) for k, v in dict(f=5e8, p=.2, b=4e6, f_edge=1e9, l=.1, o=.05, U=.1).items()}
    return p, s, a


def battery_step(multiplier=1.25, margin=.5, enabled=True, reserve_headroom=1.75):
    p = make_default_params(1, horizon=3, safety_margin_enabled=enabled, battery_margin_j=margin)
    p['actual_energy_multiplier'] = multiplier
    env = make_env(params=p)
    _, reset = env.reset(seed=7, options={'state': {'B': p['B_min'] + reserve_headroom}})
    _, _, _, _, info = env.step(np.full(7, -1.))
    return env, reset, info


def thermal_step(actual_alpha=8.):
    p = make_default_params(1, horizon=3, safety_margin_enabled=True, temperature_margin_c=2.)
    p.update(thermal_alpha=6., actual_thermal_alpha=actual_alpha, initial_temperature_c=79.)
    env = make_env(params=p)
    _, reset = env.reset(seed=7)
    raw = np.full(7, -1.); raw[0] = 0.  # 0.5 f_max; synthetic heating experiment
    _, _, _, _, info = env.step(raw)
    return env, reset, info


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf'), -float('inf'), [1.], [[1.]], 'bad', True, None])
def test_invalid_energy_multiplier_rejected(bad):
    p, s, a = physical_case(); p['actual_energy_multiplier'] = bad
    with pytest.raises(ValueError, match='actual_energy_multiplier'):
        make_env(params=p)
    with pytest.raises(ValueError, match='actual_energy_multiplier'):
        transition_metrics(s, a, p)


@pytest.mark.parametrize('multiplier', [0., .5, 1., 1.5, 2.])
def test_actual_energy_components_scale_and_work_is_unchanged(multiplier):
    p, s, a = physical_case()
    nominal = transition_metrics(s, a, p)
    actual = transition_metrics(s, a, dict(p, actual_energy_multiplier=multiplier))
    for field in ('local_energy', 'tx_energy', 'baseline_energy', 'energy', 'total_energy'):
        np.testing.assert_allclose(actual[field], nominal[field] * multiplier, rtol=1e-15)
    np.testing.assert_array_equal(actual['energy'], actual['local_energy']+actual['tx_energy']+actual['baseline_energy'])
    np.testing.assert_array_equal(actual['battery_next'], s['B']-actual['energy'])
    np.testing.assert_array_equal(actual['predicted_energy_j'], nominal['energy'])
    np.testing.assert_array_equal(actual['energy_prediction_error_j'], actual['energy']-nominal['energy'])
    for field in ('queue_next', 'completed_local_work', 'completed_offload_work', 'rejected_arrivals',
                  'local_time', 'tx_time', 'edge_time', 'rates', 'temperature_next', 'peak_temp'):
        np.testing.assert_array_equal(actual[field], nominal[field])
    if multiplier == 1.:
        for field in nominal:
            np.testing.assert_array_equal(actual[field], nominal[field])


def test_nominal_equations_preserved():
    p, s, a = physical_case()
    q = slot_quantities(s, a, p)
    expected = p['kappa'] * a['f']**3 * q['local_time'] + a['p']*q['tx_time'] + p['baseline_power']*p['delta']
    m = transition_metrics(s, a, p)
    np.testing.assert_allclose(m['energy'], expected, rtol=1e-15)
    np.testing.assert_array_equal(m['battery_next'], s['B']-m['energy'])


def test_actual_only_keys_do_not_change_standalone_prediction():
    p, s, a = physical_case()
    ordinary = evaluate_constraints(s, a, p)
    hidden = evaluate_constraints(s, a, dict(p, actual_energy_multiplier=100., actual_thermal_alpha=100.))
    for key in ('Q', 'B', 'T'):
        np.testing.assert_array_equal(ordinary['predicted_state'][key], hidden['predicted_state'][key])
    np.testing.assert_array_equal(ordinary['predicted_energy_j'], hidden['predicted_energy_j'])


def test_energy_model_hidden_from_filter_and_snapshot_separated():
    env, reset, info = battery_step()
    for key in ('actual_energy_multiplier', 'actual_thermal_alpha', 'simulator_only_config'):
        assert key not in env.params and key not in env.filter.params
        assert key not in reset['effective_config']
    assert reset['simulator_only_config']['actual_energy_multiplier'] == 1.25
    assert reset['simulator_only_config']['effective_actual_thermal_alpha'] == env.params['thermal_alpha']
    assert info['predicted_report']['feasible'] and info['actual_report']['feasible']
    r = info['prediction_errors']
    assert r['predicted_energy_j'][0] == 1.
    assert r['actual_energy_j'][0] == 1.25
    assert r['energy_error_j'][0] == .25
    assert r['battery_error_j'][0] == -.25
    assert r['battery_adverse_error_within_configured_margin'][0]
    assert r['battery_adverse_error_within_effective_margin'][0]
    assert info['elapsed_time_s'] == 1.


def test_exceeded_battery_margin_records_real_violation():
    _, _, info = battery_step(multiplier=2.)
    assert info['filter_status'] == 'nominal_accepted'
    assert info['predicted_report']['feasible']
    assert not info['actual_report']['feasible']
    assert info['episode_ending_reason'] == 'actual_constraint_violation'
    assert info['actual_report']['magnitudes']['battery_reserve_deficit_j'] == .25
    assert info['prediction_errors']['battery_error_j'][0] == -1.
    assert not info['prediction_errors']['battery_adverse_error_within_configured_margin'][0]
    assert info['post_state'] is not None and info['elapsed_time_s'] == 1.


@pytest.mark.parametrize('alpha, covered, safe', [(8., True, True), (20., False, False)])
def test_thermal_error_covered_and_exceeded(alpha, covered, safe):
    env, reset, info = thermal_step(alpha)
    assert 'actual_thermal_alpha' not in env.filter.params
    assert reset['effective_config']['thermal_alpha'] == 6.
    assert reset['simulator_only_config']['effective_actual_thermal_alpha'] == alpha
    assert info['filter_status'] == 'nominal_accepted'
    r = info['prediction_errors']
    assert r['temperature_error_c'][0] == pytest.approx(.35*(alpha-6.))
    assert bool(r['temperature_adverse_error_within_configured_margin'][0]) is covered
    assert info['actual_report']['feasible'] is safe
    matched = thermal_step(6.)[2]
    np.testing.assert_array_equal(r['predicted_temperature_c'], matched['prediction_errors']['predicted_temperature_c'])
    np.testing.assert_array_equal(r['predicted_battery_j'], matched['prediction_errors']['predicted_battery_j'])


def test_both_mismatches_simultaneously():
    p = make_default_params(1, horizon=1)
    p.update(actual_energy_multiplier=1.5, actual_thermal_alpha=2.)
    env = make_env(params=p); env.reset(seed=0)
    raw = np.full(7, -1.); raw[0] = 0.
    info = env.step(raw)[4]; r = info['prediction_errors']
    assert r['energy_error_j'][0] == .5
    assert r['temperature_error_c'][0] == pytest.approx(.35*(2.-.15))


def test_disabled_margin_coverage_is_retrospective_only():
    _, _, info = battery_step(enabled=False)
    r = info['prediction_errors']
    assert not r['safety_margins']['enabled']
    assert r['battery_adverse_error_within_configured_margin'][0]
    assert not r['battery_adverse_error_within_effective_margin'][0]


def test_favorable_error_differs_from_absolute_error():
    _, _, info = battery_step(multiplier=0., margin=.1)
    r = info['prediction_errors']
    assert r['battery_error_j'][0] == 1.
    assert r['battery_adverse_error_j'][0] == 0.
    assert r['battery_adverse_error_within_configured_margin'][0]
    assert not r['battery_absolute_error_within_configured_margin'][0]


def test_per_device_margin_coverage():
    p = make_default_params(2, battery_margin_j=[.1, .5])
    p['actual_energy_multiplier'] = 1.25
    env = make_env(params=p); env.reset(seed=0)
    r = env.step(np.full(14, -1.))[4]['prediction_errors']
    np.testing.assert_array_equal(r['battery_adverse_error_within_configured_margin'], [False, True])


def test_coverage_does_not_use_checker_tolerance():
    _, _, info = battery_step(multiplier=1.25, margin=.25-5e-10)
    r = info['prediction_errors']
    assert not r['battery_adverse_error_within_configured_margin'][0]


def test_unexecuted_has_no_comparison_or_fake_zero(tmp_path):
    env, _, info = battery_step(reserve_headroom=.5)
    assert info['executed_action'] is None and info['elapsed_time_s'] == 0.
    assert not info['infeasibility_proven']
    for key in COMPARISON_FIELDS:
        assert info['prediction_errors'][key] is None
        assert device_prediction_records(info)[0][key] is None
    row = slot_record(info, env.last_filter_result)
    assert row['energy_j'] is row['predicted_energy_j'] is row['energy_prediction_error_j'] is None
    write_json(tmp_path/'step.json', info)
    saved = json.loads((tmp_path/'step.json').read_text())
    assert saved['prediction_errors']['battery_error_j'] is None


def test_matched_errors_are_zero():
    _, _, info = battery_step(multiplier=1.)
    for key in ('temperature_error_c', 'battery_error_j', 'energy_error_j'):
        np.testing.assert_array_equal(info['prediction_errors'][key], [0.])


def test_same_seed_reproduces_exogenous_traces():
    a, _ = build_env('reference', 2, 4, 42, 1.)
    b, _ = build_env('reference', 2, 4, 42, 1., actual_energy_multiplier=2., actual_thermal_alpha=10.)
    for name in ('_arrival_trace', '_channel_trace'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    raw = np.full(14, -1.); raw[0] = 0.
    first_a = a.step(raw)[4]; first_b = b.step(raw)[4]
    for field in ACTION_FIELDS:
        np.testing.assert_array_equal(first_a['executed_action'][field], first_b['executed_action'][field])
    np.testing.assert_array_equal(first_a['post_state']['Q'], first_b['post_state']['Q'])
    assert not np.array_equal(first_a['post_state']['B'], first_b['post_state']['B'])


def test_prediction_uses_corrected_action_not_original_proposal():
    p = make_default_params(2, horizon=1)
    p['actual_energy_multiplier'] = 1.25
    env = make_env(params=p); env.reset(seed=0)
    info = env.step(np.full(14, .5))[4]
    assert info['intervention'] and info['executed_action'] is not None
    r = info['prediction_errors']
    expected = slot_quantities(info['pre_state'], info['executed_action'], env.params)['total_energy']
    np.testing.assert_array_equal(r['predicted_energy_j'], expected)
    np.testing.assert_allclose(r['actual_energy_j'], expected*1.25)
    for field in ACTION_FIELDS:
        np.testing.assert_array_equal(info['executed_action'][field], env.last_filter_result['action'][field])


@pytest.mark.parametrize('scenario', ['reference', 'battery'])
def test_runner_csv_and_config(tmp_path, scenario):
    summary = run_episode(tmp_path, scenario, 0, devices=2, horizon=2, make_plots=False,
                          actual_energy_multiplier=1.25, actual_thermal_alpha=.3, battery_margin_j=.5)
    snapshot = json.loads((tmp_path/'config_snapshot.json').read_text())
    assert snapshot['simulator_only_config']['actual_energy_multiplier'] == 1.25
    assert snapshot['simulator_only_config']['effective_actual_thermal_alpha'] == .3
    assert snapshot['controller_config']['thermal_alpha'] == .15
    assert 'actual_energy_multiplier' not in snapshot['controller_config']
    rows = list(csv.DictReader((tmp_path/'prediction_errors.csv').open()))
    assert len(rows) == 2*summary['attempted_slots']
    assert abs(summary['queue_balance_residual_work']) < 1e-9
    if scenario == 'battery':
        assert all(row['energy_error_j'] == row['actual_battery_j'] == '' for row in rows)
    else:
        assert all(float(row['energy_error_j']) > 0 for row in rows)
        assert all(row['executed'] == 'True' for row in rows)


@pytest.mark.parametrize('flag', ['--actual-energy-multiplier', '--actual-thermal-alpha'])
def test_invalid_cli_mismatch_rejected(flag, tmp_path):
    result = subprocess.run([sys.executable, 'experiments_stage1.py', '--output', str(tmp_path/'unused'),
                             flag, '-1'], text=True, capture_output=True)
    assert result.returncode == 2
    assert 'finite and nonnegative' in result.stderr
    assert not (tmp_path/'unused').exists()


def test_controlled_diagnostic_outputs(tmp_path):
    from mismatch_diagnostics import run, CASES
    summaries = {s['scenario']: s for s in run(tmp_path, seed=0)}
    assert set(summaries) == set(CASES)
    for name in ('energy_matched', 'energy_covered', 'thermal_matched', 'thermal_covered'):
        assert summaries[name]['actual_feasible'] is True
    for name in ('energy_exceeded', 'thermal_exceeded'):
        assert summaries[name]['actual_feasible'] is False
        assert summaries[name]['ending_reason'] == 'actual_constraint_violation'
    assert summaries['no_execution']['actual_feasible'] is None
    assert summaries['no_execution']['executed_slots'] == 0
    records = json.loads((tmp_path/'no_execution'/'steps.json').read_text())
    assert records[0]['info']['prediction_errors']['temperature_error_c'] is None
    for name in CASES:
        assert (tmp_path/name/'prediction_errors.csv').is_file()


def test_fallback_error_is_for_executed_idle():
    p = make_default_params(1, horizon=1)
    p['actual_energy_multiplier'] = 1.5
    env = make_env(params=p); env.reset(seed=0)
    info = env.step(np.full(7, np.nan))[4]
    assert info['filter_status'] == 'fallback_verified'
    assert info['prediction_errors']['predicted_energy_j'][0] == 1.
    assert info['prediction_errors']['actual_energy_j'][0] == 1.5
    assert info['prediction_errors']['battery_error_j'][0] == -.5


@pytest.mark.parametrize('bad', [-1., np.nan, np.inf, [1.]])
def test_invalid_thermal_override_is_rejected(bad):
    p = make_default_params(1); p['actual_thermal_alpha'] = bad
    with pytest.raises(ValueError, match='actual_thermal_alpha'):
        make_env(params=p)
