"""Batch 1: opt-in margins across all one-step authorization gates."""
from copy import deepcopy
import json

import numpy as np
import pytest

from mlbac_stage1.config import make_default_params, safety_margin_settings
from mlbac_stage1.constraints import evaluate_constraints
from mlbac_stage1.env import make_env
from mlbac_stage1.experiment_logging import write_json
from mlbac_stage1.models import ACTION_FIELDS
from mlbac_stage1.optimization import ProjectionProblem
from mlbac_stage1.safety_filter import SafetyFilter


def case(**kwargs):
    p = make_default_params(1, **kwargs)
    s = {k: np.array([v]) for k, v in dict(Q=1., B=25920., T=30., A=1., g=1e-5).items()}
    a = {k: np.zeros(1) for k in ACTION_FIELDS}
    return p, s, a


def fake_search(candidate=None):
    return {'status': 'failed', 'elapsed_s': 0., 'attempts': [{'status': 'failed'}],
            'candidates': [] if candidate is None else [{'action': candidate, 'solver_status': 'converged'}]}


@pytest.mark.parametrize('key', ['temperature_margin_c', 'battery_margin_j'])
@pytest.mark.parametrize('bad', [-1., np.nan, np.inf, -np.inf, [1., 2.], [[1.]], 'invalid'])
def test_margin_validation(key, bad):
    with pytest.raises(ValueError, match=key):
        make_default_params(1, **{key: bad})
    p = make_default_params(1)
    p[key] = bad
    with pytest.raises(ValueError, match=key):
        SafetyFilter(p)
    with pytest.raises(ValueError, match=key):
        make_env(params=p)


@pytest.mark.parametrize('value', [0., 2., [2.]])
def test_valid_margins(value):
    p = make_default_params(1, temperature_margin_c=value, battery_margin_j=value)
    assert not safety_margin_settings(p)['enabled']
    np.testing.assert_array_equal(safety_margin_settings(p)['effective_battery_margin_j'], [0.])


def test_per_device_margins():
    p = make_default_params(2, safety_margin_enabled=True,
                            temperature_margin_c=[1., 2.], battery_margin_j=3.)
    settings = safety_margin_settings(p)
    np.testing.assert_array_equal(settings['effective_temperature_margin_c'], [1., 2.])
    np.testing.assert_array_equal(settings['effective_battery_margin_j'], [3., 3.])


@pytest.mark.parametrize('bad', [1, 'yes', None])
def test_flag_must_be_boolean(bad):
    with pytest.raises(ValueError, match='boolean'):
        make_default_params(safety_margin_enabled=bad)


def test_disabled_margins_preserve_behavior():
    p, s, a = case()
    base = evaluate_constraints(s, a, p)
    p.update(temperature_margin_c=10000., battery_margin_j=1e8)
    r = evaluate_constraints(s, a, p)
    assert r['feasible'] == base['feasible']
    assert r['metrics'] == base['metrics']
    assert r['violations'] == base['violations']
    assert r['margin_checks']['margin_battery_pass'] is None
    for key in ('Q', 'B', 'T'):
        np.testing.assert_array_equal(r['predicted_state'][key], base['predicted_state'][key])


@pytest.mark.parametrize('kind', ['thermal', 'battery'])
def test_real_search_does_not_accept_ordinary_safe_proposal_unchanged(kind):
    p, s, a = case(safety_margin_enabled=True)
    a['f'][:] = 1e9
    a['l'][:] = 1.
    if kind == 'thermal':
        p['thermal_alpha'] = 60.
        s['T'][:] = 30.
        # Choose an endpoint buffer that excludes the proposal but admits idle.
        p['temperature_margin_c'] = 10.
    else:
        s['B'][:] = p['B_min'] + 2.
        p['battery_margin_j'] = .8
    ordinary = evaluate_constraints(s, a, dict(p, safety_margin_enabled=False))
    assert ordinary['feasible']
    assert not evaluate_constraints(s, a, p)['feasible']
    r = SafetyFilter(p).filter_action(s, a)
    assert r['status'] in ('corrected_candidate_verified', 'fallback_verified')
    assert evaluate_constraints(s, r['action'], p)['feasible']
    assert r['correction_norm'] > 0
    assert r['candidate_checks']
    assert r['safety_claim'] == 'one_step_predicted_with_model_margins'


def test_optimizer_targets_same_thresholds():
    p, s, a = case(safety_margin_enabled=True, temperature_margin_c=10., battery_margin_j=2.)
    problem = ProjectionProblem(s, a, p)
    x = next(problem.starts())
    report = evaluate_constraints(s, a, p)
    checks = report['margin_checks']
    np.testing.assert_array_equal(problem.temperature_threshold, checks['temperature_threshold_c'])
    np.testing.assert_array_equal(problem.battery_threshold, checks['battery_threshold_j'])
    residual = problem.residuals(x)
    assert residual[-2] == pytest.approx((problem.temperature_threshold[0] - report['predicted_state']['T'][0]) / 100.)
    assert residual[-3] == pytest.approx((report['predicted_state']['B'][0] - problem.battery_threshold[0]) / (s['B'][0]-p['B_min'][0]))


def test_solver_success_cannot_bypass_margins(monkeypatch):
    p, s, a = case(safety_margin_enabled=True, battery_margin_j=.8)
    s['B'][:] = p['B_min'] + 2.
    a['f'][:] = 1e9; a['l'][:] = 1.
    monkeypatch.setattr('mlbac_stage1.safety_filter.search_candidates', lambda *args: fake_search(a))
    r = SafetyFilter(p).filter_action(s, a)
    assert not r['candidate_checks'][0]['verified']
    assert r['status'] == 'fallback_verified'
    assert r['report']['margin_checks']['margin_battery_pass'].all()


@pytest.mark.parametrize('margin, safe', [(0.5, True), (2., False)])
def test_idle_uses_margin(monkeypatch, margin, safe):
    p, s, a = case(safety_margin_enabled=True, battery_margin_j=margin)
    s['B'][:] = p['B_min'] + 2.
    a['U'][:] = 2.  # force search then fallback
    monkeypatch.setattr('mlbac_stage1.safety_filter.search_candidates', lambda *args: fake_search())
    r = SafetyFilter(p).filter_action(s, a)
    assert r['safe'] is safe
    assert r['fallback_attempted']
    assert r['status'] == ('fallback_verified' if safe else 'no_verified_safe_continuation')
    assert not r['infeasibility_proven']


@pytest.mark.parametrize('kind', ['thermal', 'battery'])
def test_numerical_tolerance_cannot_relax_margin(kind):
    p, s, a = case(safety_margin_enabled=True)
    if kind == 'battery':
        p['battery_margin_j'] = 2.
        s['B'][:] = p['B_min'] + 3. - 5e-10  # after baseline energy: below tightened reserve
    else:
        p['temperature_margin_c'] = 2.
        s['T'][:] = 25.
        p['thermal_alpha'] = 0.
        p['T_max'][:] = 27. - 5e-10
    r = evaluate_constraints(s, a, p)
    assert not r['feasible']
    assert evaluate_constraints(s, a, dict(p, safety_margin_enabled=False))['feasible']


def test_starting_temperature_not_authorized_above_hard_limit():
    p, s, a = case(safety_margin_enabled=True)
    s['T'][:] = p['T_max'] + 5e-10
    assert evaluate_constraints(s, a, dict(p, safety_margin_enabled=False))['feasible']
    assert not evaluate_constraints(s, a, p)['feasible']
    assert SafetyFilter(p).filter_action(s, a)['action'] is None


def test_enormous_margin_stops_without_actual_state():
    p = make_default_params(1, safety_margin_enabled=True, battery_margin_j=1e8)
    env = make_env(params=p); env.reset(seed=3)
    before = deepcopy(env.state)
    _, _, done, _, info = env.step(np.full(7, -1.))
    assert done and info['filter_status'] == 'no_verified_safe_continuation'
    assert not info['infeasibility_proven']
    assert info['elapsed_time_s'] == 0.
    for key in ('post_state', 'actual_report', 'executed_action', 'transition_metrics'):
        assert info[key] is None
    for key in before:
        np.testing.assert_array_equal(env.state[key], before[key])


def test_final_gate_rechecks_margin(monkeypatch):
    p = make_default_params(1, safety_margin_enabled=True, battery_margin_j=.8)
    env = make_env(params=p)
    env.reset(seed=0, options={'state': {'B': p['B_min'] + 2.}})
    original = env.filter.filter_action
    def corrupt(state, action):
        r = original(state, action)
        r['action']['f'][:] = 1e9; r['action']['l'][:] = 1.
        return r
    monkeypatch.setattr(env.filter, 'filter_action', corrupt)
    _, _, done, _, info = env.step(np.full(7, -1.))
    assert done and info['filter_status'] == 'final_verification_failed'
    assert info['executed_action'] is None and env.timestep == 0
    assert info['predicted_report']['margin_checks']['ordinary_battery_pass'].all()


def test_execution_equals_verified_action_and_snapshot(tmp_path):
    p = make_default_params(2, safety_margin_enabled=True,
                            temperature_margin_c=[1., 2.], battery_margin_j=3.)
    env = make_env(params=p); _, reset = env.reset(seed=0)
    _, _, _, _, info = env.step(np.full(14, .5))
    assert info['executed_action'] is not None
    for key in ACTION_FIELDS:
        np.testing.assert_array_equal(info['executed_action'][key], env.last_filter_result['action'][key])
    assert evaluate_constraints(info['pre_state'], info['executed_action'], p)['feasible']
    write_json(tmp_path / 'config.json', reset['effective_config'])
    saved = json.loads((tmp_path / 'config.json').read_text())
    assert saved['safety_margin_settings']['effective_battery_margin_j'] == [3., 3.]
    assert saved['safety_margin_settings']['effective_temperature_margin_c'] == [1., 2.]
    write_json(tmp_path / 'step.json', env.last_filter_result)
    record = json.loads((tmp_path / 'step.json').read_text())
    for name in ('proposal_report', 'report'):
        assert 'margin_checks' in record[name]
        assert 'safety_margins' in record[name]


def test_verified_corrected_candidate_keeps_exact_action(monkeypatch):
    p, s, a = case(safety_margin_enabled=True, battery_margin_j=.8)
    s['B'][:] = p['B_min'] + 2.
    a['f'][:] = 1e9; a['l'][:] = 1.
    candidate = deepcopy(a)
    candidate['f'][:] = .5e9
    monkeypatch.setattr('mlbac_stage1.safety_filter.search_candidates', lambda *args: fake_search(candidate))
    r = SafetyFilter(p).filter_action(s, a)
    assert r['status'] == 'corrected_candidate_verified'
    assert r['report']['margin_checks']['margin_battery_pass'].all()
    for key in ACTION_FIELDS:
        np.testing.assert_array_equal(r['action'][key], candidate[key])


def test_enabled_zero_margin_is_strict_at_boundary():
    p, s, a = case(safety_margin_enabled=True)
    s['B'][:] = p['B_min'] + 1.
    assert not evaluate_constraints(s, a, p)['feasible']
    assert evaluate_constraints(s, a, dict(p, safety_margin_enabled=False))['feasible']


def test_runner_logs_effective_margin_config(tmp_path):
    from experiments_stage1 import run_episode
    run_episode(tmp_path, 'reference', 0, devices=1, horizon=1, make_plots=False,
                safety_margin_enabled=True, temperature_margin_c=2., battery_margin_j=.2)
    config = json.loads((tmp_path / 'config_snapshot.json').read_text())
    assert config['controller_config']['safety_margin_settings']['effective_temperature_margin_c'] == [2.]
    steps = json.loads((tmp_path / 'steps.json').read_text())
    for phase in ('proposal_report', 'report'):
        assert steps[0]['filter_result'][phase]['safety_margins']['enabled']
        assert steps[0]['filter_result'][phase]['margin_checks']['predicted_battery_j']
