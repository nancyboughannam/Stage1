"""Verification foundation tests; no optimization or PPO quality claims."""
import copy
import numpy as np
import pytest
from mlbac_stage1.config import make_default_params, action_metric_settings
from mlbac_stage1.constraints import evaluate_constraints, evaluate_post_transition
from mlbac_stage1.models import ACTION_FIELDS, transition_metrics, slot_quantities


def case(n=1):
    p = make_default_params(n)
    s = {k: np.full(n, v) for k, v in dict(Q=1., B=25920., T=30., A=1., g=1e-5).items()}
    a = {k: np.zeros(n) for k in ACTION_FIELDS}
    return p, s, a


@pytest.mark.parametrize('n', [1, 5])
def test_idle_prediction_and_transition_agree(n):
    p, s, a = case(n)
    before = copy.deepcopy(a)
    report = evaluate_constraints(s, a, p)
    actual = transition_metrics(s, a, p)
    assert report['feasible']
    for k, name in [('Q', 'queue_next'), ('B', 'battery_next'), ('T', 'temperature_next')]:
        np.testing.assert_allclose(report['predicted_state'][k], actual[name])
    np.testing.assert_array_equal(actual['total_energy'], np.ones(n))
    for k in a:
        np.testing.assert_array_equal(a[k], before[k])


@pytest.mark.parametrize('value', [np.array([np.nan]), np.array([np.inf]), np.zeros((1, 1)), np.zeros(2), 'bad'])
def test_invalid_proposals(value):
    p, s, a = case(); a['f'] = value
    report = evaluate_constraints(s, a, p)
    assert not report['feasible'] and report['input_status'] == 'invalid_proposal'
    with pytest.raises(ValueError):
        transition_metrics(s, a, p)


def test_missing_field():
    p, s, a = case(); del a['U']
    assert evaluate_constraints(s, a, p)['input_status'] == 'invalid_proposal'


@pytest.mark.parametrize('key,value,status', [('Q', -1., 'invalid_state'), ('B', 40000., 'invalid_state'),
    ('g', np.nan, 'invalid_state'), ('T', 81., 'already_unsafe'), ('B', 3000., 'already_unsafe'),
    ('Q', 11., 'already_unsafe'), ('T', -274., 'invalid_state')])
def test_state_classification(key, value, status):
    p, s, a = case(); s[key][:] = value
    report = evaluate_constraints(s, a, p)
    assert not report['feasible'] and report['input_status'] == status
    assert report['predicted_state'] is None


@pytest.mark.parametrize('kind', ['local', 'radio', 'edge'])
def test_positive_work_without_resources(kind):
    p, s, a = case()
    if kind == 'local':
        a['l'][:] = .1
    else:
        a['o'][:] = .1
        if kind == 'radio': a['f_edge'][:] = 1e9
        else:
            a['b'][:] = 1e6; a['p'][:] = .1
    with np.errstate(invalid='raise', divide='raise'):
        assert not evaluate_constraints(s, a, p)['feasible']
    with pytest.raises(ValueError): transition_metrics(s, a, p)


@pytest.mark.parametrize('kind', ['bandwidth', 'edge', 'negative_queue', 'overflow', 'reserve', 'thermal', 'admission'])
def test_constraints(kind):
    p, s, a = case(2)
    if kind == 'bandwidth': a['b'][:] = 8e6
    elif kind == 'edge': a['f_edge'][:] = 8e9
    elif kind == 'negative_queue': a['l'][:] = 2.; a['f'][:] = 1e9
    elif kind == 'overflow': s['Q'][:] = 10.; a['U'][:] = 1.
    elif kind == 'reserve': s['B'][:] = p['B_min']
    elif kind == 'thermal':
        s['T'][:] = 79.; p['thermal_alpha'] = 60.; a['f'][:] = 1e9
    else:
        s['A'][:] = 3.; a['U'][:] = 2.5
    report = evaluate_constraints(s, a, p)
    assert not report['feasible']
    if kind == 'negative_queue': assert report['violations']['queue_lower'] == [0, 1]


def test_energy_formula_and_actual_model_mismatch():
    p, s, a = case()
    a.update({k: np.array([v]) for k,v in dict(f=2e8,p=.1,b=1e6,f_edge=5e8,l=.5,o=.2,U=.4).items()})
    q = slot_quantities(s, a, p)
    np.testing.assert_allclose(q['local_energy'], p['kappa'] * a['f']**3 * q['local_time'])
    assert evaluate_constraints(s, a, p)['feasible']
    actual_p = dict(p, actual_thermal_alpha=1000.)
    actual = transition_metrics(s, a, actual_p)
    post = dict(Q=actual['queue_next'], B=actual['battery_next'], T=actual['temperature_next'])
    assert not evaluate_post_transition(post, actual, p)['feasible']


def test_tolerances_are_explicit_and_used():
    p, s, a = case(); a['b'][:] = 10e6 + .01
    assert not evaluate_constraints(s, a, p)['feasible']
    p['checker_tolerances']['bandwidth_hz'] = .02
    assert evaluate_constraints(s, a, p)['feasible']
    transition_metrics(s, a, p)
    p['checker_tolerances']['work'] = -1
    with pytest.raises(ValueError): evaluate_constraints(s, a, p)


def test_zero_range_scales_and_overrides():
    p, _, _ = case(5); p['A_max'][:] = 0; p['shared_bandwidth_hz'] = 2e6
    settings = action_metric_settings(p)
    assert not settings['active']['U'].any()
    np.testing.assert_array_equal(settings['scales']['b'], np.full(5, 2e6))


def test_nonfinite_configuration():
    with pytest.raises(ValueError): make_default_params(delta=np.nan)
