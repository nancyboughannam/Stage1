"""A verified action must not create a queue rejected at the next step."""
from copy import deepcopy
import numpy as np
import pytest
from mlbac_stage1.config import make_default_params
from mlbac_stage1.constraints import evaluate_constraints, evaluate_post_transition
from mlbac_stage1.models import transition_metrics, validate_state
from mlbac_stage1.env import make_env


def case():
    p=make_default_params(1)
    s={k:np.array([v]) for k,v in dict(Q=1.,B=25920.,T=30.,A=0.,g=1e-5).items()}
    a={k:np.array([v]) for k,v in dict(f=1e9,p=0.,b=0.,f_edge=0.,l=1.+5e-10,o=0.,U=0.).items()}
    return p,s,a


def test_tiny_negative_queue_is_not_authorized():
    p,s,a=case(); before=deepcopy(a)
    report=evaluate_constraints(s,a,p)
    assert not report['feasible']
    assert report['violations']['queue_lower']==[0]
    assert -p['checker_tolerances']['work'] < report['predicted_state']['Q'][0] < 0
    with pytest.raises(ValueError,match='physically executable'):
        transition_metrics(s,a,p)
    for k in a: np.testing.assert_array_equal(a[k],before[k])


def test_exactly_empty_queue_is_valid():
    p,s,a=case(); a['l'][:]=1.
    assert evaluate_constraints(s,a,p)['feasible']
    metrics=transition_metrics(s,a,p)
    s.update(Q=metrics['queue_next'],B=metrics['battery_next'],T=metrics['temperature_next'])
    assert s['Q'][0]==0.
    assert validate_state(s,p)['status']=='valid'


def test_actual_negative_queue_is_detected_even_below_tolerance():
    p,s,_=case(); s['Q'][:]=-5e-10
    assert not evaluate_post_transition(s,{'peak_temp':s['T']},p)['feasible']


def test_env_rejects_tiny_negative_solver_candidate_and_can_step_again(monkeypatch):
    p,s,a=case()
    env=make_env(params=p,horizon=2,deterministic=True)
    env.reset(seed=0,options={'state':s})
    def result(*args):
        return {'status':'converged','elapsed_s':0.,'attempts':[],
                'candidates':[{'action':a,'solver_status':'converged'}]}
    monkeypatch.setattr('mlbac_stage1.safety_filter.search_candidates',result)
    _,_,done,_,info=env.step(np.ones(7))
    assert not done and info['filter_status']=='fallback_verified'
    assert env.state['Q'][0]>=0
    _,_,done,_,_=env.step(np.full(7,-1.))
    assert done
