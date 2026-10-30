from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from mlbac_stage1.config import make_default_params
from mlbac_stage1.constraints import evaluate_constraints
from mlbac_stage1.env import make_env
from mlbac_stage1.models import ACTION_FIELDS
from mlbac_stage1.optimization import ProjectionProblem, search_candidates
from mlbac_stage1.safety_filter import SafetyFilter


def case(n=1):
    p = make_default_params(n)
    s = {k:np.full(n,v) for k,v in dict(Q=1.,B=25920.,T=30.,A=1.,g=1e-5).items()}
    a = {k:np.zeros(n) for k in ACTION_FIELDS}
    return p,s,a


def fake_search(candidate=None, status='failed'):
    return {'status':status, 'elapsed_s':0., 'attempts':[{'status':status}],
            'candidates':[] if candidate is None else [{'action':candidate,'solver_status':status}]}


@pytest.mark.parametrize('n',[1,5])
def test_pass_through(n):
    p,s,a = case(n); before=deepcopy(a)
    result = SafetyFilter(p).filter_action(s,a)
    assert result['status']=='nominal_accepted'
    assert result['solver_status']=='not_run' and result['objective_value']==0
    for k in a:
        np.testing.assert_array_equal(result['action'][k],before[k])
        np.testing.assert_array_equal(a[k],before[k])


@pytest.mark.parametrize('key,total',[('b',10e6),('f_edge',10e9)])
def test_known_joint_projection(key,total):
    p,s,a=case(2); a[key][:]=.8*total
    result=SafetyFilter(p).filter_action(s,a)
    assert result['status']=='corrected_candidate_verified'
    np.testing.assert_allclose(result['action'][key],.5*total,rtol=1e-7)
    assert result['objective_value']==pytest.approx(.09,abs=1e-7)
    assert not result['global_optimality_proven']
    assert evaluate_constraints(s,result['action'],p)['feasible']


def test_weighted_projection():
    p,s,a=case(2); a['b'][:]=[9e6,7e6]
    r=SafetyFilter(p).filter_action(s,a)
    np.testing.assert_allclose(r['action']['b'],[6e6,4e6],rtol=1e-6)


def test_joint_bandwidth_timing_correction():
    p,s,a=case(2); s['Q'][:]=10
    a['b'][:]=8e6; a['p'][:]=1.; a['f_edge'][:]=1e9; a['o'][:]=4.5
    only_bandwidth=deepcopy(a); only_bandwidth['b'][:]=5e6
    assert not evaluate_constraints(s,only_bandwidth,p)['feasible']
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='corrected_candidate_verified'
    assert evaluate_constraints(s,r['action'],p)['feasible']
    assert any(not np.allclose(r['action'][k],a[k]) for k in ('p','f_edge','o'))


@pytest.mark.parametrize('kind',['thermal','queue','battery','zero_resources'])
def test_correction_families(kind):
    p,s,a=case()
    if kind=='thermal':
        p['thermal_alpha']=60.; s['T'][:]=79.; a['f'][:]=.9e9; a['l'][:]=.05
    elif kind=='queue':
        s['Q'][:]=10.; a['U'][:]=1.
    elif kind=='battery':
        s['B'][:]=p['B_min']+1.01; a['f'][:]=1e9; a['l'][:]=1.
    else: a['o'][:]=.2
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='corrected_candidate_verified'
    assert evaluate_constraints(s,r['action'],p)['feasible']


@pytest.mark.parametrize('status',['failed','timeout','exception','iteration_limit'])
def test_failure_backup(monkeypatch,status):
    p,s,a=case(); a['U'][:]=2.
    monkeypatch.setattr('mlbac_stage1.safety_filter.search_candidates',lambda *args:fake_search(status=status))
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='fallback_verified' and r['solver_status']==status
    assert r['fallback_attempted'] and r['safe']


def test_solver_success_not_authorization(monkeypatch):
    p,s,a=case(); a['U'][:]=2.
    monkeypatch.setattr('mlbac_stage1.safety_filter.search_candidates',lambda *args:fake_search(a,'converged'))
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='fallback_verified' and not r['candidate_checks'][0]['verified']
    np.testing.assert_array_equal(r['action']['U'],0)


def test_nonconverged_feasible_result(monkeypatch):
    p,s,a=case(); valid=deepcopy(a); a['U'][:]=2.
    monkeypatch.setattr('mlbac_stage1.safety_filter.search_candidates',lambda *args:fake_search(valid,'iteration_limit'))
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='corrected_candidate_verified'
    assert r['selected_solver_status']=='iteration_limit'


def test_failure_without_backup(monkeypatch):
    p,s,a=case(); s['B'][:]=p['B_min']+.5
    monkeypatch.setattr('mlbac_stage1.safety_filter.search_candidates',lambda *args:fake_search())
    r=SafetyFilter(p).filter_action(s,a)
    assert r['action'] is None and not r['safe'] and not r['infeasibility_proven']
    assert r['status']=='no_verified_safe_continuation'


@pytest.mark.parametrize('bad',[np.nan,np.inf])
def test_invalid_proposal_backup(bad):
    p,s,a=case(); a['f'][:]=bad
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='fallback_verified' and r['proposal_status']=='invalid_proposal'
    assert r['solver_status']=='not_run' and r['objective_value'] is None


@pytest.mark.parametrize('field,value,status',[('B',-1,'invalid_state'),('T',81,'already_unsafe')])
def test_bad_state_no_action(field,value,status):
    p,s,a=case(); s[field][:]=value
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']==status and r['action'] is None
    assert not r['fallback_attempted']


def test_zero_arrival_range():
    p,s,a=case(); p['A_max'][:]=0.; s['A'][:]=0.; a['b'][:]=20e6
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='corrected_candidate_verified'
    assert r['action']['U'][0]==0 and np.isfinite(r['objective_value'])


def test_real_cooperative_timeout():
    p,s,a=case(); p['solver_time_budget_s']=1e-12; a['U'][:]=2.
    r=SafetyFilter(p).filter_action(s,a)
    assert r['solver_status']=='timeout' and r['status']=='fallback_verified'


def test_real_solver_exception(monkeypatch):
    p,s,a=case(); a['U'][:]=2.
    def fail(*args,**kwargs): raise RuntimeError('injected')
    monkeypatch.setattr('mlbac_stage1.optimization.minimize',fail)
    r=SafetyFilter(p).filter_action(s,a)
    assert r['solver_status']=='exception' and r['status']=='fallback_verified'


def test_auxiliary_zero_resources_do_not_create_service():
    p,s,a=case(); a['o'][:]=.2
    problem=ProjectionProblem(s,a,p)
    x=next(problem.starts())
    assert np.isfinite(problem.residuals(x)).all()
    assert problem.residuals(x).min()<0


@pytest.mark.parametrize('raw',[np.full(7,np.nan),np.full(7,np.inf),np.zeros(3)])
def test_env_invalid_raw_uses_verified_backup(raw):
    env=make_env(n_devices=1,horizon=2,deterministic=True); env.reset(seed=0)
    _,_,_,_,info=env.step(raw)
    assert info['proposal_status']=='invalid_proposal'
    assert info['filter_status']=='fallback_verified'
    assert info['actual_report']['feasible']


def test_env_already_unsafe_does_not_advance():
    env=make_env(n_devices=1,horizon=2); env.reset(seed=0,options={'state':{'T':[81.]}})
    _,_,done,_,info=env.step(np.full(7,-1.))
    assert done and info['filter_status']=='already_unsafe'
    assert info['executed_action'] is None and info['elapsed_time_s']==0


def test_env_gate_rejects_false_safe_flag(monkeypatch):
    env=make_env(n_devices=1,horizon=2); env.reset(seed=0)
    original=env.filter.filter_action
    def corrupt(state,action):
        r=original(state,action); r['action']['l'][:]=100.; return r
    monkeypatch.setattr(env.filter,'filter_action',corrupt)
    _,_,done,_,info=env.step(np.full(7,-1.))
    assert done and info['filter_status']=='final_verification_failed'
    assert info['executed_action'] is None and env.timestep==0


def test_env_executed_equals_verified():
    env=make_env(n_devices=2,horizon=2); env.reset(seed=0)
    _,_,_,_,info=env.step(np.full(14,.5))
    assert info['executed_action'] is not None
    for k in ACTION_FIELDS:
        np.testing.assert_array_equal(info['executed_action'][k],env.last_filter_result['action'][k])
    assert evaluate_constraints(info['pre_state'],info['executed_action'],env.params)['feasible']


def test_filter_demo_outputs(tmp_path):
    import json
    from filter_demo import run
    rows=run(tmp_path)
    statuses={r['scenario']:r['filter_status'] for r in rows}
    assert statuses['bandwidth']=='corrected_candidate_verified'
    assert statuses['joint_timing']=='corrected_candidate_verified'
    assert statuses['battery']=='no_verified_safe_continuation'
    assert statuses['invalid_proposal']=='fallback_verified'
    result=json.loads((tmp_path/'bandwidth.json').read_text())
    assert not result['execution_performed']
    np.testing.assert_allclose(result['filter_result']['action']['b'],[5e6,5e6],rtol=1e-7)


def test_invalid_solver_vector_is_not_executed(monkeypatch):
    p,s,a=case(); a['U'][:]=2.
    def invalid(*args,**kwargs):
        return SimpleNamespace(success=True,status=0,message='injected',nit=1,x=np.array([np.nan]))
    monkeypatch.setattr('mlbac_stage1.optimization.minimize',invalid)
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='fallback_verified'
    assert all(t['status']=='invalid_candidate' for t in r['solver_attempts'])


def test_actual_thermal_violation_not_hidden_after_projection():
    p,s,a=case(); p['thermal_alpha']=60.; s['T'][:]=79.; a['f'][:]=.9e9; a['l'][:]=.05
    from mlbac_stage1.models import transition_metrics
    from mlbac_stage1.constraints import evaluate_post_transition
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='corrected_candidate_verified'
    actual=transition_metrics(s,r['action'],dict(p,actual_thermal_alpha=100.))
    post=dict(Q=actual['queue_next'],B=actual['battery_next'],T=actual['temperature_next'])
    assert not evaluate_post_transition(post,actual,p)['feasible']


def test_five_device_joint_correction():
    p,s,a=case(5)
    a['b'][:]=4e6
    a['f_edge'][:]=4e9
    r=SafetyFilter(p).filter_action(s,a)
    assert r['status']=='corrected_candidate_verified'
    assert evaluate_constraints(s,r['action'],p)['feasible']
    np.testing.assert_allclose(r['action']['b'],2e6,rtol=1e-6)
    np.testing.assert_allclose(r['action']['f_edge'],2e9,rtol=1e-6)
