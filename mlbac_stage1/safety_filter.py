from __future__ import annotations

from copy import deepcopy
from time import perf_counter
import numpy as np

from .config import action_metric_settings
from .constraints import evaluate_constraints
from .fallback import FallbackController
from .models import ACTION_FIELDS, validate_arrays, validate_state
from .optimization import search_candidates


class SafetyFilter:
    """Local nonlinear correction with independent one-step model verification.

    Search failure is not an infeasibility certificate. Idle is not recovery.
    """

    def __init__(self, params):
        self.params = params
        defaults = {'solver_ftol':1e-10, 'solver_maxiter':200, 'solver_max_starts':2,
                    'solver_time_budget_s':2., 'solver_allocation_guard':1e-10}
        for key, value in defaults.items():
            self.params.setdefault(key, value)
        self.params.setdefault('A_max', np.ones(len(params['Q_max'])))
        self.params.setdefault('N0', 1e-12)
        for key in ('solver_ftol', 'solver_time_budget_s'):
            if not np.isfinite(params[key]) or params[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('solver_maxiter', 'solver_max_starts'):
            if not isinstance(params[key], (int,np.integer)) or params[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if params['solver_max_starts'] > 2:
            raise ValueError('At most two deterministic starts are implemented')
        if not np.isfinite(params['solver_allocation_guard']) or not 0 <= params['solver_allocation_guard'] < 1:
            raise ValueError('solver_allocation_guard must be in [0,1)')
        action_metric_settings(params)
        self.backup = FallbackController(params)
        self.last_result = None

    def distance(self, action, nominal):
        metric = action_metric_settings(self.params)
        value = 0.
        for key in ACTION_FIELDS:
            active = metric['active'][key]
            difference = (action[key][active]-np.asarray(nominal[key])[active])/metric['scales'][key][active]
            value += .5 * float(np.sum(metric['weights'][key][active]*difference**2))
        return value

    def filter_action(self, state, action):
        started = perf_counter()
        original = deepcopy(action)
        proposal_report = evaluate_constraints(state, action, self.params)
        state_status = validate_state(state, self.params)['status']
        proposal_valid = not validate_arrays(action, ACTION_FIELDS, len(self.params['Q_max']))
        proposal_status = ('invalid_proposal' if not proposal_valid else
                           'feasible' if proposal_report['feasible'] else 'constraint_violating')
        common = {'proposal_status':proposal_status, 'state_status':state_status,
                  'nominal_action':original, 'proposal_report':proposal_report,
                  'infeasibility_proven':False, 'verification_scope':'one_step',
                  'safety_claim':'one_step_predicted_zero_model_margin',
                  'solver_status':'not_run', 'solver_attempts':[], 'solver_elapsed_s':0.,
                  'fallback_reason':None, 'fallback_attempted':False,
                  'objective_value':None, 'correction_norm':None,
                  'global_optimality_proven':False, 'candidate_checks':[]}

        def finish(selected, report, status, **extra):
            result = {**common, **extra, 'action':selected, 'safe':selected is not None,
                      'status':status, 'report':report, 'filter_elapsed_s':perf_counter()-started}
            if selected is not None and proposal_valid:
                result['objective_value'] = self.distance(selected, action)
                result['correction_norm'] = float(np.sqrt(2*result['objective_value']))
            self.last_result = result
            return result

        if state_status != 'valid':
            return finish(None, proposal_report, state_status, failure_kind=state_status)
        if proposal_report['feasible']:
            selected = {k:np.asarray(action[k], dtype=float).copy() for k in ACTION_FIELDS}
            return finish(selected, proposal_report, 'nominal_accepted')

        if proposal_valid:
            try:
                search = search_candidates(state, action, self.params)
            except Exception as exc:
                search = {'candidates':[], 'status':'exception', 'elapsed_s':0.,
                          'attempts':[{'status':'exception','message':f'{type(exc).__name__}: {exc}'}]}
            common.update(solver_status=search['status'], solver_attempts=search['attempts'], solver_elapsed_s=search['elapsed_s'])
            verified = []
            for item in search['candidates']:
                candidate = item['action']
                report = evaluate_constraints(state, candidate, self.params)
                common['candidate_checks'].append({'solver_status':item['solver_status'],
                    'verified':report['feasible'], 'report':report})
                if report['feasible']:
                    verified.append((self.distance(candidate, action), candidate, report, item['solver_status']))
            if verified:
                _, selected, report, solver_status = min(verified, key=lambda item:item[0])
                return finish(selected, report, 'corrected_candidate_verified', selected_solver_status=solver_status)
            reason = 'no_verified_optimizer_candidate'
        else:
            reason = 'invalid_proposal'

        # One source for backup construction. Always checked against this state.
        idle = self.backup.action_for(state)
        report = evaluate_constraints(state, idle, self.params)
        common.update(fallback_attempted=True, fallback_reason=reason)
        if report['feasible']:
            return finish(idle, report, 'fallback_verified')
        return finish(None, report, 'no_verified_safe_continuation', failure_kind='no_verified_candidate_found')
