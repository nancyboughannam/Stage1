"""Local nonlinear candidate search. Feasibility is decided by constraints.py."""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import minimize

from .config import action_metric_settings, safety_margin_thresholds
from .models import ACTION_FIELDS, compute_rates, predict_next_temperature


class SearchTimeout(RuntimeError):
    pass


class ProjectionProblem:
    """Normalized physical actions plus auxiliary transmission/edge time fractions.

    Auxiliary times avoid division by zero in nonlinear solver constraints.
    They are not executed controls and do not enter the distance objective.
    """

    def __init__(self, state, nominal, params):
        self.state, self.params = state, params
        self.n = len(params['Q_max'])
        self.margins, self.temperature_threshold, self.battery_threshold = safety_margin_thresholds(params)
        metric = action_metric_settings(params)
        self.scales = np.column_stack([metric['scales'][k] for k in ACTION_FIELDS])
        self.weights = np.column_stack([metric['weights'][k] for k in ACTION_FIELDS])
        self.active = self.scales.ravel() > 0
        self.reference = np.column_stack([nominal[k] for k in ACTION_FIELDS])
        self.reference = np.divide(self.reference, self.scales, out=np.zeros_like(self.scales), where=self.scales > 0).ravel()[self.active]
        self.m = int(self.active.sum())
        self.bounds = [(0., 1.)] * (self.m + 2 * self.n)

    def unpack(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.m + 2*self.n,) or not np.isfinite(x).all():
            raise ValueError('Invalid solver vector')
        normalized = np.zeros(self.n * 7)
        normalized[self.active] = x[:self.m]
        physical = normalized.reshape(self.n, 7) * self.scales
        return {k: physical[:, j].copy() for j, k in enumerate(ACTION_FIELDS)}

    def objective(self, x):
        diff = x[:self.m] - self.reference
        return float(.5 * np.sum(self.weights.ravel()[self.active] * diff**2))

    def gradient(self, x):
        return np.concatenate((self.weights.ravel()[self.active] * (x[:self.m]-self.reference), np.zeros(2*self.n)))

    def residuals(self, x):
        a = self.unpack(x)
        p, s = self.params, self.state
        dt = float(p['delta'])
        tx = x[self.m:self.m+self.n] * dt
        edge = x[self.m+self.n:] * dt
        f, power, b, fe, l, o, u = (a[k] for k in ACTION_FIELDS)
        c, d = np.asarray(p['c']), np.asarray(p['d'])
        qmax = np.asarray(p['Q_max'])
        qscale = np.maximum(qmax, 1.)
        rate = compute_rates(s['g'], b, power, float(p['N0']))
        # Auxiliary tx duration bounds real tx duration from above.
        energy = np.asarray(p['kappa']) * c * l * f**2 + power*tx + np.asarray(p['baseline_power'])*dt
        budget = np.asarray(s['B']) - np.asarray(p['B_min'])
        battery_target = self.battery_threshold if self.margins['enabled'] else np.asarray(p['B_min'])
        temperature_target = self.temperature_threshold if self.margins['enabled'] else np.asarray(p['T_max'])
        next_q = np.asarray(s['Q']) - l - o + u
        next_t = predict_next_temperature(s['T'], a, p)
        bw, cpu = float(p['shared_bandwidth_hz']), float(p['shared_edge_cpu_hz'])
        # Tiny inward target for allocation sums, not a physical model-error margin.
        guard = float(p['solver_allocation_guard'])
        residuals = [
            np.atleast_1d((bw*(1-guard)-sum(b))/max(bw, 1.)),
            np.atleast_1d((cpu*(1-guard)-sum(fe))/max(cpu, 1.)),
            (np.asarray(s['Q'])-l-o)/qscale,
            (np.asarray(s['A'])-u)/np.maximum(np.asarray(p['A_max']),1.),
            next_q/qscale, (qmax-next_q)/qscale,
            (f*dt-c*l)/np.maximum(np.asarray(p['f_hw_max'])*dt, 1.),
            (rate*tx-d*o)/np.maximum(d*qscale, 1.),
            (fe*edge-c*o)/np.maximum(c*qscale, 1.),
            (dt-tx-edge)/dt,
            (np.asarray(s['B'])-energy-battery_target)/np.maximum(np.abs(budget),1.)
                if self.margins['enabled'] else (budget-energy)/np.maximum(np.abs(budget),1.),
            (temperature_target-next_t)/100.,
        ]
        if self.margins['enabled']:
            residuals.append((np.asarray(p['T_max'])-np.asarray(s['T']))/100.)
        return np.concatenate(residuals)

    def starts(self):
        x = np.concatenate((np.clip(self.reference, 0., 1.), np.full(2*self.n, .5)))
        # Keep physical reference untouched; bounded start is only a solver seed.
        yield x
        a = self.unpack(x)
        for k in ('l', 'o', 'U'):
            a[k][:] = 0.
        for k in ('f', 'p', 'b', 'f_edge'):
            j = ACTION_FIELDS.index(k)
            a[k] = .05 * self.scales[:,j] / (self.n if k in ('b','f_edge') else 1)
        matrix = np.column_stack([a[k] for k in ACTION_FIELDS])
        scaled = np.divide(matrix, self.scales, out=np.zeros_like(matrix), where=self.scales > 0)
        yield np.concatenate((scaled.ravel()[self.active], np.full(2*self.n,.25)))


def search_candidates(state, nominal, params):
    """Return local solve candidates and diagnostics; never marks an action safe.

    Time budget is cooperative, checked at Python evaluations and callbacks.
    It cannot interrupt a currently running native solver operation.
    """
    problem = ProjectionProblem(state, nominal, params)
    start = perf_counter()
    deadline = start + float(params['solver_time_budget_s'])
    candidates, attempts = [], []

    def check_time():
        if perf_counter() >= deadline:
            raise SearchTimeout('Cooperative search time budget exhausted')

    def objective(x):
        check_time()
        return problem.objective(x)

    def residuals(x):
        check_time()
        return problem.residuals(x)

    for x0 in list(problem.starts())[:params['solver_max_starts']]:
        try:
            check_time()
            result = minimize(objective, x0, jac=problem.gradient, method='SLSQP',
                bounds=problem.bounds, constraints=[{'type':'ineq','fun':residuals}],
                callback=lambda x: check_time(),
                options={'ftol':params['solver_ftol'], 'maxiter':params['solver_maxiter']})
            check_time()
            status = 'converged' if result.success else ('iteration_limit' if result.status == 9 else 'failed')
            attempt = {'status':status, 'message':str(result.message), 'iterations':int(result.nit)}
            attempts.append(attempt)
            # No clipping or rounding after solving. Final checker owns acceptance.
            try:
                candidate = problem.unpack(result.x)
                candidates.append({'action':candidate, 'solver_status':status,
                                   'solver_objective':problem.objective(result.x)})
            except (ValueError, TypeError):
                attempt['status'] = 'invalid_candidate'
        except SearchTimeout as exc:
            attempts.append({'status':'timeout','message':str(exc),'iterations':None})
            break
        except Exception as exc:
            attempts.append({'status':'exception','message':f'{type(exc).__name__}: {exc}','iterations':None})
            break
    return {'candidates':candidates, 'attempts':attempts,
            'elapsed_s':perf_counter()-start,
            'status': attempts[-1]['status'] if attempts else 'not_run'}
