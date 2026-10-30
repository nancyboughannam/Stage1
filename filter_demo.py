"""Controlled Batch B filter checks; no PPO and no physical execution."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from mlbac_stage1.config import make_default_params
from mlbac_stage1.models import ACTION_FIELDS
from mlbac_stage1.safety_filter import SafetyFilter


def json_ready(value):
    if isinstance(value, dict): return {k:json_ready(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray): return json_ready(value.tolist())
    if isinstance(value, np.generic): return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def examples():
    for name in ('normal','bandwidth','edge_cpu','joint_timing','thermal','battery','invalid_proposal'):
        n=2 if name in ('bandwidth','edge_cpu','joint_timing') else 1
        p=make_default_params(n)
        s={k:np.full(n,v) for k,v in dict(Q=1.,B=25920.,T=30.,A=1.,g=1e-5).items()}
        a={k:np.zeros(n) for k in ACTION_FIELDS}
        if name=='bandwidth': a['b'][:]=8e6
        elif name=='edge_cpu': a['f_edge'][:]=8e9
        elif name=='joint_timing':
            s['Q'][:]=10.; a['b'][:]=8e6; a['p'][:]=1.; a['f_edge'][:]=1e9; a['o'][:]=4.5
        elif name=='thermal':
            p['thermal_alpha']=60.; s['T'][:]=79.; a['f'][:]=.9e9; a['l'][:]=.05
        elif name=='battery': s['B'][:]=p['B_min']+.5
        elif name=='invalid_proposal': a['f'][:]=np.nan
        yield name,p,s,a


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,params,state,action in examples():
        result=SafetyFilter(params).filter_action(state,action)
        payload={'scenario':name,'execution_performed':False,'config':params,
                 'state':state,'nominal_action':action,'filter_result':result,
                 'serialization_note':'Nonfinite numbers serialize as null; input validity and verification statuses remain explicit.'}
        (output/f'{name}.json').write_text(json.dumps(json_ready(payload),indent=2,allow_nan=False)+'\n')
        selected=result['action']
        row={'scenario':name,'filter_status':result['status'],'proposal_status':result['proposal_status'],
             'solver_status':result['solver_status'],'selected_solver_status':result.get('selected_solver_status'),
             'verified':result['safe'],'objective_value':result['objective_value'],
             'solver_elapsed_ms':1000*result['solver_elapsed_s'],
             'filter_elapsed_ms':1000*result['filter_elapsed_s'],
             'fallback_reason':result['fallback_reason'],
             'selected_bandwidth_mhz':None if selected is None else float(sum(selected['b'])/1e6),
             'selected_edge_cpu_ghz':None if selected is None else float(sum(selected['f_edge'])/1e9)}
        rows.append(row)
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines=['Batch B: controlled filter checks (no physical execution; no trained PPO).']
    lines += [f"{r['scenario']}: {r['filter_status']} | solver={r['solver_status']} | verified={r['verified']}" for r in rows]
    lines += ['Inspect each scenario JSON for proposed/selected actions, constraint reports, and solver attempts.',
              'Runtime is measured on this machine; a feasible correction is not a global-optimality certificate.']
    (output/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines)); print(f'Outputs: {output}')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='filter_demo')
    run(parser.parse_args().output)
