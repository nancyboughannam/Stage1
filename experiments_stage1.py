"""Batch C: finite-horizon scripted-policy diagnostics, not PPO benchmarks."""
import argparse
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlbac_stage1.config import make_default_params
from mlbac_stage1.env import make_env
from mlbac_stage1.experiment_logging import write_json, write_csv, slot_record, summarize, aggregate
from reference_demo import scripted_nominal_action

SCENARIOS = ('reference', 'resource_pressure', 'weak_channel', 'thermal_matched', 'thermal_mismatch', 'battery', 'invalid_proposal')


def build_env(scenario, devices, horizon, seed, load):
    p = make_default_params(devices, horizon=horizon)
    # Fixed admission/action range across the supported load sweep (0 < load <= 2).
    p['A_max'] = np.full(devices, 4.)
    p['arrival_max_per_slot'] = 2. * load
    if scenario == 'weak_channel':
        p['channel_gain_min'], p['channel_gain_max'] = 5e-8, 2e-7
    if scenario.startswith('thermal_'):
        p['thermal_alpha'] = 60.
        p['initial_temperature_c'] = 79.
        if scenario == 'thermal_mismatch':
            p['actual_thermal_alpha'] = 90.
    env = make_env(params=p)
    options = None
    if scenario == 'battery':
        options = {'state': {'B': np.asarray(p['B_min']) + .5}}
    _, reset = env.reset(seed=seed, options=options)
    return env, reset


def policy(env, scenario):
    raw = scripted_nominal_action(env).astype(float).reshape(env.n_devices, 7)
    if scenario == 'resource_pressure':
        # Each device requests 80% of the shared bandwidth and edge CPU.
        raw[:, 2:4] = .6
    elif scenario.startswith('thermal_'):
        raw[:, 0] = .8  # 90% local frequency, intentionally thermal-stressing.
    elif scenario == 'invalid_proposal':
        raw[0, 0] = np.nan
    return raw.reshape(-1)


def plots(output, records, rows, config, initial):
    observed = [r['info'] for r in records if r['info']['post_state'] is not None]
    times = [0.] + [i['elapsed_time_s'] for i in observed]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key, limit, label in zip(axes, ('Q','B','T'), ('Q_max','B_min','T_max'), ('Queue (work units)','Battery (J)','Temperature (C)')):
        states = np.asarray([initial[key]] + [i['post_state'][key] for i in observed])
        for device in range(config['n_devices']):
            ax.plot(times, states[:,device], '.-', label=f'device {device}')
            ax.axhline(config[limit][device], color='black', ls='--', alpha=.3)
            if observed:
                predicted = [i['predicted_state'][key][device] for i in observed]
                ax.plot(times[1:], predicted, 'x', color='black', alpha=.6,
                        label='predicted endpoint' if device == 0 else None)
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=3)
        ax.set_xlim(left=0, right=max(1.,times[-1]))
    axes[-1].set_xlabel('Simulated time (s)')
    fig.suptitle('Simulated states and predictions | ' + str(rows[-1]['episode_ending_reason']))
    fig.tight_layout(); fig.savefig(output/'states.png',dpi=140); plt.close(fig)
    fig, axes = plt.subplots(4,1,figsize=(10,11),sharex=True)
    x = [r['slot'] for r in rows]
    for key in ('offered_work','admitted_work','completed_work','rejected_work'):
        axes[0].plot(x,[r[key] if r[key] is not None else np.nan for r in rows],'.-',label=key)
    axes[0].set_ylabel('Work / attempted slot'); axes[0].legend(fontsize=8)
    for phase in ('decoded_nominal_action','executed_action'):
        y = [np.sum(r['info'][phase]['U']) if r['info'][phase] and 'U' in r['info'][phase] else np.nan for r in records]
        axes[1].plot(x,y,'.-',label=phase+' admission')
    axes[1].set_ylabel('Admission (work)'); axes[1].legend(fontsize=8)
    axes[2].plot(x,[r['filter_elapsed_ms'] for r in rows],'.-',label='filter runtime')
    axes[2].set_ylabel('Wall-clock ms')
    energy = np.cumsum([r['energy_j'] if r['energy_j'] is not None else np.nan for r in rows])
    axes[3].plot(x,energy,'.-',label='cumulative device energy')
    axes[3].set_ylabel('Energy consumed (J)'); axes[3].set_xlabel('Attempted slot index')
    for ax in axes:
        ax.grid(alpha=.2); ax.set_xlim(0,max(1,x[-1])); ax.set_ylim(bottom=0.)
    if not any(r['executed'] for r in rows):
        axes[3].text(.5,.5,'No slot executed; no energy outcome observed',
                     transform=axes[3].transAxes,ha='center')
    fig.suptitle('Service and filter cost | no trained PPO; missing execution stays blank')
    fig.tight_layout(); fig.savefig(output/'service_and_filter.png',dpi=140); plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for ax, key, scale, unit in zip(axes.flat, ('f','p','b','f_edge','l','o','U'),
                                    (1e9,1.,1e6,1e9,1.,1.,1.),
                                    ('GHz','W','MHz','GHz','work','work','work')):
        for device in range(config['n_devices']):
            color = f'C{device % 10}'
            for phase, style, label in (('decoded_nominal_action','--','proposal'), ('executed_action','-','executed')):
                y = [r['info'][phase][key][device]/scale
                     if r['info'][phase] and key in r['info'][phase] else np.nan for r in records]
                ax.plot(x,y,style,marker='.',color=color,label=f'd{device} {label}')
        ax.set_ylabel(f'{key} ({unit})'); ax.grid(alpha=.2)
        ax.set_xlim(0,max(1,x[-1])); ax.set_ylim(bottom=0.); ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.legend(fontsize=6,ncol=2)
    axes.flat[-1].set_visible(False)
    for ax in axes[-1]: ax.set_xlabel('Attempted slot index')
    fig.suptitle('Decoded proposal versus executed action | blanks mean no execution')
    fig.tight_layout(); fig.savefig(output/'actions.png',dpi=140); plt.close(fig)


def run_episode(output, scenario, seed, devices=5, horizon=30, load=1., max_rejection=.2, min_throughput=.5, make_plots=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    env, reset = build_env(scenario, devices, horizon, seed, load)
    initial = env._copy_state(env.state)
    records, rows = [], []
    try:
        for _ in range(horizon):
            _, _, terminated, truncated, info = env.step(policy(env, scenario))
            record = {'info': info, 'filter_result':env.last_filter_result}
            records.append(record)
            rows.append(slot_record(info, env.last_filter_result))
            if terminated or truncated: break
        summary = summarize(rows, records, env.params, initial, horizon, max_rejection, min_throughput)
        summary.update(scenario=scenario, seed=seed, load_multiplier=load,
                       policy='scripted_reference_with_labelled_scenario_overrides_not_PPO')
        write_json(output/'config_snapshot.json', {
            'controller_config':reset['effective_config'], 'initial_state':initial,
            'simulator_only_actual_thermal_alpha':env._actual_thermal_alpha,
            'seed':seed, 'scenario':scenario, 'load_multiplier':load,
            'service_criteria':summary['criteria'], 'python':platform.python_version(),
            'versions':{k:importlib.metadata.version(k) for k in ('numpy','scipy','gymnasium')},
            'notes':['Nonfinite numbers serialize as null; input validity is separately recorded.',
                     'Thermal stress coefficients are synthetic, not calibrated.',
                     'Weak channel is observed environmental variation, not hidden channel-model error.',
                     'Scripted admission cap remains 0.5 work/device/slot; low service can be caused by this cap.']})
        # Full per-device arrays, all residuals, raw/decoded/selected/executed actions and solver attempts.
        write_json(output/'steps.json', records)
        write_csv(output/'slots.csv', rows)
        write_json(output/'summary.json', summary)
        if make_plots: plots(output, records, rows, env.params, initial)
        return summary
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='experiments_batch_c')
    parser.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--devices',type=int,default=5)
    parser.add_argument('--horizon',type=int,default=30)
    parser.add_argument('--loads',nargs='+',type=float,default=[1.])
    parser.add_argument('--max-rejection',type=float,default=.2)
    parser.add_argument('--min-throughput',type=float,default=.5)
    args=parser.parse_args()
    if args.devices<1 or args.horizon<1 or any(not 0<x<=2 for x in args.loads):
        parser.error('devices/horizon must be positive; loads must be in (0,2].')
    if len(set(args.seeds))!=len(args.seeds) or any(s<0 for s in args.seeds):
        parser.error('seeds must be distinct nonnegative integers')
    if not 0<=args.max_rejection<=1 or not np.isfinite(args.min_throughput) or args.min_throughput<0:
        parser.error('invalid service criteria')
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    groups={}
    for scenario in args.scenarios:
        for load in args.loads:
            name=f'{scenario}/load_{load:g}'
            summaries=[]
            for seed in args.seeds:
                s=run_episode(root/name/f'seed_{seed}',scenario,seed,args.devices,args.horizon,load,args.max_rejection,args.min_throughput)
                summaries.append(s)
                print(f"{name} seed={seed}: {s['executed_slots']}/{args.horizon} slots; {s['ending_reason']}",flush=True)
            groups[name]=aggregate(summaries)
    write_json(root/'aggregate.json',groups)
    print(f'Outputs: {root}. These diagnostics do not establish a maximum safe load.')


if __name__=='__main__': main()
