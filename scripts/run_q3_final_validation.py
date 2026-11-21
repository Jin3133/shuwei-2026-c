#!/usr/bin/env python3
"""Frozen Q3 validation over the official alpha-beta service grid."""
from __future__ import annotations
import argparse,csv,json,random,subprocess,sys
from collections import defaultdict
from itertools import product
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from shuwei_c.config import load_config
from shuwei_c.scheduling.simulator import Execute,Job,Review,ScenarioConfig,ScenarioSimulator
from run_q3_nominal_scheduling import aggregate,choose_action,read_csv
from run_q3_paid_review_sweep import build_predictions,can_review
from run_q3_value_of_information import learn_value

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--trials-csv',type=Path,required=True)
    a=ap.parse_args(); c=load_config(a.experiment); root=Path(c['task_root']); common=json.loads((root/'common_conditions.json').read_text()); rows=pq.read_table(c['features']).to_pylist()
    cal=json.loads(Path(c['calibration_results']).read_text()); corrections={str(r['prefix_s']):r['corrections_s'] for r in cal['results'] if r['model']=='random_forest'}
    predictions=build_predictions(rows,c,corrections); values,diagnostics=learn_value(rows,c,corrections)
    tasks=read_csv(root/'tasks.csv'); units=read_csv(root/'source_segment_index.csv'); zones={r['zone_id']:(float(r['x']),float(r['y'])) for r in read_csv(root/'service_zones.csv')}; pools=defaultdict(list)
    for r in units:
        if r['split']==c['evaluation_split'] and r['eligible']=='1' and r['unit_id'] in predictions: pools[r['family_id']].append(r)
    cases={r['case_id']:r for r in common['cases']}; conditions=[(float(x),float(y)) for x,y in product(c['service_grid']['alpha'],c['service_grid']['beta_s'])]; trials=[]
    for case_id in sorted(cases):
        task_rows=[r for r in tasks if r['case_id']==case_id]; case=cases[case_id]
        jobs=[Job(r['job_id'],float(r['release_s']),float(r['due_s']),float(r['priority_weight']),r['zone_id'],tuple(x.strip() for x in r['predecessors'].split(';') if x.strip())) for r in task_rows]
        scenario=ScenarioConfig(float(case['horizon_s']),float(case['cost_budget']),int(case['observation_limit']),zones,initial_zone=common['initial_zone'],review_delay_s=float(common['observation']['delay_s']),review_cost=float(common['observation']['cost']),max_reviews_per_job=int(common['observation']['max_per_job']),reserve_min_s=float(common['service']['reserve_min_s']),changeover_s=float(common['movement']['changeover_s']),service_fixed_cost=float(common['fees']['service_fixed']),reserve_cost_rate=float(common['fees']['reserve_per_second']),distance_cost_rate=float(common['fees']['distance_per_unit']),max_actions=int(common['max_actions']))
        for seed in common['validation_seeds']:
            rng=random.Random(int(seed)); sampled={r['job_id']:rng.choice(pools[r['family_id']]) for r in task_rows}
            for alpha,beta in conditions:
                realized={jid:alpha*float(unit['duration_s'])+beta for jid,unit in sampled.items()}
                for policy in ('no_review','final_case_adaptive_voi'):
                    sim=ScenarioSimulator(jobs,realized,scenario)
                    while not sim.state.stopped:
                        estimates={}
                        for jid,unit in sampled.items():
                            prefix=float(c['prefixes_s'][sim.state.jobs[jid].reviews]); pred=predictions[unit['unit_id']][prefix]; q50=alpha*pred['q50']; q80=alpha*pred['q80']; f=float(c['reservation_fraction'])
                            estimates[jid]={'q50':q50,'q80':q50+f*(q80-q50)}
                        action=choose_action(sim,jobs,estimates,c['ordering']['rule'],beta,float(c['ordering']['lookahead_factor']))
                        active=policy=='final_case_adaptive_voi' and case_id in c['final_policy']['active_cases']
                        if active and isinstance(action,Execute) and sim.state.jobs[action.job_id].reviews<int(c['final_policy']['max_reviews_per_job']) and can_review(sim,action.job_id):
                            prefix=float(c['prefixes_s'][sim.state.jobs[action.job_id].reviews]); gain=alpha*values[sampled[action.job_id]['unit_id']].get(prefix,-1e9)
                            job=next(j for j in jobs if j.job_id==action.job_id); finish=sim.state.now_s+estimates[action.job_id]['q50']+beta; slack=job.due_s-finish
                            decision_gain=gain*(1.0+job.weight/max(abs(slack),1.0))
                            if decision_gain>float(c['final_policy']['gain_threshold_s']): action=Review(action.job_id)
                        sim.step(action)
                    trials.append({'case_id':case_id,'seed':int(seed),'alpha':alpha,'beta_s':beta,'policy':policy,**sim.metrics(),'sampled_units':';'.join(sampled[j.job_id]['unit_id'] for j in jobs)})
    a.trials_csv.parent.mkdir(parents=True,exist_ok=True)
    with a.trials_csv.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=list(trials[0])); w.writeheader(); w.writerows(trials)
    summaries=[]
    for case_id in sorted(cases):
        for alpha,beta in conditions:
            for policy in ('no_review','final_case_adaptive_voi'):
                part=[r for r in trials if r['case_id']==case_id and r['alpha']==alpha and r['beta_s']==beta and r['policy']==policy]
                summaries.append({'case_id':case_id,'alpha':alpha,'beta_s':beta,'policy':policy,**aggregate(part)})
    nominal_alpha=float(c['nominal']['alpha']); nominal_beta=float(c['nominal']['beta_s']); nominal=[r for r in summaries if r['alpha']==nominal_alpha and r['beta_s']==nominal_beta]
    rng=np.random.default_rng(2026); paired_effects=[]
    for case_id in sorted(cases):
        base={r['seed']:r for r in trials if r['case_id']==case_id and r['alpha']==nominal_alpha and r['beta_s']==nominal_beta and r['policy']=='no_review'}
        active={r['seed']:r for r in trials if r['case_id']==case_id and r['alpha']==nominal_alpha and r['beta_s']==nominal_beta and r['policy']=='final_case_adaptive_voi'}
        for metric in ('weighted_completion_rate','weighted_on_time_rate','weighted_tardiness_s','cost','observations'):
            delta=np.asarray([float(active[s][metric])-float(base[s][metric]) for s in sorted(base)]); boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
            paired_effects.append({'case_id':case_id,'metric':metric,'mean_delta':float(delta.mean()),'bootstrap_95_ci':[float(x) for x in np.quantile(boot,[.025,.975])]})
    report={'experiment_id':c['experiment_id'],'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'frozen_policy':c['final_policy'],'value_diagnostics':diagnostics,'nominal_results':nominal,'nominal_paired_effects':paired_effects,'summaries':summaries}
    a.output.write_text(json.dumps(report,indent=2)+'\n'); print(f"wrote {len(trials)} trials, {len(summaries)} summaries")
if __name__=='__main__': main()
