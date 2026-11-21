#!/usr/bin/env python3
"""Final paired Q3 ablation under one nominal replay protocol."""
from __future__ import annotations
import argparse,csv,json,random,subprocess,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from shuwei_c.config import load_config
from shuwei_c.scheduling.simulator import Job,ScenarioConfig,ScenarioSimulator
from run_q3_nominal_scheduling import aggregate,build_predictors,choose_action,read_csv
from run_q3_paid_review_sweep import build_random_forest_predictions

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--trials-csv',type=Path,required=True)
    a=ap.parse_args(); c=load_config(a.experiment); root=Path(c['task_root']); common=json.loads((root/'common_conditions.json').read_text()); rows=pq.read_table(c['features']).to_pylist()
    gcal=json.loads(Path(c['gbdt_calibration']).read_text()); gcfg={'prefix_s':c['prefix_s'],'train_split':c['train_split'],'evaluation_split':c['evaluation_split'],'quantiles':c['quantiles'],'model':c['gbdt_model'],'features':c['features_schema']}
    base=build_predictors(rows,gcfg,float(gcal['corrections_s'][str(c['prefix_s'])]['0.8']))
    rcal=json.loads(Path(c['rf_calibration']).read_text()); corrections={str(r['prefix_s']):r['corrections_s'] for r in rcal['results'] if r['model']=='random_forest'}
    rcfg={'prefixes_s':c['prefixes_s'],'train_split':c['train_split'],'evaluation_split':c['evaluation_split'],'model':c['rf_model']}; rf=build_random_forest_predictions(rows,rcfg,corrections)
    predictions={}
    for unit_id,p in base.items():
        predictions[unit_id]={'family':(p['family_q50'],p['family_q80']),'gbdt':(p['gbdt_q50'],p['gbdt_q80']),'rf':(rf[unit_id][float(c['prefix_s'])]['q50'],rf[unit_id][float(c['prefix_s'])]['q80'])}
    tasks=read_csv(root/'tasks.csv'); units=read_csv(root/'source_segment_index.csv'); zones={r['zone_id']:(float(r['x']),float(r['y'])) for r in read_csv(root/'service_zones.csv')}; pools=defaultdict(list)
    for r in units:
        if r['split']==c['evaluation_split'] and r['eligible']=='1' and r['unit_id'] in predictions: pools[r['family_id']].append(r)
    cases={r['case_id']:r for r in common['cases']}; trials=[]; beta=float(c['scenario']['beta_s']); alpha=float(c['scenario']['alpha'])
    for case_id in sorted(cases):
        task_rows=[r for r in tasks if r['case_id']==case_id]; case=cases[case_id]; jobs=[Job(r['job_id'],float(r['release_s']),float(r['due_s']),float(r['priority_weight']),r['zone_id'],tuple(x.strip() for x in r['predecessors'].split(';') if x.strip())) for r in task_rows]
        scenario=ScenarioConfig(float(case['horizon_s']),float(case['cost_budget']),int(case['observation_limit']),zones,initial_zone=common['initial_zone'],review_delay_s=float(common['observation']['delay_s']),review_cost=float(common['observation']['cost']),max_reviews_per_job=int(common['observation']['max_per_job']),reserve_min_s=float(common['service']['reserve_min_s']),changeover_s=float(common['movement']['changeover_s']),service_fixed_cost=float(common['fees']['service_fixed']),reserve_cost_rate=float(common['fees']['reserve_per_second']),distance_cost_rate=float(common['fees']['distance_per_unit']),max_actions=int(common['max_actions']))
        for seed in common['validation_seeds']:
            rng=random.Random(int(seed)); sampled={r['job_id']:rng.choice(pools[r['family_id']]) for r in task_rows}; realized={j:alpha*float(r['duration_s'])+beta for j,r in sampled.items()}
            for v in c['variants']:
                estimates={}
                for jid,unit in sampled.items():
                    q50,q80=predictions[unit['unit_id']][v['model']]; q50*=alpha; q80*=alpha; f=float(v['reserve_fraction']); estimates[jid]={'q50':q50,'q80':q50+f*(q80-q50)}
                sim=ScenarioSimulator(jobs,realized,scenario)
                while not sim.state.stopped: sim.step(choose_action(sim,jobs,estimates,v['ordering'],beta,float(c['lookahead_factor'])))
                trials.append({'case_id':case_id,'seed':int(seed),'variant':v['name'],'model':v['model'],'ordering':v['ordering'],'reserve_fraction':v['reserve_fraction'],'ablation':v['ablation'],**sim.metrics(),'sampled_units':';'.join(sampled[j.job_id]['unit_id'] for j in jobs)})
    a.trials_csv.parent.mkdir(parents=True,exist_ok=True)
    with a.trials_csv.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=list(trials[0]),lineterminator='\n'); w.writeheader(); w.writerows(trials)
    summaries=[]
    for case_id in sorted(cases):
        for v in c['variants']: summaries.append({'case_id':case_id,'variant':v['name'],'model':v['model'],'ordering':v['ordering'],'reserve_fraction':v['reserve_fraction'],'ablation':v['ablation'],**aggregate([r for r in trials if r['case_id']==case_id and r['variant']==v['name']])})
    rng=np.random.default_rng(2026); paired=[]
    for case_id in sorted(cases):
        for challenger in ('family_atc_r025','gbdt_atc_r025'):
            final={r['seed']:r for r in trials if r['case_id']==case_id and r['variant']=='rf_atc_r025'}; other={r['seed']:r for r in trials if r['case_id']==case_id and r['variant']==challenger}
            for metric in ('weighted_completion_rate','weighted_on_time_rate','weighted_tardiness_s'):
                d=np.asarray([float(final[s][metric])-float(other[s][metric]) for s in sorted(final)]); boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
                better=d>0 if metric!='weighted_tardiness_s' else d<0
                paired.append({'case_id':case_id,'final_minus':challenger,'metric':metric,'mean_delta':float(d.mean()),'bootstrap_95_ci':[float(x) for x in np.quantile(boot,[.025,.975])],'seed_win_rate':float(np.mean(better))})
    report={'experiment_id':c['experiment_id'],'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'summaries':summaries,'paired_effects':paired}; a.output.write_text(json.dumps(report,indent=2)+'\n'); print(f'wrote {len(trials)} trials, {len(summaries)} summaries')
if __name__=='__main__': main()
