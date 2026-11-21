#!/usr/bin/env python3
"""Train an OOF value-of-information model and evaluate it in Q3 scheduling."""
from __future__ import annotations
import argparse,csv,json,random,subprocess,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from shuwei_c.config import load_config
from shuwei_c.scheduling.simulator import Execute,Job,Review,ScenarioConfig,ScenarioSimulator
from run_q3_nominal_scheduling import aggregate,choose_action,read_csv
from run_q3_paid_review_sweep import build_predictions,can_review

EXCLUDED={"prefix_id","segment_id","episode_uuid","dataset","split","family_id","target_duration_s","prefix_duration_s"}

def tree_q(model,x,q): return np.quantile(np.asarray([t.predict(x) for t in model.estimators_]),q,axis=0)

def learn_value(rows,cfg,corrections):
    with Path(cfg["official_records"]).open(encoding="utf-8-sig",newline="") as f: env={r["episode_uuid"]:r["environment_id"] for r in csv.DictReader(f)}
    numeric=[n for n in rows[0] if n not in EXCLUDED]; families=sorted({r["family_id"] for r in rows}); prefixes=list(map(float,cfg["prefixes_s"]))
    train_by={p:[r for r in rows if r["split"]==cfg["train_split"] and np.isclose(r["prefix_duration_s"],p)] for p in prefixes}
    valid_by={p:[r for r in rows if r["split"]==cfg["evaluation_split"] and np.isclose(r["prefix_duration_s"],p)] for p in prefixes}
    matrices={}; medians={}
    for p in prefixes:
        medians[p]={n:float(np.nanmedian([r[n] for r in train_by[p] if r[n] is not None])) for n in numeric}
        def mat(part,p=p): return np.asarray([[medians[p][n] if r[n] is None else r[n] for n in numeric]+[float(r["family_id"]==f) for f in families] for r in part],float)
        matrices[p]=(mat(train_by[p]),mat(valid_by[p]))
    oof={}; final={}
    for p in prefixes:
        tr=train_by[p]; x,z=matrices[p]; y=np.asarray([r["target_duration_s"] for r in tr]); groups=np.asarray([env[r["episode_uuid"]] for r in tr]); pred=np.zeros((len(tr),2))
        for fit,held in GroupKFold(cfg["n_group_folds"]).split(x,y,groups):
            m=RandomForestRegressor(**cfg["model"]).fit(x[fit],y[fit]); pred[held,0]=tree_q(m,x[held],.5); pred[held,1]=tree_q(m,x[held],.8)
        corr=np.asarray(corrections[str(p)]); oof[p]=np.maximum.accumulate(pred+corr,axis=1)
        m=RandomForestRegressor(**cfg["model"]).fit(x,y); final[p]=np.maximum.accumulate(np.stack([tree_q(m,z,.5),tree_q(m,z,.8)],axis=1)+corr,axis=1)
    scores=defaultdict(dict); diagnostics=[]
    for i,p in enumerate(prefixes[:-1]):
        nxt=prefixes[i+1]; tr=train_by[p]; va=valid_by[p]; truth=np.asarray([r["target_duration_s"] for r in tr])
        target=np.abs(truth-oof[p][:,0])-np.abs(truth-oof[nxt][:,0])
        x,z=matrices[p]; xmeta=np.column_stack([x,oof[p][:,0],oof[p][:,1]-oof[p][:,0]]); zmeta=np.column_stack([z,final[p][:,0],final[p][:,1]-final[p][:,0]])
        meta=RandomForestRegressor(**cfg["value_model"]).fit(xmeta,target); predicted=meta.predict(zmeta)
        valid_truth=np.asarray([r["target_duration_s"] for r in va]); actual=np.abs(valid_truth-final[p][:,0])-np.abs(valid_truth-final[nxt][:,0])
        corr=float(np.corrcoef(predicted,actual)[0,1]) if np.std(predicted)>0 and np.std(actual)>0 else 0.0
        diagnostics.append({"from_prefix_s":p,"to_prefix_s":nxt,"train_target_mean_s":float(target.mean()),"validation_actual_gain_mean_s":float(actual.mean()),
            "value_prediction_mae_s":float(np.mean(np.abs(actual-predicted))),"value_prediction_correlation":corr,"predicted_positive_rate":float(np.mean(predicted>0)),"actual_positive_rate":float(np.mean(actual>0))})
        for j,r in enumerate(va): scores[r["segment_id"]][p]=float(predicted[j])
    return scores,diagnostics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--trials-csv',type=Path,required=True)
    a=ap.parse_args(); c=load_config(a.experiment); root=Path(c['task_root']); common=json.loads((root/'common_conditions.json').read_text()); rows=pq.read_table(c['features']).to_pylist()
    cal=json.loads(Path(c['calibration_results']).read_text()); corrections={str(r['prefix_s']):r['corrections_s'] for r in cal['results'] if r['model']=='random_forest'}
    predictions=build_predictions(rows,c,corrections); values,diagnostics=learn_value(rows,c,corrections)
    tasks=read_csv(root/'tasks.csv'); units=read_csv(root/'source_segment_index.csv'); zones={r['zone_id']:(float(r['x']),float(r['y'])) for r in read_csv(root/'service_zones.csv')}; pools=defaultdict(list)
    for r in units:
        if r['split']==c['evaluation_split'] and r['eligible']=='1' and r['unit_id'] in predictions: pools[r['family_id']].append(r)
    cases={r['case_id']:r for r in common['cases']}; trials=[]
    for case_id in sorted(cases):
        tr=[r for r in tasks if r['case_id']==case_id]; case=cases[case_id]; jobs=[Job(r['job_id'],float(r['release_s']),float(r['due_s']),float(r['priority_weight']),r['zone_id'],tuple(x.strip() for x in r['predecessors'].split(';') if x.strip())) for r in tr]
        scenario=ScenarioConfig(float(case['horizon_s']),float(case['cost_budget']),int(case['observation_limit']),zones,initial_zone=common['initial_zone'],review_delay_s=float(common['observation']['delay_s']),review_cost=float(common['observation']['cost']),max_reviews_per_job=int(common['observation']['max_per_job']),reserve_min_s=float(common['service']['reserve_min_s']),changeover_s=float(common['movement']['changeover_s']),service_fixed_cost=float(common['fees']['service_fixed']),reserve_cost_rate=float(common['fees']['reserve_per_second']),distance_cost_rate=float(common['fees']['distance_per_unit']),max_actions=int(common['max_actions']))
        for seed in common['validation_seeds']:
            rng=random.Random(int(seed)); sampled={r['job_id']:rng.choice(pools[r['family_id']]) for r in tr}; realized={j:float(r['duration_s'])+float(c['scenario']['beta_s']) for j,r in sampled.items()}
            for policy in c['policies']:
                sim=ScenarioSimulator(jobs,realized,scenario)
                while not sim.state.stopped:
                    estimates={}
                    for jid,unit in sampled.items():
                        prefix=float(c['prefixes_s'][sim.state.jobs[jid].reviews]); pred=predictions[unit['unit_id']][prefix]; f=float(c['reservation_fraction']); estimates[jid]={'q50':pred['q50'],'q80':pred['q50']+f*(pred['q80']-pred['q50'])}
                    action=choose_action(sim,jobs,estimates,c['ordering']['rule'],float(c['scenario']['beta_s']),float(c['ordering']['lookahead_factor']))
                    policy_review_limit=int(policy.get('max_reviews_per_job',scenario.max_reviews_per_job))
                    case_active=case_id in policy.get('active_cases',[case_id])
                    if policy['mode']=='voi' and case_active and isinstance(action,Execute) and sim.state.jobs[action.job_id].reviews<policy_review_limit and can_review(sim,action.job_id):
                        rt=sim.state.jobs[action.job_id]; prefix=float(c['prefixes_s'][rt.reviews]); gain=values[sampled[action.job_id]['unit_id']].get(prefix,-1e9)
                        job=next(j for j in jobs if j.job_id==action.job_id); estimated_finish=sim.state.now_s+estimates[action.job_id]['q50']+float(c['scenario']['beta_s']); slack=job.due_s-estimated_finish
                        decision_gain=gain*(1.0+job.weight/max(abs(slack),1.0))
                        if decision_gain>float(policy['gain_threshold_s']): action=Review(action.job_id)
                    sim.step(action)
                trials.append({'case_id':case_id,'seed':int(seed),'policy':policy['name'],**sim.metrics(),'sampled_units':';'.join(sampled[j.job_id]['unit_id'] for j in jobs)})
    a.trials_csv.parent.mkdir(parents=True,exist_ok=True)
    with a.trials_csv.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=list(trials[0])); w.writeheader(); w.writerows(trials)
    summaries=[]
    for case_id in sorted(cases):
        for p in c['policies']: summaries.append({'case_id':case_id,'policy':p['name'],**aggregate([r for r in trials if r['case_id']==case_id and r['policy']==p['name']])})
    report={'experiment_id':c['experiment_id'],'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'value_diagnostics':diagnostics,'summaries':summaries}; a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
