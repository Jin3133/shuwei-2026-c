#!/usr/bin/env python3
"""Cross-fitted horizon shrinkage for frozen Q1 GRU forecasts."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from shuwei_c.config import load_config
from run_q1_formal_validation import neural_prediction,summarize

def fit_alpha(pred,target,mask,horizons,apply_horizons,lo,hi):
    out=[]
    for h in range(pred.shape[1]):
        if float(horizons[h]) not in apply_horizons:
            out.append(1.0); continue
        valid=mask[:,h]; denominator=np.sum(pred[valid,h]**2); value=np.sum(pred[valid,h]*target[valid,h])/max(denominator,1e-12); out.append(float(np.clip(value,lo,hi)))
    return np.asarray(out,float)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); c=load_config(a.experiment); horizons=np.asarray(c['horizons_s'],float)
    meta=pq.read_table(c['source'],filters=[('split','=',c['split'])],columns=['episode_uuid']).to_pylist(); episodes=np.asarray([r['episode_uuid'] for r in meta])
    with Path(c['official_records']).open(encoding='utf-8-sig',newline='') as f: episode_env={r['episode_uuid']:r['environment_id'] for r in csv.DictReader(f)}
    environments=np.asarray([episode_env[e] for e in episodes]); unique=np.unique(environments); data=np.load(c['validation_npz']); x=np.asarray(data['x'],np.float32); target=np.asarray(data['y'],np.float32); mask=np.asarray(data['target_mask'],bool)
    predictions={}; variances={}
    for name,spec in c['models'].items(): predictions[name],variances[name],_=neural_prediction(spec,None,x)
    per_horizon=[]; coefficients={}; distances={}
    for name,pred in predictions.items():
        cross=np.zeros_like(pred); fold={}
        for env in unique:
            fit=environments!=env; held=environments==env; alpha=fit_alpha(pred[fit],target[fit],mask[fit],horizons,set(map(float,c['shrinkage']['apply_horizons_s'])),float(c['shrinkage']['lower']),float(c['shrinkage']['upper'])); cross[held]=pred[held]*alpha[None,:,None]; fold[env]=alpha.tolist()
        full=fit_alpha(pred,target,mask,horizons,set(map(float,c['shrinkage']['apply_horizons_s'])),float(c['shrinkage']['lower']),float(c['shrinkage']['upper'])); coefficients[name]={'full_validation_for_frozen_test':full.tolist(),'leave_one_environment_out':fold}
        for label,z,var in [(name,pred,variances[name]),(name+'_crossfit_shrink',cross,variances[name])]:
            summary,d=summarize(label,z,target,mask,horizons,var); per_horizon+=summary; distances[label]=d
    paired=[]; rng=np.random.default_rng(c['bootstrap']['seed'])
    for name in predictions:
        raw=distances[name]; shrunk=distances[name+'_crossfit_shrink']; delta=[]
        for env in unique:
            ids=environments==env; delta.append(float(shrunk[ids].mean()-raw[ids].mean()))
        delta=np.asarray(delta); boot=delta[rng.integers(0,len(delta),(c['bootstrap']['replicates'],len(delta)))].mean(axis=1)
        paired.append({'model':name,'metric':'ade_m','environments':len(unique),'mean_delta_m':float(delta.mean()),'environment_bootstrap_95_ci_m':[float(v) for v in np.quantile(boot,[.025,.975])],'environment_win_rate':float(np.mean(delta<0))})
    report={'experiment_id':c['experiment_id'],'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'split':c['split'],'windows':len(x),'environments':len(unique),'coefficients':coefficients,'per_horizon':per_horizon,'paired_effects':paired}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
