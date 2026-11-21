#!/usr/bin/env python3
"""Environment-grouped validation of Q1 trajectory uncertainty calibration."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from shuwei_c.config import load_config
from run_q1_formal_validation import neural_prediction

def conformal_quantile(values,coverage):
    values=np.sort(np.asarray(values,float)); n=len(values)
    if not n: raise ValueError('Empty calibration group')
    rank=min(n,int(np.ceil(coverage*(n+1))))
    return float(values[rank-1])

def load_split(cfg,split,npz_path):
    cols=['episode_uuid','target_displacement','target_ref_ok']
    rows=pq.read_table(cfg['source'],filters=[('split','=',split)],columns=cols).to_pylist()
    x=np.asarray(np.load(npz_path)['x'],np.float32)
    if len(rows)!=len(x): raise ValueError(f'{split} parquet/NPZ mismatch')
    pred,var,_=neural_prediction(cfg['model'],None,x)
    target=np.asarray([r['target_displacement'] for r in rows],np.float32)
    mask=np.asarray([r['target_ref_ok'] for r in rows],bool)
    error=np.linalg.norm(target-pred,axis=-1)
    base=np.linalg.norm(np.sqrt(np.maximum(var,1e-6)),axis=-1)
    return rows,mask,error,base

def summarize(error,base,mask,scales,horizons,label):
    out=[]
    for h,horizon in enumerate(horizons):
        valid=mask[:,h]; radius=base[valid,h]*scales[h]
        out.append({'method':label,'horizon_s':float(horizon),'n':int(valid.sum()),'scale':float(scales[h]),'coverage':float(np.mean(error[valid,h]<=radius)),'mean_radius_m':float(np.mean(radius))})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); c=load_config(a.experiment)
    horizons=np.asarray(c['horizons_s'],float); coverage=float(c['coverage'])
    with Path(c['official_records']).open(encoding='utf-8-sig',newline='') as f: env_map={r['episode_uuid']:r['environment_id'] for r in csv.DictReader(f)}
    vr,vm,ve,vb=load_split(c,'validation',c['validation_npz']); venv=np.asarray([env_map[r['episode_uuid']] for r in vr]); environments=np.unique(venv)
    crossfit=[]
    for held in environments:
        train=venv!=held; test=venv==held
        pooled=[]; robust=[]
        for h in range(len(horizons)):
            pooled.append(conformal_quantile((ve[:,h]/np.maximum(vb[:,h],1e-8))[train&vm[:,h]],coverage))
            env_scales=[conformal_quantile((ve[:,h]/np.maximum(vb[:,h],1e-8))[(venv==env)&vm[:,h]],coverage) for env in environments if env!=held]
            robust.append(max(env_scales))
        for method,scales in [('pooled',pooled),('worst_environment',robust)]:
            for row in summarize(ve,vb,test[:,None]&vm,np.asarray(scales),horizons,method): row['held_environment']=held; crossfit.append(row)
    full_env_scales=[]; pooled_scales=[]
    score=ve/np.maximum(vb,1e-8)
    for h in range(len(horizons)):
        pooled_scales.append(conformal_quantile(score[vm[:,h],h],coverage))
        full_env_scales.append(max(conformal_quantile(score[(venv==env)&vm[:,h],h],coverage) for env in environments))
    tr,tm,te,tb=load_split(c,'test',c['test_npz'])
    test_results=summarize(te,tb,tm,np.asarray(pooled_scales),horizons,'pooled_validation')+summarize(te,tb,tm,np.asarray(full_env_scales),horizons,'worst_validation_environment')
    report={'experiment_id':c['experiment_id'],'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'target_coverage':coverage,'validation_environments':len(environments),'crossfit_validation':crossfit,'frozen_scales':{'pooled_validation':pooled_scales,'worst_validation_environment':full_env_scales},'test_sensitivity_analysis':test_results,'interpretation_rule':'Method comparison is based on leave-one-environment-out validation. Test is reported only as a sensitivity analysis and is not used to fit a scale.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
