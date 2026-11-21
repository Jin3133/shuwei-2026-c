#!/usr/bin/env python3
"""Formal Q1 validation with physical-unit baselines and frozen neural models."""
from __future__ import annotations
import argparse,json,math,subprocess,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from shuwei_c.config import load_config
from shuwei_c.models import build_model

def kalman_forecast(position,velocity,quality,timestamps,horizons,q_var,r_var):
    out=np.empty((len(position),len(horizons),3),dtype=np.float32); eye3=np.eye(3); H=np.block([eye3,np.zeros((3,3))]); R=eye3*r_var
    for n in range(len(position)):
        state=np.r_[position[n,0],velocity[n,0]].astype(float); P=np.eye(6)
        for i in range(1,position.shape[1]):
            dt=max(float(timestamps[n,i]-timestamps[n,i-1]),1e-6); F=np.block([[eye3,dt*eye3],[np.zeros((3,3)),eye3]])
            state=F@state; P=F@P@F.T+np.eye(6)*q_var
            if quality[n,i]>0:
                residual=position[n,i]-H@state; S=H@P@H.T+R; K=P@H.T@np.linalg.inv(S); state=state+K@residual; P=(np.eye(6)-K@H)@P
        anchor=position[n,-1]
        for j,h in enumerate(horizons): out[n,j]=state[:3]+float(h)*state[3:]-anchor
    return out

def neural_prediction(spec,data_cfg,x):
    cfg=load_config(Path(spec['config'])); ckpt=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False); model=build_model(cfg['model'],cfg['data']); model.load_state_dict(ckpt['model']); model.eval(); means=[]; variances=[]
    with torch.no_grad():
        for start in range(0,len(x),1024):
            y=model(torch.from_numpy(x[start:start+1024])); means.append(y.mean.numpy())
            if y.variance is not None: variances.append(y.variance.numpy())
    return np.concatenate(means), (np.concatenate(variances) if variances else None), ckpt['metrics']

def summarize(name,pred,target,mask,horizons,variance=None):
    distance=np.linalg.norm(pred-target,axis=-1); rows=[]
    for i,h in enumerate(horizons):
        valid=mask[:,i]; d=distance[valid,i]; row={'model':name,'horizon_s':h,'n':int(valid.sum()),'mean_error_m':float(d.mean()),'rmse_m':float(np.sqrt(np.mean(d*d))),'median_error_m':float(np.quantile(d,.5)),'p90_error_m':float(np.quantile(d,.9))}
        if variance is not None:
            var=np.maximum(variance[valid,i],1e-8); err=target[valid,i]-pred[valid,i]; sigma=np.sqrt(var); joint=np.all(np.abs(err)<=1.96*sigma,axis=-1)
            row.update(gaussian_nll=float(np.mean(.5*(np.log(2*np.pi*var)+err*err/var))),raw_joint_95_coverage=float(joint.mean()),mean_axis_std_m=float(sigma.mean()),uncertainty_error_correlation=float(np.corrcoef(sigma.mean(axis=1),d)[0,1]))
        rows.append(row)
    return rows,distance

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); c=load_config(a.experiment); horizons=np.asarray(c['horizons_s'],float)
    cols=['episode_uuid','dataset','hand','input_timestamps_s','input_relative_position','input_velocity','input_quality','target_displacement','target_ref_ok']
    rows=pq.read_table(c['source'],filters=[('split','=',c['split'])],columns=cols).to_pylist(); npz=np.load(c['validation_npz']); x=np.asarray(npz['x'],np.float32)
    position=np.asarray([r['input_relative_position'] for r in rows],np.float32); velocity=np.asarray([r['input_velocity'] for r in rows],np.float32); quality=np.asarray([r['input_quality'] for r in rows],np.float32)[...,0]; timestamps=np.asarray([r['input_timestamps_s'] for r in rows],float)
    target=np.asarray([r['target_displacement'] for r in rows],np.float32); mask=np.asarray([r['target_ref_ok'] for r in rows],bool)
    if len(rows)!=len(x): raise ValueError('Parquet and NPZ validation order/length mismatch')
    predictions={'last_position':np.zeros_like(target),'constant_velocity':velocity[:,-1,None,:]*horizons[None,:,None]}
    predictions['kalman']=kalman_forecast(position,velocity,quality,timestamps,horizons,float(c['kalman']['process_variance']),float(c['kalman']['measurement_variance']))
    variance={}; checkpoints={}
    for name in ('mse_gru','gaussian_gru'):
        predictions[name],variance[name],checkpoints[name]=neural_prediction(c['models'][name],None,x)
    per_horizon=[]; distances={}
    for name,pred in predictions.items():
        rows_summary,dist=summarize(name,pred,target,mask,horizons,variance.get(name)); per_horizon+=rows_summary; distances[name]=dist
    strata=[]
    datasets=sorted({r['dataset'] for r in rows}); hands=sorted({r['hand'] for r in rows})
    for field,levels in [('dataset',datasets),('hand',hands)]:
        for level in levels:
            ids=np.asarray([r[field]==level for r in rows])
            for name,d in distances.items():
                strata.append({'stratum':field,'level':level,'model':name,'n_windows':int(ids.sum()),'ade_m':float(d[ids].mean()),'fde_1s_m':float(d[ids,-1].mean())})
    episodes=np.asarray([r['episode_uuid'] for r in rows]); unique=np.unique(episodes); rng=np.random.default_rng(c['bootstrap']['seed']); paired=[]
    for challenger in ('last_position','constant_velocity','kalman','mse_gru'):
        per_episode=[]
        for episode in unique:
            ids=episodes==episode; per_episode.append(float(distances['gaussian_gru'][ids].mean()-distances[challenger][ids].mean()))
        delta=np.asarray(per_episode); boot=delta[rng.integers(0,len(delta),(c['bootstrap']['replicates'],len(delta)))].mean(axis=1)
        paired.append({'final_minus':challenger,'metric':'ade_m','episodes':len(unique),'mean_delta_m':float(delta.mean()),'episode_bootstrap_95_ci_m':[float(v) for v in np.quantile(boot,[.025,.975])],'episode_win_rate':float(np.mean(delta<0))})
    report={'experiment_id':c['experiment_id'],'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'split':c['split'],'windows':len(rows),'episodes':len(unique),'quality_note':'All delivered Q1 validation input-quality flags equal 1; quality stratification is not identifiable in Q1 windows.','checkpoint_metrics':checkpoints,'per_horizon':per_horizon,'stratified':strata,'paired_effects':paired}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
