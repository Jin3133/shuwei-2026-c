#!/usr/bin/env python3
"""One-time frozen Q1 test evaluation."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from shuwei_c.config import load_config
from run_q1_formal_validation import kalman_forecast,neural_prediction,summarize

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); c=load_config(a.experiment); horizons=np.asarray(c['horizons_s'],float)
    cols=['episode_uuid','dataset','hand','input_timestamps_s','input_relative_position','input_velocity','input_quality','target_displacement','target_ref_ok']; rows=pq.read_table(c['source'],filters=[('split','=',c['split'])],columns=cols).to_pylist(); data=np.load(c['test_npz']); x=np.asarray(data['x'],np.float32)
    position=np.asarray([r['input_relative_position'] for r in rows],np.float32); velocity=np.asarray([r['input_velocity'] for r in rows],np.float32); quality=np.asarray([r['input_quality'] for r in rows],np.float32)[...,0]; timestamps=np.asarray([r['input_timestamps_s'] for r in rows],float); target=np.asarray([r['target_displacement'] for r in rows],np.float32); mask=np.asarray([r['target_ref_ok'] for r in rows],bool)
    if len(rows)!=len(x): raise ValueError('Parquet and test NPZ mismatch')
    predictions={'last_position':np.zeros_like(target),'constant_velocity':velocity[:,-1,None,:]*horizons[None,:,None]}; predictions['kalman']=kalman_forecast(position,velocity,quality,timestamps,horizons,float(c['kalman']['process_variance']),float(c['kalman']['measurement_variance']))
    variances={}; checkpoints={}
    for name,spec in c['models'].items(): predictions[name],variances[name],checkpoints[name]=neural_prediction(spec,None,x)
    shrink=json.loads(Path(c['shrinkage_results']).read_text())['coefficients']
    for name in ('mse_gru','gaussian_gru'):
        alpha=np.asarray(shrink[name]['full_validation_for_frozen_test'],float); predictions[name+'_shrink']=predictions[name]*alpha[None,:,None]; variances[name+'_shrink']=variances[name]
    per_horizon=[]; distances={}
    for name,pred in predictions.items():
        summary,d=summarize(name,pred,target,mask,horizons,variances.get(name)); per_horizon+=summary; distances[name]=d
    calibration=json.loads(Path(c['conformal_report']).read_text())
    calibration_rows=calibration['per_horizon']; scales=np.asarray([r['conformal_scale'] for r in calibration_rows],float)
    target_coverage=float(calibration_rows[0]['conformal_target_coverage'])
    gaussian_radius=np.linalg.norm(np.sqrt(np.maximum(variances['gaussian_gru'],1e-6)),axis=-1)*scales[None,:]
    gaussian_error=distances['gaussian_gru']
    frozen_conformal=[]
    for i,horizon in enumerate(horizons):
        valid=mask[:,i]
        frozen_conformal.append({'horizon_s':float(horizon),'n':int(valid.sum()),'target_coverage':target_coverage,'validation_frozen_scale':float(scales[i]),'test_coverage':float(np.mean(gaussian_error[valid,i]<=gaussian_radius[valid,i])),'mean_radius_m':float(np.mean(gaussian_radius[valid,i]))})
    with Path(c['official_records']).open(encoding='utf-8-sig',newline='') as f: episode_env={r['episode_uuid']:r['environment_id'] for r in csv.DictReader(f)}
    environments=np.asarray([episode_env[r['episode_uuid']] for r in rows]); unique=np.unique(environments); strata=[]
    for field in ('dataset','hand'):
        levels=sorted({r[field] for r in rows})
        for level in levels:
            ids=np.asarray([r[field]==level for r in rows])
            for name,d in distances.items(): strata.append({'stratum':field,'level':level,'model':name,'n_windows':int(ids.sum()),'ade_m':float(d[ids].mean()),'fde_1s_m':float(d[ids,-1].mean())})
    rng=np.random.default_rng(c['bootstrap']['seed']); paired=[]; final='mse_gru'
    for challenger in ('last_position','constant_velocity','kalman','gaussian_gru','mse_gru_shrink','gaussian_gru_shrink'):
        delta=[]
        for env in unique:
            ids=environments==env; delta.append(float(distances[final][ids].mean()-distances[challenger][ids].mean()))
        delta=np.asarray(delta); boot=delta[rng.integers(0,len(delta),(c['bootstrap']['replicates'],len(delta)))].mean(axis=1)
        paired.append({'final_minus':challenger,'metric':'ade_m','environments':len(unique),'mean_delta_m':float(delta.mean()),'environment_bootstrap_95_ci_m':[float(v) for v in np.quantile(boot,[.025,.975])],'environment_win_rate':float(np.mean(delta<0))})
    report={'experiment_id':c['experiment_id'],'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'split':'test','windows':len(rows),'episodes':len({r['episode_uuid'] for r in rows}),'environments':len(unique),'final_roles':{'point_prediction':'mse_gru','uncertainty_estimation':'gaussian_gru_with_validation_frozen_conformal_radius','rejected_ablation':'horizon_shrinkage'},'frozen_shrinkage':{n:shrink[n]['full_validation_for_frozen_test'] for n in shrink},'frozen_conformal':frozen_conformal,'checkpoint_metrics':checkpoints,'per_horizon':per_horizon,'stratified':strata,'paired_effects':paired,'freeze_note':'No hyperparameter, coefficient, calibration-scale, or model-fitting change was made using test targets.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
