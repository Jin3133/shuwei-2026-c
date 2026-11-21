#!/usr/bin/env python3
"""Replay Q2 with sample-specific Gaussian uncertainty and frozen calibration scales."""
from __future__ import annotations
import argparse,json,subprocess,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from shuwei_c.config import load_config
from shuwei_c.models import build_model
from run_q2_reobservation import replay

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); c=load_config(a.experiment)
    mc=load_config(c['model_config']); checkpoint=torch.load(c['checkpoint'],map_location='cpu',weights_only=False); model=build_model(mc['model'],mc['data']); model.load_state_dict(checkpoint['model']); model.eval()
    norm=json.loads(Path(c['normalization']).read_text()); calibration=json.loads(Path(c['robust_calibration_report']).read_text())
    scale_sets={name:np.asarray(values,float) for name,values in calibration['frozen_scales'].items()}
    pm,ps=np.asarray(norm['position']['mean'],np.float32),np.asarray(norm['position']['scale'],np.float32); vm,vs=np.asarray(norm['velocity']['mean'],np.float32),np.asarray(norm['velocity']['scale'],np.float32)
    horizons=np.asarray(mc['data']['target_horizons_s'],float); table=pq.read_table(c['private_scoring_table']); rows=table.filter(pc.equal(table['split'],c['split'])).to_pylist()
    threshold_sets=c.get('calibration_thresholds_m',{name:c['uncertainty_thresholds_m'] for name in scale_sets})
    policies=[('none',None,None),('fixed',None,c['fixed_interval_s'])]+[(name,'uncertainty',float(t)) for name in scale_sets for t in threshold_sets[name]]
    collected=defaultdict(list)
    for start in range(0,len(rows),512):
        batch=rows[start:start+512]; pos=np.asarray([r['history_position'] for r in batch],np.float32); vel=np.asarray([r['history_velocity'] for r in batch],np.float32); quality=np.asarray([r['history_quality'] for r in batch],np.float32)[:,:,0]; hand=np.asarray([r['hand']=='right' for r in batch],np.float32)
        x=np.zeros((len(batch),pos.shape[1],10),np.float32); x[:,:,:3]=(pos-pm)/ps; x[:,:,3:6]=(vel-vm)/vs; x[:,:,6]=1; x[:,:,8]=quality; x[:,:,9]=hand[:,None]
        with torch.no_grad(): out=model(torch.from_numpy(x)); means=out.mean.numpy(); base=np.linalg.norm(np.sqrt(np.maximum(out.variance.numpy(),1e-6)),axis=-1)
        shrink=np.exp(-c['shrinkage_decay']*horizons)
        for i,row in enumerate(batch):
            initial=means[i]*shrink[:,None]; duration=float(row['mask_duration_s'])
            for scale_name,kind,threshold in policies:
                if kind=='uncertainty': policy=('uncertainty',threshold); radii=base[i]*scale_sets[scale_name]; name=f'{scale_name}_{threshold:.2f}'
                elif scale_name=='fixed': policy=('fixed',threshold); radii=np.zeros_like(horizons); name='fixed_interval'
                else: policy=('none',None); radii=np.zeros_like(horizons); name='no_request'
                result=replay(row,initial,horizons,radii,policy,c); collected[duration,name].append(result)
    by_duration=[]
    for (duration,name),values in sorted(collected.items()):
        by_duration.append({'mask_duration_s':duration,'policy':name,'n':len(values),**{f'mean_{key}':float(np.mean([v[key] for v in values])) for key in values[0]}})
    aggregate=[]
    for name in sorted({name for _,name in collected}):
        values=[v for (duration,policy),group in collected.items() if policy==name for v in group]; error=float(np.mean([v['path_error'] for v in values])); requests=float(np.mean([v['requests'] for v in values]))
        aggregate.append({'policy':name,'n':len(values),'mean_path_error_m':error,'mean_endpoint_error_m':float(np.mean([v['endpoint_error'] for v in values])),'mean_requests':requests,'mean_cost':float(np.mean([v['cost'] for v in values])),'mean_wait_exposure_s':float(np.mean([v['wait_exposure'] for v in values])),'selection_objective':error+c['objective_request_weight_m']*requests})
    pareto=[]
    for row in aggregate:
        dominated=any(other['mean_path_error_m']<=row['mean_path_error_m'] and other['mean_requests']<=row['mean_requests'] and (other['mean_path_error_m']<row['mean_path_error_m'] or other['mean_requests']<row['mean_requests']) for other in aggregate)
        if not dominated: pareto.append(row['policy'])
    sensitivity=[]
    for weight in c['objective_request_weights_m']:
        ranked=sorted((row['mean_path_error_m']+weight*row['mean_requests'],row['policy']) for row in aggregate)
        sensitivity.append({'request_weight_m':float(weight),'best_policy':ranked[0][1],'best_objective':float(ranked[0][0]),'runner_up_policy':ranked[1][1],'runner_up_objective':float(ranked[1][0])})
    report={'experiment_id':c['experiment_id'],'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'split':c['split'],'uncertainty_definition':'sample Gaussian radial standard deviation multiplied by validation-frozen conformal scale','frozen_scales':{k:v.tolist() for k,v in scale_sets.items()},'conditions':{k:c[k] for k in ('return_delay_s','request_cost','request_budget','fixed_interval_s','objective_request_weight_m')},'pareto_policies':pareto,'objective_sensitivity':sensitivity,'aggregate':aggregate,'by_duration':by_duration}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
