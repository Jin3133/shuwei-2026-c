#!/usr/bin/env python3
"""One-time frozen duration evaluation on Q3 test environments."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from shuwei_c.config import load_config
from run_q3_paid_review_sweep import build_random_forest_predictions

EXCLUDED={"prefix_id","segment_id","episode_uuid","dataset","split","family_id","target_duration_s"}
def pinball(y,p,q):
    e=y-p; return float(np.mean(np.maximum(q*e,(q-1)*e)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--predictions-csv',type=Path,required=True)
    a=ap.parse_args(); c=load_config(a.experiment); rows=pq.read_table(c['source']).to_pylist(); train=[r for r in rows if r['split']==c['train_split']]; test=[r for r in rows if r['split']==c['evaluation_split']]
    rfcal=json.loads(Path(c['rf_calibration']).read_text()); rfcorr={str(r['prefix_s']):r['corrections_s'] for r in rfcal['results'] if r['model']=='random_forest'}
    rfcfg={'prefixes_s':c['prefixes_s'],'train_split':c['train_split'],'evaluation_split':c['evaluation_split'],'model':c['rf_model']}; rfpred=build_random_forest_predictions(rows,rfcfg,rfcorr)
    gcal=json.loads(Path(c['gbdt_calibration']).read_text()); numeric=[n for n in rows[0] if n not in EXCLUDED]; families=sorted({r['family_id'] for r in rows}); med={n:float(np.nanmedian([r[n] for r in train if r[n] is not None])) for n in numeric}
    def matrix(part): return np.asarray([[med[n] if r[n] is None else r[n] for n in numeric]+[float(r['family_id']==f) for f in families] for r in part],float)
    xt=matrix(train); yt=np.asarray([r['target_duration_s'] for r in train]); gmodels={}
    for q in (.5,.8): gmodels[q]=GradientBoostingRegressor(loss='quantile',alpha=q,**c['gbdt_model']).fit(xt,yt)
    family=defaultdict(list)
    for r in train:
        if np.isclose(r['prefix_duration_s'],.2): family[r['family_id']].append(float(r['target_duration_s']))
    records=[]
    for prefix in map(float,c['prefixes_s']):
        part=[r for r in test if np.isclose(r['prefix_duration_s'],prefix)]; x=matrix(part); q50=gmodels[.5].predict(x); q80=gmodels[.8].predict(x)+float(gcal['corrections_s'][str(prefix)]['0.8'])
        for i,r in enumerate(part):
            fq50=float(np.quantile(family[r['family_id']],.5,method='higher')); fq80=float(np.quantile(family[r['family_id']],.8,method='higher')); rp=rfpred[r['segment_id']][prefix]
            for model,p50,p80 in [('family',fq50,fq80),('gbdt',float(q50[i]),max(float(q50[i]),float(q80[i]))),('random_forest',rp['q50'],rp['q80'])]:
                records.append({'segment_id':r['segment_id'],'episode_uuid':r['episode_uuid'],'split':r['split'],'family_id':r['family_id'],'prefix_s':prefix,'model':model,'q50_s':p50,'q80_s':p80,'target_duration_s':r['target_duration_s']})
    summaries=[]
    for prefix in map(float,c['prefixes_s']):
        for model in ('family','gbdt','random_forest'):
            part=[r for r in records if r['prefix_s']==prefix and r['model']==model]; y=np.asarray([r['target_duration_s'] for r in part],float); p50=np.asarray([r['q50_s'] for r in part]); p80=np.asarray([r['q80_s'] for r in part])
            summaries.append({'prefix_s':prefix,'model':model,'n':len(part),'q50_mae_s':float(np.mean(np.abs(y-p50))),'q50_rmse_s':float(np.sqrt(np.mean((y-p50)**2))),'q80_coverage':float(np.mean(y<=p80)),'q80_pinball':pinball(y,p80,.8),'q80_idle_s':float(np.mean(np.maximum(p80-y,0))),'q80_overrun_s':float(np.mean(np.maximum(y-p80,0)))})
    a.predictions_csv.parent.mkdir(parents=True,exist_ok=True)
    with a.predictions_csv.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n'); w.writeheader(); w.writerows(records)
    with Path(c['official_records']).open(encoding='utf-8-sig',newline='') as f: environment={r['episode_uuid']:r['environment_id'] for r in csv.DictReader(f)}
    report={'experiment_id':c['experiment_id'],'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'test_segments':len({r['segment_id'] for r in test}),'test_episodes':len({r['episode_uuid'] for r in test}),'test_environments':len({environment[r['episode_uuid']] for r in test}),'summaries':summaries,'scheduling_status':'not_run_missing_organizer_test_seeds'}
    a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
