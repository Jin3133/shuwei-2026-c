#!/usr/bin/env python3
"""Fit and evaluate independent Q3 duration models for each visible prefix."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shuwei_c.config import load_config

def qhigher(x, q): return float(np.quantile(x, q, method="higher"))
def pinball(y, p, q):
    e=y-p; return float(np.mean(np.maximum(q*e,(q-1)*e)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--experiment",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); c=load_config(a.experiment); rows=pq.read_table(c["source"]).to_pylist()
    with Path(c["official_records"]).open(encoding="utf-8-sig",newline="") as f: env={r["episode_uuid"]:r["environment_id"] for r in csv.DictReader(f)}
    excluded=set(c["features"]["excluded"])|set(c["features"]["categorical"]); numeric=[n for n in rows[0] if n not in excluded]
    families=sorted({r["family_id"] for r in rows}); results=[]; corrections={}
    for prefix in map(float,c["prefixes_s"]):
        tr=[r for r in rows if r["split"]==c["train_split"] and np.isclose(r["prefix_duration_s"],prefix)]
        va=[r for r in rows if r["split"]==c["evaluation_split"] and np.isclose(r["prefix_duration_s"],prefix)]
        med={n:float(np.nanmedian([r[n] for r in tr if r[n] is not None])) for n in numeric}
        def mat(part): return np.asarray([[med[n] if r[n] is None else r[n] for n in numeric]+[float(r["family_id"]==f) for f in families] for r in part],float)
        xt,xv=mat(tr),mat(va); yt=np.asarray([r["target_duration_s"] for r in tr]); yv=np.asarray([r["target_duration_s"] for r in va]); groups=np.asarray([env[r["episode_uuid"]] for r in tr])
        raw=[]; calibrated=[]; corrections[str(prefix)]={}
        for q in map(float,c["quantiles"]):
            oof=np.zeros(len(tr))
            for fit,held in GroupKFold(n_splits=c["n_group_folds"]).split(xt,yt,groups):
                m=GradientBoostingRegressor(loss="quantile",alpha=q,**c["model"]); m.fit(xt[fit],yt[fit]); oof[held]=m.predict(xt[held])
            correction=qhigher(yt-oof,q); corrections[str(prefix)][str(q)]=correction
            m=GradientBoostingRegressor(loss="quantile",alpha=q,**c["model"]); m.fit(xt,yt); pred=m.predict(xv)
            raw.append(pred); calibrated.append(pred+correction)
        raw=np.maximum.accumulate(np.asarray(raw).T,axis=1); calibrated=np.maximum.accumulate(np.asarray(calibrated).T,axis=1)
        for label,pred in (("raw",raw),("oof_calibrated",calibrated)):
            row={"prefix_s":prefix,"variant":label,"n":len(va),"q50_mae_s":float(np.mean(np.abs(yv-pred[:,0]))),"q50_rmse_s":float(np.sqrt(np.mean((yv-pred[:,0])**2)))}
            for i,q in enumerate(map(float,c["quantiles"])):
                row[f"q{int(q*100)}_coverage"]=float(np.mean(yv<=pred[:,i])); row[f"q{int(q*100)}_pinball"]=pinball(yv,pred[:,i],q)
                row[f"q{int(q*100)}_idle_s"]=float(np.mean(np.maximum(pred[:,i]-yv,0))); row[f"q{int(q*100)}_overrun_s"]=float(np.mean(np.maximum(yv-pred[:,i],0)))
            results.append(row)
    report={"experiment_id":c["experiment_id"],"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"corrections_s":corrections,"results":results}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
