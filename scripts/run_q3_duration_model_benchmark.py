#!/usr/bin/env python3
"""Compact prefix-specific duration model benchmark."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import GroupKFold
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from shuwei_c.config import load_config

def qh(x,q): return float(np.quantile(x,q,method="higher"))
def tree_quantiles(model,x,qs):
    values=np.asarray([tree.predict(x) for tree in model.estimators_]).T
    return np.stack([np.quantile(values,q,axis=1) for q in qs],axis=1)
def fit_predict(name,params,x,y,z,qs):
    if name=="gradient_boosting":
        out=[]
        for q in qs:
            m=GradientBoostingRegressor(loss="quantile",alpha=q,**params).fit(x,y); out.append(m.predict(z))
        return np.stack(out,axis=1)
    cls=RandomForestRegressor if name=="random_forest" else ExtraTreesRegressor
    return tree_quantiles(cls(**params).fit(x,y),z,qs)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--experiment",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); c=load_config(a.experiment)
    rows=pq.read_table(c["source"]).to_pylist();
    with Path(c["official_records"]).open(encoding="utf-8-sig",newline="") as f: env={r["episode_uuid"]:r["environment_id"] for r in csv.DictReader(f)}
    numeric=[n for n in rows[0] if n not in set(c["excluded"])]; families=sorted({r["family_id"] for r in rows}); qs=list(map(float,c["quantiles"])); results=[]
    for prefix in map(float,c["prefixes_s"]):
        tr=[r for r in rows if r["split"]==c["train_split"] and np.isclose(r["prefix_duration_s"],prefix)]; va=[r for r in rows if r["split"]==c["evaluation_split"] and np.isclose(r["prefix_duration_s"],prefix)]
        med={n:float(np.nanmedian([r[n] for r in tr if r[n] is not None])) for n in numeric}
        def mat(part): return np.asarray([[med[n] if r[n] is None else r[n] for n in numeric]+[float(r["family_id"]==f) for f in families] for r in part],float)
        x,z=mat(tr),mat(va); y=np.asarray([r["target_duration_s"] for r in tr]); truth=np.asarray([r["target_duration_s"] for r in va]); groups=np.asarray([env[r["episode_uuid"]] for r in tr])
        for name,params in c["models"].items():
            oof=np.zeros((len(tr),len(qs)))
            for fit,held in GroupKFold(c["n_group_folds"]).split(x,y,groups): oof[held]=fit_predict(name,params,x[fit],y[fit],x[held],qs)
            correction=np.asarray([qh(y-oof[:,i],q) for i,q in enumerate(qs)]); raw=fit_predict(name,params,x,y,z,qs); pred=np.maximum.accumulate(raw+correction,axis=1)
            e=truth-pred[:,0]
            results.append({"prefix_s":prefix,"model":name,"n":len(va),"q50_mae_s":float(np.mean(np.abs(e))),"q50_rmse_s":float(np.sqrt(np.mean(e**2))),
                "q80_coverage":float(np.mean(truth<=pred[:,1])),"q80_pinball":float(np.mean(np.maximum(qs[1]*(truth-pred[:,1]),(qs[1]-1)*(truth-pred[:,1])))),
                "q80_idle_s":float(np.mean(np.maximum(pred[:,1]-truth,0))),"q80_overrun_s":float(np.mean(np.maximum(truth-pred[:,1],0))),"corrections_s":correction.tolist()})
    report={"experiment_id":c["experiment_id"],"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"results":results}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
