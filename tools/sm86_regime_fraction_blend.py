#!/usr/bin/env python3
"""Blend neutral/high endpoint sources from independent clock-burst fraction."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np

TARGETS=("ncu_gpc_cycles","ncu_duration_ns")

def load(path):
 if path.is_dir():
  with (path/'performance.csv').open(newline='') as s: return {r['workload']:r for r in csv.DictReader(s) if r.get('status')=='completed'}
 d=json.loads(path.read_text()) if path.is_file() else None
 rows=(d or {}).get('predictions') or (d or {}).get('rows') or []
 return {r['workload']:r for r in rows if r.get('workload')}

def pred(row,target):
 v=row.get(f'{target}_predicted')
 if v is None and isinstance(row.get('endpoints'),dict):
  e=row['endpoints'].get(target,{})
  if isinstance(e,dict): v=e.get('elastic',e.get('reference'))
 if v is None and isinstance(row.get('predictions'),dict): v=row['predictions'].get(target)
 v=float(v)
 if not math.isfinite(v) or v<=0: raise RuntimeError('invalid source prediction')
 return v

def run(a):
 obs=json.loads(a.clock_observation.read_text()); frac=float(obs.get('active_window_high_clock_fraction',0.0) or 0.0); alpha=min(1.0,max(0.0,(frac-a.low_fraction)/max(a.high_fraction-a.low_fraction,1e-9)))
 neutral=load(a.neutral); high=load(a.high); score=load(a.score_campaign); names=sorted(set(neutral)&set(high)&set(score)); rows=[]
 for n in names:
  item={'workload':n,'status':'candidate','regime_fraction':frac,'blend_alpha':alpha,'coverage_status':'fraction_blend_covered'}
  for t in TARGETS:
   p=(1-alpha)*pred(neutral[n],t)+alpha*pred(high[n],t); actual=float(score[n].get(f'{t}_actual',score[n].get(t))); item[f'{t}_predicted']=p; item[f'{t}_actual']=actual; item[f'{t}_ape']=abs(p-actual)/actual
  rows.append(item)
 metrics={}
 for t in TARGETS:
  x=[r[f'{t}_ape'] for r in rows]; metrics[t]={'count':len(x),'mape':float(np.mean(x)),'max_ape':float(np.max(x))}
 out={'schema':'sm86_regime_fraction_blend_v1','phase':a.phase,'neutral_source':str(a.neutral),'high_source':str(a.high),'clock_observation':str(a.clock_observation),'high_fraction':frac,'blend_alpha':alpha,'policy':'linear neutral/high interpolation from independent burst fraction; no score NCU route selection','metrics':metrics,'predictions':rows}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); return out

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',required=True); p.add_argument('--neutral',type=Path,required=True); p.add_argument('--high',type=Path,required=True); p.add_argument('--score-campaign',type=Path,required=True); p.add_argument('--clock-observation',type=Path,required=True); p.add_argument('--low-fraction',type=float,default=.10); p.add_argument('--high-fraction',type=float,default=.20); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); r=run(a); print(json.dumps({'phase':r['phase'],'high_fraction':r['high_fraction'],'blend_alpha':r['blend_alpha'],'metrics':r['metrics'],'output':str(a.output)},indent=2))
if __name__=='__main__': main()
