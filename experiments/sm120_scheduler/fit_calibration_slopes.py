#!/usr/bin/env python3
"""Fit simple calibration slopes for timing-only dependent chains.

Fits are descriptive calibration surfaces for the probe family only. They are
not architectural constants and are not applied to evaluation rows here.
"""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

def fit(points):
 n=len(points);sx=sum(x for x,y in points);sy=sum(y for x,y in points);sxx=sum(x*x for x,y in points);sxy=sum(x*y for x,y in points);den=n*sxx-sx*sx
 slope=(n*sxy-sx*sy)/den if den else None; intercept=(sy-slope*sx)/n if slope is not None else None
 return {'slope_cycles_per_round':slope,'intercept_cycles':intercept,'points':points}

def main():
 p=argparse.ArgumentParser();p.add_argument('--depth',type=Path,required=True);p.add_argument('--ncu',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=json.loads((a.depth/'summary.json').read_text());fits={}
 for mode in ('fma','mad','add'):
  pts=[]
  for r in d['rows']:
   if r['name'].startswith(mode+'_depth_'):
    rounds=int(r['args'][3]);cycles=statistics.median(w['elapsed_cycles'] for o in r['observations'] for w in o['warps_observed']);pts.append((rounds,cycles))
  fits[mode]=fit(sorted(pts))
 ncu=json.loads(a.ncu.read_text())
 result={'schema_version':1,'model':'sm120_scheduler_calibration_surface_v1','clock_slope_fits':fits,'ncu_rows':ncu['rows'],'parameter_use':'calibration-only; no evaluation target or simulator parameter was fit here','non_claims':['physical scheduler port topology','scoreboard depth','warp-to-scheduler mapping']}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'modes':list(fits),'ncu_rows':len(ncu['rows'])}))
if __name__=='__main__':main()
