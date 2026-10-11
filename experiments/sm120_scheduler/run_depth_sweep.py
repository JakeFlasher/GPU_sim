#!/usr/bin/env python3
"""Vary dependent-loop depth for intercept/slope calibration observations."""
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
 for mode in ('fma','mad','add'):
  for rounds in (64,128,256,512,1024,2048,4096,8192):
   args=[mode,'1','16',str(rounds),'1','1','1','3','2'];t=time.perf_counter_ns();q=subprocess.run([str(a.binary),*args],capture_output=True,text=True,timeout=120,check=False);obs=[json.loads(x) for x in q.stdout.splitlines() if x.startswith('{')];row={'name':f'{mode}_depth_r{rounds}','args':args,'returncode':q.returncode,'elapsed_ns':time.perf_counter_ns()-t,'observations':obs,'stdout_tail':q.stdout[-1000:],'stderr_tail':q.stderr[-1000:]};rows.append(row);print(row['name'],q.returncode,len(obs),flush=True)
 r={'schema_version':1,'model':'sm120_scheduler_depth_calibration_v1','rows':rows};(a.out/'summary.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':main()
