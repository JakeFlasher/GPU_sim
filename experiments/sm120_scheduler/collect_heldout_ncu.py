#!/usr/bin/env python3
"""Collect NCU rows for the fresh scheduler held-out trace matrix."""
from __future__ import annotations
import argparse,csv,json,subprocess,time
from pathlib import Path

METRICS=['gpu__time_duration.sum','sm__cycles_elapsed.avg','smsp__inst_executed.sum','smsp__inst_issued.sum','smsp__warps_active.sum','smsp__warps_eligible.sum','smsp__warp_issue_stalled_not_selected_per_warp_active','smsp__warp_issue_stalled_short_scoreboard_per_warp_active','smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active','smsp__warp_issue_stalled_barrier_per_warp_active']
CASES=[('heldout_fma_k8',['fma','8','16','1024','1','1','1','3','2']),('heldout_mad_k8',['mad','8','16','1024','1','1','1','3','2']),('heldout_mix_k8',['mix','8','16','1024','1','1','1','3','2']),('heldout_warp_density_w16_small',['fma','4','16','256','16','65535','1','3','2']),('heldout_barrier_w16',['barrier','1','16','128','16','65535','1','3','2']),('heldout_footprint_u256',['fma','4','256','16','1','1','1','3','2'])]
def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--ncu',type=Path,default=Path('/usr/local/cuda-13.2/bin/ncu'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
 for name,args in CASES:
  raw=a.out/(name+'.raw.csv');cmd=[str(a.ncu),'--csv','--page','raw','--print-units','base','--profile-from-start','off','--target-processes','all','--launch-count','1','--metrics',','.join(METRICS),'--log-file',str(raw),str(a.binary),*args];t=time.perf_counter_ns()
  try:q=subprocess.run(cmd,capture_output=True,text=True,timeout=300,check=False);to=False
  except subprocess.TimeoutExpired as e:q=subprocess.CompletedProcess(cmd,124,e.stdout or '',e.stderr or '');to=True
  (a.out/(name+'.stdout')).write_text(q.stdout or '');(a.out/(name+'.stderr')).write_text(q.stderr or '');rows.append({'name':name,'args':args,'command':cmd,'returncode':q.returncode,'timed_out':to,'elapsed_ns':time.perf_counter_ns()-t,'raw_csv':str(raw),'raw_csv_present':raw.is_file() and raw.stat().st_size>0});print(name,q.returncode,rows[-1]['raw_csv_present'],flush=True)
 r={'schema_version':1,'model':'sm120_scheduler_heldout_ncu_v1','metrics':METRICS,'rows':rows,'target_used_as_prediction_input':False};(a.out/'summary.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows),'profiles':sum(x['raw_csv_present'] for x in rows)}))
if __name__=='__main__':main()
