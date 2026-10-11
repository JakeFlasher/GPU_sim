#!/usr/bin/env python3
"""Collect independent NCU scheduler observations for selected probes."""
from __future__ import annotations
import argparse,csv,json,subprocess,time
from pathlib import Path

METRICS=[
 'gpu__time_duration.sum','sm__cycles_elapsed.avg','smsp__inst_executed.sum',
 'smsp__inst_issued.sum','smsp__warps_active.sum','smsp__warps_eligible.sum',
 'smsp__warp_issue_stalled_not_selected_per_warp_active','smsp__warp_issue_stalled_short_scoreboard_per_warp_active',
 'smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active','smsp__warp_issue_stalled_barrier_per_warp_active',
]

def cases():
 out=[]
 for mode in ('fma','mad','mix'):
  for k in (1,4,16): out.append((f'fixedwork_{mode}_k{k}',[mode,str(k),'16',str(8192//k),'1','1','1','3','2']))
 for w in (1,8,32): out.append((f'warp_density_w{w}',['fma','4','16','2048',str(w),str(0xffffffff if w==32 else (1<<w)-1),'1','3','2']))
 for w in (1,8,32): out.append((f'barrier_w{w}',['barrier','1','16','128',str(w),str(0xffffffff if w==32 else (1<<w)-1),'1','3','2']))
 for u,r in ((64,64),(1024,4)): out.append((f'footprint_u{u}',['fma','4',str(u),str(r),'1','1','1','3','2']))
 for mask,name in ((0x55,'even'),(0xaa,'odd')): out.append((f'placement_{name}_allocated8',['fma','4','16','2048','8',str(mask),'1','3','2']))
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--ncu',type=Path,default=Path('/usr/local/cuda-13.2/bin/ncu'));p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=float,default=300);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for name,args in cases():
  raw=a.out/(name+'.raw.csv'); stdout=a.out/(name+'.stdout'); stderr=a.out/(name+'.stderr');
  cmd=[str(a.ncu),'--csv','--page','raw','--print-units','base','--profile-from-start','off','--target-processes','all','--launch-count','1','--metrics',','.join(METRICS),'--log-file',str(raw),str(a.binary),*args]
  t=time.perf_counter_ns()
  try:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=a.timeout,check=False);to=False
  except subprocess.TimeoutExpired as e:
   p=subprocess.CompletedProcess(cmd,124,e.stdout or '',e.stderr or '');to=True
  stdout.write_text(p.stdout or '',encoding='utf-8');stderr.write_text(p.stderr or '',encoding='utf-8')
  rows.append({'name':name,'args':args,'command':cmd,'returncode':p.returncode,'timed_out':to,'elapsed_ns':time.perf_counter_ns()-t,'raw_csv':str(raw),'raw_csv_present':raw.is_file() and raw.stat().st_size>0})
  print(name,p.returncode,rows[-1]['raw_csv_present'],flush=True)
 result={'schema_version':1,'model':'sm120_scheduler_ncu_campaign_v1','metrics':METRICS,'rows':rows,'target_used_as_prediction_input':False}
 (a.out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'rows':len(rows),'profiles':sum(r['raw_csv_present'] for r in rows),'returncodes':sorted(set(r['returncode'] for r in rows))}))
if __name__=='__main__':main()
