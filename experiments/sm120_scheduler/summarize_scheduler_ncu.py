#!/usr/bin/env python3
"""Parse selected NCU scheduler rows and join only for reporting."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path

def parse(path):
 rows=[]
 with path.open(newline='',encoding='utf-8',errors='replace') as f:
  allrows=list(csv.reader(line for line in f if not line.startswith('==')))
 try: h=next(i for i,r in enumerate(allrows) if r and r[0]=='ID')
 except StopIteration:return {}
 if h+2>=len(allrows):return {}
 data=allrows[h+2]
 out={}
 for k,v in zip(allrows[h],data):
  try: out[k]=float(v)
  except ValueError: out[k]=v
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--ncu',type=Path,required=True);p.add_argument('--timing',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 timing={x['name']:x for x in json.loads(a.timing.read_text())['rows']};rows=[]
 for raw in sorted(a.ncu.glob('*.raw.csv')):
  name=raw.name[:-8]; obs=parse(raw); t=timing.get(name,{})
  row={'name':name,'ncu_raw_csv':str(raw),'ncu_metrics':{k:obs.get(k) for k in ('gpu__time_duration.sum','sm__cycles_elapsed.avg','smsp__inst_executed.sum','smsp__inst_issued.sum','smsp__warps_active.sum','smsp__warps_eligible.sum','smsp__warp_issue_stalled_not_selected_per_warp_active.ratio','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio','smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio','smsp__warp_issue_stalled_barrier_per_warp_active.ratio')},'timing_event_ms_median':t.get('event_ms_median'),'clock_cycles_median':t.get('clock_cycles_median'),'body_ops_per_active_warp':t.get('body_ops_per_active_warp'),'ncu_observed':bool(obs)}
  rows.append(row)
 result={'schema_version':1,'model':'sm120_scheduler_ncu_join_v1','rows':rows,'target_used_as_prediction_input':False,'interpretation':'NCU values are independent observations; no fit or simulator parameter update is performed by this join.'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'rows':len(rows),'ncu_observed':sum(r['ncu_observed'] for r in rows)}))
if __name__=='__main__':main()
