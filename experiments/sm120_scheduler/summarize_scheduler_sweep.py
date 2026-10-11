#!/usr/bin/env python3
"""Summarize timing-only scheduler interventions without fitting hardware topology."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

def med(xs): return statistics.median(xs) if xs else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=json.loads((a.input/'summary.json').read_text())
 rows=[]
 for r in d['rows']:
  obs=r['observations']; clocks=[w['elapsed_cycles'] for o in obs for w in o['warps_observed']]
  event=[o['event_ms'] for o in obs]
  ops=[o['body_ops_per_active_warp'] for o in obs]
  rows.append({'name':r['name'],'argv':r['argv'],'observations':len(obs),'event_ms_median':med(event),'clock_cycles_median':med(clocks),'body_ops_per_active_warp':ops[0] if ops else None,'cycles_per_body_op':med(clocks)/ops[0] if clocks and ops and ops[0] else None,'registers':obs[0]['registers'] if obs else None,'local_bytes':obs[0]['local_bytes'] if obs else None,'driver_occupancy_blocks':obs[0]['driver_occupancy_blocks'] if obs else None,'warps_observed':sum(len(o['warps_observed']) for o in obs)})
 result={'schema_version':1,'model':'sm120_scheduler_intervention_summary_v1','case_count':len(rows),'rows':rows,'interpretation':['Timing windows and clock slopes are observations, not unique scheduler-port or scoreboard-depth identification.','Placement fields are samples; logical warp index is not a guaranteed physical scheduler ID.','No NCU target or correctness oracle was used in this timing-only summary.']}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'rows':len(rows),'median_event_ms':med([x['event_ms_median'] for x in rows if x['event_ms_median'] is not None])}))
if __name__=='__main__':main()
