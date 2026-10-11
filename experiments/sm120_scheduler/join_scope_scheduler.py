#!/usr/bin/env python3
"""Join scope-aligned timing, simulator replay, trace and NCU observations."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path

def parse_ncu(path):
 rows=[]
 with path.open(newline='',encoding='utf-8',errors='replace') as f: data=list(csv.reader(line for line in f if not line.startswith('==')))
 try:i=next(i for i,r in enumerate(data) if r and r[0]=='ID')
 except StopIteration:return {}
 if i+2>=len(data):return {}
 for k,v in zip(data[i],data[i+2]):
  try:rows.append((k,float(v)))
  except ValueError:pass
 return dict(rows)
def rel(p,n):
 if n is None:return {'status':'unavailable','predicted':p,'observed':None}
 return {'status':'relative_error','predicted':p,'observed':n,'signed_relative_error':(p-n)/n if n else None,'absolute_relative_error':abs(p-n)/abs(n) if n else None}
def main():
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--ncu',type=Path,required=True);p.add_argument('--timing',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();timing={r['name']:r for r in json.loads(a.timing.read_text())['rows']};ncu={x.name[:-8]:parse_ncu(x) for x in a.ncu.glob('*.raw.csv')};rows=[]
 for line in a.replay.read_text().splitlines():
  if not line.strip():continue
  doc=json.loads(line);name=doc['capture_name'];sim=doc['simulator'];tim=timing.get(name,{});o=(tim.get('observations') or [{}])[0];summary=((sim.get('result') or {}).get('fast_frontier') or [{}])[0].get('summary') or {};m=ncu.get(name,{})
  event=o.get('event_ms');wall=sim.get('wall_ms');pred_cycles=(sim.get('completion_cycles_by_launch') or [None])[0];ncu_cycles=m.get('sm__cycles_elapsed.avg');pred_inst=summary.get('candidate_instructions');ncu_inst=m.get('smsp__inst_executed.sum');rows.append({'name':name,'event_ms':event,'simulator_wall_ms':wall,'replay_speedup_event_over_sim_wall':event/wall if event and wall else None,'simulator_completion_cycles':pred_cycles,'ncu_sm_cycles_avg':ncu_cycles,'simulator_vs_ncu_cycles':rel(pred_cycles,ncu_cycles) if pred_cycles is not None else None,'trace_candidate_instructions':pred_inst,'ncu_inst_executed':ncu_inst,'simulator_vs_ncu_instructions':rel(pred_inst,ncu_inst) if pred_inst is not None else None,'ncu_metrics':m,'scope':'one measured kernel; one trace launch; one NCU launch','ncu_used_as_input':False})
 result={'schema_version':1,'model':'sm120_scheduler_scope_join_v1','rows':rows,'interpretation':'Primary replay speed uses uninstrumented CUDA event divided by simulator process replay for the same one-launch trace. NCU comparisons are independent diagnostics; no fit or parameter update is performed.'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows),'speedups':[r['replay_speedup_event_over_sim_wall'] for r in rows]}))
if __name__=='__main__':main()
