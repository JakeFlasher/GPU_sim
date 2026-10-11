#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--timing',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();tim=json.loads(a.timing.read_text());
 event={}
 for row in tim.get('rows',[]):
  name=row.get('workload') or row.get('family'); markers=row.get('sm120_event_markers',[])
  if name and markers:event[name]=float(markers[0].get('elapsed_ms'))
 rows=[]
 for line in a.replay.read_text().splitlines():
  if not line.strip():continue
  doc=json.loads(line);name=doc['capture_name'];sim=doc['simulator'];pl=sim.get('per_launch',[{}])[0];m=pl.get('ncu_metrics',{});ncu_cycles=(m.get('sm__cycles_elapsed.avg') or {}).get('value');pred=pl.get('completion_cycles');hw=event.get(name)
  rows.append({'family':name,'cuda_event_ms':hw,'simulator_wall_ms':sim.get('wall_ms'),'full_replay_speedup':hw/sim.get('wall_ms') if hw and sim.get('wall_ms') else None,'simulator_completion_cycles':pred,'ncu_sm_cycles_avg':ncu_cycles,'signed_cycle_error':(pred-ncu_cycles)/ncu_cycles if pred is not None and ncu_cycles else None,'absolute_cycle_ape':abs(pred-ncu_cycles)/abs(ncu_cycles) if pred is not None and ncu_cycles else None,'ncu_instruction_error_rows':sim.get('ncu_metric_errors',[]),'scope':'one P4 launch; full trace replay; NCU identity/ordered row','ncu_used_as_input':False})
 aps=[r['absolute_cycle_ape'] for r in rows if r['absolute_cycle_ape'] is not None];ss=[r['full_replay_speedup'] for r in rows if r['full_replay_speedup'] is not None]
 out={'schema_version':1,'model':'sm120_full_route_async_join_v1','rows':rows,'summary':{'rows':len(rows),'median_cycle_ape':statistics.median(aps) if aps else None,'max_cycle_ape':max(aps) if aps else None,'median_full_replay_speedup':statistics.median(ss) if ss else None,'min_full_replay_speedup':min(ss) if ss else None,'max_full_replay_speedup':max(ss) if ss else None},'fast_path_status':'frozen_not_used','physical_non_claim':['cache_geometry','TMA_completion_timestamp','mbarrier_visibility_timestamp']};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out['summary']))
if __name__=='__main__':main()
