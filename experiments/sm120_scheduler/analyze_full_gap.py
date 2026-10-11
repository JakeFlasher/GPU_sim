#!/usr/bin/env python3
"""Map full-replay memory/async gaps by operation class without fitting NCU."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
 for line in a.replay.read_text().splitlines():
  if not line.strip():continue
  d=json.loads(line);s=d['simulator'];pl=(s.get('per_launch') or [{}])[0];t=((s.get('result') or {}).get('launches') or [{}])[0].get('timing',{});lim=(t.get('readiness') or {}).get('limiting_cycles',{});trace=((s.get('result') or {}).get('launches') or [{}])[0].get('trace_metrics',{});mem=trace.get('memory',{});ncu=pl.get('ncu_metrics',{});ncu_cycles=(ncu.get('sm__cycles_elapsed.avg') or {}).get('value');rows.append({'family':d['capture_name'],'completion_cycles':pl.get('completion_cycles'),'ncu_cycles':ncu_cycles,'cycle_ape':abs(pl.get('completion_cycles')-ncu_cycles)/abs(ncu_cycles) if ncu_cycles else None,'limiting_cycles':lim,'memory_trace':mem,'async_affected_opcodes':((t.get('alpha_approximation') or {}).get('affected_opcode_counts') or {}),'tma_sidecar':(d.get('async_sidecar_observation') or {}).get('tma_byte_count_sum'),'sync_records':(d.get('async_sidecar_observation') or {}).get('sync_records'),'ncu_sector_metrics':{k:v for k,v in ncu.items() if isinstance(v,dict) and v.get('unit') in ('sector','cycle') and any(x in k for x in ('l1tex__t_sectors','lts__t_sectors','dram__sectors','sm__cycles_elapsed'))},'scope':'full trace replay; NCU independent'} )
 out={'schema_version':1,'model':'sm120_full_replay_gap_map_v1','rows':rows,'interpretation':['Counts and NCU values diagnose mechanisms; they are not fitted prediction inputs.','Full replay remains the only active simulator path.','No physical cache geometry, MSHR, DRAM queue or native async timestamp is inferred.']};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':main()
