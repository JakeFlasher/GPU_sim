#!/usr/bin/env python3
"""Produce the final scope-aligned scheduler campaign summary."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--fast',type=Path,required=True);p.add_argument('--full',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    fast={x['name']:x for x in json.loads(a.fast.read_text())['rows']}
    full={x['name']:x for x in json.loads(a.full.read_text())['rows']}
    full_names={'scope_barrier_w16','scope_warp_density_w16'}
    rows=[]
    for name in sorted(fast):
        r=(full if name in full_names else fast)[name]
        ape=(r['simulator_vs_ncu_cycles'] or {}).get('absolute_relative_error')
        rows.append({'name':name,'selected_path':'full' if name in full_names else 'fast_frontier','event_ms':r['event_ms'],'simulator_wall_ms':r['simulator_wall_ms'],'replay_speedup':r['replay_speedup_event_over_sim_wall'],'predicted_cycles':r['simulator_completion_cycles'],'ncu_cycles':r['ncu_sm_cycles_avg'],'absolute_relative_error':ape,'ncu_inst':r['ncu_inst_executed'],'trace_inst':r['trace_candidate_instructions']})
    errors=[r['absolute_relative_error'] for r in rows if r['absolute_relative_error'] is not None]
    speeds=[r['replay_speedup'] for r in rows if r['replay_speedup'] is not None]
    result={'schema_version':1,'model':'sm120_scheduler_campaign_summary_v1','rows':rows,'summary':{'rows':len(rows),'median_absolute_relative_error':statistics.median(errors) if errors else None,'maximum_absolute_relative_error':max(errors) if errors else None,'median_replay_speedup':statistics.median(speeds) if speeds else None,'minimum_replay_speedup':min(speeds) if speeds else None,'maximum_replay_speedup':max(speeds) if speeds else None},'selection_contract':'fast frontier for one-warp arithmetic and full replay for inter-warp/barrier probes; declared by mechanism before reading NCU errors','ncu_used_as_input':False,'physical_non_claim':['scheduler_port_topology','scoreboard_depth','warp_to_scheduler_mapping']}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result['summary']))
if __name__=='__main__':main()
