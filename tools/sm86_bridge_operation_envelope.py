#!/usr/bin/env python3
"""Emit a conservative operation-family endpoint envelope for bridged bodies."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

def f(r,k):
 try:return float(r.get(k,''))
 except:return 0.0

def main():
 p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--mixed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.matrix.open(newline='') as s: rows=list(csv.DictReader(s))
 with a.mixed.open(newline='') as s: mixed=list(csv.DictReader(s))[-1]
 targets=['gpu__time_duration.sum','gpc__cycles_elapsed.max','smsp__inst_executed.sum','smsp__inst_executed_pipe_alu.sum','smsp__inst_executed_pipe_lsu.sum','smsp__inst_executed_pipe_xu.sum','pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg']
 envelope={k:{'min':min(f(r,k) for r in rows),'max':max(f(r,k) for r in rows)} for k in targets}
 mixed_obs={k:f(mixed,k) for k in targets}
 resource=['smsp__inst_executed.sum','smsp__inst_executed_pipe_alu.sum','smsp__inst_executed_pipe_lsu.sum','smsp__inst_executed_pipe_xu.sum']
 # normalized nearest-family distance in aggregate resource counts; no route labels.
 scale={k:max(abs(f(r,k)) for r in rows) or 1.0 for k in resource}
 distances=[]
 for r in rows:
  distances.append({'body':r.get('body',''),'distance':math.sqrt(sum(((f(r,k)-mixed_obs[k])/scale[k])**2 for k in resource))})
 nearest=min(distances,key=lambda x:x['distance'])
 out={'schema':'sm86_bridge_operation_envelope_v1','matrix':str(a.matrix),'mixed_observation':mixed_obs,'independent_family_envelope':envelope,'nearest_independent_body':nearest,'target_labels_excluded_from_envelope':True,'limitations':['aggregate operation populations do not identify per-PC routes','template-derived ABI metadata remains bounded','envelope is descriptive and refuses universal opcode latency']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
