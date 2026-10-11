#!/usr/bin/env python3
"""Side-by-side local comparison for manual analysis only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def read(path):
 out={}
 for l in path.read_text().splitlines():
  if l.strip():
   d=json.loads(l);s=d['simulator'];out[d['capture_name']]={'wall_ms':s.get('wall_ms'),'completion_cycles':(s.get('completion_cycles_by_launch') or [None])[0]}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--full',type=Path,required=True);p.add_argument('--fast',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();f=read(a.full);q=read(a.fast);rows=[]
 for n in sorted(set(f)|set(q)):rows.append({'capture_name':n,'full':f.get(n),'fast':q.get(n)})
 r={'schema_version':1,'model':'sm120_full_fast_side_by_side_v1','rows':rows,'interpretation':'Manual side-by-side diagnostic; full replay remains the primary execution path.'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows)}))
if __name__=='__main__':main()
