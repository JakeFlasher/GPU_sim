#!/usr/bin/env python3
"""Run the compact P10 footprint/selected-warp matrix."""
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
CASES=[('p10_footprint_u64',['fma','4','64','64','1','1','1','3','2']),('p10_footprint_u256',['fma','4','256','16','1','1','1','3','2']),('p10_footprint_u1024',['fma','4','1024','4','1','1','1','3','2']),('p10_select_w0',['fma','4','16','256','16','1','1','3','2']),('p10_select_w3',['fma','4','16','256','16','8','1','3','2']),('p10_select_w7',['fma','4','16','256','16','128','1','3','2']),('p10_select_w12',['fma','4','16','256','16','4096','1','3','2']),('p10_pair_low',['fma','4','16','256','16','3','1','3','2']),('p10_pair_high',['fma','4','16','256','16','49152','1','3','2']),('p10_pair_interleave',['fma','4','16','256','16','195','1','3','2'])]
def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
    for n,args in CASES:
        t=time.perf_counter_ns();q=subprocess.run([str(a.binary),*args],capture_output=True,text=True,timeout=180,check=False);obs=[json.loads(x) for x in q.stdout.splitlines() if x.startswith('{')];r={'name':n,'args':args,'returncode':q.returncode,'elapsed_ns':time.perf_counter_ns()-t,'observations':obs,'stdout_tail':q.stdout[-1500:],'stderr_tail':q.stderr[-1000:]};rows.append(r);(a.out/(n+'.json')).write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(n,q.returncode,len(obs),flush=True)
    result={'schema_version':1,'model':'sm120_p10_compact_matrix_v1','rows':rows,'trace_budget_contract':'one measured launch, <=16 allocated warps, <=256 runtime rounds for placement'}; (a.out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps({'rows':len(rows),'returncodes':sorted(set(r['returncode'] for r in rows))}))
if __name__=='__main__':main()
