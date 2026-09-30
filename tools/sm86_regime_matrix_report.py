#!/usr/bin/env python3
"""Consolidate fixed-regime route metrics without refitting models."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path

def run(args):
    started=time.perf_counter(); cells=[]
    for spec in args.cell:
        phase,regime,route,path=spec.split("=",3); report=json.loads(Path(path).read_text()); cells.append({"phase":phase,"regime":regime,"route":route,"source":path,"metrics":report.get("metrics"),"coverage":report.get("coverage")})
    out={"schema":"sm86_regime_matrix_report_v1","cells":cells,"model_seconds":time.perf_counter()-started,"posthoc_only":True,"limitations":["cells use prior fixed route reports; no new fitting","clock regime is external-observation dependent","NCU targets remain score-only comparisons"]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+'\n'); return out

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--cell',action='append',required=True,help='phase=regime=route=summary.json'); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); r=run(a); print(json.dumps({"cells":len(r['cells']),"output":str(a.output)},indent=2))
if __name__=='__main__': main()
