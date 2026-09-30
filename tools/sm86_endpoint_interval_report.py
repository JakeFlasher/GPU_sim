#!/usr/bin/env python3
"""Summarize candidate interval coverage after predictions are formed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")


def run(args):
    report = json.loads(args.input.read_text())
    rows = report.get("predictions") or report.get("rows") or []
    output = {"schema": "sm86_endpoint_interval_report_v1", "input": str(args.input), "targets_score_only": True, "metrics": {}}
    for target in TARGETS:
        usable=[]
        for row in rows:
            actual=row.get(f"{target}_actual"); low=row.get(f"{target}_interval_low"); high=row.get(f"{target}_interval_high")
            if actual is None or low is None or high is None: continue
            usable.append((float(actual),float(low),float(high)))
        if not usable:
            output["metrics"][target]={"count":0,"coverage":None,"median_relative_width":None}
            continue
        a=np.asarray([x[0] for x in usable]); lo=np.asarray([x[1] for x in usable]); hi=np.asarray([x[2] for x in usable])
        widths=(hi-lo)/np.maximum(a,1.0)
        output["metrics"][target]={"count":len(a),"coverage":float(np.mean((lo<=a)&(a<=hi))),"median_relative_width":float(np.median(widths)),"min_relative_width":float(np.min(widths)),"max_relative_width":float(np.max(widths))}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(output,indent=2)+"\n"); return output


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--input',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); report=run(args); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
