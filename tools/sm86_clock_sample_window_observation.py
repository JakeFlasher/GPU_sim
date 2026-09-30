#!/usr/bin/env python3
"""Reduce an external nvidia-smi clock stream over profiler process windows."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time


def _samples(path):
    rows=[]
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                rows.append({"timestamp_ns": int(row["timestamp_ns"]), "sm_clock_mhz": float(row["sm_clock_mhz"]), "mem_clock_mhz": float(row["mem_clock_mhz"]), "pstate": row.get("pstate", "").strip()})
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _windows(root):
    windows={}
    for result_path in root.glob("*/result.json"):
        result=json.loads(result_path.read_text()); name=result.get("id",result_path.parent.name); spans=[]
        for label,item in result.get("processes",{}).items():
            if not label.startswith("ncu_") or not label.endswith(tuple(str(i) for i in range(10))):
                continue
            start=item.get("start_unix_ns"); end=item.get("end_unix_ns")
            if isinstance(start,int) and isinstance(end,int) and end>=start: spans.append({"label":label,"start_unix_ns":start,"end_unix_ns":end})
        if spans: windows[name]=spans
    return windows


def run(args):
    started=time.perf_counter(); samples=_samples(args.samples); windows=_windows(args.campaign)
    all_clocks=[row["sm_clock_mhz"] for row in samples]
    window_rows=[]; workload_rows=[]
    for workload,spans in sorted(windows.items()):
        selected=[]
        for row in samples:
            if any(span["start_unix_ns"] <= row["timestamp_ns"] <= span["end_unix_ns"] for span in spans): selected.append(row)
        clocks=[row["sm_clock_mhz"] for row in selected]
        if clocks:
            workload_rows.append({"workload":workload,"window_count":len(spans),"sample_count":len(clocks),"sm_clock_median_mhz":statistics.median(clocks),"sm_clock_min_mhz":min(clocks),"sm_clock_max_mhz":max(clocks),"high_fraction":sum(clock>=args.anchor_clock_mhz*args.high_threshold for clock in clocks)/len(clocks)})
            window_rows.extend(selected)
    selected_clocks=[row["sm_clock_mhz"] for row in window_rows]
    active=statistics.median(selected_clocks) if selected_clocks else None
    high_count=sum(clock >= args.anchor_clock_mhz*args.high_threshold for clock in selected_clocks)
    high_fraction=high_count / len(selected_clocks) if selected_clocks else 0.0
    regime="unknown"
    if active is not None:
        if active>=args.anchor_clock_mhz*args.high_threshold or (high_fraction >= args.high_fraction_threshold and max(selected_clocks) >= args.anchor_clock_mhz*args.high_threshold): regime="high"
        else: regime="low"
    report={"schema":"sm86_clock_sample_window_observation_v2","campaign":str(args.campaign),"samples":str(args.samples),"anchor_clock_mhz":args.anchor_clock_mhz,"high_threshold":args.high_threshold,"high_fraction_threshold":args.high_fraction_threshold,"sampler_sample_count":len(samples),"sampler_all_clock_mhz":{"median":statistics.median(all_clocks) if all_clocks else None,"min":min(all_clocks) if all_clocks else None,"max":max(all_clocks) if all_clocks else None},"profiler_window_sample_count":len(selected_clocks),"active_window_clock_mhz":{"median":active,"min":min(selected_clocks) if selected_clocks else None,"max":max(selected_clocks) if selected_clocks else None},"active_window_high_clock_fraction":high_fraction,"classification_basis":"median_or_high_fraction_and_maximum","regime":regime,"workloads":workload_rows,"windows":windows,"runtime_seconds":time.perf_counter()-started,"independent_external_sampler":True,"ncu_frequency_features_consumed":False,"limitations":["host sampler is asynchronous and 100-ms scale; it is not launch-synchronized","process windows are profiler-tool windows, not NCU internal semaphore intervals"]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+"\n"); return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign',type=Path,required=True); p.add_argument('--samples',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--anchor-clock-mhz',type=float,default=1350.0); p.add_argument('--high-threshold',type=float,default=1.15); p.add_argument('--high-fraction-threshold',type=float,default=0.10); a=p.parse_args(); r=run(a); print(json.dumps({"regime":r['regime'],"active_window_clock_mhz":r['active_window_clock_mhz'],"active_window_high_clock_fraction":r['active_window_high_clock_fraction'],"sampler_sample_count":r['sampler_sample_count'],"profiler_window_sample_count":r['profiler_window_sample_count'],"output":str(a.output)},indent=2))


if __name__=='__main__': main()
