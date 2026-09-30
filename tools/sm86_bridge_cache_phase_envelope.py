#!/usr/bin/env python3
"""Derive a conservative warm/cold endpoint envelope for bridged bodies."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path

def n(v):
 try:return float(v)
 except:return 0.0

def read_last(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))[-1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--family-matrix',type=Path,required=True);p.add_argument('--mixed-cold',type=Path,required=True);p.add_argument('--mixed-warm',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.family_matrix.open(newline='') as f: rows=list(csv.DictReader(f))
 duration=[n(r['warm_cold_duration_ratio']) for r in rows]; gpc=[n(r['warm_cold_gpc_ratio']) for r in rows]
 cold=read_last(a.mixed_cold);warm=read_last(a.mixed_warm)
 obs={'duration_ratio':n(warm['gpu__time_duration.sum'])/n(cold['gpu__time_duration.sum']),'gpc_ratio':n(warm['gpc__cycles_elapsed.max'])/n(cold['gpc__cycles_elapsed.max'])}
 med={'duration_ratio':statistics.median(duration),'gpc_ratio':statistics.median(gpc)}
 pred={'duration_ns':n(cold['gpu__time_duration.sum'])*1000*med['duration_ratio'],'gpc_cycles':n(cold['gpc__cycles_elapsed.max'])*med['gpc_ratio']}
 actual={'duration_ns':n(warm['gpu__time_duration.sum'])*1000,'gpc_cycles':n(warm['gpc__cycles_elapsed.max'])}
 out={'schema':'sm86_bridge_cache_phase_envelope_v1','independent_family_count':len(rows),'family_ratio_envelope':{'duration_ratio':{'min':min(duration),'max':max(duration),'median':med['duration_ratio']},'gpc_ratio':{'min':min(gpc),'max':max(gpc),'median':med['gpc_ratio']}},'mixed_observation':{'cold':cold,'warm':warm,'observed_ratios':obs,'median_transfer_prediction':pred,'warm_actual':actual,'prediction_ape':{k:abs(pred[k]-actual[k])/actual[k] for k in pred},'target_warm_excluded_from_family_fit':True},'limitations':['phase ratio is endpoint-specific and body-local','does not identify cache ownership/read-from state','aggregate family labels are not a general ABI model']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
