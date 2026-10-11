#!/usr/bin/env python3
"""Collect uninstrumented CUDA-event timing for the held-out scheduler cases."""
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path
CASES=[('heldout_fma_k8',['fma','8','16','1024','1','1','1','3','2']),('heldout_mad_k8',['mad','8','16','1024','1','1','1','3','2']),('heldout_mix_k8',['mix','8','16','1024','1','1','1','3','2']),('heldout_warp_density_w16_small',['fma','4','16','256','16','65535','1','3','2']),('heldout_barrier_w16',['barrier','1','16','128','16','65535','1','3','2']),('heldout_footprint_u256',['fma','4','256','16','1','1','1','3','2'])]
def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
 for n,args in CASES:
  t=time.perf_counter_ns();q=subprocess.run([str(a.binary),*args],capture_output=True,text=True,timeout=120,check=False);obs=[json.loads(x) for x in q.stdout.splitlines() if x.startswith('{')];rows.append({'name':n,'args':args,'returncode':q.returncode,'elapsed_ns':time.perf_counter_ns()-t,'observations':obs,'stdout_tail':q.stdout[-2000:],'stderr_tail':q.stderr[-1000:]});print(n,q.returncode,len(obs),flush=True)
 r={'schema_version':1,'model':'sm120_scheduler_heldout_timing_v1','rows':rows,'timing_source':'uninstrumented_binary_cuda_event_and_clock64'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
