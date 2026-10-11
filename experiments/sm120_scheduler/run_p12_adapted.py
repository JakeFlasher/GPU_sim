#!/usr/bin/env python3
"""Adapt existing legal simbench route/async binaries to one-launch P12 scope."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
CASES=[('global_vectorized_128b_copy','global_vectorized_128b_copy'),('dram_streaming_dot_product','dram_streaming_dot_product'),('shared_bank_conflict_stride','shared_bank_conflict_stride'),('cp_async_shared_tile_copy','cp_async_shared_tile_copy'),('tma_tensor_copy_2d','tma_tensor_copy_2d'),('warp_divergence_reconvergence','warp_divergence_reconvergence')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('/root/GPU_simulator_rebase'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
 for name,stem in CASES:
  binary=a.repo/'workloads/simbench_bw/l1'/name/(stem+'_run');src=a.repo/'workloads/simbench_bw/l1'/name/(stem+'.cu');main_src=a.repo/'workloads/simbench_bw/l1'/name/'main.cu';env=os.environ.copy();env['SM120_EVENT_REPS']='1';t=time.perf_counter_ns();q=subprocess.run([str(binary)],cwd=a.repo,env=env,capture_output=True,text=True,timeout=180,check=False);out=(q.stdout or '')[-8000:];(a.out/(name+'.stdout')).write_text(out);(a.out/(name+'.stderr')).write_text(q.stderr or '');rows.append({'family':name,'binary':str(binary),'source':str(src),'main_source':str(main_src),'returncode':q.returncode,'elapsed_ns':time.perf_counter_ns()-t,'stdout_tail':out,'stderr_tail':(q.stderr or '')[-2000:]});print(name,q.returncode,flush=True)
 r={'schema_version':1,'model':'sm120_p12_adapted_legal_workloads_v1','rows':rows,'scope':'one event-repetition; existing legal route/async binaries','ncu_used_as_input':False};a.out.parent.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows),'success':sum(x['returncode']==0 for x in rows)}))
if __name__=='__main__':main()
