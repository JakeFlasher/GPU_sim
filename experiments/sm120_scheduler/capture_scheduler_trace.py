#!/usr/bin/env python3
"""Capture a compact scheduler-probe matrix with the existing NVBit producer."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path

CASES=[
 ('heldout_fma_k8',['fma','8','16','1024','1','1','1','1','2']),
 ('heldout_mad_k8',['mad','8','16','1024','1','1','1','1','2']),
 ('heldout_mix_k8',['mix','8','16','1024','1','1','1','1','2']),
 ('heldout_warp_density_w16',['fma','4','16','2048','16',str((1<<16)-1),'1','1','2']),
 ('heldout_barrier_w16',['barrier','1','16','128','16',str((1<<16)-1),'1','1','2']),
 ('heldout_footprint_u256',['fma','4','256','16','1','1','1','1','2']),
]

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path('/root/GPU_simulator_rebase'));p.add_argument('--out-root',type=Path,required=True);p.add_argument('--timeout',type=float,default=300);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True)
 tracer=a.repo/'tools/nvbit_tracer/tracer_tool/tracer_tool.so';pb=a.repo/'.toolchain/pb/lib';rows=[]
 for name,args in CASES:
  dst=a.out_root/name;dst.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env.update({'CUDA_INJECTION64_PATH':str(tracer),'USER_DEFINED_FOLDERS':'1','TRACES_FOLDER':str(dst),'ACTIVE_FROM_START':'1','NVBIT_ALPHA_NO_HASH':'1','SM120_MEMORY_TENSOR_ONLY':'1','SM120_CAPTURE_OPERANDS':'0','SM120_ALLOW_UNREGISTERED_GLOBALS':'0','SKIP_ENHANCED_TRACER':'1'});env['LD_LIBRARY_PATH']=':'.join([str(pb),'/usr/local/cuda-13.2/targets/x86_64-linux/lib',env.get('LD_LIBRARY_PATH','')]);cmd=[str(a.binary),*args];t=time.perf_counter_ns()
  try:q=subprocess.run(cmd,cwd=a.repo,env=env,capture_output=True,text=True,timeout=a.timeout,check=False);to=False
  except subprocess.TimeoutExpired as e:
   stdout=e.stdout.decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or '')
   stderr=e.stderr.decode(errors='replace') if isinstance(e.stderr,bytes) else (e.stderr or '')
   q=subprocess.CompletedProcess(cmd,124,stdout,stderr);to=True
  row={'name':name,'args':args,'command':cmd,'returncode':q.returncode,'timed_out':to,'elapsed_ns':time.perf_counter_ns()-t,'capture_dir':str(dst),'dynamic_trace_present':(dst/'dynamic_trace.pb').is_file(),'stdout_tail':(q.stdout or '')[-3000:],'stderr_tail':(q.stderr or '')[-3000:]};rows.append(row);print(name,q.returncode,row['dynamic_trace_present'],flush=True)
 r={'schema_version':1,'model':'sm120_scheduler_nvbit_capture_v1','rows':rows,'trace_count':sum(x['dynamic_trace_present'] for x in rows)};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows),'trace_count':r['trace_count']}))
if __name__=='__main__':main()
