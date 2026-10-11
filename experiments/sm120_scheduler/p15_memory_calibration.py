#!/usr/bin/env python3
"""Independent full-replay memory-service calibration matrix."""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
CASES=[('p15_scalar32','global_scalar_32b_copy',['--samples','4096','--bins','4','--seed','31','--shape-point','p15_scalar32']),('p15_vector64','global_vectorized_64b_copy',['--samples','2048','--bins','4','--seed','37','--shape-point','p15_vector64']),('p15_shared_stride2','shared_bank_conflict_stride',['--rows','64','--cols','64','--stride','2','--seed','41','--shape-point','p15_shared_stride2']),('p15_shared_stride4','shared_bank_conflict_stride',['--rows','64','--cols','64','--stride','4','--seed','43','--shape-point','p15_shared_stride4']),('p15_dram_mix1','dram_stream_stride_mix',['--warps','8','--accesses','32','--stride-classes','1','--seed','47','--shape-point','p15_dram_mix1']),('p15_dram_mix4','dram_stream_stride_mix',['--warps','8','--accesses','32','--stride-classes','4','--seed','53','--shape-point','p15_dram_mix4'])]
METRICS='gpu__time_duration.sum,sm__cycles_elapsed.avg,smsp__inst_executed.sum,smsp__inst_issued.sum,smsp__warps_active.sum,smsp__warps_eligible.sum,l1tex__t_sectors.sum,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,dram__sectors_op_read.sum,dram__sectors_op_write.sum'
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('/root/GPU_simulator_rebase'));p.add_argument('--out-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['timing','ncu','nvbit'],required=True);p.add_argument('--ncu',type=Path,default=Path('/usr/local/cuda-13.2/bin/ncu'));a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True);rows=[];tr=a.repo/'tools/nvbit_tracer/tracer_tool/tracer_tool.so';pb=a.repo/'.toolchain/pb/lib'
 for n,stem,args in CASES:
  binary=a.repo/'workloads/simbench_bw/l1'/stem/(stem+'_run')
  if a.mode=='timing':
   e=os.environ.copy();e['SM120_EVENT_REPS']='1';q=subprocess.run([str(binary),*args],cwd=a.repo,env=e,capture_output=True,text=True,timeout=300,check=False);rows.append({'name':n,'workload':stem,'args':args,'returncode':q.returncode,'stdout_tail':q.stdout[-3000:],'stderr_tail':q.stderr[-1000:]})
  elif a.mode=='ncu':
   raw=a.out_root/(n+'.raw.csv');cmd=[str(a.ncu),'--csv','--page','raw','--print-units','base','--target-processes','all','--metrics',METRICS,'--log-file',str(raw),str(binary),*args];q=subprocess.run(cmd,capture_output=True,text=True,timeout=500,check=False);rows.append({'name':n,'workload':stem,'args':args,'returncode':q.returncode,'raw_csv':str(raw),'raw_present':raw.is_file() and raw.stat().st_size>0});(a.out_root/(n+'.stdout')).write_text(q.stdout or '');(a.out_root/(n+'.stderr')).write_text(q.stderr or '')
  else:
   dst=a.out_root/n;dst.mkdir(parents=True,exist_ok=True);e=os.environ.copy();e.update({'CUDA_INJECTION64_PATH':str(tr),'USER_DEFINED_FOLDERS':'1','TRACES_FOLDER':str(dst),'ACTIVE_FROM_START':'1','NVBIT_ALPHA_NO_HASH':'1','SM120_MEMORY_TENSOR_ONLY':'1','SM120_CAPTURE_OPERANDS':'0','SM120_ALLOW_UNREGISTERED_GLOBALS':'0','SKIP_ENHANCED_TRACER':'1'});e['LD_LIBRARY_PATH']=':'.join([str(pb),'/usr/local/cuda-13.2/targets/x86_64-linux/lib',e.get('LD_LIBRARY_PATH','')]);q=subprocess.run([str(binary),*args],cwd=a.repo,env=e,capture_output=True,text=True,timeout=500,check=False);rows.append({'name':n,'workload':stem,'args':args,'returncode':q.returncode,'capture_dir':str(dst),'trace_present':(dst/'dynamic_trace.pb').is_file()})
  print(n,q.returncode,flush=True)
 r={'schema_version':1,'model':f'sm120_p15_memory_calibration_{a.mode}_v1','rows':rows,'mode':a.mode,'role':'calibration_only_not_evaluation','fast_path_used':False};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows),'success':sum(x['returncode']==0 for x in rows)}))
if __name__=='__main__':main()
