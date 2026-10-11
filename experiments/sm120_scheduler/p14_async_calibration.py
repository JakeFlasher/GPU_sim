#!/usr/bin/env python3
"""P14 independent async calibration matrix; full replay is the only model path."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
CASES=[('p14_cp32','cp_async_shared_tile_copy',['--rows','32','--cols','32','--seed','31','--shape-point','p14_cp32']),('p14_cp128','cp_async_shared_tile_copy',['--rows','128','--cols','128','--seed','37','--shape-point','p14_cp128']),('p14_tma64','tma_tensor_copy_2d',['--rows','64','--cols','64','--seed','41','--shape-point','p14_tma64']),('p14_tma256','tma_tensor_copy_2d',['--rows','256','--cols','256','--seed','43','--shape-point','p14_tma256'])]
METRICS='gpu__time_duration.sum,sm__cycles_elapsed.avg,smsp__inst_executed.sum,smsp__inst_issued.sum,smsp__warps_active.sum,smsp__warps_eligible.sum,dram__sectors_op_read.sum,dram__sectors_op_write.sum,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum'
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('/root/GPU_simulator_rebase'));p.add_argument('--out-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['timing','ncu','nvbit'],required=True);p.add_argument('--ncu',type=Path,default=Path('/usr/local/cuda-13.2/bin/ncu'));a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True);rows=[];tr=a.repo/'tools/nvbit_tracer/tracer_tool/tracer_tool.so';pb=a.repo/'.toolchain/pb/lib'
 for n,stem,args in CASES:
  binary=a.repo/'workloads/simbench_bw/l1'/stem/(stem+'_run')
  if a.mode=='timing':
   e=os.environ.copy();e['SM120_EVENT_REPS']='1';q=subprocess.run([str(binary),*args],cwd=a.repo,env=e,capture_output=True,text=True,timeout=300,check=False);markers=[]
   for line in q.stdout.splitlines():
    if line.startswith('SM120_'):markers.append(line)
   rows.append({'name':n,'workload':stem,'args':args,'returncode':q.returncode,'markers':markers,'stdout_tail':q.stdout[-1000:]})
  elif a.mode=='ncu':
   raw=a.out_root/(n+'.raw.csv');cmd=[str(a.ncu),'--csv','--page','raw','--print-units','base','--profile-from-start','off','--target-processes','all','--launch-count','1','--metrics',METRICS,'--log-file',str(raw),str(binary),*args];q=subprocess.run(cmd,capture_output=True,text=True,timeout=400,check=False);rows.append({'name':n,'workload':stem,'args':args,'returncode':q.returncode,'raw_csv':str(raw),'raw_present':raw.is_file() and raw.stat().st_size>0})
  else:
   dst=a.out_root/n;dst.mkdir(parents=True,exist_ok=True);e=os.environ.copy();e.update({'CUDA_INJECTION64_PATH':str(tr),'USER_DEFINED_FOLDERS':'1','TRACES_FOLDER':str(dst),'ACTIVE_FROM_START':'1','NVBIT_ALPHA_NO_HASH':'1','SM120_MEMORY_TENSOR_ONLY':'1','SM120_CAPTURE_OPERANDS':'0','SM120_ALLOW_UNREGISTERED_GLOBALS':'0','SKIP_ENHANCED_TRACER':'1'});e['LD_LIBRARY_PATH']=':'.join([str(pb),'/usr/local/cuda-13.2/targets/x86_64-linux/lib',e.get('LD_LIBRARY_PATH','')]);q=subprocess.run([str(binary),*args],cwd=a.repo,env=e,capture_output=True,text=True,timeout=400,check=False);rows.append({'name':n,'workload':stem,'args':args,'returncode':q.returncode,'capture_dir':str(dst),'trace_present':(dst/'dynamic_trace.pb').is_file()})
  print(n,rows[-1]['returncode'],flush=True)
 r={'schema_version':1,'model':f'sm120_p14_async_calibration_{a.mode}_v1','rows':rows,'mode':a.mode,'role':'calibration_only_not_evaluation'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows),'success':sum(x['returncode']==0 for x in rows)}))
if __name__=='__main__':main()
