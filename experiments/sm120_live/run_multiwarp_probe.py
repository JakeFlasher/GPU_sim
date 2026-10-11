#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
MODES=[(3,'store32'),(4,'store8'),(5,'atom_return'),(6,'red')]
METRICS='gpu__time_duration.sum,sm__cycles_elapsed.avg,smsp__inst_executed.sum,smsp__inst_issued.sum,smsp__warps_active.sum,smsp__warps_eligible.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,dram__sectors_op_read.sum,dram__sectors_op_write.sum'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode',choices=['timing','ncu','nvbit'],required=True);ap.add_argument('--repo',type=Path,default=Path('/root/GPU_simulator_rebase'));ap.add_argument('--ncu',type=Path,default=Path('/usr/local/cuda-13.2/bin/ncu'));a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[];tr=a.repo/'tools/nvbit_tracer/tracer_tool/tracer_tool.so';pb=a.repo/'.toolchain/pb/lib'
 for m,n in MODES:
  argv=[str(a.binary),str(m),'4096','32','1','0','128','1']
  if a.mode=='timing':q=subprocess.run(argv,capture_output=True,text=True,timeout=120,check=False);row={'name':n,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
  elif a.mode=='ncu':
   raw=a.out/(n+'.raw.csv');q=subprocess.run([str(a.ncu),'--csv','--page','raw','--print-units','base','--profile-from-start','off','--target-processes','all','--launch-count','1','--metrics',METRICS,'--log-file',str(raw),*argv],capture_output=True,text=True,timeout=240,check=False);row={'name':n,'returncode':q.returncode,'raw_csv':str(raw),'raw_present':raw.is_file() and raw.stat().st_size>0}
  else:
   dst=a.out/n;dst.mkdir(parents=True,exist_ok=True);e=os.environ.copy();e.update({'CUDA_INJECTION64_PATH':str(tr),'USER_DEFINED_FOLDERS':'1','TRACES_FOLDER':str(dst),'ACTIVE_FROM_START':'1','NVBIT_ALPHA_NO_HASH':'1','SM120_MEMORY_TENSOR_ONLY':'1','SM120_CAPTURE_OPERANDS':'1','SM120_ALLOW_UNREGISTERED_GLOBALS':'0','SKIP_ENHANCED_TRACER':'1'});e['LD_LIBRARY_PATH']=':'.join([str(pb),'/usr/local/cuda-13.2/targets/x86_64-linux/lib',e.get('LD_LIBRARY_PATH','')]);q=subprocess.run(argv,cwd=a.repo,env=e,capture_output=True,text=True,timeout=240,check=False);row={'name':n,'returncode':q.returncode,'capture_dir':str(dst),'trace_present':(dst/'dynamic_trace.pb').is_file()}
  rows.append(row);print(n,row['returncode'],flush=True)
 (a.out/'summary.json').write_text(json.dumps({'schema_version':1,'model':f'sm120_q11_multiwarp_{a.mode}_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'shape':'one 128-thread block (four warps), 32 rounds','rows':rows},indent=2)+'\n')
if __name__=='__main__':main()
