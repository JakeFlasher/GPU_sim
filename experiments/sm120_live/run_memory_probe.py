#!/usr/bin/env python3
"""Collect one-launch timing/NCU/NVBit observations for SM120 memory routes."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
MODES=[(0,'ld_ca'),(1,'ld_cg'),(2,'load_store'),(3,'store32'),(4,'store8'),(5,'atom_return'),(6,'red'),(7,'cp_async_g1'),(8,'cp_async_g2'),(9,'cp_async_g4')]
METRICS='gpu__time_duration.sum,sm__cycles_elapsed.avg,smsp__inst_executed.sum,smsp__inst_issued.sum,smsp__warps_active.sum,smsp__warps_eligible.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,dram__sectors_op_read.sum,dram__sectors_op_write.sum'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode',choices=['timing','ncu','nvbit'],required=True);ap.add_argument('--repo',type=Path,default=Path('/root/GPU_simulator_rebase'));ap.add_argument('--ncu',type=Path,default=Path('/usr/local/cuda-13.2/bin/ncu'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);rows=[]; tracer=args.repo/'tools/nvbit_tracer/tracer_tool/tracer_tool.so'; pb=args.repo/'.toolchain/pb/lib'
 for mode,name in MODES:
  # one 32-thread block, 4K-word footprint, 64 iterations; atom/red use same address.
  argv=[str(args.binary),str(mode),'4096','64','1','0','32','1']
  if args.mode=='timing':
   q=subprocess.run(argv,capture_output=True,text=True,timeout=120,check=False); obs=[json.loads(x) for x in q.stdout.splitlines() if x.startswith('{')]; row={'mode':mode,'name':name,'argv':argv,'returncode':q.returncode,'observations':obs,'stderr_tail':q.stderr[-1000:]}
  elif args.mode=='ncu':
   raw=args.out/(name+'.raw.csv'); cmd=[str(args.ncu),'--csv','--page','raw','--print-units','base','--profile-from-start','off','--target-processes','all','--launch-count','1','--metrics',METRICS,'--log-file',str(raw),*argv];q=subprocess.run(cmd,capture_output=True,text=True,timeout=240,check=False);row={'mode':mode,'name':name,'argv':argv,'returncode':q.returncode,'raw_csv':str(raw),'raw_present':raw.is_file() and raw.stat().st_size>0};(args.out/(name+'.stdout')).write_text(q.stdout or '');(args.out/(name+'.stderr')).write_text(q.stderr or '')
  else:
   dst=args.out/name;dst.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env.update({'CUDA_INJECTION64_PATH':str(tracer),'USER_DEFINED_FOLDERS':'1','TRACES_FOLDER':str(dst),'ACTIVE_FROM_START':'1','NVBIT_ALPHA_NO_HASH':'1','SM120_MEMORY_TENSOR_ONLY':'1','SM120_CAPTURE_OPERANDS':'1','SM120_ALLOW_UNREGISTERED_GLOBALS':'0','SKIP_ENHANCED_TRACER':'1'});env['LD_LIBRARY_PATH']=':'.join([str(pb),'/usr/local/cuda-13.2/targets/x86_64-linux/lib',env.get('LD_LIBRARY_PATH','')]);q=subprocess.run(argv,cwd=args.repo,env=env,capture_output=True,text=True,timeout=240,check=False);row={'mode':mode,'name':name,'argv':argv,'returncode':q.returncode,'capture_dir':str(dst),'trace_present':(dst/'dynamic_trace.pb').is_file(),'stderr_tail':q.stderr[-1000:]}
  rows.append(row);print(name,row['returncode'],flush=True)
 result={'schema_version':1,'model':f'sm120_live_memory_probe_{args.mode}_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'rows':rows,'scope':'one 32-thread launch per mode; separate timing/NCU/NVBit processes','policy':['calibration/evidence only','full replay first when traces exist','no NCU targets passed into model','Accel-Sim banned']};(args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'mode':args.mode,'rows':len(rows),'success':sum(x['returncode']==0 for x in rows)}))
if __name__=='__main__':main()
