#!/usr/bin/env python3
"""Execute one-launch scope-aligned P10 timing/NCU/NVBit matrix."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
CASES=[('p10_footprint_u64',['fma','4','64','64','1','1','1','1','0']),('p10_footprint_u256',['fma','4','256','16','1','1','1','1','0']),('p10_footprint_u1024',['fma','4','1024','4','1','1','1','1','0']),('p10_select_w0',['fma','4','16','256','16','1','1','1','0']),('p10_select_w3',['fma','4','16','256','16','8','1','1','0']),('p10_select_w7',['fma','4','16','256','16','128','1','1','0']),('p10_select_w12',['fma','4','16','256','16','4096','1','1','0']),('p10_pair_low',['fma','4','16','256','16','3','1','1','0']),('p10_pair_high',['fma','4','16','256','16','49152','1','1','0']),('p10_pair_interleave',['fma','4','16','256','16','195','1','1','0'])]

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path('/root/GPU_simulator_rebase'));p.add_argument('--out-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['timing','ncu','nvbit'],required=True);p.add_argument('--ncu',type=Path,default=Path('/usr/local/cuda-13.2/bin/ncu'));a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True);rows=[]
 tr=a.repo/'tools/nvbit_tracer/tracer_tool/tracer_tool.so';pb=a.repo/'.toolchain/pb/lib'
 for n,args in CASES:
  if a.mode=='timing':
   t=time.perf_counter_ns();q=subprocess.run([str(a.binary),*args],capture_output=True,text=True,timeout=180,check=False);obs=[json.loads(x) for x in q.stdout.splitlines() if x.startswith('{')];row={'name':n,'args':args,'returncode':q.returncode,'elapsed_ns':time.perf_counter_ns()-t,'observations':obs}
  elif a.mode=='ncu':
   raw=a.out_root/(n+'.raw.csv');cmd=[str(a.ncu),'--csv','--page','raw','--print-units','base','--profile-from-start','off','--target-processes','all','--launch-count','1','--metrics','gpu__time_duration.sum,sm__cycles_elapsed.avg,smsp__inst_executed.sum,smsp__inst_issued.sum,smsp__warps_active.sum,smsp__warps_eligible.sum,smsp__warp_issue_stalled_not_selected_per_warp_active,smsp__warp_issue_stalled_short_scoreboard_per_warp_active,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active,smsp__warp_issue_stalled_barrier_per_warp_active','--log-file',str(raw),str(a.binary),*args];q=subprocess.run(cmd,capture_output=True,text=True,timeout=300,check=False);row={'name':n,'args':args,'returncode':q.returncode,'raw_csv':str(raw),'raw_csv_present':raw.is_file() and raw.stat().st_size>0};(a.out_root/(n+'.stdout')).write_text(q.stdout or '');(a.out_root/(n+'.stderr')).write_text(q.stderr or '')
  else:
   dst=a.out_root/n;dst.mkdir(parents=True,exist_ok=True);e=os.environ.copy();e.update({'CUDA_INJECTION64_PATH':str(tr),'USER_DEFINED_FOLDERS':'1','TRACES_FOLDER':str(dst),'ACTIVE_FROM_START':'1','NVBIT_ALPHA_NO_HASH':'1','SM120_MEMORY_TENSOR_ONLY':'1','SM120_CAPTURE_OPERANDS':'0','SM120_ALLOW_UNREGISTERED_GLOBALS':'0','SKIP_ENHANCED_TRACER':'1'});e['LD_LIBRARY_PATH']=':'.join([str(pb),'/usr/local/cuda-13.2/targets/x86_64-linux/lib',e.get('LD_LIBRARY_PATH','')]);q=subprocess.run([str(a.binary),*args],cwd=a.repo,env=e,capture_output=True,text=True,timeout=300,check=False);row={'name':n,'args':args,'returncode':q.returncode,'capture_dir':str(dst),'dynamic_trace_present':(dst/'dynamic_trace.pb').is_file()}
  rows.append(row);print(n,q.returncode,row.get('raw_csv_present',row.get('dynamic_trace_present',len(row.get('observations',[])))),flush=True)
 r={'schema_version':1,'model':f'sm120_p10_scope_{a.mode}_v1','rows':rows,'mode':a.mode,'scope_contract':'one measured launch'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows),'mode':a.mode,'success':sum(x['returncode']==0 for x in rows)}))
if __name__=='__main__':main()
