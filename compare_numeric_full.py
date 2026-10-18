from __future__ import annotations
import collections,json,sys,math
from pathlib import Path
from typing import Any
sys.path.insert(0,'tools'); from jsonl_io import open_jsonl_reader

def rows(path):
 d={}
 with open_jsonl_reader(Path(path)) as f:
  for line in f:
   r=json.loads(line); d[str(r['canonical_capture']).replace('/', '__')]=r
 return d

def ignore(path):
 if not path:return False
 n=path[-1]
 if n in {'schema_version','reference_clock_hz','service_rate_bytes_per_sec','sim_total_wall_ns','sim_replay_wall_ns','driver_wall_ms'}: return True
 if n.endswith('_wall_ns') or n.endswith('_wall_ms'):return True
 if n in {'capture_dir','replay_environment_removed','semantics','detail','reason','role','status','unit','ncu_metric','key','arch','source_key','model_status','model_unit','comparison_target_unit','profile_id','scope','observation_scope','configuration_source','non_claim','timing_scope','geometry_status','partition_mapping_status','mshr_count','queue_policy','residency','outstanding_demand','dram_service','issue','transaction_complete','barrier_arrival','wait_or_test','destination_visibility','payload_observation','cache_scope','memory_mode','architecture','carded_dram_bandwidth','sm120_alpha_profile_id','sm120_alpha','simulator_version','canonical_capture','schema_version'}: return True
 # Ignore new observer arrays/counters whose absence is expected across branch output schemas.
 if any(x in n for x in ('request_mask_audit','youngest_priority_order_mismatches','pending_pressure','store_packet_candidate','pending_merge_sectors','new_fill_sectors','l1_unrequested_subline','wavefront_mask_lookup','constant_memory_','local_canonical','local_frame256','local_payload','atomic_conflict','shared_ideal','shared_legacy','tma_shadow','sync_sidecar','global_store_adjacent','slice_accesses')): return True
 return False

def num(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(float(v))
def rec(a,b,path,out):
 if ignore(path):return
 if isinstance(a,dict) and isinstance(b,dict):
  for k in set(a)&set(b): rec(a[k],b[k],path+(str(k),),out)
 elif isinstance(a,list) and isinstance(b,list):
  if all(isinstance(x,dict) and 'key' in x for x in a+b):
   am={str(x['key']):x for x in a}; bm={str(x['key']):x for x in b}
   for k in set(am)&set(bm):rec(am[k],bm[k],path+(k,),out)
  else:
   for i,(x,y) in enumerate(zip(a,b)):rec(x,y,path+(str(i),),out)
 elif num(a) and num(b) and a!=b:
  out.append((path,a,b))

def run(label,b,u):
 base=rows(b); uni=rows(u); c=collections.Counter(); ex={}; rowsdiff=0
 for k in sorted(set(base)&set(uni)):
  ds=[];rec(base[k],uni[k],(),ds)
  if ds:rowsdiff+=1
  for p,a,x in ds:
   key='.'.join(p);c[key]+=1;ex.setdefault(key,(k,a,x))
 return {'label':label,'rows':len(base),'common_rows':len(set(base)&set(uni)),'rows_with_numeric_diffs':rowsdiff,'numeric_diff_count':sum(c.values()),'top_numeric_diffs':[{'path':p,'count':n,'example':ex[p]} for p,n in c.most_common(100)]}
ps=[('sm120_disabled','/home/jakeshea/5090_branch/final_results/dram_disabled/full.jsonl.zst','full_replay_runs/fix3_sm120_dram_disabled/full.jsonl.zst'),('sm120_enabled','/home/jakeshea/5090_branch/final_results/dram_enabled/full.jsonl.zst','full_replay_runs/fix3_sm120_dram_enabled/full.jsonl.zst'),('sm86','/home/jakeshea/5090_branch/final_results/3090ti/full.jsonl.zst','full_replay_runs/fix6_sm86_full/full.jsonl.zst')]
res=[]
for x,b,u in ps:
 print('run',x,file=sys.stderr);res.append(run(x,b,u))
Path('FULL_NUMERIC_COMPARISON.json').write_text(json.dumps(res,indent=2)+'\n')
for r in res: print(json.dumps({k:r[k] for k in ('label','rows','common_rows','rows_with_numeric_diffs','numeric_diff_count')},indent=2)); print(json.dumps(r['top_numeric_diffs'][:30],indent=2))
