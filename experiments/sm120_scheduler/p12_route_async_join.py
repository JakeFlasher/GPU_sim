#!/usr/bin/env python3
"""Join existing exact-scope route/async evidence into logical P12 rows."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--timing',type=Path,required=True);p.add_argument('--score',type=Path,required=True);p.add_argument('--service',type=Path,required=True);p.add_argument('--async-ledger',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 timing=json.loads(a.timing.read_text());score=json.loads(a.score.read_text());service=json.loads(a.service.read_text());ledger=json.loads(a.async_ledger.read_text())
 times={r['workload']:r for r in timing.get('rows',timing.get('timing',[])) if isinstance(r,dict) and r.get('workload')}
 score_rows=score.get('rows',score.get('families',score.get('prediction_rows',[])))
 score_map={r.get('workload',r.get('family',r.get('capture_name'))):r for r in score_rows if isinstance(r,dict)}
 service_map={r.get('capture_name'):r for r in service.get('prediction_rows',[]) if isinstance(r,dict)}
 async_status='unknown'
 for capture in ledger.get('captures',[]) if isinstance(ledger,dict) else []:
  if isinstance(capture,dict) and capture.get('payload_observation_status'):
   async_status=capture['payload_observation_status']
 rows=[]
 names=['cp_async_shared_tile_copy','dram_streaming_dot_product','global_vectorized_128b_copy','shared_bank_conflict_stride','tma_tensor_copy_2d','warp_divergence_reconvergence']
 for name in names:
  rows.append({'family':name,'timing':times.get(name,{}),'score_diagnostic':score_map.get(name,{}),'logical_service_rows':[r for r in service.get('prediction_rows',[]) if r.get('capture_name')==name],'async_payload_status':'observed_after_kernel_sync' if name=='tma_tensor_copy_2d' and async_status=='observed_after_kernel_sync' else ('not_applicable' if name!='tma_tensor_copy_2d' else async_status),'physical_non_claim':['cache_set_hash','associativity','MSHR_count','native_TMA_completion_timestamp','mbarrier_visibility_instant']})
 out={'schema_version':1,'model':'sm120_p12_route_async_join_v1','rows':rows,'ncu_used_as_input':False,'interpretation':'Logical route/service and async ledger observations are joined by family; NCU remains an independent comparison domain.'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(rows),'tma_payload_status':async_status}))
if __name__=='__main__':main()
