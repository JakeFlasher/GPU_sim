from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0,'tools')
from jsonl_io import open_jsonl_reader

def rows(path):
    with open_jsonl_reader(Path(path)) as f:
        return [json.loads(line) for line in f]
def by_name(path, name):
    for row in rows(path):
        if row.get('canonical_capture') == name or str(row.get('canonical_capture','')).endswith('/'+name):
            return row
    raise KeyError((path,name))
def select(row):
    launch=row.get('launches') or []
    # Keep a compact, behavior-relevant comparison surface.
    out={'status':row.get('status'),'architecture':row.get('architecture'),'memory_mode':row.get('memory_mode'),
         'launch_count':len(launch),'schema_version':row.get('schema_version')}
    if launch:
        l=launch[0]; c=l.get('counters',{}); out.update({
          'instructions_issued':c.get('instructions_issued'),'records_seen':c.get('records_seen'),
          'front_end':c.get('front_end'), 'l1': {k:c.get('l1',{}).get(k) for k in ('hits','misses','merges','line_fills','store_sectors')} if c.get('l1') else None,
          'l2': {k:c.get('l2',{}).get(k) for k in ('hits','misses','lookup_hits','lookup_requests')} if c.get('l2') else None,
          'dram': {k:c.get('dram',{}).get(k) for k in ('read_sectors','service_rate_bytes_per_sec')} if c.get('dram') else None,
          'timing_status': (l.get('timing') or {}).get('completion_frontier',{}).get('status'),
          'metric_prediction_count':len(l.get('metric_predictions',[])),
        })
    return out
pairs=[
 ('sm86 cp.async','cp_async_shared_tile_copy','/home/jakeshea/5090_branch/final_results/3090ti/full.jsonl.zst','verification_outputs/sm86_cp_async_v4/full.jsonl.zst'),
 ('sm120 TMA','simbench_l1_tma_tensor_copy_2d','/home/jakeshea/5090_branch/final_results/dram_enabled/full.jsonl.zst','verification_outputs/sm120_tma_v5/full.jsonl.zst'),
 ('sm86 scalar','simbench_l1_global_scalar_32b_copy','/home/jakeshea/5090_branch/final_results/3090ti/full.jsonl.zst','verification_outputs/sm86_scalar_v4/full.jsonl.zst'),
 ('sm120 scalar','simbench_l1_global_scalar_32b_copy','/home/jakeshea/5090_branch/final_results/dram_enabled/full.jsonl.zst','verification_outputs/sm120_scalar_v5/full.jsonl.zst'),
]
lines=['# Unified replay verification','', 'Comparisons use copied real traces under `verification_inputs/`; source capture and final-result trees were not modified.', '', '| Case | Baseline | Unified | Selected counters equal? |', '|---|---|---|---|']
for label,name,base,unified in pairs:
    b=select(by_name(base,name)); u=select(by_name(unified,name));
    keys=('status','architecture','memory_mode','launch_count','instructions_issued','records_seen','front_end','l1','l2','dram','timing_status','metric_prediction_count')
    equal=all(b.get(k)==u.get(k) for k in keys if k not in ('dram','metric_prediction_count'))
    equal = equal and b.get('dram',{}).get('read_sectors') == u.get('dram',{}).get('read_sectors')
    if b.get('architecture') == 'sm_86':
        equal = equal and b.get('dram',{}).get('service_rate_bytes_per_sec') == u.get('dram',{}).get('service_rate_bytes_per_sec')
    equal = equal and b.get('metric_prediction_count') == u.get('metric_prediction_count')
    lines.append(f'| {label} | `{b["status"]}/{b["architecture"]}` | `{u["status"]}/{u["architecture"]}` | {"yes" if equal else "no (see details)"} |')
    lines += ['', f'### {label}', '', '```json', json.dumps({'baseline':b,'unified':u},indent=2,sort_keys=True), '```', '']
Path('VERIFICATION_REPORT.md').write_text('\n'.join(lines))
