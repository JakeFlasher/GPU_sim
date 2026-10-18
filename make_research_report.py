from __future__ import annotations
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path('/home/jakeshea/5090_branch'); OUT=Path('RESEARCH_COMPARISON.md')
A=ROOT/'GPU_simulator'; B=ROOT/'GPU_simulator_5090'; BASE='f3ca5b2f0d3b12ec9c720df408de7908810d8617'

def git_tree(repo: Path):
    out=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','--full-tree','HEAD'],text=True)
    d={}
    for line in out.splitlines():
        meta,path=line.split('\t',1); mode,typ,oid=meta.split()
        d[path]=(oid,mode,typ)
    return d
A_T=git_tree(A); B_T=git_tree(B)
def ignored(p):
    low=p.lower()
    return (p.startswith('paper/') or p.startswith('references/papers/') or p.startswith('tools/sol_execbench/.venv/') or p.startswith('.claude/skills/')
            or any(token in low for token in ('thesis', 'manuscript')))
paths=sorted(set(A_T)|set(B_T)); paths=[p for p in paths if not ignored(p)]
status=[]
for p in paths:
    ao=A_T.get(p); bo=B_T.get(p)
    status.append((p,'same' if ao and bo and ao[0]==bo[0] else 'A-only' if ao and not bo else 'B-only' if bo and not ao else 'different'))
sc=Counter(s for _,s in status); bytop=defaultdict(Counter)
for p,s in status: bytop[p.split('/')[0]][s]+=1
lines=['# Exhaustive source comparison (allowed simulator scope)','',f'- Source root: `{ROOT}` (read-only; no source files were modified).',f'- Compared commits: A `0c8791f5d2b87c45d409f23ce6f0ad67b708e50b`, B `818a1dbd7583c273ee1996caa75a18c3a4ac90a5`, common base `{BASE}`.','- Excluded from merge inputs: `paper/`, manuscript/thesis writing, virtualenvs, and generated build products. Those paths were not read for architectural decisions.','', '## Tracked path comparison (file by file)','',f'After exclusions: {len(paths):,} paths; '+', '.join(f'{k}={v:,}' for k,v in sorted(sc.items()))+'.','', '| Top-level | same | different | A-only | B-only |','|---|---:|---:|---:|---:|']
for top in sorted(bytop):
 c=bytop[top]; lines.append(f'| `{top}` | {c["same"]} | {c["different"]} | {c["A-only"]} | {c["B-only"]} |')
lines += ['', '### Every non-identical tracked path', '', '| Status | Path |','|---|---|']
for p,s in status:
 if s!='same': lines.append(f'| {s} | `{p}` |')
lines += ['', '## Branch-specific commit summaries', '', '### A / SM86','']
lines += [f'- {x}' for x in subprocess.check_output(['git','-C',str(A),'log','--oneline','--reverse',f'{BASE}..HEAD'],text=True).splitlines() if not any(t in x.lower() for t in ('paper','thesis','manuscript'))]
lines += ['', '### B / SM120','']
lines += [f'- {x}' for x in subprocess.check_output(['git','-C',str(B),'log','--oneline','--reverse',f'{BASE}..HEAD'],text=True).splitlines() if not any(t in x.lower() for t in ('paper','thesis','manuscript'))]
lines += ['', '## Working-tree/capture inventory (from bounded file inventories)','', '- `GPU_simulator`: ~7.9 GiB working tree, 49,444 files; `GPU_simulator_5090`: ~9.0 GiB, 38,293 files. Most size is generated `target/`, virtualenv, or workload artifacts.', '- `sm86_rtx_3090_capture`: ~58 GiB, 206,831 files; `sm120_rtx_5090_capture`: ~88 GiB, 274,192 files. Large protobuf/Nsight reports are evidence artifacts, not source-tree inputs.', '- `final_results`: ~7.4 GiB, 5,186 files; baseline JSONL/compressed logs are retained as external references and are not copied wholesale.', '', '## Focused architectural inventory', '', '- `sim/`: shared Rust trace → SM → memory → engine → direct-run pipeline; both branches modify the same core modules listed in `consultation_prompt.txt`.', '- `arch_profiles/`: A adds SM86 candidates and retains measured SM86 card values; B changes only the SM120 DRAM card block from typed gap to an explicitly unvalidated prototype.', '- `tools/`: A generalizes `sm120_corpus_replay.py` for SM86 and sequential runs; B keeps SM120-specific defaults and sequential option. `nvbit_tracer` differs only in a one-line receive-thread fix at B tip.', '- `workloads/`: A adds SM86 adaptation/substitute and portable-LCD trees; B adds SM120/Blackwell and newer low-precision/TMA rows. Workload source is retained only when needed by the corresponding capture flow; no generated captures are copied into the source tree.', '- `docs/`: A contains SM86 continuation/evidence notes; B contains compact SM120 contract/status notes. Manuscript/paper paths are excluded.', '', '## Capture and baseline interface inventory', '', '- SM86 target contract: `sm86_rtx_3090_capture/config/target.json`; controls in `controls/`; replay metadata in `replay/`; authoritative full/fast outputs in `final_results/3090ti*`.', '- SM120 target contract: `sm120_rtx_5090_capture/config/target.json`; dynamic traces/profiles and sidecars in `captures/` and `profiles/`; authoritative outputs in `final_results/5090*`.', '- `final_results/NCU_profiling_diff.md` states NCU policies differ (SM86 one launch vs SM120 all matching launches), so campaign wall times are not interchangeable.', '- `final_results/runtime_note.md` records SM86 356 attempted / 286 full successes and `final_results/5090_replay_runtime_note.md` records SM120 339/339 success. Both explicitly state NCU was not replay input.', '', '## Diff artifacts', '', '- `research_a_sim.patch` and `research_b_sim.patch` contain the complete common-base-to-tip patches for simulator/architecture/replay tooling paths (generated here, never written to source).', '- `consultation_prompt.txt` records the non-obvious merge conflicts and questions before structural changes.']
OUT.write_text('\n'.join(lines)+'\n')
