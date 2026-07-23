---
Use when: choosing `-arch` / `-gencode` for nvcc on a Blackwell build; diagnosing "no kernel image available for execution on the device" on SM120 / SM100; deciding whether to rely on PTX JIT vs native cubin; migrating a Hopper-era build to Blackwell; understanding CC 10.0 vs 12.0 capability gating; deciding between sm_120 and sm_120a (architecture-conditional opt-in for FP4 / NVFP4 block-scaled MMA, tcgen05).
Primary raw source: references/NVIDIA_Manuals/blackwell_compatibility/01_blackwell-architecture-compatibility.md.
Supporting sources: references/NVIDIA_Manuals/nvcc_guide/05_gpu-compilation.md (fatbin assembly, virtual vs real arch).
Repo touchpoints: workloads/simbench_bw/, tools/cloud/compile_path_inspector.py, tools/nvbit_tracer/, nvcc build invocations in active workload Makefiles.
Out of scope: Blackwell tuning numbers (in `blackwell-tuning.md`); per-mnemonic PTX / SASS semantics.
Version family: CUDA 13.x; Blackwell compute capability 10.0 (B200) and 12.0 (SM120 consumer). Mirror snapshot date: 2026-04-13. Not authoritative; recheck raw source on CUDA upgrade.
---

# Blackwell Compatibility — Forward Compat, Cubin vs PTX, `-gencode`

Two orthogonal compatibility axes that both bite at kernel launch:

1. **Cubin** is **bound to a major CC revision**. A cubin built for
   `sm_X.Y` runs on `sm_X.Z` where `Z ≥ Y`, but **not** on a different
   major (e.g. sm_80 → sm_90 fails).
2. **PTX** is **forward-compatible**. A PTX built for `compute_X` runs
   on any CC `≥ X` via driver JIT.

Ship PTX in every binary unless you have a reason not to.

## The short rule

> **For Blackwell, include at least `compute_100` PTX. Also include native
> `sm_100` and/or `sm_120` cubin where you can afford the binary bloat.**

This is the raw guide's §1.4 recommendation, simplified.

## Architecture table (what this repo cares about)

| Arch | CC | nvcc target | Appears on |
|------|----|-------------|------------|
| Pascal | 6.0 / 6.1 | `sm_60` / `sm_61` | Old baselines only |
| Volta | 7.0 | `sm_70` | Some academic papers |
| Turing | 7.5 | `sm_75` | — |
| Ampere | 8.0 / 8.6 / 8.7 | `sm_80` / `sm_86` / `sm_87` | |
| Ada | 8.9 | `sm_89` | Consumer FP8 debut |
| Hopper | 9.0 | `sm_90` / `sm_90a` | Academic paper target |
| **Blackwell** (B200) | **10.0** | `sm_100` / `sm_100a` / `sm_100f` | Datacenter |
| **Blackwell** (consumer) | **12.0** | `sm_120` / `sm_120a` | **Primary cloud target for this repo** |

`a` = **architecture-conditional features** (tcgen05, FP4, block-scaled MMA). `f` = family feature set (added in PTX 8.8).

## Cubin compatibility — the binding rule

From raw guide §1.2:

- Cubin for CC `X.Y` runs on any CC `X.Z` where `Z ≥ Y`.
- Cubin for CC `X.Y` does **not** run on CC `W.V` where `W ≠ X`.

Consequences:

- An `sm_80` cubin runs on `sm_86` / `sm_87` but **not** on `sm_90` / `sm_100` / `sm_120`.
- An `sm_90` cubin runs on `sm_90a` but **not** `sm_100` (different major).
- An `sm_100` cubin runs on `sm_100a` / `sm_100f` but **not** on `sm_120` (different major under CUDA's compatibility matrix — CC 10.x and CC 12.x are separate lineages).
- An `sm_120` cubin runs on `sm_120a` but **not** on `sm_100` (backward major is blocked).

**Repo policy:** active Blackwell workload binaries should build native sm_120
artifacts for the cloud RTX PRO 6000 Blackwell target and separate sm_86
artifacts for the local RTX 3090 Ti smoke target. If cross-testing on B200
becomes needed, rebuild with explicit `-gencode` entries covering both CC 10.0
and CC 12.0.

## PTX compatibility — the forward rule

- PTX for `compute_X.Y` runs on any CC `≥ X.Y` via driver JIT.
- Caveat: `compute_100a` (architecture-conditional) **is not forward-compat**. PTX for `compute_90a` **cannot** run on sm_100 / sm_120.

The raw guide §1.2 explicitly calls this out:

> "Application binaries that include PTX version of kernels with architecture conditional features using `sm_100a` or `compute_100a` … are not forward or backward compatible."

**Repo observation:** Blackwell FP4 and tcgen05 paths may require
architecture-conditional targets such as `sm_120a`. Keep those builds separate
from portable sm_120 or sm_86 smoke binaries, and do not assume conditional PTX
is forward-compatible across major CC families.

## Verifying an existing binary is Blackwell-compatible

Raw guide §1.3.1 gives a one-liner:

```bash
# If the app still runs with JIT forced, the binary carries PTX and is compat.
CUDA_FORCE_PTX_JIT=1 ./my_app
```

Interpretation:

- **Runs cleanly** → binary has PTX; driver JITed it to Blackwell cubin.
- **Fails to launch** → binary only has pre-Blackwell cubin; rebuild.
- Unset `CUDA_FORCE_PTX_JIT` after testing.

Repo workflow to cross-check a single binary:

```bash
cuobjdump -sass -ptx workloads/simbench_bw/path/to/binary | head -40
```

Expect to see both `.target sm_120` (cubin) and an `arch = sm_120a` PTX
header (if the source used conditional features).

## Building for Blackwell — `-gencode` recipes

### CUDA Toolkit 12.8+ (native Blackwell cubin)

From raw guide §1.4.2. The canonical multi-arch recipe includes PTX for forward compat:

```bash
nvcc \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_80,code=sm_80 \
  -gencode=arch=compute_90,code=sm_90 \
  -gencode=arch=compute_100,code=sm_100 \
  -gencode=arch=compute_100,code=compute_100 \
  -O2 -o out.o -c kernel.cu
```

### CUDA Toolkit 12.7 or earlier (no native Blackwell cubin)

Raw guide §1.4.1: the last `-gencode` must emit PTX (`code=compute_XX`) for future compat; Blackwell will JIT from that PTX.

### `-arch=sm_XX` shorthand caveat

`-arch=sm_XX` expands to `-gencode=arch=compute_XX,code=sm_XX` + `-gencode=arch=compute_XX,code=compute_XX`. Two gotchas:

- You can only pass **one** `-arch=` at a time (multi-arch requires `-gencode`).
- Only one PTX version ends up in the binary (the last `-gencode` for that CC wins).

**Repo policy:** for cloud-only sm_120 workloads, `-arch=sm_120 -O2` may be
sufficient. For anything reused across cloud and local smoke hosts, use explicit
`-gencode` stanzas.

## Key `-gencode` semantics

| Clause | Meaning |
|--------|---------|
| `arch=compute_X` | **Front-end** PTX target. Must always be a PTX version. |
| `code=sm_X` | **Back-end** cubin target for a real arch. |
| `code=compute_X` | **Back-end** PTX retained in the binary (for forward JIT). |
| `code=sm_X,compute_X` (inline CSV) | Both cubin and PTX for the same arch in one stanza. |

**Only the back-end target(s) specified by `code=` are retained in the binary**; `arch=` alone produces nothing. At least one `code=compute_X` is required for forward compat.

## Independent Thread Scheduling (Volta+ behavior) compat

Since Volta, warps no longer execute in lockstep. If code relies on pre-Volta warp-synchronous behavior, the Blackwell compatibility guide §1.4.3 offers a Pascal-scheduling opt-in:

```bash
nvcc -gencode=arch=compute_60,code=sm_100 ...
```

**Repo observation:** this repo does **not** rely on warp-synchronous programming; all warp-level synchronization is explicit (`__syncwarp`, `cg::coalesced_threads`, `WARPSYNC`). Do not add the Pascal opt-in.

## Failure modes you will actually hit

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA error: no kernel image is available for execution on the device` | Binary has no cubin for the host's CC AND no PTX (or PTX is `compute_Xa` conditional) | Rebuild with `-gencode=arch=compute_100,code=compute_100` (or the right target). |
| Kernel launch succeeds but numerics differ from Hopper | Independent Thread Scheduling interaction; or FP4/FP8 rounding differences in tcgen05 | Confirm with `CUDA_FORCE_PTX_JIT=1` that it's not a cubin/PTX path difference; then audit warp-level assumptions. |
| `CUDA_FORCE_PTX_JIT=1` run succeeds, normal run fails on Blackwell | Binary has compatible PTX but stale pre-Blackwell cubin that the driver prefers | Rebuild with a native Blackwell `sm_100` / `sm_120` cubin. |
| Tensor-core kernel fails on SM120 but works on SM100 | Source built `-arch=sm_100a`; SM120 has CC 12.0 major → no cubin match, and `compute_100a` PTX is not forward-compat to 12.0 | Rebuild with `-arch=sm_120a`. |

## See also

- `.claude/knowledge/blackwell/blackwell-tuning.md` — occupancy / shared-mem / cluster numbers once the binary builds cleanly.
- `.claude/knowledge/sass/sass-tooling.md` — `cuobjdump -sass -ptx` / `nvdisasm` for inspecting what ended up in the binary.
- `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md` — which PTX features need `sm_100a` / `sm_120a` architecture-conditional gating.
- `.claude/knowledge/INDEX.md` — full agent KB router.

## Escalate to raw when

- You need per-SKU `sm_120`-family details beyond CC 12.0 (consumer variants like sm_121, sm_122 if they ship) — the raw compatibility guide updates per CUDA release; trust it over this card.
- You need the CUDA 12.8 vs 13.x build matrix for every supported arch — the guide §1.4 evolves per toolkit version.
- You hit an "architecture-conditional" (`a`) or "family" (`f`) feature not covered here — the raw guide's §1.2 paragraph on `sm_Xa` / `compute_Xa` is the authoritative source.
- Independent-Thread-Scheduling opt-in behavior differs from what's described — §1.4.3 of the raw guide is authoritative; the opt-in rules may change per CUDA release.
