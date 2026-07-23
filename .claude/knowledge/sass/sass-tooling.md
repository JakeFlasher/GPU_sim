---
Use when: inspecting a `.cubin`, `.fatbin`, or host binary to see what SASS / PTX the compiler produced; diagnosing an unsupported-opcode failure in the NVBit tracer under `tools/nvbit_tracer/`; building static features for non-root prediction.
Primary raw source: references/NVIDIA_Manuals/cuda_binary_utility/02_cuobjdump.md + 03_nvdisasm.md.
Supporting sources: references/NVIDIA_Manuals/cuda_binary_utility/05_cu-filt.md (demangle); references/NVIDIA_Manuals/cuda_binary_utility/06_nvprune.md (slim libs); tools/nvbit_tracer/README.md (repo tracer).
Repo touchpoints: tools/nvbit_tracer/, workloads/simbench_bw/, tools/cloud/compile_path_inspector.py.
Out of scope: per-mnemonic SASS semantics, PTX semantics, NCU profiling.
Version family: CUDA 13.x (nvcc / nvdisasm / cuobjdump shipped with CUDA ≥ 13.2). Mirror snapshot date: 2026-04-13. Not authoritative; recheck raw source on CUDA upgrade.
---

# SASS Binary Tooling (cuobjdump / nvdisasm / cu++filt / nvprune)

Four CUDA-shipped tools that operate on compiled artifacts — host binaries,
fat binaries, cubins, and **static libraries / relocatable objects**. None
of them accept a `.cu` source.

| Tool | Input | Output | Primary purpose |
|------|-------|--------|-----------------|
| `cuobjdump` | host binary, fatbin, standalone cubin | SASS / PTX / ELF sections / strings / symbols | Discover what device code is inside a shipped binary |
| `nvdisasm` | **standalone cubin only** | SASS + CFG (+ DOT) | Deep per-kernel disassembly with control-flow analysis |
| `cu++filt` | text stream of **mangled** identifiers | demangled names | Demangle CUDA-C++ kernel names for readability |
| `nvprune` | **relocatable host object or static library** (not executables) | slimmed object / static library | Strip architecture slices from a cuBLAS-style library |

## 1. cuobjdump — "what is in this binary?"

### Core flags

| Flag | Output |
|------|--------|
| `-sass` | Disassemble every embedded cubin to SASS |
| `-ptx` | Emit any embedded PTX text |
| `-elf` | Dump CUDA ELF sections in readable form |
| `-symbols` | List function and global symbols |
| `-all-fatbin` / `-all` | Dump every arch slice, not just the host's |
| `-arch sm_XX` | Restrict output to one arch |
| `-function NAME` | Restrict to one kernel by mangled name |
| `-xelf all` / `-xptx all` | Extract every arch's cubin / PTX as separate files on disk |

### Workflows

```bash
# What SASS + PTX does this host binary carry?
cuobjdump -sass -ptx my_host_binary

# Same, pinned to sm_120 slice
cuobjdump -sass -arch sm_120 my_host_binary

# Extract every arch slice (writes files like my_host_binary.sm_XX.cubin)
cuobjdump -xelf all my_host_binary
cuobjdump -xptx all my_host_binary
```

### SASS line format

```
/*0000*/   LDC R1, c[0x0][0x37c] ;       /* 0x0000df00ff017b82 */
                                         /* 0x000fe20000000800 */
```

Two hex blocks per instruction: opcode + operands, then schedule/stall
metadata. NVBit's `tracer_tool.so` surfaces both. An unknown-opcode
failure there means the tracer's opcode matrix (`blackwell_opcode.h`,
etc.) hasn't been taught that SASS mnemonic yet.

### When cuobjdump is the wrong tool

- Only have a cubin, not a fatbin → use `nvdisasm` (CFG analysis, happier with bare cubins).
- Need control-flow → `nvdisasm -cfg` (next section); `cuobjdump` only linearizes.
- Want to strip an arch slice before shipping → `nvprune`, not `cuobjdump`.

## 2. nvdisasm — "disassemble + analyze this cubin"

**Accepts only standalone cubins.** For fat binaries, extract slices first via `cuobjdump -xelf all`.

### Core flags

| Flag (long) | Short | Output |
|-------------|-------|--------|
| `--print-code` | `-c` | Only print executable sections (suppress ELF header noise) |
| `--cuda-function-index <idx>` | `-fun` | Restrict output to one function **by symbol index**, not name |
| `--output-control-flow-graph` | `-cfg` | CFG in DOT language |
| `--output-control-flow-graph-with-basic-blocks` | `-bbcfg` | Basic-block CFG |
| `--no-dataflow` | `-ndf` | Skip control-flow analysis (required if reloc info is missing) |
| `--print-instruction-encoding` | `-hex` | Print raw encoding alongside each instruction |
| `--emit-json` | `-json` | JSON-format disassembly (not compatible with `-cfg` / `-bbcfg` / `-life-*`) |
| `--binary <SMxy>` | | Force a specific SM target |

### Control-flow analysis requires relocation info

`nvdisasm` errors with "relocations required" on bare cubins unless relocation
info was preserved. Two fixes:

- Pass `-ndf` → no CFG, just the linear instruction stream.
- Rebuild with `ptxas -preserve-relocs` / `nvlink -preserve-relocs` to keep the reloc section.

### Workflows

```bash
# Plain SASS
nvdisasm a.cubin

# Per-kernel CFG as PNG
nvdisasm -cfg a.cubin | dot -ocfg.png -Tpng

# SASS with raw encoding (what NVBit captures)
nvdisasm -hex a.cubin

# Disasm one function by symbol index (use nvdisasm a.cubin | grep .global to find indices)
nvdisasm -fun 0 -c a.cubin
```

### Output layout

Each kernel gets a `.text._ZN...` ELF section header, a
`.sectioninfo @"SHI_REGISTERS=N"` line (per-kernel reg count, useful cross-check
against `ptxas -v`), then the instruction stream. `.L_N:` labels mark branch
targets. `EXIT` terminates.

## 3. cu++filt — demangle CUDA C++ identifiers

Input: text stream containing **mangled** CUDA C++ identifiers (or
alphanumeric tokens that look like them). Output: demangled where
possible; original token otherwise.

| Flag | Effect |
|------|--------|
| `-p` | Drop parameter types from demangled output (`f<int>(long)` → `f<int>`) |
| `-_` | Strip leading underscore before demangling (target-dependent) |
| `-h` | Help |

### Workflows

```bash
nvdisasm a.cubin | cu++filt                 # demangle SASS stream
cuobjdump -sass app | cu++filt              # demangle cuobjdump output
nm my_host_binary | cu++filt                # demangle symbol list
echo '_Z1fIiEbl' | cu++filt                 # one-off
```

CUDA C++ demangling differs from host C++ (kernel-launch signatures,
`__global__` / `__device__` annotations). Use `cu++filt`, not plain
`c++filt`.

## 4. nvprune — strip arch slices from a **static library or object**

**Does NOT accept host executables.** Input must be a **relocatable host
object (.o)** or **static library (.a)**. Output is the same format,
slimmed down. The most common use is pruning a distributed static
library (`libcublas_static.a`) to a single arch before linking.

| Flag (long) | Short | Effect |
|-------------|-------|--------|
| `--arch <gpu arch>,…` | `-arch` | Keep these architectures |
| `--generate-code arch=compute_XX,code=sm_XX` | `-gencode` | `nvcc`-style archspec; only `code=` values are used as targets |
| `--no-relocatable-elf` | `-no-relocatable-elf` | Drop relocatable ELF sections too |
| `--output-file PATH` | `-o` | Output path (required) |
| `--options-file PATH` | `-optf` | Read options from a file |

### Workflow

```bash
# Keep only the sm_120 slice
nvprune -arch sm_120 libcublas_static.a -o libcublas_static120.a

# Keep multiple slices
nvprune -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_120,code=sm_120 \
        libcublas_static.a -o libcublas_static_slim.a
```

The output library runs **only on the specified arches**. Don't use this on libraries you'll ship broadly.

## Repo workflow: NVBit tracer "unknown opcode" diagnosis

When `tools/nvbit_tracer/tracer_tool.so` reports an unknown opcode for a `.cu` kernel:

1. Build with `nvcc -arch=sm_120 -O2 -keep` to retain the intermediate `.cubin` / `.ptx`.
2. `cuobjdump -sass -arch sm_120 <binary>` — confirm the SM120-slice mnemonic matches what the tracer saw.
3. `nvdisasm -hex <cubin>` — get the raw encoding (second hex block); compare against `tools/nvbit_tracer/blackwell_opcode.h` to see whether that opcode's encoding entry is wrong or missing.
4. Cross-reference with `sass-instruction-reference.md` (per-arch mnemonic index).
5. If the mnemonic is SM120-only, also check `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md` for the PTX-level equivalent.

Use this workflow on current `workloads/simbench_bw/` binaries and NVBit tracer
smoketests. Historical cluster/D6 archives are not active project inputs.

## Repo workflow: verifying a build is genuinely sm_120

```bash
cuobjdump -sass workloads/simbench_bw/path/to/binary \
  | head -3
# Expect:   code for sm_120
#           .target sm_120
```

**Repo observation:** nvcc may also embed compat fallbacks (multiple `code for sm_XX`
blocks). Harmless, but the tracer runs against whatever arch the CUDA
runtime selects at load time — you can't force a slice from the tracer
side.

## See also

- `.claude/knowledge/sass/sass-instruction-reference.md` — per-arch SASS mnemonic index (which ops are Blackwell-new, how to route an unknown opcode).
- `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md` and `ptx-isa-*.md` siblings — PTX-level equivalent of SASS mnemonics.
- `tools/nvbit_tracer/README.md` — tracer build, patches, and CPU→CPU ordering invariant.
- `.claude/knowledge/INDEX.md` — full agent KB router.

## Escalate to raw when

- You need the full option list for any tool (every `--help` line) — the raw `02_cuobjdump.md` / `03_nvdisasm.md` / `05_cu-filt.md` / `06_nvprune.md` have complete tables.
- You need to inspect an unexpected fatbin section list — `cuobjdump -lelf` and `-xelf all` are the way; §2.2 of `02_cuobjdump.md` enumerates section conventions.
- You are writing an opcode decoder and need bit-layout detail — that's in `cuda_binary_utility/04_instruction-set-reference.md`, distilled under `sass-instruction-reference.md`.
- A flag you expect is missing — CUDA 12.x → 13.x renamed several; check the installed tool's `--help` directly.
