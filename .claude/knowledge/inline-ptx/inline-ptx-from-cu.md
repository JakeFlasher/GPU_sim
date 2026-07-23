---
Use when: writing or auditing an `asm(...)` inline-PTX block inside a `.cu` kernel; debugging a mismatch between what a CUDA kernel tries to emit and what `ptxas` parses; extending a simbench workload to a new PTX mnemonic; chasing an "error: asm operand type size(N) does not match..." diagnostic.
Primary raw source: references/NVIDIA_Manuals/cuda_inline_ptx/01_using-inline-ptx-assembly-in-cuda.md.
Supporting sources: references/NVIDIA_Manuals/ptx_isa/09_instruction-set.md (the PTX mnemonics you inline); `.claude/knowledge/ptx/ptx-isa-*.md` (distilled PTX cards).
Repo touchpoints: workloads/simbench_bw/, tools/nvbit_tracer/.
Out of scope: PTX ISA semantics (which instruction does what); pure-PTX authoring (`.ptx` files); writing PTX through the driver's PTX JIT.
Version family: CUDA 13.x. Mirror snapshot date: 2026-04-13. Not authoritative; recheck raw source on CUDA upgrade.
---

# Inline PTX in `.cu` Kernels — `asm(...)` Essentials

Inline PTX is the fastest way to exercise a specific PTX mnemonic from a
CUDA C++ kernel. Historical tensor-core microbenches used it to surface
`tcgen05.*` and block-scaled MMA variants without waiting for CUDA C++
intrinsics.

## Syntax

```cpp
asm("template" : "constraint"(output), ... : "constraint"(input), ... : "clobbers");
```

Four colon-separated fields:

| Field | Contents |
|-------|----------|
| Template | PTX text with `%0`, `%1`, … referring to operands by position |
| Outputs | `(output_var)` pairs; the constraint is prefixed with `=` (write-only) or `+` (read-write) |
| Inputs | `(input_var)` pairs; no `=` prefix |
| Clobbers | Optional. `"memory"` is the most common — forces ordering w.r.t. surrounding memory ops |

### Minimal forms

```cpp
asm("membar.gl;");                                         // no operands
asm("mov.s32 %0, 2;" : "=r"(i));                           // no inputs → drop final colon
asm("mov.s32 r1, %0;" :: "r"(i));                          // no outputs → adjacent colons
asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));    // 1 out, 2 in
asm("add.s32 %0, %0, %1;" : "+r"(i) : "r"(j));             // read-write (+)
```

Operand indexing: `%0` = first output, `%1` = first input (if no outputs) / second listed operand (if 1 output), etc. **Outputs are indexed first**, then inputs, in list order. `%%` escapes a literal `%` (for `%clock`, `%tid`, etc.).

## Constraints (raw guide §1.1.2)

One letter per PTX register type. A constraint binds the C++ variable to a register of that type; the compiler emits the necessary `ld`/`st` glue around the `asm()` block.

| Letter | Register type | Matching C++ types (type AND size must match) |
|--------|--------------|----------------------------------------------|
| `h` | `.u16` | `uint16_t`, `int16_t`, `short` (16-bit integer types) |
| `r` | `.u32` | `uint32_t`, `int32_t`, `int` (32-bit integer types) |
| `l` | `.u64` | `uint64_t`, `int64_t`, `long long`, pointers |
| `q` | `.u128` | `__int128` (platforms where it is available) |
| `f` | `.f32` | `float` |
| `d` | `.f64` | `double` |
| `n` | Immediate integer (known at compile time) | `constexpr int` / `int` literal |
| `C` | `const char[]` array (compile-time known) | static `constexpr char[]` storage |

**Do not "bit-cast" across letters.** The raw guide §1.3 explicitly
rejects this as an error: passing a `float` with `"=r"` yields `"asm
operand type size(4) does not match type/size implied by constraint 'r'"`
(an integer-register class for an FP value). For FP values, use `"f"`
(FP32) or `"d"` (FP64). For `__half`, cast on the C++ side into a
`uint16_t` / `short` and pass that with `"h"` — the `asm()` block then
sees a `.u16` register. On the PTX side, **scalar** `.f16` instructions
accept either `.f16` or `.b16` operands (PTX ISA §9), so a 16-bit half
carried through an `"h"` operand can still feed ops like `mov.f16` or
`add.f16`. **Packed** `.f16x2` ops are different — they take 32-bit
operands (`.b32` / `.f16x2`), not a single 16-bit lane — so a packed
operation needs an `"r"`-bound 32-bit value, not an `"h"`-bound 16-bit
one. Match the mnemonic's operand type to the one the ISA specifies for
the instruction you actually want.

**There is no constraint letter for 8-bit registers.** PTX ops of 8-bit
type accept operands wider than the instruction (raw guide end-of-§1.1.2):

```cpp
__device__ void copy_u8(char* in, char* out) {
    int d;
    asm("ld.u8 %0, [%1];" : "=r"(d) : "l"(in) : "memory");
    *out = d;
}
```

### The `"C"` constraint (compile-time string substitution)

`"C"` lets you bake compile-time choices into the PTX template. Canonical use: rounding-mode suffixes that depend on a template parameter.

```cpp
template<> struct helper<mode_rn> {
    static constexpr const char mode[] = ".rn";
};
asm("add.f32%1 %0, %2, %3;"
    : "=f"(result)
    : "C"(helper<mode>::mode), "f"(a), "f"(b));
// At compile time: generates `add.f32.rn %0, %1, %2;` or similar per mode.
```

The referenced variable must have **static storage duration**, be **constant-initialized**, and have type `const char[]`. Device-code only.

## Multi-instruction blocks

Split across multiple lines using C++ string concatenation; end each line except the last with `\n\t` for readable generated PTX:

```cpp
__device__ int cube(int x) {
    int y;
    asm("{\n\t"                      // local scope → reusable temp names
        " .reg .u32 t1;\n\t"
        " mul.lo.u32 t1, %1, %1;\n\t"
        " mul.lo.u32 %0, t1, %1;"
        "}"
        : "=r"(y) : "r"(x));
    return y;
}
```

Wrap temporaries in `{ ... }` **scope braces** — otherwise inlining the function multiple times in one TU produces "duplicate definition of t1" errors (raw guide §1.2.1).

## Pitfalls (raw guide §1.2)

### Namespace / scope conflicts

Multiple inlinings of the same `asm()` declaring a `.reg` will collide. Always use `{...}` scope braces around temp-register declarations.

### Memory-space awareness

`asm()` **does not know** which CUDA memory space a pointer is in. For `sm_20`+, any pointer is passed as a generic address. If you need a specific state-space prefix (`.global`, `.shared`), emit it explicitly in the PTX template.

### Compiler motion and optimization

The compiler assumes `asm()` has **no side effects** beyond writing the output operands. To prevent the compiler from removing, reordering, or CSE-ing an `asm()`:

- Add `volatile`: `asm volatile ("…" : "=r"(x));`
- Add a `"memory"` clobber to also block memory-op reordering around the asm:
  ```cpp
  asm volatile ("mov.u32 %0, %%clock;" : "=r"(x) :: "memory");
  asm          ("st.u32 [%0], %1;"    :: "l"(p), "r"(x) : "memory");
  ```

### PTX-level errors are deferred to ptxas

The C++ front-end **does not parse the PTX template**. Typos, wrong constraints, or unsupported operand modifiers surface only when `ptxas` runs. Keep PTX strings short; test-compile with `nvcc -ptx` frequently while iterating.

## Front-end error checks (raw guide §1.3)

The front-end does validate a few things:

| Check | Error text |
|-------|-----------|
| Multiple constraint letters per operand | "an asm operand may specify only one constraint letter in a `__device__`/`__global__` function" |
| Aggregate as operand (`struct`, `int4`) | "an asm operand must have scalar type" |
| Constraint size mismatch | "asm operand type size(N) does not match type/size implied by constraint 'X'" |

Fix by explicit-cast or extract-field:

```cpp
char ci, cj, ck;
int temp = ci;
asm("add.s32 %0, %1, %2;" : "=r"(temp) : "r"((int)cj), "r"((int)ck));
ci = temp;
```

## Repo workflow: adding a new PTX mnemonic to a D6 bench

1. Author the PTX in isolation: write the target instruction on paper / in the distilled PTX card (`.claude/knowledge/ptx/ptx-isa-*.md`); confirm operand shapes and CC gating.
2. Wrap in an `asm volatile(...)` block inside a `__device__` helper.
3. Compile with `nvcc -arch=sm_120a -ptx kernel.cu` to catch ptxas errors immediately.
4. `nvcc -arch=sm_120a -O2 -keep kernel.cu` to retain `.ptx` and `.cubin` for inspection.
5. `cuobjdump -sass` to confirm the resulting SASS matches expectations (see `sass-tooling.md`).
6. Trace with NVBit (see `tools/nvbit_tracer/`); if the opcode is unknown, update `blackwell_opcode.h` per the diagnosis workflow in `sass-tooling.md`.

**Repo observation:** the D6 microbenches use this pattern to drive `tcgen05.mma.*`, `cp.async.bulk.*`, and block-scaled MMA variants that do not yet have stable CUDA C++ intrinsics.

## See also

- `.claude/knowledge/ptx/ptx-isa-tensor-cores.md` — mma / wgmma / tcgen05 mnemonics.
- `.claude/knowledge/ptx/ptx-isa-async-copy.md` — `cp.async.*` family.
- `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md` — SM100 / SM120-specific PTX deltas.
- `.claude/knowledge/sass/sass-tooling.md` — `cuobjdump` / `nvdisasm` for verifying what the inline PTX lowered to.
- `.claude/knowledge/nvcc/nvcc-flags.md` — `-keep`, `-ptx`, `-Xptxas -v` for iteration.
- `.claude/knowledge/INDEX.md` — full agent KB router.

## Escalate to raw when

- You need the full constraint table with every combinator (e.g. `+`, `=`, `&` for early-clobber — though the raw guide does not document `&` for CUDA) — raw §1.1.2 is authoritative.
- You are using `"C"` for non-trivial compile-time string assembly and need edge-case semantics (class statics, templated arrays, `extern const char[]`) — raw §1.1.2 example block covers these explicitly.
- You hit a "cannot find a register in class" ptxas diagnostic — that's a register-pressure problem, not an inline-PTX syntax problem; see `perf-occupancy-launch.md`'s register-budget discussion.
- You want to write standalone `.ptx` files instead of inline asm — that's outside this card; `nvcc_guide/02_compilation-phases.md` covers the `.ptx` input path.
