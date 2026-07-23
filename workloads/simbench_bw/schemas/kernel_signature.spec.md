# `kernel_signature_sha256` — canonical-input spec

This document is the authoritative source of `kernel_signature_sha256`.
Both the CUPTI-side collector
(`workloads/simbench_bw/cupti_activity_collector.cu`) and the NVBit-side
join helper consumed by `merge_evidence.py join-cupti-nvbit` consume
this spec — neither implementation owns the algorithm. Cross-implementation
drift breaks the offline two-run join. Conformance is enforced by
`workloads/simbench_bw/tests/test_kernel_signature_conformance.py`,
which contains the spec's authoritative reference implementation and
parametrizes itself over every committed fixture under
`workloads/simbench_bw/golden/kernel_signature_fixtures/`.

`kernel_signature_sha256` canonical inputs are spec-owned + golden-fixture-anchored,
not collector-header-owned: production-side implementations (CUPTI .cu,
NVBit join helper) consume the spec and are conformance-tested against
the fixtures.

## Canonical input fields (10 fields, fixed order)

The hash is computed over a canonical JSON serialization of these
fields, in this exact order:

| # | Field | Type | Notes |
|---|---|---|---|
| 1 | `kernel_name` | string | Demangled kernel symbol name as reported by CUPTI / cuobjdump (e.g. `void rmsnorm_fwd_bf16(float const*, float*, int, int)`). UTF-8. |
| 2 | `grid_x` | uint32 | Grid dim X. >= 1. |
| 3 | `grid_y` | uint32 | Grid dim Y. >= 1. |
| 4 | `grid_z` | uint32 | Grid dim Z. >= 1. |
| 5 | `block_x` | uint32 | Block dim X. >= 1. |
| 6 | `block_y` | uint32 | Block dim Y. >= 1. |
| 7 | `block_z` | uint32 | Block dim Z. >= 1. |
| 8 | `shared_mem_static_bytes` | uint64 | Static shared-memory allocation per block, bytes. >= 0. |
| 9 | `shared_mem_dynamic_bytes` | uint64 | Dynamic shared-memory allocation per block, bytes. >= 0. |
| 10 | `regs_per_thread` | uint32 | Per-thread register usage as reported by CUPTI / `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS, ...)`. >= 0. |

These fields are chosen because they are stable identifiers of the
SAME kernel launch shape across the NVBit and CUPTI runs. Per-launch
variables (correlation IDs, stream IDs, repeat indices, context IDs,
timestamps) are NOT included — those are the join dimensions, not
the signature. Adding fields to this list bumps the spec to v2 (no
v2 currently exists; v1 is the only schema version).

## Canonical serialization

The fields are emitted as a single line of UTF-8 JSON in this exact
form (no whitespace inside the object, no trailing newline):

```
{"kernel_name":"<NAME>","grid_x":<U32>,"grid_y":<U32>,"grid_z":<U32>,"block_x":<U32>,"block_y":<U32>,"block_z":<U32>,"shared_mem_static_bytes":<U64>,"shared_mem_dynamic_bytes":<U64>,"regs_per_thread":<U32>}
```

with these rules:

1. **Field order is fixed** by the table above. The spec's reference
   implementation iterates over a tuple constant in the exact order
   listed; production implementations MUST do the same. JSON's
   "key order is unordered" property does not apply here — the bytes
   on the wire are what gets hashed, not the abstract object.
2. **Numeric fields** are written as decimal integers without leading
   zeros, no `+` sign, no decimal point, no exponent. JSON's `json.dumps(int)`
   produces this naturally.
3. **`kernel_name`** is JSON-string-escaped per RFC 8259 §7:
   * `"` is escaped to `\"`
   * `\` is escaped to `\\`
   * control characters U+0000–U+001F are escaped as `\u00XX` (lowercase hex)
   * non-ASCII characters (U+0080+) are emitted verbatim as UTF-8 bytes
     (the spec does NOT use `\uXXXX` for non-ASCII)
   * template-syntax characters (`<`, `>`, `,`, `(`, `)`, `*`, etc.)
     pass through as their literal UTF-8 bytes — they are NOT
     HTML-escaped or URL-escaped.

   **Canonical encoder implementations.** The byte-for-byte authoritative
   reference encoders are:
   * **Python**: `_spec_strict_escape_kernel_name(s)` defined in
     `workloads/simbench_bw/tests/test_kernel_signature_conformance.py`
     (the spec's reference implementation) and mirrored verbatim in
     `workloads/simbench_bw/nvbit_trace_summary_adapter.py` (the
     NVBit-side production implementation).
   * **C++**: `json_escape_kernel_name(const std::string&)` defined in
     `workloads/simbench_bw/cupti_activity_collector.cu` (the CUPTI-side
     production implementation).

   All three encoders produce identical bytes for any input string;
   the cross-implementation conformance suites under
   `workloads/simbench_bw/tests/test_kernel_signature_conformance{,_nvbit_side,_cupti_side}.py`
   enforce this byte-equality across every fixture in
   `workloads/simbench_bw/golden/kernel_signature_fixtures/`.

   **Common implementation pitfall.** Python's stdlib
   `json.dumps(s, ensure_ascii=False)` is the natural shorthand
   reach but is NOT sufficient on its own: while it correctly emits
   non-ASCII characters as verbatim UTF-8, it emits RFC 8259's short
   escapes (`\t`, `\n`, `\r`, `\b`, `\f`) for the corresponding
   control characters in U+0000–U+001F instead of the spec-required
   `\u00XX` lowercase hex. Both forms are RFC 8259-valid but produce
   different bytes, so a kernel signature whose `kernel_name`
   contains any of those control characters would hash to two
   different SHA-256 values across the Python and C++ encoders
   under the naive `json.dumps` approach. New Python implementations
   MUST use `_spec_strict_escape_kernel_name(s)` (or replicate its
   logic) rather than relying on `json.dumps`. C++ implementations
   typically need the same explicit branch on `c < 0x20`.
4. **No whitespace** anywhere in the object: no spaces around `:` or
   `,`, no newlines, no leading/trailing whitespace.
5. **Field labels** are double-quoted ASCII identifiers from the
   table above, in the exact spelling shown.

## Hash algorithm

```
canonical_bytes      = canonical_serialize(inputs).encode("utf-8")
kernel_signature_sha256 = SHA-256(canonical_bytes), hex-lowercase, 64 chars
```

## Golden fixtures

Each fixture under `workloads/simbench_bw/golden/kernel_signature_fixtures/`
is a directory containing exactly three files:

| File | Content |
|---|---|
| `inputs.json` | The 10 canonical input fields as a JSON object. Field order in this file does NOT matter (the reference implementation re-serializes); it exists for human readability. |
| `expected_canonical.txt` | The single-line canonical serialization the spec implementation must emit for these inputs. No trailing newline. |
| `expected_sha256.txt` | The expected SHA-256 hash, hex-lowercase, 64 chars, with a single trailing newline. |

`workloads/simbench_bw/tests/test_kernel_signature_conformance.py`
auto-discovers every directory under `kernel_signature_fixtures/` and
parametrizes its conformance assertions across them. At least three
fixtures MUST be committed at all times so the spec is exercised
across distinct kernel shapes; the test guards this with
`test_at_least_three_fixtures_present`.

Currently committed fixtures:

* `small_pointwise/` — small one-block pointwise kernel, no shared mem.
* `medium_rmsnorm_bf16/` — `rmsnorm_fwd_bf16` representative shape (the
  vertical-slice candidate kernel for the local-host CUPTI runtime).
* `cluster_dsmem_4kb/` — sm_120a cluster-launched kernel with non-trivial
  shared-memory footprint.

## Adding a new fixture

1. Choose a representative kernel shape (small / medium / large /
   cluster / tensor-core / etc.).
2. Write `inputs.json` with the 10 canonical input fields.
3. Compute the expected canonical serialization and SHA-256 using
   the spec's reference implementation:

   ```bash
   python3 - <<'PY'
   import sys
   sys.path.insert(0, "workloads/simbench_bw/tests")
   from test_kernel_signature_conformance import (
       canonical_serialize, compute_kernel_signature_sha256,
   )
   import json
   inputs = json.load(open("workloads/simbench_bw/golden/kernel_signature_fixtures/<NAME>/inputs.json"))
   print("expected_canonical:", canonical_serialize(inputs))
   print("expected_sha256   :", compute_kernel_signature_sha256(inputs))
   PY
   ```
4. Save the canonical line to `expected_canonical.txt` (no trailing
   newline) and the hash + trailing newline to `expected_sha256.txt`.
5. Run the conformance test (`python3 -m pytest
   workloads/simbench_bw/tests/test_kernel_signature_conformance.py -v`)
   to confirm the new fixture passes.

## Not in scope

* Hashing the cubin or SASS. The spec's input fields are launch-config
  metadata, not the kernel's binary identity. A separate `binary_sha256`
  field on the parent sidecar / measurement manifest carries that.
* Hashing per-launch variables (correlation IDs, stream IDs, repeat
  indices, context IDs). Those are the offline-join dimensions, not
  part of the signature.
* Determining what a "kernel launch" is in NVBit's vs CUPTI's view.
  Both surfaces report the launch shape; the spec just specifies the
  signature.
