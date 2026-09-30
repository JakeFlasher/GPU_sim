# SM86 native value-dependent differential

This probe compiles `value.ptx` with `ptxas -arch=sm_86`, runs the resulting
single-CTA/32-lane kernel through the CUDA driver, and uses a pinned NVBit
injection tool to capture register/predicate values immediately before each
dynamic SASS instruction. The independent runner records input/output words;
`validate_capture.py` checks the capture against the runner's unsigned
predicate/add/subtract oracle.

The retained `native_capture_delta_2pow31_seed17.out` was captured on:

- NVIDIA GeForce RTX 3090 Ti;
- compute capability 8.6;
- PCI `0000:01:00.0`;
- CUDA driver API 13040;
- one CTA and exactly 32 lanes;
- delta `2^31`, including signed-boundary inputs.

`native_artifact_manifest.json` binds the PTX, SM86 cubin, disassembly,
capture, runner output, validation output, and the generated payload sidecar by
SHA-256.

Validation checks 32 lanes × 13 dynamic PCs, zero capture overflow, `LDG.E`
effective addresses and return values, `ISETP` outcomes, predicated `IADD3`
results, and `STG.E` effective addresses/payload values. It reports 15
predicate-true lanes and status `ok`.

`native_capture_to_payload.py` converts the 32 captured stores into
`native_value_payload_sidecar_v1` for retained legacy captures, or v2 for new
captures. V2 preserves full 64-bit addresses, static PC/guard, per-lane
occurrence, launch/function/CTA/warp metadata, and explicit input/output
baseline metadata. Load-return observations are retained separately. The
sidecar explicitly marks ownership and flush completion as unknown, and the
simulator does not silently join it to native replay events.

`compare_native_interpreter.py` executes the same retained SM86 disassembly in
the finite interpreter and compares all 32 native lanes on dynamic PCs,
load-return values, store addresses/payloads, and predicate outcomes. The
retained result is `native_interpreter_validation_delta_2pow31_seed17.json`.

This closes a native value-capture differential slice, not the full value-
dependent SASS limitation. Calls, atomics, fences, races, allocation-wide
memory provenance, arbitrary control flow, and timing remain outside the contract.
It does not identify physical pipes or prove that runtime values regenerate
static SASS bytes.

The capture tool lives in `capture.cu`/`inject.cu`; `runner.cpp` uses the CUDA
driver API. Build with the repository's pinned NVBit core and CUDA 13.4 toolkit;
run only on SM86 and refuse any other device.
