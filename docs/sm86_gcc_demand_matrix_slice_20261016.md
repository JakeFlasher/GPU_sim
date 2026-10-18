# SM86 GCC/constant-client demand matrix — H85 — 2026-10-16

## Protocol

Independent real hardware probe under the locked RTX 3090 Ti protocol:

* GPU UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6,
  driver 615.71.09;
* CUDA 13.2, g++-14, `-arch=sm_86`, graphics 1350 MHz, memory 10501 MHz;
* 84 blocks × 1024 threads; binary and raw NCU CSVs are in
  `/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/gcc_demand_h85/`.

The matrix varies mode (arithmetic, one constant load, two constant loads),
repetitions, and constant footprint. NCU is independent comparison evidence;
none of its values entered replay or parameter selection.

## Results

Selected rows (`dram`, total GCC requests, GCC constant requests, L2 GCC
sectors, DRAM read sectors):

| mode/reps/footprint | DRAM sectors | GCC requests | GCC constant requests | LTS GCC sectors |
|---|---:|---:|---:|---:|
| arithmetic / 1 / 4 KiB | 164 | 1,967 | 1,005 | 1,232 |
| arithmetic / 128 / 4 KiB | 164 | 1,974 | 1,003 | 1,232 |
| one-LDC / 1 / 4 KiB | 296 | 7,143 | 6,268 | 2,128 |
| one-LDC / 8 / 4 KiB | 300 | 30,368 | 29,522 | 2,184 |
| one-LDC / 128 / 4 KiB | 316 | 590,880 | 590,041 | 2,240 |
| two-LDC / 128 / 4 KiB | 452 | 2,127,353 | 2,126,307 | 3,248 |
| one-LDC / 128 / 64 B | 172 | 2,016 | 1,169 | 1,288 |
| one-LDC / 128 / 1 KiB | 292 | 562,727 | 561,855 | 2,128 |
| one-LDC / 128 / 16 KiB | 308 | 586,203 | 585,375 | 2,240 |

Full matrix is machine-readable at
`.../gcc_demand_h85/matrix_summary.csv`.

## Inferences and boundary

1. Dynamic GCC request count scales by orders of magnitude with repeated LDCs,
   but L2 GCC sectors grow only modestly (2,128→2,240 for one-LDC reps
   1→128); request count is therefore not a DRAM-sector proxy.
2. DRAM read sectors remain nearly flat (296→316) while constant requests grow
   7,143→590,880, demonstrating that GCC cache requests, GCC L2 sectors, and
   DRAM read fetches are distinct endpoints with cache reuse and instruction
   residency.
3. Two LDCs raise the GCC L2 sector and DRAM domains, but still do not scale
   with the 2.1M request count. Footprint changes alter the low-demand floor
   without establishing a universal linear law.

This falsifies the naive H73/P4 rule “add trace constant requests to
`dram__sectors_read`.” It supports retaining GCC/constant observers and typed
intervals, while keeping aggregate carded DRAM read service unchanged until a
dynamic source-client completion witness and held-out matrix exist.
