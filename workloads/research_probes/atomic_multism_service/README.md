# sm_86 multi-SM atomic service probe

One-warp CTAs sweep 1, 4, 16, 42, 84, and 168 blocks. Each warp aliases one
32-bit word internally; blocks either share that word or use per-block words at
128-, 256-, 4096-, or 4100-byte spacing. A runtime index table prevents NVCC
warp aggregation and an optional permutation changes block-to-address order
without changing the address multiset.

This probe distinguishes a per-SM service candidate from distributed/shared L2
atomic-unit behavior and tests the 256-byte/page-offset effects reported by
Maucher et al. (PLOS 2025).

The 106-cell matrix includes direct placement at every block count plus
address-order permutations at 84 blocks. Coarse W84 and all W168 cells receive
three NCU profiles; all cells receive seven long and seven trace-length
correctness runs, NVBit capture, and two simulator replays.

A 27-point direct B84 spacing sweep covers powers of two and offsets around
128, 256, and 4096 bytes. Its non-coarse cells receive one NCU profile because
their seven direct CLOCK trials are the primary timing observation.

`pair_collision.py` adds a direct collision oracle: 84 CTAs are split evenly
between address zero and one swept address. Its 288 offsets cover every aligned
word below 128 bytes and every 128-byte line through 32 KiB.

`pair_phase.py` then shifts both addresses through every aligned word in a
256-byte window for deltas 128, 256, 260, and 4096 bytes, testing whether the
collision relation is a fixed offset rule or depends on absolute address phase.
