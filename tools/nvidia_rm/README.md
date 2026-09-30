# NVIDIA RM research seam

`mem_op_d_encoder.py` is deliberately non-submitting. It emits the public
Ampere C56F/C76F `MEM_OP_A..D` words for `L2_FLUSH_DIRTY` (`0x10`).
It also emits the contiguous grouped A-D form (`0x2004000a` plus four data
words) used by NVIDIA's UVM host helper, and a correctly DRF-packed example
GPFIFO entry.

The authorized local RTX 3090 Ti submission used current 615.71.09 headers,
the current 376-byte `NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS`, a secondary
`UVM_MM_INITIALIZE` FD, UVM channel envelope `0x203400000..0x212b40000`, and
the Turing usermode doorbell. Reproducible source and bounded output are
retained at:

`results/alpha/rm_mem_op_d_semaphore_3090ti_20260928/`

That corrected witness establishes RM/channel allocation, correctly packed
pushbuffer mapping, schedule, doorbell, and a post-packet semaphore timestamp
(flush span 480 timer ticks versus a 64-tick NOP baseline). It does not
establish proprietary cache line ownership, dirty payload/read-from state, or
complete writeback of every dirty line. The earlier GP_GET-only artifact is
retained as superseded output.
