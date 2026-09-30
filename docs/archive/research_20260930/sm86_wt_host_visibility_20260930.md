# SM86 system-memory write-through visibility subcase

**Date:** 2026-09-30  
**GPU:** RTX 3090 Ti / SM86 / driver 615.71.09 / clocks 1350/10501 MHz.

A temporary CUDA kernel used inline st.global.wt.u32 to a
cudaHostAllocMapped system-memory pointer, followed by membar.sys. The body
also recorded %globaltimer before/after the operation. Sixteen bounded launches
published values 0xabc00000 through 0xabc0000f to host memory.

Observed GPU spans were 1,024 or 2,048 ns after the first setup launch; host
synchronization spans were approximately 6.1--7.4 us after setup. The host
mapping read the newly written value after each synchronization.

This is a bounded architectural path observation:

* st.wt targets system-memory write-through semantics;
* membar.sys supplies system-scope ordering;
* host visibility follows CUDA stream synchronization.

It does not prove a device-memory cache-line owner, a per-line DRAM
acknowledgement, or the behavior of ordinary st.wb/st.cg device-memory
stores. It therefore narrows the cache model only for the explicit
system-memory/write-through intervention.

Artifacts: results/alpha/sm86_wt_host_visibility_goal_20260930/.
