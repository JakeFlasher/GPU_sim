# Shifted NAK CUDA-ELF template cache observation

The shifted three-parameter template keeps the same bridged instruction/resource
population but changes cache/endpoint observations:

- cold: 2.496 us / 3,341 GPC cycles, 220 DRAM read sectors, L2 16 hit / 16 miss
- warm: 2.432 us / 3,242 GPC cycles, 80 DRAM read sectors, L2 20 hit / 12 miss

This is template-specific cache evidence and does not identify ownership,
read-from source, eviction, or complete DRAM writeback.
