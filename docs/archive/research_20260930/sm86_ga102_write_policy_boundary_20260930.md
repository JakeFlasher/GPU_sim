# GA102 write-policy boundary

The open 615.71.09 RM source assigns GA102 to
`FB_CACHE_WRITE_MODE_WRITETHROUGH`. The live `GET_GPU_CACHE_INFO` query on the
RTX 3090 Ti returns the corresponding enum. This is a useful **RM policy
classification**, not a live cache-line observation.

The same source retains generic dirty-flush paths because the public cache
control surface is shared across chips and memory targets. On the local card:

* video-memory invalidate and evict are accepted;
* video-memory writeback-only is rejected;
* dirty-memory and system-memory combinations are accepted by the public
  control, but several internal or operation-specific paths can be no-ops or
  unsupported.

Therefore the model may narrow one architectural branch to “GA102 RM
write-through policy,” but it must not infer:

* that every L1/L2 line has no dirty state;
* the physical read-from source of a hit;
* whether a line was resident, bypassed, compressed, or evicted;
* that a successful generic flush returned a per-line DRAM acknowledgement.

NVIDIA’s public Ampere material describes L2 as a shared cache and exposes
residency controls, while not exposing line-owner/read-from telemetry ([Ampere
L2 controls](https://developer.nvidia.com/blog/?p=17431), [GA102
whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)).

This is a narrower cache-policy world, not closure of the cache blocker.
