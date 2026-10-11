# F3 — direct trace features and logical locality

**Status:** complete for the retained trace contract (logical scenarios)  
**Output:** `/dev/shm/sm120_rtx_5090_work/f3_trace_features_all_v2.json`

`tools/sm120_trace_features.py` reads `dynamic_trace.pb` with the authenticated
protobuf binding, walks the retained per-CTA/per-warp instruction records,
joins native static `(PC, opcode)` labels and launch-local TMA descriptors, and
emits address/sector demand. It computes bounded fully-associative LRU curves,
sampled stack distances, and explicit receive-order/reverse-order/warm-pass
scenarios. The path does not consume NCU or runtime labels.

The retained corpus produced 12 captures / 29 launches. The ordering scenarios
are logical bounds: the trace does not reveal the hardware cache interleaving,
set hash, associativity, dirty policy or MSHR population.

# F5 — finite logical memory service

`tools/sm120_memory_service_model.py` consumes the direct feature output and
prices L1/L2/DRAM demand with declared scenario parameters, finite admission
batches, and TMA sector demand. It emits 29 prediction rows at
`/dev/shm/sm120_rtx_5090_work/f5_memory_service_predictions.json` and keeps
NCU sector rows in a separate diagnostic join; NCU values are not predictor
inputs. Physical geometry and arbitration remain explicit non-claims.
