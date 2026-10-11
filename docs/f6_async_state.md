# F6 — logical asynchronous/TMA completion state

**Output:** `/dev/shm/sm120_rtx_5090_work/f6_async_state_campaign_v2.json`

`tools/sm120_async_state_model.py` streams the retained UTMA/TMA and sync
sidecars, binds records by kernel/function/warp/PC/instruction index, tracks
descriptor bytes, source/load versus destination/store demand, barrier
addresses, arrival/phase-check/wait populations and logical service cycles.
The ledger keeps issue, service, wait/generation and visibility fields
separate; only a bounded event prefix is serialized while aggregate counts
cover the full sidecar.

Twenty-four launches carry async sidecars. Five retained launches have no
sidecar rows and remain unavailable rather than zero-filled. Destination
visibility and completion timestamps remain unobserved without a payload
consumer litmus. No descriptor-byte field is used as a payload witness.

The Rust replay's existing async-group/readiness event state remains the timing
backend for supported native forms; this sidecar ledger supplies a direct
SM120 observation/model seam while preserving the non-RTL boundary.
