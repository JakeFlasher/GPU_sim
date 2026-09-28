# SM86 message-passing coherence litmus

`message_passing.cu` launches two blocks. Block 0 writes a value, executes
`__threadfence()`, and publishes a flag with `atomicExch`. Block 1 polls the
flag with an atomic read and then reads the data. The host repeats the kernel
with a fresh value and records stale-read or timeout outcomes.

This is an independent RTX 3090 Ti silicon value observation for the declared
CUDA device-scope message-passing contract. It does not prove the simulator's
coherence implementation, exact cache timing, or proprietary protocol state.
It should be compared against a value-bearing simulator litmus with explicit
read-from/fence identities, not against a counter formula.

`l1_invalidation.cu` adds three controls: an ordinary non-volatile preload /
reload, a volatile reload, and a consumer-fence reload. The ordinary path is
retained as a negative control because it is not a valid CUDA visibility
contract; the run observed stale values after the producer update. Both the
volatile and consumer-fence controls produced zero failures over 1000
iterations. This separates a missing visibility qualifier from a claim of
hardware protocol failure and motivates explicit proxy/volatile/read-from
metadata in the simulator contract.
