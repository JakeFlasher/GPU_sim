# Native-event/frontier endpoint crosswalk (2026-09-30)

`tools/sm86_native_event_frontier_crosswalk.py` fits a family-disjoint
log-response surface for `NCU endpoint / native CUDA-event time`.  Replay
frontiers provide scale and tail context; native event time supplies the
independent elapsed endpoint.  No NCU score counter is used as a feature for
its held-out workload.

The ridge and feature contract are fixed before scoring: log replay event,
log replay GPC envelope, log replay instruction population, and replay tail
per instruction, with ridge 1.0. Every row receives a candidate or widened
candidate-fallback interval.

## Live results

| phase | rows | GPC MAPE | duration MAPE |
|---|---:|---:|---:|
| cold | 15/15 | **11.16%** | **10.99%** |
| warm | 15/15 | **11.23%** | **12.09%** |

Large-family holdouts are especially strong (5.98%/4.33% cold and
8.46%/9.38% warm GPC/duration); base-family holdouts are weaker but still
finite (15.70%/16.82% cold and 13.66%/14.46% warm).

This is a measurement crosswalk, not a unified clock. CUDA events and NCU
WaitForIdle/semaphore duration remain distinct, and no physical pipe, cache
owner, read-from, or complete DRAM semantics are inferred. It remains
non-promoted relative to the refined frontier/archetype model.

The response-surface idea follows the controlled microbenchmark/DVFS approach
of [Wang and Chu](https://arxiv.org/abs/1701.05308) and the newer
microbenchmark-driven analytical modeling direction
([Jarmusch and Chandrasekaran](https://arxiv.org/abs/2605.04178)); the local
model is intentionally weaker because the endpoint domains are not unified.
