---
title: "8.7. Morally strong operations"
section: "8.7"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#morally-strong-operations"
---

## [8.7. Morally strong operations](https://docs.nvidia.com/cuda/parallel-thread-execution#morally-strong-operations)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#morally-strong-operations "Permalink to this headline")

Two operations are said to be _morally strong_ relative to each other if they satisfy all of the
following conditions:

1. The operations are related in _program order_ (i.e, they are both executed by the same thread),
or each operation is _strong_ and specifies a _scope_ that includes the thread executing the
other operation.
2. Both operations are performed via the same _proxy_.
3. If both are memory operations, then they overlap completely.

Most (but not all) of the axioms in the memory consistency model depend on relations between
_morally strong_ operations.

### [8.7.1. Conflict and Data-races](https://docs.nvidia.com/cuda/parallel-thread-execution#conflict-and-data-races)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#conflict-and-data-races "Permalink to this headline")

Two _overlapping_ memory operations are said to _conflict_ when at least one of them is a _write_.

Two _conflicting_ memory operations are said to be in a _data-race_ if they are not related in
_causality order_ and they are not _morally strong_.

### [8.7.2. Limitations on Mixed-size Data-races](https://docs.nvidia.com/cuda/parallel-thread-execution#mixed-size-limitations)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#mixed-size-limitations "Permalink to this headline")

A _data-race_ between operations that _overlap_ completely is called a _uniform-size data-race_,
while a _data-race_ between operations that _overlap_ partially is called a _mixed-size data-race_.

The axioms in the memory consistency model do not apply if a PTX program contains one or more
_mixed-size data-races_. But these axioms are sufficient to describe the behavior of a PTX program
with only _uniform-size data-races_.

**Atomicity of mixed-size RMW operations**

In any program with or without _mixed-size data-races_, the following property holds for every pair
of _overlapping atomic_ operations A1 and A2 such that each specifies a _scope_ that includes the
other: Either the _read-modify-write_ operation specified by A1 is performed completely before A2 is
initiated, or vice versa. This property holds irrespective of whether the two operations A1 and A2
overlap partially or completely.
