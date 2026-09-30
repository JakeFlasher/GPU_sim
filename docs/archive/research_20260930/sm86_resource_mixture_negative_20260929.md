# SM86 resource-mixture projection: bounded negative result

A pipe-agnostic alternative was evaluated using static trace opcode/resource
fractions, dynamic trace-population terms, and the existing issue/event/launch
frontiers. It deliberately does not assign a dynamic instruction to a physical
pipe. The features contain no NCU counters or aggregate pipe counters.

On the held-out large cohort, the direct resource-mixture ridge surface reached:

| target | MAPE |
|---|---:|
| NCU GPC cycles | 3.782% |
| NCU duration | 3.813% |

The existing frontier-archetype model remains lower at 2.586% / 2.698% after
independent flash-shape refinement. Therefore the resource mixture is **not
promoted as the point predictor**. It remains useful as a route-probability
and uncertainty surface:

- class-local pipe probes constrain feasible ALU/FMA/LSU/XU/tensor mixtures;
- static control bits and scoreboards constrain issue/dependence ordering;
- the endpoint model consumes the resulting interval/partial order rather than
  fabricating per-PC physical route tags.

This is evidence that aggregate resource classes alone do not close the mixed
physical-pipe blocker. It does support retaining route ambiguity as a bounded
latent variable instead of forcing a universal opcode-to-pipe map.
