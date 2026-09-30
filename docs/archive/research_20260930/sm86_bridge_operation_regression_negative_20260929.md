# Mixed NAK bridge aggregate-resource regression: negative result

A six-row independent operation-family matrix was used to fit a linear surface
from aggregate instruction/ALU/LSU/XU populations to the mixed bridged body.
The mixed target row was excluded from fitting.

| metric | predicted | observed | APE |
|---|---:|---:|---:|
| duration | 2.880 us | 2.336 us | 23.3% |
| GPC cycles | 3,826 | 3,085 | 24.0% |
| PM long-scoreboard average | 2.196 | 2.205 | 0.41% |

This separates two effects: aggregate populations can predict the short-body
stall statistic, but they do not predict the endpoint duration/GPC cost of the
mixed bridged body. The result strengthens the decision to retain ABI metadata,
cache phase, and hidden scheduling as interval/latent effects rather than
claiming a universal resource-to-time law.
