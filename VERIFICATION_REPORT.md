# Unified replay verification

Comparisons use copied real traces under `verification_inputs/`; source capture and final-result trees were not modified.

| Case | Baseline | Unified | Selected counters equal? |
|---|---|---|---|
| sm86 cp.async | `ok/sm_86` | `ok/sm_86` | yes |

### sm86 cp.async

```json
{
  "baseline": {
    "architecture": "sm_86",
    "dram": {
      "read_sectors": 128,
      "service_rate_bytes_per_sec": 825000000000
    },
    "front_end": {
      "compulsory_instruction_line_read_sectors": 44,
      "compulsory_instruction_read_sectors": 39,
      "static_instruction_line_read_sectors": 52
    },
    "instructions_issued": 644,
    "l1": {
      "hits": 0,
      "line_fills": 32,
      "merges": 96,
      "misses": 128,
      "store_sectors": 128
    },
    "l2": {
      "hits": 96,
      "lookup_hits": 128,
      "lookup_requests": 256,
      "misses": 160
    },
    "launch_count": 1,
    "memory_mode": "l1-l2",
    "metric_prediction_count": 8,
    "records_seen": 644,
    "schema_version": 1,
    "status": "ok",
    "timing_status": "incomplete_unmodeled_retirement_tails"
  },
  "unified": {
    "architecture": "sm_86",
    "dram": {
      "read_sectors": 128,
      "service_rate_bytes_per_sec": 825000000000
    },
    "front_end": {
      "compulsory_instruction_line_read_sectors": 44,
      "compulsory_instruction_read_sectors": 39,
      "static_instruction_line_read_sectors": 52
    },
    "instructions_issued": 644,
    "l1": {
      "hits": 0,
      "line_fills": 32,
      "merges": 96,
      "misses": 128,
      "store_sectors": 128
    },
    "l2": {
      "hits": 96,
      "lookup_hits": 128,
      "lookup_requests": 256,
      "misses": 160
    },
    "launch_count": 1,
    "memory_mode": "l1-l2",
    "metric_prediction_count": 8,
    "records_seen": 644,
    "schema_version": 1,
    "status": "ok",
    "timing_status": "incomplete_unmodeled_retirement_tails"
  }
}
```

| sm120 TMA | `ok/sm_120` | `ok/sm_120` | yes |

### sm120 TMA

```json
{
  "baseline": {
    "architecture": "sm_120",
    "dram": {
      "read_sectors": 0,
      "service_rate_bytes_per_sec": null
    },
    "front_end": {
      "compulsory_constant_read_sectors": 2,
      "compulsory_instruction_read_sectors": 33
    },
    "instructions_issued": 69,
    "l1": {
      "hits": 0,
      "line_fills": 0,
      "merges": 0,
      "misses": 0,
      "store_sectors": 0
    },
    "l2": {
      "hits": 0,
      "lookup_hits": 0,
      "lookup_requests": 0,
      "misses": 0
    },
    "launch_count": 1,
    "memory_mode": "l1-l2",
    "metric_prediction_count": 8,
    "records_seen": 69,
    "schema_version": 1,
    "status": "ok",
    "timing_status": "complete"
  },
  "unified": {
    "architecture": "sm_120",
    "dram": {
      "read_sectors": 0,
      "service_rate_bytes_per_sec": 1792000000000
    },
    "front_end": {
      "compulsory_constant_read_sectors": 2,
      "compulsory_instruction_read_sectors": 33
    },
    "instructions_issued": 69,
    "l1": {
      "hits": 0,
      "line_fills": 0,
      "merges": 0,
      "misses": 0,
      "store_sectors": 0
    },
    "l2": {
      "hits": 0,
      "lookup_hits": 0,
      "lookup_requests": 0,
      "misses": 0
    },
    "launch_count": 1,
    "memory_mode": "l1-l2",
    "metric_prediction_count": 8,
    "records_seen": 69,
    "schema_version": 1,
    "status": "ok",
    "timing_status": "complete"
  }
}
```

| sm86 scalar | `ok/sm_86` | `ok/sm_86` | yes |

### sm86 scalar

```json
{
  "baseline": {
    "architecture": "sm_86",
    "dram": {
      "read_sectors": 32,
      "service_rate_bytes_per_sec": 825000000000
    },
    "front_end": {
      "compulsory_instruction_line_read_sectors": 32,
      "compulsory_instruction_read_sectors": 28,
      "static_instruction_line_read_sectors": 40
    },
    "instructions_issued": 79,
    "l1": {
      "hits": 0,
      "line_fills": 8,
      "merges": 24,
      "misses": 32,
      "store_sectors": 32
    },
    "l2": {
      "hits": 24,
      "lookup_hits": 32,
      "lookup_requests": 64,
      "misses": 40
    },
    "launch_count": 1,
    "memory_mode": "l1-l2",
    "metric_prediction_count": 8,
    "records_seen": 79,
    "schema_version": 1,
    "status": "ok",
    "timing_status": "incomplete_unmodeled_retirement_tails"
  },
  "unified": {
    "architecture": "sm_86",
    "dram": {
      "read_sectors": 32,
      "service_rate_bytes_per_sec": 825000000000
    },
    "front_end": {
      "compulsory_instruction_line_read_sectors": 32,
      "compulsory_instruction_read_sectors": 28,
      "static_instruction_line_read_sectors": 40
    },
    "instructions_issued": 79,
    "l1": {
      "hits": 0,
      "line_fills": 8,
      "merges": 24,
      "misses": 32,
      "store_sectors": 32
    },
    "l2": {
      "hits": 24,
      "lookup_hits": 32,
      "lookup_requests": 64,
      "misses": 40
    },
    "launch_count": 1,
    "memory_mode": "l1-l2",
    "metric_prediction_count": 8,
    "records_seen": 79,
    "schema_version": 1,
    "status": "ok",
    "timing_status": "incomplete_unmodeled_retirement_tails"
  }
}
```

| sm120 scalar | `ok/sm_120` | `ok/sm_120` | yes |

### sm120 scalar

```json
{
  "baseline": {
    "architecture": "sm_120",
    "dram": {
      "read_sectors": 32,
      "service_rate_bytes_per_sec": null
    },
    "front_end": {
      "compulsory_constant_read_sectors": 3,
      "compulsory_instruction_read_sectors": 33
    },
    "instructions_issued": 87,
    "l1": {
      "hits": 0,
      "line_fills": 8,
      "merges": 24,
      "misses": 32,
      "store_sectors": 32
    },
    "l2": {
      "hits": 24,
      "lookup_hits": 0,
      "lookup_requests": 64,
      "misses": 40
    },
    "launch_count": 1,
    "memory_mode": "l1-l2",
    "metric_prediction_count": 8,
    "records_seen": 87,
    "schema_version": 1,
    "status": "ok",
    "timing_status": "incomplete_unmodeled_retirement_tails"
  },
  "unified": {
    "architecture": "sm_120",
    "dram": {
      "read_sectors": 32,
      "service_rate_bytes_per_sec": 1792000000000
    },
    "front_end": {
      "compulsory_constant_read_sectors": 3,
      "compulsory_instruction_read_sectors": 33
    },
    "instructions_issued": 87,
    "l1": {
      "hits": 0,
      "line_fills": 8,
      "merges": 24,
      "misses": 32,
      "store_sectors": 32
    },
    "l2": {
      "hits": 24,
      "lookup_hits": 0,
      "lookup_requests": 64,
      "misses": 40
    },
    "launch_count": 1,
    "memory_mode": "l1-l2",
    "metric_prediction_count": 8,
    "records_seen": 87,
    "schema_version": 1,
    "status": "ok",
    "timing_status": "incomplete_unmodeled_retirement_tails"
  }
}
```
