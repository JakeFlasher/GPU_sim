# CTA-barrier issue-pipeline probe

This one-CTA `sm_86` probe separates barrier instruction issue from CTA
arrival/rendezvous semantics. Its four campaign protocols are compile-time
kernel specializations:

| Protocol | Measured loop body | Purpose |
|---|---|---|
| `b00` | control arithmetic, no barrier | dispatch/control subtraction |
| `b10` | one CTA barrier per generation | barrier issue throughput |
| `b20` | two consecutive CTA barriers | generation/re-arm pressure |
| `b11` | warp-0 dependent skew chain, then one barrier | controlled arrival spread |

The diagnostic `cal` protocol applies the same dependent chain to every warp
without a barrier so its standalone cost can be measured. Skew 32 contains
five dependent integer SASS operations; skew 64 contains seven. The long
binary uses 16,384 generations and the bounded NVBit binary uses 64.

Build and run a correctness sample:

```bash
make -C workloads/research_probes/barrier_pipeline
workloads/research_probes/barrier_pipeline/barrier_pipeline b10 8 0 7
```

Arguments are `PROTOCOL WARPS SKEW [TRIALS]`. Campaign widths are W1-W8, W12,
and W16. `b00`, `b10`, and `b20` require skew 0; `b11` and `cal` require skew
32 or 64.

`multi_cta_barrier` holds eight active warps per SM while changing CTA shape:
one 8-warp CTA, two 4-warp CTAs, or four 2-warp CTAs. It requests 96, 48, or
24 KiB of dynamic shared memory per CTA and launches exactly `SM_count ×
CTAs_per_SM` blocks. Every run reports the observed block count for each SM and
fails correctness unless all 84 SMs receive the declared number of blocks.

Run its six-cell campaign with:

```bash
.venv/bin/python workloads/research_probes/barrier_pipeline/multi_cta_campaign.py correctness
.venv/bin/python workloads/research_probes/barrier_pipeline/multi_cta_campaign.py ncu
.venv/bin/python workloads/research_probes/barrier_pipeline/multi_cta_campaign.py source
.venv/bin/python workloads/research_probes/barrier_pipeline/multi_cta_campaign.py nvbit
.venv/bin/python workloads/research_probes/barrier_pipeline/multi_cta_campaign.py static
.venv/bin/python workloads/research_probes/barrier_pipeline/multi_cta_campaign.py simulator
.venv/bin/python workloads/research_probes/barrier_pipeline/multi_cta_analyze.py
```

The campaign matrix and model-selection procedure are defined in
`docs/STATUS.md`. NCU and NVBit run as
separate foreground processes; no profiler metric is read by simulator replay.

Run the resumable stages with:

```bash
.venv/bin/python workloads/research_probes/barrier_pipeline/campaign.py manifest
.venv/bin/python workloads/research_probes/barrier_pipeline/campaign.py correctness
.venv/bin/python workloads/research_probes/barrier_pipeline/campaign.py calibration
.venv/bin/python workloads/research_probes/barrier_pipeline/campaign.py ncu
.venv/bin/python workloads/research_probes/barrier_pipeline/campaign.py source
.venv/bin/python workloads/research_probes/barrier_pipeline/campaign.py nvbit
.venv/bin/python workloads/research_probes/barrier_pipeline/campaign.py static
cargo build --release -p sim-run --manifest-path sim/Cargo.toml
.venv/bin/python workloads/research_probes/barrier_pipeline/campaign.py simulator
.venv/bin/python workloads/research_probes/barrier_pipeline/analyze.py
```

After the barrier-only resource is selected, the coupled W2/W4/W8
async/barrier and launch-chain validation uses a separate 18-cell root:

```bash
.venv/bin/python workloads/research_probes/barrier_pipeline/coupled_campaign.py correctness
.venv/bin/python workloads/research_probes/barrier_pipeline/coupled_campaign.py ncu
.venv/bin/python workloads/research_probes/barrier_pipeline/coupled_campaign.py source
.venv/bin/python workloads/research_probes/barrier_pipeline/coupled_campaign.py nvbit
.venv/bin/python workloads/research_probes/barrier_pipeline/coupled_campaign.py static
.venv/bin/python workloads/research_probes/barrier_pipeline/coupled_campaign.py simulator
.venv/bin/python workloads/research_probes/barrier_pipeline/coupled_campaign.py production
.venv/bin/python workloads/research_probes/barrier_pipeline/coupled_campaign.py predecessor
.venv/bin/python workloads/research_probes/barrier_pipeline/coupled_analyze.py
```
