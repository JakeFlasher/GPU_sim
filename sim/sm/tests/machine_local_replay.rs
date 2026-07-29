#![allow(clippy::unwrap_used)] // test binary: unwrap is an assertion
//! Machine-local replay smoke: join a real captured trace with its
//! control-bits sidecar and replay it end-to-end under the issue model
//! with the sim-mem L1 attached (cache-op roles come from the sidecar's
//! own opcode text; geometry from the committed pchase static card).
//!
//! The M1 evidence is gitignored machine-local data, so inputs come from
//! the environment (Rule-1 guard: no annex/label path literals inside
//! sim/):
//!
//! ```sh
//! SIM_TRACE_SMOKE_DIR=<trace_<shape> dir> \
//! SIM_CONTROL_BITS_DIR=<control-bits sidecar dir> \
//!   cargo test --release -p sim-sm --test machine_local_replay \
//!   real_trace_replays_under_issue_model -- --ignored --exact --nocapture
//! ```
//!
//! The control-bits sidecar is located from the trace's own metadata (the
//! exec info's `benchmark_name` minus a trailing `_run` +
//! `.control_bits.json`). The ordinary workspace suite ignores this test; the
//! authoritative hardware gate supplies both environment values and executes
//! it nonvacuously. Missing values on a direct ignored-test invocation fail.

use std::path::{Path, PathBuf};

use sim_arch::BuildArch;
use sim_calib::ArchCard;
use sim_mem::{L2Config, PchaseCard};
use sim_sm::{
    ControlSidecar, IssueModelParams, KernelReplayOutcome, ReplayMemMode, build_schedules,
    replay_kernel,
};
use sim_trace::{ExecInfo, TraceDir};

fn sm86_card() -> ArchCard {
    // The card is COMMITTED repo data (not machine-local evidence), so an
    // in-repo path is fine here. The M1 traces are sm_86 captures.
    let path = PathBuf::from("../../arch_profiles/cards/sm_86.json");
    ArchCard::load(&path).expect("committed sm_86 card loads")
}

fn sm86_pchase_card() -> PchaseCard {
    // Committed P-chase static card: the L1 geometry source the Phase-1
    // walker resolves from (capacity has no default; MSHR documented
    // default when typed).
    let path = PathBuf::from("../../data/pchase/sm_86_static_card.json");
    PchaseCard::load(&path).expect("committed pchase card loads")
}

#[test]
#[ignore = "requires machine-local M1 evidence; run tools/dev/hw_evidence_gate.sh"]
fn real_trace_replays_under_issue_model() {
    let trace_dir = std::env::var("SIM_TRACE_SMOKE_DIR")
        .expect("SIM_TRACE_SMOKE_DIR must be set by tools/dev/hw_evidence_gate.sh");
    let cb_dir = std::env::var("SIM_CONTROL_BITS_DIR")
        .expect("SIM_CONTROL_BITS_DIR must be set by tools/dev/hw_evidence_gate.sh");

    let td = TraceDir::open(&trace_dir).expect("open trace dir");
    let exec = ExecInfo::load_from_trace_dir(td.root()).expect("exec info loads");
    let card = sm86_card();
    let identity = card
        .identity(BuildArch::Sm86)
        .expect("sm_86 card identity resolves");
    let stem = exec
        .benchmark_name
        .strip_suffix("_run")
        .unwrap_or(&exec.benchmark_name);
    let sidecar_path = Path::new(&cb_dir).join(format!("{stem}.control_bits.json"));
    let sidecar =
        ControlSidecar::load_for_arch(&sidecar_path, identity.deploy()).expect("sidecar loads");

    let schedules = build_schedules(&exec, &sidecar).expect("(function, pc) join");

    let lat = card
        .required_latencies()
        .expect("sm_86 card has no latency gaps");
    let params = IssueModelParams::from_card_latencies_for_arch(
        identity.deploy(),
        lat.l1_hit,
        lat.l2_hit,
        lat.dram,
        lat.shared,
    )
    .expect("sm_86 issue parameters resolve");

    // L1 mode: geometry (capacity + MSHR) from the committed pchase
    // static card's fail-closed resolvers; the hit/miss LATENCIES come
    // from `params` — one source, so the pair can never drift. The
    // resolvers' own latency audit (hit strictly below miss) still runs
    // as part of geometry resolution. SIM_L1_DISABLE=1 exercises the
    // flat-latency ablation path end-to-end (the owner runner runs one
    // cell in this mode so `classify_flat` never goes machine-untested).
    // Typed mode selection: flat (SIM_L1_DISABLE=1) → L1-const
    // (SIM_L2_DISABLE=1, the pre-L2 lockstep surface) → default full
    // hierarchy (shared L2 from the arch card's vendor-documented
    // `l2_geometry`; assoc/hash/policy documented defaults).
    let mode = if std::env::var("SIM_L1_DISABLE").is_ok() {
        ReplayMemMode::Flat
    } else {
        let pchase = sm86_pchase_card();
        assert_eq!(pchase.arch(), identity.deploy());
        let geometry = pchase
            .resolve_walk_geometry_for_arch(identity.deploy(), None)
            .expect("pchase geometry resolves");
        let l1 = params.l1_config(geometry.capacity.line_capacity, geometry.mshr.count);
        if std::env::var("SIM_L2_DISABLE").is_ok() {
            ReplayMemMode::L1Const(l1)
        } else {
            let g = card
                .require_l2_geometry()
                .expect("sm_86 card l2_geometry present");
            let l2 = L2Config::from_carded_geometry_for_arch(
                identity.deploy(),
                g.capacity_bytes,
                g.line_bytes,
                g.sector_bytes,
                g.num_slices,
                g.policy.as_deref(),
            )
            .expect("carded L2 geometry valid");
            // DRAM read-bandwidth covariate from the card's locked
            // operating point (fail-closed: a card without the block
            // cannot run the full hierarchy). SIM_DRAM_BW_DISABLE=1 is
            // the latency-only ablation (the pre-covariate model).
            let dram = if std::env::var("SIM_DRAM_BW_DISABLE").is_ok() {
                sim_sm::DramBw::Unmodeled {
                    arch: identity.deploy(),
                }
            } else {
                let bw = card
                    .require_dram_bandwidth()
                    .expect("sm_86 card dram_bandwidth present");
                sim_sm::DramBw::Carded {
                    arch: identity.deploy(),
                    bytes_per_sec: bw
                        .validated_peak_bytes_per_second()
                        .expect("carded peak rate valid"),
                    sm_clock_hz: bw.locked_sm_clock_hz().expect("carded SM clock valid"),
                }
            };
            ReplayMemMode::L1SharedL2 { l1, l2, dram }
        }
    };

    let mut kernels = 0usize;
    let mut records: u64 = 0;
    let mut issued: u64 = 0;
    let mut diagnostic_last_issue_cycles: u64 = 0;
    let mut sms = 0usize;
    let mut l1_totals: Option<sim_mem::L1Counters> = None;
    let mut l2_totals: Option<sim_mem::L2Counters> = None;
    let mut split_totals = sim_sm::L2SourceSplit::for_arch(identity.deploy());
    let mut dram_totals: Option<sim_sm::DramStats> = None;
    for loc in td.kernels().collect::<Vec<_>>() {
        let outcome = replay_kernel(&td, &loc, &schedules, &params, mode).expect("replay");
        let full = match outcome {
            KernelReplayOutcome::Full(full) => full,
            KernelReplayOutcome::SampledObserved(sampled) => panic!(
                "machine-local full replay received sampled coverage {}/{}",
                sampled.coverage().blocks_observed(),
                sampled.coverage().population_blocks()
            ),
        };
        assert_eq!(
            full.arch(),
            identity.deploy(),
            "every launch result must retain the composition architecture"
        );
        kernels = kernels.checked_add(1).expect("kernel count fits usize");
        records = records
            .checked_add(full.records_seen())
            .expect("record count fits u64");
        issued = issued
            .checked_add(full.instructions_issued())
            .expect("instruction count fits u64");
        diagnostic_last_issue_cycles =
            diagnostic_last_issue_cycles.max(full.diagnostic_last_issue_cycles());
        sms = sms.max(full.per_sm().len());
        assert_eq!(full.instructions_issued(), full.records_seen());
        if let Some(c) = full.l1() {
            match &mut l1_totals {
                Some(total) => total.add_checked(c).expect("matching L1 architecture"),
                None => l1_totals = Some(*c),
            }
        }
        if let Some(c) = full.l2() {
            match &mut l2_totals {
                Some(total) => total.add_checked(c).expect("matching L2 architecture"),
                None => l2_totals = Some(*c),
            }
        }
        if let Some(split) = full.l2_by_source() {
            split_totals
                .add_checked(split)
                .expect("matching L2 source architecture");
        }
        if let Some(d) = full.dram() {
            match &mut dram_totals {
                Some(total) => total.add_checked(d).expect("matching DRAM architecture"),
                None => dram_totals = Some(*d),
            }
        }
    }
    eprintln!(
        "smoke: replay kernels={kernels} records={records} issued={issued} \
         diagnostic_last_issue_cycles={diagnostic_last_issue_cycles} sms={sms}"
    );
    if let Some(c) = &l1_totals {
        eprintln!(
            "smoke: l1 hits={} misses={} merges={} stalls={} line_alloc={} sector_res={} \
             bypass={} bypass_atom={} untagged={} store={} store_hits={} store_tag_hits={}",
            c.hits,
            c.misses,
            c.merges,
            c.stall_events,
            c.line_alloc_misses,
            c.sector_resident_misses,
            c.bypass_sectors,
            c.bypass_atom_sectors,
            c.untagged_sectors,
            c.store_sectors,
            c.store_hits,
            c.store_tag_hits,
        );
    }
    if let Some(c) = &l2_totals {
        let s = split_totals;
        eprintln!(
            "smoke: l2 hits={} misses={} line_misses={} sector_fill_misses={} write_tag_hits={} \
             fill=({},{}) bypass=({},{}) untagged=({},{}) store=({},{})",
            c.hits,
            c.misses,
            c.line_misses,
            c.sector_fill_misses,
            c.write_tag_hits,
            s.sectors[0],
            s.hits[0],
            s.sectors[1],
            s.hits[1],
            s.sectors[2],
            s.hits[2],
            s.sectors[3],
            s.hits[3],
        );
    }
    if let Some(d) = &dram_totals {
        eprintln!(
            "smoke: dram read_sectors={} bw_delayed_responses={}",
            d.read_sectors, d.bw_delayed_responses
        );
    }

    // NCU-metric emission over the cell's summed counters (sim-emit contract
    // layer). Rates compose from summed numerators/denominators — so emission
    // runs on the AGGREGATE, never on per-kernel rates. One machine-parseable
    // line per metric drives the accuracy sweep (labels stay outside sim/).
    let snap = sim_emit::CounterSnapshot {
        arch: identity.deploy(),
        instructions_issued: issued,
        l1: l1_totals.as_ref(),
        l2: l2_totals.as_ref(),
        l2_by_source: l2_totals.as_ref().map(|_| &split_totals),
        dram: dram_totals.as_ref(),
    };
    for p in sim_emit::emit_all(&snap).expect("emission contracts hold") {
        use sim_emit::{EmitOutcome, MetricValue};
        let payload = match p.outcome {
            EmitOutcome::Predicted(MetricValue::Exact(v)) => format!("count:{v}"),
            EmitOutcome::Predicted(MetricValue::Fraction {
                value,
                numerator,
                denominator,
            }) => match value {
                Some(x) => format!("frac:{x}:{numerator}/{denominator}"),
                None => format!("frac:na:{numerator}/{denominator}"),
            },
            EmitOutcome::Unavailable(u) => format!("unavailable:{u:?}"),
        };
        eprintln!("smoke: emit {} = {}", p.key, payload);
    }

    assert!(kernels > 0 && issued > 0 && diagnostic_last_issue_cycles > 0 && sms > 0);
}
