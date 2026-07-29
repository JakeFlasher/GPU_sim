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
//! SIM_LAUNCH_SET_SCOPE=<authority-bound ASCII scope> \
//!   cargo test --release -p sim-sm --test machine_local_replay \
//!   real_trace_replays_under_issue_model -- --ignored --exact --nocapture
//! ```
//!
//! The control-bits sidecar is located from the trace's own metadata (the
//! exec info's `benchmark_name` minus a trailing `_run` +
//! `.control_bits.json`). The ordinary workspace suite ignores this test; the
//! authoritative hardware gate supplies all three environment values and executes
//! it nonvacuously. Missing values on a direct ignored-test invocation fail.

use std::path::{Path, PathBuf};

use sim_arch::BuildArch;
use sim_calib::ArchCard;
use sim_mem::{L2Config, PchaseCard};
use sim_sm::{
    ControlSidecar, FullCoverageSource, FullLaunchSetDeclaration, IssueModelParams,
    KernelReplayOutcome, ReplayConfiguration, ReplayLaunchCoverageBinding, ReplayMemMode,
    aggregate_full_launch_set, build_schedules, replay_kernel,
};
use sim_trace::{ExecInfo, TraceDir, kernel_grid};

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

/// Build the declaration roster directly from the retained root protobuf,
/// independently of both `TraceDir::kernels` and replay-produced coverage.
fn expected_launch_roster(td: &TraceDir) -> Vec<ReplayLaunchCoverageBinding> {
    let mut roster = Vec::new();
    for (&device_id, device) in &td.trace().gpu_device {
        for (&stream_id, stream) in &device.streams {
            for kernel in &stream.kernels {
                let grid = kernel_grid(kernel).expect("launch grid is valid");
                let block = kernel
                    .block_dim
                    .as_ref()
                    .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
                    .map(|dim| (dim.x as u32, dim.y as u32, dim.z as u32));
                let cluster = match kernel.cluster_dim.as_ref() {
                    None => (1, 1, 1),
                    Some(dim) if (dim.x, dim.y, dim.z) == (0, 0, 0) => (1, 1, 1),
                    Some(dim) if dim.x > 0 && dim.y > 0 && dim.z > 0 => {
                        (dim.x as u32, dim.y as u32, dim.z as u32)
                    }
                    Some(dim) => panic!(
                        "invalid root cluster geometry ({}, {}, {})",
                        dim.x, dim.y, dim.z
                    ),
                };
                roster.push(
                    ReplayLaunchCoverageBinding::checked(
                        device_id,
                        stream_id,
                        kernel.id,
                        kernel.function_unique_id,
                        kernel.name.clone(),
                        grid,
                        block,
                        cluster,
                    )
                    .expect("root launch identity is declarable"),
                );
            }
        }
    }
    roster
}

fn l1_config_json(config: sim_mem::L1Config) -> serde_json::Value {
    serde_json::json!({
        "arch": config.arch.token(),
        "line_capacity": config.line_capacity,
        "mshr_count": config.mshr_count,
        "hit_latency": config.hit_latency,
        "miss_latency": config.miss_latency,
    })
}

fn l2_config_json(config: sim_mem::L2Config) -> serde_json::Value {
    let replacement = match config.replacement {
        sim_mem::L2Replacement::Lru => "lru",
        sim_mem::L2Replacement::Srrip => "srrip",
        sim_mem::L2Replacement::Bip => "bip",
    };
    let slice_hash = match config.slice_hash {
        sim_mem::SliceHash::LineMod => "line_mod",
        sim_mem::SliceHash::SectorMod => "sector_mod",
        sim_mem::SliceHash::XorFolded => "xor_folded",
        sim_mem::SliceHash::KnuthMult => "knuth_mult",
        sim_mem::SliceHash::ShiftMod => "shift_mod",
        sim_mem::SliceHash::BitReverse => "bit_reverse",
    };
    serde_json::json!({
        "arch": config.arch.token(),
        "capacity_bytes": config.capacity_bytes,
        "line_bytes": config.line_bytes,
        "sector_bytes": config.sector_bytes,
        "num_slices": config.num_slices,
        "associativity": config.associativity,
        "replacement": replacement,
        "slice_hash": slice_hash,
    })
}

fn replay_configuration_json(configuration: ReplayConfiguration) -> serde_json::Value {
    let issue = configuration.issue_params();
    let memory = match configuration.memory_mode() {
        ReplayMemMode::Flat => serde_json::json!({"kind": "flat"}),
        ReplayMemMode::L1Const(l1) => serde_json::json!({
            "kind": "l1_const",
            "l1": l1_config_json(l1),
        }),
        ReplayMemMode::L1SharedL2 { l1, l2, dram } => {
            let dram = match dram {
                sim_sm::DramBw::Unmodeled { arch } => serde_json::json!({
                    "kind": "unmodeled",
                    "arch": arch.token(),
                }),
                sim_sm::DramBw::Carded {
                    arch,
                    bytes_per_sec,
                    sm_clock_hz,
                } => serde_json::json!({
                    "kind": "carded",
                    "arch": arch.token(),
                    "bytes_per_sec": bytes_per_sec,
                    "sm_clock_hz": sm_clock_hz,
                }),
            };
            serde_json::json!({
                "kind": "l1_shared_l2",
                "l1": l1_config_json(l1),
                "l2": l2_config_json(l2),
                "dram": dram,
            })
        }
    };
    serde_json::json!({
        "issue": {
            "arch": issue.arch.token(),
            "l1_hit_latency": issue.l1_hit_latency,
            "l1_miss_latency": issue.l1_miss_latency,
            "global_flat_latency": issue.global_flat_latency,
            "shared_write_barrier_latency": issue.shared_write_barrier_latency,
            "non_memory_write_barrier_latency": issue.non_memory_write_barrier_latency,
            "read_barrier_latency": issue.read_barrier_latency,
            "max_resident_warps_per_sm": issue.max_resident_warps_per_sm,
        },
        "memory": memory,
    })
}

fn coverage_json(coverage: &sim_sm::FullLaunchCoverage) -> serde_json::Value {
    let launch = coverage.launch();
    let source = match coverage.source() {
        FullCoverageSource::UnsampledComplete => {
            serde_json::json!({"kind": "unsampled_complete"})
        }
        FullCoverageSource::LosslessCompressedFull(provenance) => serde_json::json!({
            "kind": "lossless_compressed_full",
            "selected_block_ids_sha256": provenance.selected_block_ids_sha256(),
            "sidecar_sha256": provenance.sidecar_sha256(),
        }),
    };
    serde_json::json!({
        "launch": {
            "device_id": launch.device_id(),
            "stream_id": launch.stream_id(),
            "kernel_id": launch.kernel_id(),
            "function_unique_id": launch.function_unique_id(),
            "kernel_mangled_name": launch.kernel_mangled_name(),
            "grid": [launch.grid().0, launch.grid().1, launch.grid().2],
            "block": launch.block().map(|block| [block.0, block.1, block.2]),
            "cluster": [launch.cluster().0, launch.cluster().1, launch.cluster().2],
        },
        "blocks_observed": coverage.blocks_observed(),
        "source": source,
    })
}

fn metric_json(prediction: &sim_emit::MetricPrediction) -> serde_json::Value {
    use sim_emit::{EmitOutcome, MemUnit, MetricValue, Unavailable, Unit};

    let unit = match prediction.unit() {
        Unit::Count => "count",
        Unit::Pct => "pct",
    };
    let outcome = match prediction.outcome() {
        EmitOutcome::Predicted(MetricValue::Exact(value)) => serde_json::json!({
            "kind": "predicted_exact",
            "value": value,
        }),
        EmitOutcome::Predicted(MetricValue::Fraction {
            value,
            numerator,
            denominator,
        }) => serde_json::json!({
            "kind": "predicted_fraction",
            "value": value,
            "numerator": numerator,
            "denominator": denominator,
        }),
        EmitOutcome::Unavailable(Unavailable::ArchitectureUnsupported { arch }) => {
            serde_json::json!({
                "kind": "unavailable_architecture",
                "arch": arch.token(),
            })
        }
        EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit { needs }) => {
            let needs = match needs {
                MemUnit::L1Tex => "l1tex",
                MemUnit::Lts => "lts",
                MemUnit::Dram => "dram",
            };
            serde_json::json!({
                "kind": "unavailable_mem_unit",
                "needs": needs,
            })
        }
    };
    serde_json::json!({
        "arch": prediction.arch().token(),
        "key": prediction.key(),
        "ncu_metric": prediction.ncu_metric(),
        "unit": unit,
        "outcome": outcome,
    })
}

fn emission_bundle_json(predictions: &sim_emit::FullLaunchSetPredictions) -> serde_json::Value {
    let estimator = match predictions.estimator_provenance() {
        sim_sm::FullCounterEstimatorProvenance::ExactFullCoverageReplay => {
            "exact_full_coverage_replay"
        }
    };
    let tails = predictions.unmodeled_tails();
    serde_json::json!({
        "schema": "sim_emit_full_launch_set_v1",
        "scope": predictions.scope().as_str(),
        "arch": predictions.arch().token(),
        "configuration": replay_configuration_json(predictions.configuration()),
        "launch_coverages": predictions
            .launch_coverages()
            .iter()
            .map(coverage_json)
            .collect::<Vec<_>>(),
        "estimator_provenance": estimator,
        "unmodeled_tails": {
            "red_rmw": tails.red_rmw,
            "dirty_writeback": tails.dirty_writeback,
            "coherence": tails.coherence,
        },
        "metrics": predictions.metrics().iter().map(metric_json).collect::<Vec<_>>(),
    })
}

#[test]
#[ignore = "requires machine-local M1 evidence; run tools/dev/hw_evidence_gate.sh"]
fn real_trace_replays_under_issue_model() {
    let trace_dir = std::env::var("SIM_TRACE_SMOKE_DIR")
        .expect("SIM_TRACE_SMOKE_DIR must be set by tools/dev/hw_evidence_gate.sh");
    let cb_dir = std::env::var("SIM_CONTROL_BITS_DIR")
        .expect("SIM_CONTROL_BITS_DIR must be set by tools/dev/hw_evidence_gate.sh");
    let launch_set_scope = std::env::var("SIM_LAUNCH_SET_SCOPE")
        .expect("SIM_LAUNCH_SET_SCOPE must be set by the hardware evidence owner");

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

    let configuration =
        ReplayConfiguration::checked(params, mode).expect("replay configuration is coherent");
    let mut kernels = 0usize;
    let mut diagnostic_last_issue_cycles: u64 = 0;
    let mut sms = 0usize;
    let expected_launches = expected_launch_roster(&td);
    let locations = td.kernels().collect::<Vec<_>>();
    assert_eq!(
        locations.len(),
        expected_launches.len(),
        "TraceDir kernel iterator must match the retained root roster"
    );
    let mut full_results = Vec::new();
    for loc in locations {
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
        assert_eq!(full.configuration(), configuration);
        kernels = kernels.checked_add(1).expect("kernel count fits usize");
        diagnostic_last_issue_cycles =
            diagnostic_last_issue_cycles.max(full.diagnostic_last_issue_cycles());
        sms = sms.max(full.per_sm().len());
        assert_eq!(full.instructions_issued(), full.records_seen());
        full_results.push(full);
    }
    let declaration =
        FullLaunchSetDeclaration::checked(launch_set_scope, configuration, expected_launches)
            .expect("machine-local launch roster is declarable");
    let counters = aggregate_full_launch_set(declaration, full_results)
        .expect("full launch-set counter aggregation succeeds");
    let records = counters.records_seen();
    let issued = counters.instructions_issued();
    eprintln!(
        "smoke: replay kernels={kernels} records={records} issued={issued} \
         diagnostic_last_issue_cycles={diagnostic_last_issue_cycles} sms={sms}"
    );
    if let Some(c) = counters.l1() {
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
    if let (Some(c), Some(s)) = (counters.l2(), counters.l2_by_source()) {
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
    if let Some(d) = counters.dram() {
        eprintln!(
            "smoke: dram read_sectors={} bw_delayed_responses={}",
            d.read_sectors, d.bw_delayed_responses
        );
    }

    // Rates compose from summed numerators/denominators, so emission runs on
    // the checked aggregate. One JSON record retains every binding field with
    // the metric vector; downstream must never consume detached key/value lines.
    let predictions = sim_emit::emit_all(&counters).expect("emission contracts hold");
    let bundle = serde_json::to_string(&emission_bundle_json(&predictions))
        .expect("emission bundle serializes");
    eprintln!("smoke: emit_bundle={bundle}");

    assert!(kernels > 0 && issued > 0 && diagnostic_last_issue_cycles > 0 && sms > 0);
}
