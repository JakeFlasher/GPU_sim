#![allow(clippy::unwrap_used)] // integration-test assertions

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use prost::Message;
use serde_json::{Value, json};
use sim_arch::ArchId;
use sim_mem::L2Config;
use sim_run::emit::{
    EMITTED_METRIC_COUNT, EmitOutcome, EmittedMetricKey, MemUnit, MetricValue,
    RequiredUnimplementedBlocker, Unavailable, emit_all, emit_launch, index_by_key,
};
use sim_run::run_manifest;
use sim_sm::{
    AdmittedLaunchEvidence, DramBw, FullKernelReplayResult, FullLaunchSetDeclaration,
    IssueModelParams, KernelReplayOutcome, ReplayMemMode, admit_launch_evidence,
    aggregate_full_launch_set, replay_kernel,
};
use sim_trace::{Sha256Digest, TraceDirSnapshot, canonical_json, canonical_json_utf8};
use tempfile::TempDir;

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/addr_carry_canary_bundle")
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type().unwrap();
        if file_type.is_dir() {
            copy_tree(&source_path, &destination_path);
        } else {
            assert!(
                file_type.is_file(),
                "fixture must contain regular files only"
            );
            fs::copy(source_path, destination_path).unwrap();
        }
    }
}

fn copied_fixture() -> (TempDir, PathBuf) {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("bundle");
    copy_tree(&fixture_root(), &root);
    let manifest = root.join("run_manifest.v2.json");
    (temp, manifest)
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn artifact_claim(root: &Path, relative: &str) -> Value {
    let bytes = fs::read(root.join(relative)).unwrap();
    json!({
        "path": relative,
        "size_bytes": bytes.len(),
        "sha256": Sha256Digest::of(&bytes),
    })
}

fn refresh_trace_artifact_claims(root: &Path, launch_manifest: &mut Value) {
    let snapshot = TraceDirSnapshot::capture(root.join("trace")).unwrap();
    let payload = snapshot.artifact_identity().threadblock_payload();
    launch_manifest["artifacts"]["dynamic_trace_pb"] =
        artifact_claim(root, "trace/dynamic_trace.pb");
    launch_manifest["artifacts"]["threadblock_payload"] = json!({
        "path": "trace/threadblocks",
        "file_count": payload.file_count(),
        "raw_bytes": payload.raw_bytes(),
        "sorted_path_manifest_sha256": payload.sorted_path_manifest_sha256(),
        "sorted_path_stream_sha256": payload.sorted_path_stream_sha256(),
    });
    launch_manifest["artifacts"]["stats_csv"] = artifact_claim(root, "trace/stats.csv");
}

const CANARY_KERNEL_NAME: &str = "_Z17addr_carry_canaryPKiPiS1_S1_i___0";

fn fixture_launch_key_for_stream(process_launch_ordinal: u64, stream_id: u64) -> Value {
    json!({
        "workload_id": "fixture-only.addr-carry-canary",
        "matched_execution_id": "fixture-only.execution.addr-carry-canary",
        "process_launch_ordinal": process_launch_ordinal,
        "kernel_mangled_name": CANARY_KERNEL_NAME,
        "grid": {"x": 1, "y": 1, "z": 1},
        "block": {"x": 32, "y": 1, "z": 1},
        "device_id": 0,
        "stream_id": stream_id,
        "binary_architecture": "sm_86"
    })
}

fn fixture_launch_key(process_launch_ordinal: u64) -> Value {
    fixture_launch_key_for_stream(process_launch_ordinal, 0)
}

const SAMPLED_BLOCKS: [i32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59];
const CYCLE_STUDY_SECTIONS: [&str; 4] = [
    "D1 — NCU observable contract",
    "D2 — Simulator diagnostic candidates",
    "D3 — SM/GPC topology and spatial mapping",
    "D4 — Launch composition and emission boundary",
];
const CYCLE_STUDY_SUBSECTIONS: [&str; 4] = [
    "Target-free findings",
    "Hypotheses only",
    "Silicon-blocked decisions",
    "Promotion forbidden until",
];

fn replay_modes(
    evidence: &AdmittedLaunchEvidence,
) -> (IssueModelParams, [(ReplayMemMode, &'static str); 3]) {
    let arch = ArchId::Sm86;
    let latencies = evidence
        .arch_card()
        .required_latencies()
        .expect("fixture has carded latencies");
    let issue_params = IssueModelParams::from_legacy_card_latencies_for_arch(
        arch,
        latencies.l1_hit,
        latencies.l2_hit,
        latencies.dram,
        latencies.shared,
    )
    .expect("fixture card resolves strict issue parameters");
    let geometry = evidence
        .pchase_card()
        .resolve_walk_geometry_for_arch(arch, None)
        .expect("fixture pchase card resolves L1 geometry");
    let l1 = issue_params.l1_config(geometry.capacity.line_capacity, geometry.mshr.count);
    let card_l2 = evidence
        .arch_card()
        .require_l2_geometry()
        .expect("fixture has carded L2 geometry");
    let l2 = L2Config::from_carded_geometry_for_arch(
        arch,
        card_l2.capacity_bytes,
        card_l2.line_bytes,
        card_l2.sector_bytes,
        card_l2.num_slices,
        card_l2.policy.as_deref(),
    )
    .expect("fixture card resolves L2 configuration");
    (
        issue_params,
        [
            (ReplayMemMode::Flat, "flat"),
            (ReplayMemMode::L1Const(l1), "l1-const"),
            (
                ReplayMemMode::L1SharedL2 {
                    l1,
                    l2,
                    dram: DramBw::Unmodeled { arch },
                },
                "l1-shared-l2",
            ),
        ],
    )
}

fn full_replay(
    evidence: &AdmittedLaunchEvidence,
    issue_params: &IssueModelParams,
    mode: ReplayMemMode,
) -> FullKernelReplayResult {
    match replay_kernel(evidence, issue_params, mode).expect("fixture replay succeeds") {
        KernelReplayOutcome::Full(result) => result,
        KernelReplayOutcome::SampledObserved(_) => {
            panic!("unsampled fixture must produce full coverage")
        }
    }
}

#[derive(Clone, Copy)]
struct ExpectedModeledOutcomes {
    instructions: u64,
    global_load_sectors: u64,
    global_store_sectors: u64,
    l1_read_fraction: (Option<f64>, u64, u64),
    l1_unified_fraction: (Option<f64>, u64, u64),
    l2_fraction: (Option<f64>, u64, u64),
    dram_read_sectors: u64,
}

fn exact(value: u64) -> EmitOutcome {
    EmitOutcome::Predicted(MetricValue::Exact(value))
}

fn fraction(value: (Option<f64>, u64, u64)) -> EmitOutcome {
    EmitOutcome::Predicted(MetricValue::Fraction {
        value: value.0,
        numerator: value.1,
        denominator: value.2,
    })
}

fn unavailable(needs: MemUnit) -> EmitOutcome {
    EmitOutcome::Unavailable(Unavailable::MemModeLacksUnit { needs })
}

fn expected_mode_outcomes(
    mode: ReplayMemMode,
    expected: ExpectedModeledOutcomes,
) -> BTreeMap<EmittedMetricKey, EmitOutcome> {
    let (global_loads, global_stores, l1_read, l1_unified) = match mode {
        ReplayMemMode::Flat => (
            unavailable(MemUnit::L1Tex),
            unavailable(MemUnit::L1Tex),
            unavailable(MemUnit::L1Tex),
            unavailable(MemUnit::L1Tex),
        ),
        ReplayMemMode::L1Const(_) | ReplayMemMode::L1SharedL2 { .. } => (
            exact(expected.global_load_sectors),
            exact(expected.global_store_sectors),
            fraction(expected.l1_read_fraction),
            fraction(expected.l1_unified_fraction),
        ),
    };
    let (l2, dram) = match mode {
        ReplayMemMode::Flat | ReplayMemMode::L1Const(_) => {
            (unavailable(MemUnit::Lts), unavailable(MemUnit::Dram))
        }
        ReplayMemMode::L1SharedL2 { .. } => (
            fraction(expected.l2_fraction),
            exact(expected.dram_read_sectors),
        ),
    };
    BTreeMap::from([
        (
            EmittedMetricKey::SmspInstExecutedSum,
            exact(expected.instructions),
        ),
        (EmittedMetricKey::TrafficGmemLdSectors, global_loads),
        (EmittedMetricKey::TrafficGmemStSectors, global_stores),
        (EmittedMetricKey::TrafficDramReadSectors, dram),
        (EmittedMetricKey::CacheHitRateL1TexReadSector, l1_read),
        (EmittedMetricKey::CacheHitRateL1TexSector, l1_unified),
        (EmittedMetricKey::CacheHitRateL2Sector, l2),
        (
            EmittedMetricKey::ExecCyclesGpcMax,
            EmitOutcome::RequiredUnimplemented {
                blocking: RequiredUnimplementedBlocker::AggregationSemanticsStudy,
            },
        ),
    ])
}

fn assert_full_launch_emission_matches_singleton_aggregate(
    result: FullKernelReplayResult,
    mode_name: &str,
    expected: ExpectedModeledOutcomes,
) {
    // BL-20260708-vacuous-trace-validator: these counts are taken from the exact
    // replay result consumed below, not from a separate fixture declaration.
    assert!(result.coverage().blocks_observed() > 0);
    assert!(result.records_seen() > 0);
    assert_eq!(result.instructions_issued(), result.records_seen());

    let per_launch = emit_launch(&result).expect("emit full launch");
    assert_eq!(per_launch.configuration(), result.configuration());
    assert_eq!(per_launch.arch(), result.arch());
    assert_eq!(per_launch.coverage(), result.coverage());
    assert_eq!(
        per_launch.diagnostic_last_issue_cycles(),
        result.diagnostic_last_issue_cycles()
    );
    assert_eq!(
        per_launch.terminal_modeled_event_time(),
        result.terminal_modeled_event_time()
    );
    assert_eq!(per_launch.unmodeled_tails(), result.unmodeled_tails());
    assert!(per_launch.diagnostic_last_issue_cycles() > 0);
    assert!(per_launch.terminal_modeled_event_time().is_some());

    let declaration = FullLaunchSetDeclaration::checked(
        format!("fixture/per-launch-emission-{mode_name}:v1"),
        result.configuration(),
        [result.coverage().launch().clone()],
    )
    .expect("declare singleton full launch set");
    let aggregate = aggregate_full_launch_set(declaration, [result.clone()])
        .expect("aggregate singleton full launch set");
    assert_eq!(
        aggregate.l2().map(|counters| counters.lookup_hits),
        result.l2().map(|counters| counters.lookup_hits),
        "singleton aggregation must preserve L2 lookup hits"
    );
    let aggregate_predictions = emit_all(&aggregate).expect("emit singleton aggregate");

    let per_launch_keys = per_launch
        .metrics()
        .iter()
        .map(|prediction| prediction.emitted_key())
        .collect::<Vec<_>>();
    assert_eq!(per_launch_keys, EmittedMetricKey::V2_ORDER);

    let expected_outcomes = expected_mode_outcomes(result.configuration().memory_mode(), expected);
    assert_eq!(per_launch.metrics().len(), EMITTED_METRIC_COUNT);
    assert_eq!(expected_outcomes.len(), EMITTED_METRIC_COUNT);
    let aggregate_by_key = index_by_key(&aggregate_predictions);
    for expected_key in EmittedMetricKey::V2_ORDER {
        let prediction = per_launch
            .metrics()
            .iter()
            .find(|prediction| prediction.emitted_key() == expected_key)
            .expect("frozen v2 key is present");
        let expected_outcome = &expected_outcomes[&expected_key];
        assert_eq!(
            prediction,
            aggregate_by_key[expected_key.as_str()],
            "singleton mode={mode_name}, key={expected_key:?}"
        );
        assert_eq!(
            prediction.outcome(),
            expected_outcome,
            "mode={mode_name}, key={expected_key:?}"
        );
        assert_eq!(
            aggregate_by_key[expected_key.as_str()].outcome(),
            expected_outcome,
            "aggregate mode={mode_name}, key={expected_key:?}"
        );
    }
}

fn replace_trace_with_one_nonmemory_instruction(root: &Path) {
    let instruction = sim_trace::pb::Instruction {
        pc: 16,
        active_mask: Some(1),
        predicate_mask: Some(1),
        function_unique_id: 1,
        vpc: Some(16),
        sm_id: Some(0),
        warpid_sm: Some(0),
        warpid_tb: Some(0),
        ..Default::default()
    };
    let warp = sim_trace::pb::Warp {
        id: 0,
        instructions: vec![instruction],
    };
    let threadblock = sim_trace::pb::Threadblock {
        block_id: Some(sim_trace::pb::Dim3d { x: 0, y: 0, z: 0 }),
        warps: [(0_i32, warp)].into_iter().collect(),
    };
    let payload_path =
        root.join("trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
    fs::write(payload_path, threadblock.encode_to_vec()).unwrap();
    fs::write(
        root.join("trace/stats.csv"),
        b"total_insts,total_reported_insts\n1,1\n",
    )
    .unwrap();

    let launch_manifest_path = root.join("launch_evidence.json");
    let mut launch_manifest = read_json(&launch_manifest_path);
    refresh_trace_artifact_claims(root, &mut launch_manifest);
    write_json(&launch_manifest_path, &launch_manifest);
}

#[derive(Debug)]
struct StudySubsection {
    heading: String,
    body: String,
}

#[derive(Debug)]
struct StudySection {
    heading: String,
    subsections: Vec<StudySubsection>,
}

fn cycle_study_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/cycles_semantics_study.md")
}

fn parse_cycle_study(markdown: &str) -> Result<Vec<StudySection>, String> {
    let mut sections = Vec::new();
    let mut current: Option<StudySection> = None;

    for line in markdown.lines() {
        if let Some(heading) = line.strip_prefix("## ") {
            if let Some(section) = current.take() {
                sections.push(section);
            }
            current = Some(StudySection {
                heading: heading.to_owned(),
                subsections: Vec::new(),
            });
            continue;
        }

        if let Some(heading) = line.strip_prefix("### ") {
            let section = current
                .as_mut()
                .ok_or_else(|| format!("subsection {heading:?} precedes every study section"))?;
            section.subsections.push(StudySubsection {
                heading: heading.to_owned(),
                body: String::new(),
            });
            continue;
        }

        let Some(section) = current.as_mut() else {
            continue;
        };
        let Some(subsection) = section.subsections.last_mut() else {
            if line.trim().is_empty() {
                continue;
            }
            return Err(format!(
                "study section {:?} has content outside an exact subsection",
                section.heading
            ));
        };
        subsection.body.push_str(line);
        subsection.body.push('\n');
    }

    if let Some(section) = current {
        sections.push(section);
    }
    Ok(sections)
}

fn study_section_text(section: &StudySection) -> String {
    let mut text = String::new();
    for subsection in &section.subsections {
        text.push_str(&subsection.body);
    }
    text
}

fn sampled_threadblock(block_x: i32) -> Vec<u8> {
    let instruction = sim_trace::pb::Instruction {
        pc: 16,
        active_mask: Some(1),
        predicate_mask: Some(1),
        function_unique_id: 1,
        vpc: Some(16),
        sm_id: Some(0),
        warpid_sm: Some(0),
        warpid_tb: Some(0),
        ..Default::default()
    };
    let warp = sim_trace::pb::Warp {
        id: 0,
        instructions: vec![instruction],
    };
    sim_trace::pb::Threadblock {
        block_id: Some(sim_trace::pb::Dim3d {
            x: block_x,
            y: 0,
            z: 0,
        }),
        warps: [(0_i32, warp)].into_iter().collect(),
    }
    .encode_to_vec()
}

fn convert_to_sampled_fixture(root: &Path, launch_manifest: &mut Value) {
    let dynamic_path = root.join("trace/dynamic_trace.pb");
    let mut trace =
        sim_trace::pb::Trace::decode(fs::read(&dynamic_path).unwrap().as_slice()).unwrap();
    trace
        .gpu_device
        .get_mut(&0)
        .unwrap()
        .streams
        .get_mut(&0)
        .unwrap()
        .kernels
        .first_mut()
        .unwrap()
        .grid_dim = Some(sim_trace::pb::Dim3d { x: 60, y: 1, z: 1 });
    fs::write(&dynamic_path, trace.encode_to_vec()).unwrap();

    let trace_root = root.join("trace");
    let threadblocks_root = trace_root.join("threadblocks");
    fs::remove_dir_all(&threadblocks_root).unwrap();
    let mut payloads = Vec::new();
    for block_x in SAMPLED_BLOCKS {
        let relative =
            format!("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_{block_x},0,0.pb");
        let raw = sampled_threadblock(block_x);
        let path = trace_root.join(&relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, &raw).unwrap();
        payloads.push((relative, block_x, raw));
    }
    payloads.sort_by(|left, right| left.0.cmp(&right.0));

    let mut raw_bytes = 0_u64;
    let mut certainty_bytes = 0_u64;
    let mut stream = Vec::new();
    let mut payload_manifest = String::from("trace-payload-file-manifest-v1\n");
    for (relative, block_x, raw) in &payloads {
        let size = u64::try_from(raw.len()).unwrap();
        raw_bytes = raw_bytes.checked_add(size).unwrap();
        if [0, 1, 2, 57, 58, 59].contains(block_x) {
            certainty_bytes = certainty_bytes.checked_add(size).unwrap();
        }
        stream.extend_from_slice(raw);
        payload_manifest.push_str(&format!("{relative}\t{size}\t{}\n", Sha256Digest::of(raw)));
    }

    let stats_raw = b"total_insts,total_reported_insts\n60,12\n";
    fs::write(trace_root.join("stats.csv"), stats_raw).unwrap();
    let selected_json = canonical_json(&json!(SAMPLED_BLOCKS)).unwrap();
    let certainty_scaled = certainty_bytes.checked_mul(5).unwrap();
    let max_trace_bytes = raw_bytes.max(certainty_scaled.checked_add(1).unwrap() / 2);
    let full_roster = (0_u64..60).collect::<Vec<_>>();
    let mut sidecar = json!({
        "schema_version": 2,
        "cell_id": "runner-sampled-fixture",
        "kernel": {
            "mangled_name": CANARY_KERNEL_NAME,
            "args_hash": "fixture-args-hash-unavailable-test-token",
        },
        "method": "wave_segment_stratified_subset",
        "reason": "byte_budget",
        "shape": {
            "original_grid": {"x": 60, "y": 1, "z": 1},
            "original_block": {"x": 32, "y": 1, "z": 1},
            "captured_grid": {"x": 60, "y": 1, "z": 1},
            "captured_block": {"x": 32, "y": 1, "z": 1},
            "reduced_shape_variant": null,
        },
        "gates": {
            "coop": false,
            "wave_count": 60,
            "n_measured_waves": 40,
            "tail_measured_waves": 2,
            "atomic_density": 0.0,
            "hotline_share": 0.0,
            "skew_top1pct_share": 0.0,
            "gini": 0.0,
            "per_sm_block_count_cv": 0.0,
            "n_phases": 1,
            "skew_heavy_block_count": 0,
            "certainty_bytes": certainty_bytes,
            "max_trace_bytes": max_trace_bytes,
            "certainty_budget_exceeded": false,
            "verdict": "subset_admissible",
        },
        "wave_model": {
            "reconstruction": "launch_index",
            "per_sm_resident_blocks": [{"sm_id": 0, "resident_blocks": 1}],
            "n_waves": 60,
            "segment_len": 3,
            "ghost_depth": 1,
        },
        "per_sm_slot_map": [{"sm_id": 0, "block_ids": full_roster}],
        "rng": {"algo": "pcg64", "seed": 0},
        "tracer": {
            "version": 5,
            "features": {"atomic_addr": true, "ldgsts_addr": true},
            "stats_pass_version": 2,
        },
        "bytes": {
            "raw": raw_bytes,
            "compressed": raw_bytes,
            "codec": "zstd",
            "level": 19,
            "long": 27,
            "dict_id": null,
        },
        "records": {"expected_full": 60, "captured": 12},
        "hashes": {
            "selected_block_ids_sha256": Sha256Digest::of(selected_json.as_bytes()),
            "decompressed_payload_sha256": Sha256Digest::of(&stream),
            "per_block_raw_sha256_manifest": payload_manifest,
        },
        "label_pairing": "full_shape_label",
        "segments": [
            {"id": 0, "block_ids": [0, 1, 2]},
            {"id": 19, "block_ids": [57, 58, 59]},
            {"id": 1, "block_ids": [3, 4, 5]},
            {"id": 2, "block_ids": [6, 7, 8]},
        ],
        "selected_segment_ids": [0, 19, 1, 2],
        "certainty_segments": [0, 19],
        "force_include_blocks": [],
        "strata": [{
            "id": 0,
            "phase_id": 0,
            "launch_tercile": 0,
            "segment_ids": [1, 2],
            "N_h": 6,
            "n_h": 2,
            "pi": 0.3333333333333333,
            "sector_stddev": 0.0,
            "neyman_weight": 0.0,
        }],
        "selection_reconciliation": {
            "status": "fixed_point",
            "attempts": 1,
            "max_attempts": 3,
            "capture_stats_sha256": Sha256Digest::of(stats_raw),
        },
        "selection_budget": {
            "max_trace_bytes": max_trace_bytes,
            "basis": "raw_threadblock_pb",
            "certainty_fraction_numerator": 2,
            "certainty_fraction_denominator": 5,
            "certainty_bytes": certainty_bytes,
        },
        "selection_auxiliary": {
            "name": "l2_sector_transactions_32b",
            "sector_bytes": 32,
            "scope": "pass2_all_blocks",
            "expected_full": 0,
            "selected": 0,
        },
        "estimator": {
            "mode": "weighted_full_shape_estimate",
            "targets": ["sectors"],
            "not_recoverable": ["cycles"],
            "convergence": {"status": "converged"},
        },
    });
    let unsigned = canonical_json(&sidecar).unwrap();
    sidecar["hashes"]["sidecar_sha256"] = json!(Sha256Digest::of(unsigned.as_bytes()));
    let sampling_path = trace_root.join("trace_sampling.json");
    write_json(&sampling_path, &sidecar);

    launch_manifest["artifacts"]["trace_sampling_json"] = json!({
        "status": "present",
        "artifact": artifact_claim(root, "trace/trace_sampling.json"),
    });
    refresh_trace_artifact_claims(root, launch_manifest);
}

#[test]
fn p8_cycle_study_and_real_runner_guard_are_nonvacuous() {
    let study_path = cycle_study_path();
    let markdown = fs::read_to_string(&study_path).expect("read checked-in P8 cycle study");
    let sections = parse_cycle_study(&markdown).expect("parse P8 cycle study sections");

    assert_eq!(
        sections.len(),
        CYCLE_STUDY_SECTIONS.len(),
        "study must contain exactly D1-D4"
    );
    for (section_index, expected_heading) in CYCLE_STUDY_SECTIONS.iter().enumerate() {
        let section = &sections[section_index];
        assert_eq!(&section.heading, expected_heading);
        assert_eq!(
            section.subsections.len(),
            CYCLE_STUDY_SUBSECTIONS.len(),
            "{} must contain exactly the four required subsections",
            section.heading
        );
        for (subsection_index, expected_heading) in CYCLE_STUDY_SUBSECTIONS.iter().enumerate() {
            let subsection = &section.subsections[subsection_index];
            assert_eq!(&subsection.heading, expected_heading);
            assert!(
                !subsection.body.trim().is_empty(),
                "{} / {} must be nonempty",
                section.heading,
                subsection.heading
            );
        }
    }

    let d1 = study_section_text(&sections[0]);
    assert!(d1.contains("`gpc__cycles_elapsed.max`"));
    assert!(d1.contains("counting unit `GPC`"));
    assert!(d1.contains("maximum counter value across all unit instances"));
    assert!(d1.contains("unit's clock domain"));

    let d2 = study_section_text(&sections[1]);
    assert!(d2.contains("`diagnostic_last_issue_cycles`"));
    assert!(d2.contains("`terminal_modeled_event_time`"));
    assert!(d2.contains("nonbinding diagnostics"));
    assert!(d2.contains("selects neither diagnostic"));

    let d3 = study_section_text(&sections[2]);
    assert!(d3.contains("total `sm_count`"));
    assert!(d3.contains("no admitted `gpc_count`"));
    assert!(d3.contains("SM-to-GPC map"));
    assert!(d3.contains("invents no GPC count"));

    let d4 = study_section_text(&sections[3]);
    assert!(d4.contains("one timing-diagnostic record per launch"));
    assert!(d4.contains("no scalar temporal reducer"));
    assert!(d4.contains("`RequiredUnimplemented(AggregationSemanticsStudy)`"));
    assert!(d4.contains("Completion of this target-free study does not itself promote emission"));

    let output = run_manifest(fixture_root().join("run_manifest.v2.json"))
        .expect("run existing nonempty canary through real v2 consumer");
    let result: Value = serde_json::from_str(output.canonical_json()).unwrap();
    assert_eq!(result["schema_version"], "sim_result.v2");

    let launches = result["launches"]
        .as_array()
        .expect("runner result launches are an array");
    assert_eq!(launches.len(), 1, "existing canary must produce one launch");
    let aggregate = result["aggregate"]
        .as_object()
        .expect("runner result aggregate is an object");
    let aggregate_launch_count = aggregate["launch_count"]
        .as_u64()
        .expect("aggregate launch_count is numeric");
    assert_eq!(
        usize::try_from(aggregate_launch_count).unwrap(),
        launches.len()
    );
    assert!(
        aggregate["metrics"]["smsp_inst_executed_sum"]["value"]["value"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "canary replay must be nonempty"
    );

    for launch in launches {
        assert!(
            launch["metrics"]["smsp_inst_executed_sum"]["value"]["value"]
                .as_u64()
                .is_some_and(|count| count > 0)
        );
    }
    let diagnostics = result["diagnostics"]["per_launch"]
        .as_array()
        .expect("v2 diagnostics remain per-launch");
    assert_eq!(diagnostics.len(), launches.len());
    for diagnostic in diagnostics {
        assert!(
            diagnostic["diagnostic_last_issue_cycles"]
                .as_u64()
                .is_some_and(|cycles| cycles > 0)
        );
        assert_eq!(
            diagnostic["terminal_modeled_event_time"]["status"],
            "modeled"
        );
        assert!(
            diagnostic["terminal_modeled_event_time"]["value"]
                .as_u64()
                .is_some_and(|time| time > 0)
        );
    }
    assert!(
        result["diagnostics"]
            .get("internal_phase_timings_ns")
            .is_none()
    );

    let cycle_metric = &aggregate["metrics"]["exec_cycles_gpc_max"];
    assert_eq!(
        cycle_metric,
        &json!({
            "status": "required_unimplemented",
            "blocking": "aggregation_semantics_study",
        })
    );
    assert!(cycle_metric.get("value").is_none());
    assert!(
        aggregate["metrics"]
            .get("diagnostic_last_issue_cycles")
            .is_none()
    );
    assert!(
        aggregate["metrics"]
            .get("terminal_modeled_event_time")
            .is_none()
    );
}

#[test]
fn full_launch_emission_matches_singleton_aggregate_on_real_consumer_path() {
    let evidence = admit_launch_evidence(fixture_root().join("launch_evidence.json"))
        .expect("admit nonempty full-coverage canary");
    let (issue_params, modes) = replay_modes(&evidence);

    // The canary's cache-all LDGSTS source now follows its decoded L1 route:
    // eight source sectors coalesce behind one four-sector L1 line fill while
    // the five structurally bypassing sectors retain their independent path.
    // The resulting shared-L2 observer sees nine sectors, not the historical
    // all-bypass thirteen; all L1 unified accounting remains thirteen.
    let expected = ExpectedModeledOutcomes {
        instructions: 54,
        global_load_sectors: 8,
        global_store_sectors: 0,
        l1_read_fraction: (Some(0.0), 0, 8),
        l1_unified_fraction: (Some(0.0), 0, 13),
        l2_fraction: (Some(5.0 / 9.0), 5, 9),
        dram_read_sectors: 9,
    };
    for (mode, mode_name) in modes {
        let result = full_replay(&evidence, &issue_params, mode);
        assert_full_launch_emission_matches_singleton_aggregate(result, mode_name, expected);
    }
}

#[test]
fn full_launch_rates_keep_zero_denominator_nulls_on_real_consumer_path() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    replace_trace_with_one_nonmemory_instruction(root);
    let evidence = admit_launch_evidence(root.join("launch_evidence.json"))
        .expect("admit nonempty full-coverage nonmemory trace");
    let (issue_params, modes) = replay_modes(&evidence);

    let expected = ExpectedModeledOutcomes {
        instructions: 1,
        global_load_sectors: 0,
        global_store_sectors: 0,
        l1_read_fraction: (None, 0, 0),
        l1_unified_fraction: (None, 0, 0),
        l2_fraction: (None, 0, 0),
        dram_read_sectors: 0,
    };
    for (mode, mode_name) in modes {
        let result = full_replay(&evidence, &issue_params, mode);
        assert_full_launch_emission_matches_singleton_aggregate(result, mode_name, expected);
    }
}

#[test]
fn committed_v2_canary_is_deterministic_canonical_and_sealed() {
    let root = fixture_root();
    let manifest = root.join("run_manifest.v2.json");
    let first = run_manifest(&manifest).expect("run admitted v2 canary");
    let second = run_manifest(&manifest).expect("repeat admitted v2 canary");
    assert_eq!(first.canonical_json(), second.canonical_json());
    assert_eq!(first.result_sha256(), second.result_sha256());

    let mut result: Value = serde_json::from_str(first.canonical_json()).unwrap();
    assert_eq!(result["schema_version"], "sim_result.v2");
    assert_eq!(
        result["exit_status"],
        json!({"status": "success", "exit_code": 0})
    );
    assert_eq!(result["launches"].as_array().unwrap().len(), 1);
    assert_eq!(result["launches"][0]["launch_key"], fixture_launch_key(0));
    assert_eq!(
        result["aggregate"]["metrics"]["smsp_inst_executed_sum"]["value"]["value"],
        54
    );
    assert_eq!(
        result["launches"][0]["metrics"]["exec_cycles_gpc_max"]["status"],
        "required_unimplemented"
    );
    assert_eq!(
        result["aggregate"]["metrics"]["exec_cycles_gpc_max"]["blocking"],
        "aggregation_semantics_study"
    );
    assert_eq!(
        result["diagnostics"]["per_launch"][0]["terminal_modeled_event_time"],
        json!({"status": "modeled", "value": 54})
    );
    assert!(
        result["diagnostics"]
            .get("internal_phase_timings_ns")
            .is_none()
    );

    let seal = result
        .as_object_mut()
        .unwrap()
        .remove("result_sha256")
        .unwrap();
    let unsigned = canonical_json_utf8(&result).unwrap();
    assert_eq!(seal, first.result_sha256().to_string());
    assert_eq!(Sha256Digest::of(unsigned.as_bytes()), first.result_sha256());
}

#[test]
fn manifest_selects_every_typed_memory_mode() {
    for (mode, expected_l1, expected_l2) in [
        (json!({"mode": "flat"}), "unmodeled", "unmodeled"),
        (json!({"mode": "l1_const"}), "modeled", "unmodeled"),
        (
            json!({"mode": "l1_shared_l2", "dram_bandwidth": "unmodeled"}),
            "modeled",
            "modeled",
        ),
        (
            json!({"mode": "l1_shared_l2", "dram_bandwidth": "carded"}),
            "modeled",
            "modeled",
        ),
    ] {
        let (_temp, manifest_path) = copied_fixture();
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["replay_mode"] = mode;
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let output = run_manifest(&manifest_path).expect("typed replay mode runs");
        let result: Value = serde_json::from_str(output.canonical_json()).unwrap();
        assert_eq!(
            result["resolved_configuration"]["l1"]["status"],
            expected_l1
        );
        assert_eq!(
            result["resolved_configuration"]["l2"]["status"],
            expected_l2
        );
        assert_eq!(
            result["aggregate"]["metrics"]["exec_cycles_gpc_max"]["status"],
            "required_unimplemented"
        );
    }
}

#[test]
fn sampled_launch_is_rejected_before_exact_aggregation_and_emission() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let launch_manifest_path = root.join("launch_evidence.json");
    let mut launch_manifest = read_json(&launch_manifest_path);
    convert_to_sampled_fixture(root, &mut launch_manifest);
    write_json(&launch_manifest_path, &launch_manifest);

    let admitted =
        admit_launch_evidence(&launch_manifest_path).expect("admit sealed sampled trace");
    let identity: Value =
        serde_json::from_str(admitted.identity().canonical_json()).expect("parse sampled identity");
    assert_eq!(identity["coverage"]["type"], "estimated_sampled");
    assert_eq!(identity["coverage"]["population_blocks"], 60);
    assert_eq!(identity["coverage"]["blocks_observed"], 12);

    let mut run_document = read_json(&manifest_path);
    run_document["launches"][0]["expected_identity_sha256"] =
        json!(admitted.identity().seal_sha256());
    run_document["launches"][0]["launch_key"]["grid"] = json!({
        "x": identity["geometry"]["grid_dim"][0],
        "y": identity["geometry"]["grid_dim"][1],
        "z": identity["geometry"]["grid_dim"][2],
    });
    write_json(&manifest_path, &run_document);

    let error = run_manifest(&manifest_path).unwrap_err();
    assert_eq!(error.failure_type(), "REPLAY_FAILED");
    assert_eq!(error.stage(), "replay");
    assert_eq!(error.exit_code(), 1);
    assert!(error.to_string().contains("sampled (12/60)"));
    let document: Value = serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
    assert_eq!(document["error"]["failure_type"], "REPLAY_FAILED");
    assert_eq!(document["error"]["stage"], "replay");
    assert_eq!(document["exit_status"]["exit_code"], 1);
}

#[test]
fn two_launches_aggregate_in_canonical_roster_order() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let dynamic_path = root.join("trace/dynamic_trace.pb");
    let mut trace =
        sim_trace::pb::Trace::decode(fs::read(&dynamic_path).unwrap().as_slice()).unwrap();
    let device = trace.gpu_device.get_mut(&0).unwrap();
    let mut second_stream = device.streams.get(&0).unwrap().clone();
    second_stream.id = 1;
    device.streams.insert(1, second_stream);
    fs::write(&dynamic_path, trace.encode_to_vec()).unwrap();

    let first_payload_relative =
        "trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb";
    let second_payload_relative =
        "trace/threadblocks/device_0/stream_1/kernel_1/d_0_s_1_k_1_0,0,0.pb";
    let second_payload_path = root.join(second_payload_relative);
    fs::create_dir_all(second_payload_path.parent().unwrap()).unwrap();
    fs::copy(root.join(first_payload_relative), &second_payload_path).unwrap();
    fs::write(
        root.join("trace/stats.csv"),
        b"total_insts,total_reported_insts\n108,108\n",
    )
    .unwrap();

    let first_launch_path = root.join("launch_evidence.json");
    let mut first_launch = read_json(&first_launch_path);
    refresh_trace_artifact_claims(root, &mut first_launch);
    write_json(&first_launch_path, &first_launch);

    let second_launch_path = root.join("launch_evidence_2.json");
    let mut second_launch = first_launch.clone();
    second_launch["launch_selector"]["stream_id"] = json!(1);
    write_json(&second_launch_path, &second_launch);

    let first_admitted = admit_launch_evidence(&first_launch_path).unwrap();
    let second_admitted = admit_launch_evidence(&second_launch_path).unwrap();
    let mut run_document = read_json(&manifest_path);
    run_document["scope"] = json!("fixture-only/addr-carry-canary-two-launch:v2");
    run_document["launches"] = json!([
        {
            "launch_evidence_manifest_path": "launch_evidence_2.json",
            "expected_identity_sha256": second_admitted.identity().seal_sha256(),
            "launch_key": fixture_launch_key_for_stream(1, 1),
        },
        {
            "launch_evidence_manifest_path": "launch_evidence.json",
            "expected_identity_sha256": first_admitted.identity().seal_sha256(),
            "launch_key": fixture_launch_key(0),
        }
    ]);
    write_json(&manifest_path, &run_document);

    let output = run_manifest(&manifest_path).expect("two admitted launches aggregate");
    let result: Value = serde_json::from_str(output.canonical_json()).unwrap();
    assert_eq!(result["aggregate"]["launch_count"], 2);
    assert_eq!(
        result["aggregate"]["metrics"]["smsp_inst_executed_sum"]["value"]["value"],
        108
    );
    assert_eq!(result["launches"][0]["launch_key"], fixture_launch_key(0));
    assert_eq!(
        result["launches"][1]["launch_key"],
        fixture_launch_key_for_stream(1, 1)
    );
    assert_eq!(
        result["aggregate"]["metrics"]["exec_cycles_gpc_max"]["status"],
        "required_unimplemented"
    );
}

#[test]
fn invalid_supplied_card_value_is_a_failure_not_a_missing_prerequisite() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let arch_card_path = root.join("cards/sm_86.json");
    let mut arch_card = read_json(&arch_card_path);
    arch_card["latencies_cycles"]["l1_hit"]["value"] = json!(0);
    write_json(&arch_card_path, &arch_card);

    let launch_manifest_path = root.join("launch_evidence.json");
    let mut launch_manifest = read_json(&launch_manifest_path);
    let arch_card_claim = artifact_claim(root, "cards/sm_86.json");
    let arch_card_sha256 = arch_card_claim["sha256"].clone();
    launch_manifest["artifacts"]["arch_card"] = arch_card_claim;
    write_json(&launch_manifest_path, &launch_manifest);

    let admitted = admit_launch_evidence(&launch_manifest_path)
        .expect("invalid latency value remains identity-bound evidence");
    let mut run_document = read_json(&manifest_path);
    run_document["configuration_evidence"]["arch_card_sha256"] = arch_card_sha256;
    run_document["launches"][0]["expected_identity_sha256"] =
        json!(admitted.identity().seal_sha256());
    write_json(&manifest_path, &run_document);

    let error = run_manifest(&manifest_path).unwrap_err();
    assert_eq!(error.failure_type(), "ARCH_CARD_ADMISSION_FAILED");
    assert_eq!(error.stage(), "arch_card_admission");
    assert_eq!(error.exit_code(), 1);
    let document: Value = serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
    assert_eq!(
        document["error"]["failure_type"],
        "ARCH_CARD_ADMISSION_FAILED"
    );
    assert_eq!(document["exit_status"]["exit_code"], 1);
}

#[test]
fn architecture_card_l2_policy_failures_keep_stage_and_availability() {
    for (policy, expected_exit) in [(None, 2), (Some("unsupported_policy"), 1)] {
        let (_temp, manifest_path) = copied_fixture();
        let root = manifest_path.parent().unwrap();
        let arch_card_path = root.join("cards/sm_86.json");
        let mut arch_card = read_json(&arch_card_path);
        let l2_geometry = arch_card["l2_geometry"].as_object_mut().unwrap();
        match policy {
            Some(value) => {
                l2_geometry.insert("policy".to_owned(), json!(value));
            }
            None => {
                assert!(l2_geometry.remove("policy").is_some());
            }
        }
        write_json(&arch_card_path, &arch_card);

        let launch_manifest_path = root.join("launch_evidence.json");
        let mut launch_manifest = read_json(&launch_manifest_path);
        let arch_card_claim = artifact_claim(root, "cards/sm_86.json");
        let arch_card_sha256 = arch_card_claim["sha256"].clone();
        launch_manifest["artifacts"]["arch_card"] = arch_card_claim;
        write_json(&launch_manifest_path, &launch_manifest);

        let admitted = admit_launch_evidence(&launch_manifest_path)
            .expect("L2 policy remains identity-bound architecture evidence");
        let mut run_document = read_json(&manifest_path);
        run_document["configuration_evidence"]["arch_card_sha256"] = arch_card_sha256;
        run_document["launches"][0]["expected_identity_sha256"] =
            json!(admitted.identity().seal_sha256());
        run_document["replay_mode"] =
            json!({"mode": "l1_shared_l2", "dram_bandwidth": "unmodeled"});
        write_json(&manifest_path, &run_document);

        let error = run_manifest(&manifest_path).unwrap_err();
        assert_eq!(error.failure_type(), "ARCH_CARD_ADMISSION_FAILED");
        assert_eq!(error.stage(), "arch_card_admission");
        assert_eq!(error.exit_code(), expected_exit);
        let document: Value = serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
        assert_eq!(
            document["error"]["failure_type"],
            "ARCH_CARD_ADMISSION_FAILED"
        );
        assert_eq!(document["error"]["stage"], "arch_card_admission");
        assert_eq!(document["exit_status"]["exit_code"], expected_exit);
    }
}

#[test]
fn malformed_supplied_pchase_value_is_a_failure_not_a_missing_prerequisite() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let pchase_card_path = root.join("cards/sm_86_static_card.json");
    let mut pchase_card = read_json(&pchase_card_path);
    assert!(
        pchase_card["properties"]["l1_capacity_bytes"]
            .as_object_mut()
            .unwrap()
            .remove("value")
            .is_some()
    );
    write_json(&pchase_card_path, &pchase_card);

    let launch_manifest_path = root.join("launch_evidence.json");
    let mut launch_manifest = read_json(&launch_manifest_path);
    let pchase_card_claim = artifact_claim(root, "cards/sm_86_static_card.json");
    let pchase_card_sha256 = pchase_card_claim["sha256"].clone();
    launch_manifest["artifacts"]["pchase_card"] = pchase_card_claim;
    write_json(&launch_manifest_path, &launch_manifest);

    let admitted = admit_launch_evidence(&launch_manifest_path)
        .expect("malformed P-chase value remains identity-bound evidence");
    let mut run_document = read_json(&manifest_path);
    run_document["configuration_evidence"]["pchase_card_sha256"] = pchase_card_sha256;
    run_document["launches"][0]["expected_identity_sha256"] =
        json!(admitted.identity().seal_sha256());
    run_document["replay_mode"] = json!({"mode": "l1_const"});
    write_json(&manifest_path, &run_document);

    let error = run_manifest(&manifest_path).unwrap_err();
    assert_eq!(error.failure_type(), "CONFIGURATION_RESOLUTION_FAILED");
    assert_eq!(error.stage(), "configuration_resolution");
    assert_eq!(error.exit_code(), 1);
    let document: Value = serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
    assert_eq!(
        document["error"]["failure_type"],
        "CONFIGURATION_RESOLUTION_FAILED"
    );
    assert_eq!(document["exit_status"]["exit_code"], 1);
}

#[test]
fn production_consumer_rejects_historical_v1_manifest_without_fallback() {
    let error = run_manifest(fixture_root().join("run_manifest.json")).unwrap_err();
    assert_eq!(error.failure_type(), "MANIFEST_ADMISSION_FAILED");
    assert_eq!(error.stage(), "manifest_admission");
    assert_eq!(error.exit_code(), 1);
    assert!(error.to_string().contains("run manifest"));
}

#[test]
fn launch_identity_substitution_is_trace_admission() {
    let (_temp, manifest_path) = copied_fixture();
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["launches"][0]["expected_identity_sha256"] =
        json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let error = run_manifest(&manifest_path).unwrap_err();
    assert_eq!(error.failure_type(), "TRACE_ADMISSION_FAILED");
    assert_eq!(error.stage(), "trace_admission");
    assert_eq!(error.exit_code(), 1);
    assert!(
        error
            .to_string()
            .contains("launch expected_identity_sha256")
    );
}

#[test]
fn schema_fixed_sm120_input_is_manifest_admission_failure() {
    let (_temp, manifest_path) = copied_fixture();
    let mut manifest = read_json(&manifest_path);
    manifest["architecture"] = json!({
        "arch_id": "sm_120",
        "sku": "rtx_5060_laptop"
    });
    write_json(&manifest_path, &manifest);

    let error = run_manifest(&manifest_path).unwrap_err();
    assert_eq!(error.failure_type(), "MANIFEST_ADMISSION_FAILED");
    assert_eq!(error.stage(), "manifest_admission");
    assert_eq!(error.exit_code(), 1);
}

#[test]
fn missing_artifacts_preserve_typed_admission_boundaries() {
    for (relative, failure_type, stage) in [
        (
            "cards/sm_86.json",
            "ARCH_CARD_ADMISSION_FAILED",
            "arch_card_admission",
        ),
        (
            "cards/sm_86_static_card.json",
            "CONFIGURATION_RESOLUTION_FAILED",
            "configuration_resolution",
        ),
        (
            "trace/dynamic_trace.pb",
            "TRACE_ADMISSION_FAILED",
            "trace_admission",
        ),
        (
            "launch_evidence.json",
            "TRACE_ADMISSION_FAILED",
            "trace_admission",
        ),
    ] {
        let (_temp, manifest_path) = copied_fixture();
        fs::remove_file(manifest_path.parent().unwrap().join(relative)).unwrap();
        let error = run_manifest(&manifest_path).unwrap_err();
        assert_eq!(error.failure_type(), failure_type, "artifact {relative}");
        assert_eq!(error.stage(), stage, "artifact {relative}");
        assert_eq!(error.exit_code(), 2, "artifact {relative}");
    }
}

#[test]
fn malformed_trace_artifact_is_trace_admission() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let dynamic_path = root.join("trace/dynamic_trace.pb");
    fs::write(&dynamic_path, b"not-a-protobuf-trace").unwrap();
    let launch_manifest_path = root.join("launch_evidence.json");
    let mut launch_manifest = read_json(&launch_manifest_path);
    launch_manifest["artifacts"]["dynamic_trace_pb"] =
        artifact_claim(root, "trace/dynamic_trace.pb");
    write_json(&launch_manifest_path, &launch_manifest);

    let error = run_manifest(&manifest_path).unwrap_err();
    assert_eq!(error.failure_type(), "TRACE_ADMISSION_FAILED");
    assert_eq!(error.stage(), "trace_admission");
    assert_eq!(error.exit_code(), 1);
}

#[test]
fn malformed_memory_capability_vector_is_not_a_missing_prerequisite() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let dynamic_path = root.join("trace/dynamic_trace.pb");
    let mut trace =
        sim_trace::pb::Trace::decode(fs::read(&dynamic_path).unwrap().as_slice()).unwrap();
    trace
        .memory_payload_classes
        .push(*trace.memory_payload_classes.first().unwrap());
    fs::write(&dynamic_path, trace.encode_to_vec()).unwrap();

    let launch_manifest_path = root.join("launch_evidence.json");
    let mut launch_manifest = read_json(&launch_manifest_path);
    refresh_trace_artifact_claims(root, &mut launch_manifest);
    write_json(&launch_manifest_path, &launch_manifest);

    let error = run_manifest(&manifest_path).unwrap_err();
    assert_eq!(error.failure_type(), "TRACE_ADMISSION_FAILED");
    assert_eq!(error.stage(), "trace_admission");
    assert_eq!(error.exit_code(), 1);
    assert!(error.to_string().contains("invalid memory_payload_classes"));
}

#[test]
fn referenced_allocations_with_absent_sidecar_are_a_missing_prerequisite() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    fs::remove_file(root.join("trace/extra_info/allocations.json")).unwrap();

    let launch_manifest_path = root.join("launch_evidence.json");
    let mut launch_manifest = read_json(&launch_manifest_path);
    launch_manifest["artifacts"]["allocations_json"] = json!({"status": "absent"});
    write_json(&launch_manifest_path, &launch_manifest);

    let error = run_manifest(&manifest_path).unwrap_err();
    assert_eq!(error.failure_type(), "TRACE_ADMISSION_FAILED");
    assert_eq!(error.stage(), "trace_admission");
    assert_eq!(error.exit_code(), 2);
    assert!(error.to_string().contains("allocations_sidecar_missing"));
}

#[test]
fn oversized_finite_card_latency_stays_an_arch_card_value_failure() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let arch_card_path = root.join("cards/sm_86.json");
    let mut arch_card = read_json(&arch_card_path);
    arch_card["latencies_cycles"]["l1_hit"]["value"] = json!(1.0e20);
    write_json(&arch_card_path, &arch_card);

    let launch_manifest_path = root.join("launch_evidence.json");
    let mut launch_manifest = read_json(&launch_manifest_path);
    let arch_card_claim = artifact_claim(root, "cards/sm_86.json");
    let arch_card_sha256 = arch_card_claim["sha256"].clone();
    launch_manifest["artifacts"]["arch_card"] = arch_card_claim;
    write_json(&launch_manifest_path, &launch_manifest);

    let admitted = admit_launch_evidence(&launch_manifest_path)
        .expect("oversized finite latency remains identity-bound architecture evidence");
    let mut run_document = read_json(&manifest_path);
    run_document["configuration_evidence"]["arch_card_sha256"] = arch_card_sha256;
    run_document["launches"][0]["expected_identity_sha256"] =
        json!(admitted.identity().seal_sha256());
    write_json(&manifest_path, &run_document);

    let error = run_manifest(&manifest_path).unwrap_err();
    assert_eq!(error.failure_type(), "ARCH_CARD_ADMISSION_FAILED");
    assert_eq!(error.stage(), "arch_card_admission");
    assert_eq!(error.exit_code(), 1);
    assert!(
        error
            .to_string()
            .contains("latency must be finite, positive")
    );
}

#[test]
fn missing_manifest_is_a_prerequisite_exit() {
    let temp = TempDir::new().unwrap();
    let error = run_manifest(temp.path().join("absent.json")).unwrap_err();
    assert_eq!(error.failure_type(), "MANIFEST_ADMISSION_FAILED");
    assert_eq!(error.stage(), "manifest_admission");
    assert_eq!(error.exit_code(), 2);
    let document: Value = serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
    assert_eq!(
        document["error"]["failure_type"],
        "MANIFEST_ADMISSION_FAILED"
    );
    assert_eq!(document["exit_status"]["exit_code"], 2);
}
