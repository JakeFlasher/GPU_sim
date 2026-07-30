#![allow(clippy::unwrap_used)] // integration-test assertions

use std::fs;
use std::path::{Path, PathBuf};

use prost::Message;
use serde_json::{Value, json};
use sim_runner::{RunError, run_manifest};
use sim_sm::admit_launch_evidence;
use sim_trace::{Sha256Digest, TraceDirSnapshot, canonical_json};
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
    let manifest = root.join("run_manifest.json");
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
const SAMPLED_BLOCKS: [i32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59];

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
fn canary_result_is_byte_stable_canonical_and_golden() {
    let root = fixture_root();
    let manifest = root.join("run_manifest.json");
    let first = run_manifest(&manifest).expect("run admitted canary");
    let second = run_manifest(&manifest).expect("repeat admitted canary");
    assert_eq!(first.canonical_json(), second.canonical_json());
    assert_eq!(first.result_sha256(), second.result_sha256());
    assert_eq!(
        first.canonical_json(),
        include_str!("fixtures/addr_carry_canary_bundle/result.golden.json")
    );

    let mut result: Value = serde_json::from_str(first.canonical_json()).unwrap();
    assert_eq!(result["schema_version"], "sim_result.v1");
    assert_eq!(result["scientific_authority"], "not_conferred");
    assert_eq!(result["hardware_readiness"], "not_asserted");
    assert_eq!(result["launches"].as_array().unwrap().len(), 1);
    assert_eq!(result["aggregate"]["counters"]["instructions_issued"], 54);
    assert_eq!(result["aggregate"]["counters"]["l1"]["status"], "unmodeled");
    assert_eq!(
        result["aggregate"]["unmodeled_tails"]["ldgsts_dst_shared_timing"],
        true
    );
    assert_eq!(
        result["launches"][0]["timing_diagnostics"]["terminal_modeled_event_time"],
        json!({"status": "modeled", "value": 54})
    );
    assert_eq!(
        result["launches"][0]["unmodeled_tails"]["ldgsts_dst_shared_timing"],
        true
    );
    assert!(result["aggregate"].get("cycles").is_none());
    let cycles = result["emissions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["key"] == "exec_cycles_gpc_max")
        .unwrap();
    assert_eq!(cycles["outcome"]["status"], "required_unimplemented");
    assert_eq!(cycles["outcome"]["blocking"], "aggregation_semantics_study");

    let seal = result
        .as_object_mut()
        .unwrap()
        .remove("result_sha256")
        .unwrap();
    let unsigned = canonical_json(&result).unwrap();
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
            result["emissions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["key"] == "exec_cycles_gpc_max")
                .unwrap()["outcome"]["status"],
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
    write_json(&manifest_path, &run_document);

    let error = run_manifest(&manifest_path).unwrap_err();
    match &error {
        RunError::CoverageNotAggregatable {
            identity_sha256,
            population_blocks,
            blocks_observed,
        } => {
            assert_eq!(*identity_sha256, admitted.identity().seal_sha256());
            assert_eq!((*population_blocks, *blocks_observed), (60, 12));
        }
        other => panic!("unexpected sampled-run error: {other:?}"),
    }
    assert_eq!(error.code(), "coverage_not_aggregatable");
    assert_eq!(error.exit_code(), 1);
    let document: Value = serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
    assert_eq!(document["category"], "failure");
    assert_eq!(document["exit_code"], 1);
}

#[test]
fn two_launches_aggregate_in_canonical_roster_order() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let dynamic_path = root.join("trace/dynamic_trace.pb");
    let mut trace =
        sim_trace::pb::Trace::decode(fs::read(&dynamic_path).unwrap().as_slice()).unwrap();
    let stream = trace
        .gpu_device
        .get_mut(&0)
        .unwrap()
        .streams
        .get_mut(&0)
        .unwrap();
    let mut second_kernel = stream.kernels.first().unwrap().clone();
    second_kernel.id = 2;
    stream.kernels.push(second_kernel);
    fs::write(&dynamic_path, trace.encode_to_vec()).unwrap();

    let first_payload_relative =
        "trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb";
    let second_payload_relative =
        "trace/threadblocks/device_0/stream_0/kernel_2/d_0_s_0_k_2_0,0,0.pb";
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
    second_launch["launch_selector"]["kernel_id"] = json!(2);
    write_json(&second_launch_path, &second_launch);

    let first_admitted = admit_launch_evidence(&first_launch_path).unwrap();
    let second_admitted = admit_launch_evidence(&second_launch_path).unwrap();
    let mut run_document = read_json(&manifest_path);
    run_document["scope"] = json!("fixture/addr-carry-canary-two-launch:v1");
    run_document["launches"] = json!([
        {
            "launch_evidence_manifest_path": "launch_evidence_2.json",
            "expected_identity_sha256": second_admitted.identity().seal_sha256(),
        },
        {
            "launch_evidence_manifest_path": "launch_evidence.json",
            "expected_identity_sha256": first_admitted.identity().seal_sha256(),
        }
    ]);
    write_json(&manifest_path, &run_document);

    let output = run_manifest(&manifest_path).expect("two admitted launches aggregate");
    let result: Value = serde_json::from_str(output.canonical_json()).unwrap();
    assert_eq!(result["aggregate"]["launch_count"], 2);
    assert_eq!(result["aggregate"]["counters"]["instructions_issued"], 108);
    assert_eq!(result["aggregate"]["counters"]["records_seen"], 108);
    assert_eq!(result["launches"][0]["launch"]["kernel_id"], 1);
    assert_eq!(result["launches"][1]["launch"]["kernel_id"], 2);
    assert_eq!(
        result["launches"][0]["launch_evidence_manifest_path"],
        "launch_evidence.json"
    );
    assert_eq!(
        result["launches"][1]["launch_evidence_manifest_path"],
        "launch_evidence_2.json"
    );
    assert!(result["aggregate"].get("cycles").is_none());
    let cycles = result["emissions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["key"] == "exec_cycles_gpc_max")
        .unwrap();
    assert_eq!(cycles["outcome"]["status"], "required_unimplemented");
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
    assert!(matches!(error, RunError::ConfigurationRejected { .. }));
    assert_eq!(error.code(), "configuration_rejected");
    assert_eq!(error.exit_code(), 1);
    let document: Value = serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
    assert_eq!(document["category"], "failure");
    assert_eq!(document["exit_code"], 1);
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
    assert!(matches!(error, RunError::ConfigurationRejected { .. }));
    assert_eq!(error.code(), "configuration_rejected");
    assert_eq!(error.exit_code(), 1);
    let document: Value = serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
    assert_eq!(document["category"], "failure");
    assert_eq!(document["exit_code"], 1);
}

#[test]
fn unknown_target_like_input_is_rejected_before_admission() {
    let temp = TempDir::new().unwrap();
    let manifest = temp.path().join("run_manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "scope": "negative/unknown-input:v1",
            "architecture": {"arch_id": "sm_86", "sku": "rtx_3090_ti"},
            "configuration_evidence": {
                "arch_card_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "pchase_card_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            },
            "launches": [{
                "launch_evidence_manifest_path": "missing.json",
                "expected_identity_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            }],
            "replay_mode": {"mode": "flat"},
            "labels": {"forbidden_target_value": 1}
        }))
        .unwrap(),
    )
    .unwrap();
    let error = run_manifest(&manifest).unwrap_err();
    assert!(matches!(error, RunError::RunManifestRejected { .. }));
    assert_eq!(error.code(), "run_manifest_rejected");
    assert_eq!(error.exit_code(), 1);
}

#[test]
fn launch_identity_substitution_is_typed() {
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
    assert!(matches!(error, RunError::IdentityMismatch { .. }));
    assert_eq!(error.code(), "identity_mismatch");
    assert_eq!(error.exit_code(), 1);
}

#[test]
fn sm120_is_typed_disabled_before_artifact_admission() {
    let temp = TempDir::new().unwrap();
    let manifest = temp.path().join("run_manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "scope": "negative/sm120-disabled:v1",
            "architecture": {"arch_id": "sm_120", "sku": "rtx_5060_laptop"},
            "configuration_evidence": {
                "arch_card_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "pchase_card_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            },
            "launches": [{
                "launch_evidence_manifest_path": "missing.json",
                "expected_identity_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            }],
            "replay_mode": {"mode": "flat"}
        }))
        .unwrap(),
    )
    .unwrap();
    let error = run_manifest(&manifest).unwrap_err();
    assert!(matches!(
        error,
        RunError::ArchitectureNotEnabled {
            arch: sim_arch::ArchId::Sm120
        }
    ));
    assert_eq!(error.exit_code(), 2);
}

#[test]
fn missing_manifest_is_a_prerequisite_exit() {
    let temp = TempDir::new().unwrap();
    let error = run_manifest(temp.path().join("absent.json")).unwrap_err();
    assert!(matches!(error, RunError::RunManifestRead { .. }));
    assert_eq!(error.exit_code(), 2);
    let document: Value = serde_json::from_str(&error.canonical_json().unwrap()).unwrap();
    assert_eq!(document["category"], "prerequisite_missing");
}
