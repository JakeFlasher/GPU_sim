use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use prost::Message;
use serde_json::{Value, json};
use sim_emit::{EmitOutcome, RequiredUnimplementedBlocker, emit_all};
use sim_sm::{
    BinaryDigestGap, FullLaunchSetDeclaration, IssueModelParams, KernelReplayOutcome,
    ManifestError, ReplayError, ReplayMemMode, admit_launch_evidence, aggregate_full_launch_set,
    replay_kernel,
};
use sim_trace::{Sha256Digest, TraceDirSnapshot, canonical_json};
use tempfile::TempDir;

const KERNEL_NAME: &str = "_Z17addr_carry_canaryPKiPiS1_S1_i___0";
const FUNCTION_NAME: &str = "_Z17addr_carry_canaryPKiPiS1_S1_i";

struct Fixture {
    _temp: TempDir,
    root: PathBuf,
    manifest_path: PathBuf,
    manifest: Value,
}

fn copy_file(source: impl AsRef<Path>, destination: impl AsRef<Path>) {
    let destination = destination.as_ref();
    fs::create_dir_all(
        destination
            .parent()
            .expect("fixture destination has a parent"),
    )
    .expect("create fixture parent");
    fs::copy(source, destination).expect("copy fixture artifact");
}

fn write_json(path: &Path, value: &Value) {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).expect("serialize fixture JSON"),
    )
    .expect("write fixture JSON");
}

fn artifact(root: &Path, relative: &str) -> Value {
    let raw = fs::read(root.join(relative)).expect("read fixture artifact");
    json!({
        "path": relative,
        "size_bytes": raw.len(),
        "sha256": Sha256Digest::of(&raw),
    })
}

fn set_fixture_block_dim(fixture: &mut Fixture, block: (i32, i32, i32)) {
    let dynamic_path = fixture.root.join("trace/dynamic_trace.pb");
    let raw = fs::read(&dynamic_path).expect("read dynamic trace");
    let mut trace = sim_trace::pb::Trace::decode(raw.as_slice()).expect("decode dynamic trace");
    trace
        .gpu_device
        .get_mut(&0)
        .expect("fixture device")
        .streams
        .get_mut(&0)
        .expect("fixture stream")
        .kernels
        .first_mut()
        .expect("fixture kernel")
        .block_dim = Some(sim_trace::pb::Dim3d {
        x: block.0,
        y: block.1,
        z: block.2,
    });
    fs::write(&dynamic_path, trace.encode_to_vec()).expect("rewrite dynamic trace");
    fixture.manifest["artifacts"]["dynamic_trace_pb"] =
        artifact(&fixture.root, "trace/dynamic_trace.pb");
    write_json(&fixture.manifest_path, &fixture.manifest);
}

fn refresh_trace_artifacts(fixture: &mut Fixture) {
    let trace_root = fixture.root.join("trace");
    let snapshot = TraceDirSnapshot::capture(&trace_root).expect("recapture fixture trace");
    let payload = snapshot.artifact_identity().threadblock_payload();
    fixture.manifest["artifacts"]["dynamic_trace_pb"] =
        artifact(&fixture.root, "trace/dynamic_trace.pb");
    fixture.manifest["artifacts"]["threadblock_payload"] = json!({
        "path": "trace/threadblocks",
        "file_count": payload.file_count(),
        "raw_bytes": payload.raw_bytes(),
        "sorted_path_manifest_sha256": payload.sorted_path_manifest_sha256(),
        "sorted_path_stream_sha256": payload.sorted_path_stream_sha256(),
    });
    fixture.manifest["artifacts"]["stats_csv"] = artifact(&fixture.root, "trace/stats.csv");
    write_json(&fixture.manifest_path, &fixture.manifest);
}

fn opcode(pc: u64) -> &'static str {
    match pc {
        80 => "LDG.E R0, [R2.64]",
        224 | 544 | 608 => "ATOM.E.ADD R0, [R2.64], R3",
        640 => "ATOMS.E.ADD R0, [R2], R3",
        736 | 752 | 768 => "LDS.U.32 R0, [R2]",
        784 => "LDGSTS.E.128 [R1], [R2.64]",
        _ => "IADD3 R0, R1, R2, RZ",
    }
}

fn fixture() -> Fixture {
    let temp = TempDir::new().expect("create fixture tempdir");
    let root = temp.path().to_path_buf();
    let trace_root = root.join("trace");
    let source = Path::new("../trace/tests/fixtures/addr_carry_canary");
    copy_file(
        source.join("dynamic_trace.pb"),
        trace_root.join("dynamic_trace.pb"),
    );
    copy_file(
        source.join("extra_info/allocations.json"),
        trace_root.join("extra_info/allocations.json"),
    );
    copy_file(
        source.join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb"),
        trace_root.join("threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb"),
    );
    fs::write(
        trace_root.join("stats.csv"),
        b"total_insts,total_reported_insts\n54,54\n",
    )
    .expect("write producer stats");

    let pcs = (0_u64..=848).step_by(16).collect::<Vec<_>>();
    let instructions = pcs
        .iter()
        .map(|pc| json!({"pc_string_hex": format!("{pc:x}"), "pc_num_dec": pc}))
        .collect::<Vec<_>>();
    write_json(
        &trace_root.join("extra_info/enhanced_execution_info.json"),
        &json!({
            "benchmark_name": "canary_sm86_run",
            "kernels": [{
                "kernel_name": KERNEL_NAME,
                "unique_function_id": 1,
                "architecture_version": 86,
                "instructions": instructions,
            }],
        }),
    );

    let control_word = 1_u64 | (7_u64 << 5) | (7_u64 << 8);
    let high_word = format!("0x{:016x}", control_word << 41);
    let records = pcs
        .iter()
        .map(|pc| {
            json!({
                "pc": pc,
                "predicate": null,
                "opcode_text": opcode(*pc),
                "high_word": high_word,
                "status": "decoded",
                "control": {
                    "stall": 1,
                    "yield_flag": 0,
                    "write_barrier": 7,
                    "read_barrier": 7,
                    "wait_mask": 0,
                    "reuse": 0,
                },
            })
        })
        .collect::<Vec<_>>();
    let control_path = root.join("control/canary_sm86.control_bits.json");
    fs::create_dir_all(control_path.parent().expect("control parent")).expect("create control dir");
    write_json(
        &control_path,
        &json!({
            "arch": "sm_86",
            "binary": "bin/canary_sm86",
            "functions": {FUNCTION_NAME: records},
            "parser_contract": {"control_layout": {
                "source": "high_word >> 41 (23 bits)",
                "stall": "bits 0:3",
                "yield_flag": "bit 4",
                "write_barrier": "bits 5:7 (7 = none)",
                "read_barrier": "bits 8:10 (7 = none)",
                "wait_mask": "bits 11:16",
                "reuse": "bits 17:20",
                "reserved_must_be_zero": "bits 21:22",
            }},
            "summary": {
                "total_decoded": pcs.len(),
                "total_unknown_encoding": 0,
            },
        }),
    );

    copy_file(
        Path::new("../../arch_profiles/cards/sm_86.json"),
        root.join("cards/sm_86.json"),
    );
    copy_file(
        Path::new("../../data/pchase/sm_86_static_card.json"),
        root.join("cards/sm_86_static_card.json"),
    );

    let trace_snapshot = TraceDirSnapshot::capture(&trace_root).expect("capture fixture trace");
    let trace_identity = trace_snapshot.artifact_identity();
    let payload = trace_identity.threadblock_payload();
    let allocations = trace_identity
        .allocations_json()
        .expect("canary allocations sidecar");
    let manifest = json!({
        "schema_version": 2,
        "artifacts": {
            "dynamic_trace_pb": artifact(&root, "trace/dynamic_trace.pb"),
            "threadblock_payload": {
                "path": "trace/threadblocks",
                "file_count": payload.file_count(),
                "raw_bytes": payload.raw_bytes(),
                "sorted_path_manifest_sha256": payload.sorted_path_manifest_sha256(),
                "sorted_path_stream_sha256": payload.sorted_path_stream_sha256(),
            },
            "allocations_json": {
                "status": "present",
                "artifact": {
                    "path": "trace/extra_info/allocations.json",
                    "size_bytes": allocations.size_bytes(),
                    "sha256": allocations.sha256(),
                },
            },
            "trace_sampling_json": {"status": "absent"},
            "stats_csv": artifact(&root, "trace/stats.csv"),
            "enhanced_execution_info_json": artifact(
                &root,
                "trace/extra_info/enhanced_execution_info.json",
            ),
            "control_bits_sidecar": artifact(
                &root,
                "control/canary_sm86.control_bits.json",
            ),
            "arch_card": artifact(&root, "cards/sm_86.json"),
            "pchase_card": artifact(&root, "cards/sm_86_static_card.json"),
        },
        "launch_selector": {
            "device_id": 0,
            "stream_id": 0,
            "kernel_id": 1,
            "function_unique_id": 1,
        },
        "architecture": {
            "arch_id": "sm_86",
            "sku": "rtx_3090_ti",
        },
        "binary_digest": {
            "status": "typed_gap",
            "gap": "trace_contract_does_not_emit_binary_digest",
        },
    });
    let manifest_path = root.join("launch_evidence.json");
    write_json(&manifest_path, &manifest);
    Fixture {
        _temp: temp,
        root,
        manifest_path,
        manifest,
    }
}

fn add_lossless_sampling_sidecar(fixture: &mut Fixture) -> Sha256Digest {
    let relative_payload = "threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb";
    let payload =
        fs::read(fixture.root.join("trace").join(relative_payload)).expect("read canary payload");
    let payload_sha256 = Sha256Digest::of(&payload);
    let payload_manifest = format!(
        "trace-payload-file-manifest-v1\n{relative_payload}\t{}\t{payload_sha256}\n",
        payload.len()
    );
    let selected_block_ids_sha256 = Sha256Digest::of(b"[0]");
    let mut sidecar = json!({
        "schema_version": 2,
        "cell_id": "canary-cell",
        "kernel": {
            "mangled_name": KERNEL_NAME,
            "args_hash": "canary-args-sha256-unavailable-test-token",
        },
        "method": "lossless_compressed_full",
        "reason": "byte_budget",
        "shape": {
            "original_grid": {"x": 1, "y": 1, "z": 1},
            "original_block": {"x": 32, "y": 1, "z": 1},
            "captured_grid": {"x": 1, "y": 1, "z": 1},
            "captured_block": {"x": 32, "y": 1, "z": 1},
            "reduced_shape_variant": null,
        },
        "gates": {
            "coop": false,
            "wave_count": 1,
            "n_measured_waves": 0,
            "tail_measured_waves": 0,
            "atomic_density": 0.0,
            "hotline_share": 0.0,
            "skew_top1pct_share": 0.0,
            "gini": 0.0,
            "per_sm_block_count_cv": 0.0,
            "n_phases": 1,
            "skew_heavy_block_count": 0,
            "certainty_bytes": null,
            "max_trace_bytes": null,
            "certainty_budget_exceeded": false,
            "verdict": "subset_admissible",
        },
        "wave_model": {
            "reconstruction": "launch_index",
            "per_sm_resident_blocks": [{"sm_id": 0, "resident_blocks": 1}],
            "n_waves": 1,
            "segment_len": 3,
            "ghost_depth": 1,
        },
        "per_sm_slot_map": [{"sm_id": 0, "block_ids": [0]}],
        "rng": {"algo": "pcg64", "seed": 0},
        "label_pairing": "full_shape_label",
        "tracer": {
            "version": 5,
            "features": {"atomic_addr": true, "ldgsts_addr": true},
            "stats_pass_version": 2,
        },
        "bytes": {
            "raw": payload.len(),
            "compressed": payload.len(),
            "codec": "zstd",
            "level": 19,
            "long": 27,
            "dict_id": null,
        },
        "records": {"expected_full": 54, "captured": 54},
        "hashes": {
            "selected_block_ids_sha256": selected_block_ids_sha256,
            "decompressed_payload_sha256": payload_sha256,
            "per_block_raw_sha256_manifest": payload_manifest,
        },
    });
    let canonical = canonical_json(&sidecar).expect("canonicalize unsigned sampling sidecar");
    let sidecar_seal = Sha256Digest::of(canonical.as_bytes());
    sidecar["hashes"]["sidecar_sha256"] = json!(sidecar_seal);

    let sidecar_path = fixture.root.join("trace/trace_sampling.json");
    write_json(&sidecar_path, &sidecar);
    fixture.manifest["artifacts"]["trace_sampling_json"] = json!({
        "status": "present",
        "artifact": artifact(&fixture.root, "trace/trace_sampling.json"),
    });
    write_json(&fixture.manifest_path, &fixture.manifest);
    sidecar_seal
}

#[test]
fn canary_admits_to_one_stable_canonical_seal() {
    let fixture = fixture();
    let first = admit_launch_evidence(&fixture.manifest_path).expect("admit canary manifest");
    let first_seal = first.identity().seal_sha256();
    assert_eq!(
        first_seal.to_string(),
        "ab173443f81eaaa5c26fabe24258d02cab2c5266207ced0f174ebd043a62ef45"
    );
    assert_eq!(
        first.identity().arch().expect("sealed arch"),
        sim_arch::ArchId::Sm86
    );
    assert_eq!(
        first.identity().binary_digest_gap(),
        BinaryDigestGap::TraceContractDoesNotEmitBinaryDigest
    );
    assert_eq!(
        first.identity().arch_card_sha256(),
        Sha256Digest::parse(
            fixture.manifest["artifacts"]["arch_card"]["sha256"]
                .as_str()
                .expect("fixture arch-card digest is a string")
        )
        .expect("fixture arch-card digest parses")
    );
    assert_eq!(
        first.identity().pchase_card_sha256(),
        Sha256Digest::parse(
            fixture.manifest["artifacts"]["pchase_card"]["sha256"]
                .as_str()
                .expect("fixture P-chase-card digest is a string")
        )
        .expect("fixture P-chase-card digest parses")
    );
    let binding = first
        .identity()
        .launch_binding()
        .expect("admitted identity reconstructs a declaration binding");
    assert_eq!(binding.device_id(), 0);
    assert_eq!(binding.stream_id(), 0);
    assert_eq!(binding.kernel_id(), 1);
    assert_eq!(binding.function_unique_id(), 1);
    assert_eq!(binding.kernel_mangled_name(), KERNEL_NAME);
    assert_eq!(binding.grid(), (1, 1, 1));
    assert_eq!(binding.block(), Some((32, 1, 1)));
    assert_eq!(binding.cluster(), (1, 1, 1));
    assert!(
        first
            .identity()
            .canonical_json()
            .contains(&format!(r#""identity_sha256":"{first_seal}""#))
    );

    let reversed = format!(
        "{{\"binary_digest\":{},\"architecture\":{},\"launch_selector\":{},\"artifacts\":{},\"schema_version\":2}}",
        serde_json::to_string(&fixture.manifest["binary_digest"]).expect("serialize binary"),
        serde_json::to_string(&fixture.manifest["architecture"]).expect("serialize arch"),
        serde_json::to_string(&fixture.manifest["launch_selector"]).expect("serialize selector"),
        serde_json::to_string(&fixture.manifest["artifacts"]).expect("serialize artifacts"),
    );
    fs::write(&fixture.manifest_path, reversed).expect("rewrite reordered manifest");
    let reordered =
        admit_launch_evidence(&fixture.manifest_path).expect("admit reordered manifest");
    assert_eq!(reordered.identity().seal_sha256(), first_seal);
    assert_eq!(
        reordered.identity().canonical_json(),
        first.identity().canonical_json()
    );
}

#[test]
fn admitted_evidence_mints_the_public_replay_result() {
    let fixture = fixture();
    let evidence =
        admit_launch_evidence(&fixture.manifest_path).expect("admit canary launch evidence");
    let latencies = evidence
        .arch_card()
        .required_latencies()
        .expect("fixture architecture card has required latencies");
    let params = IssueModelParams::from_legacy_card_latencies_for_arch(
        evidence
            .identity()
            .arch()
            .expect("sealed architecture parses"),
        latencies.l1_hit,
        latencies.l2_hit,
        latencies.dram,
        latencies.shared,
    )
    .expect("fixture issue parameters resolve");

    let outcome =
        replay_kernel(&evidence, &params, ReplayMemMode::Flat).expect("admitted launch replays");
    let KernelReplayOutcome::Full(result) = outcome else {
        panic!("unsampled admitted fixture must produce full coverage");
    };
    assert_eq!(result.instructions_issued(), result.records_seen());
    assert_eq!(result.coverage().blocks_observed(), 1);

    let launch = result.coverage().launch().clone();
    let diagnostic_last_issue_cycles = result.diagnostic_last_issue_cycles();
    let terminal_modeled_event_time = result.terminal_modeled_event_time();
    let declaration = FullLaunchSetDeclaration::checked(
        "fixture/admitted-emission:v1",
        result.configuration(),
        [launch.clone()],
    )
    .expect("declare exact fixture launch set");
    let aggregate =
        aggregate_full_launch_set(declaration, [result]).expect("aggregate admitted launch");
    assert_eq!(aggregate.launch_timing_diagnostics().len(), 1);
    assert_eq!(aggregate.launch_timing_diagnostics()[0].launch(), &launch);
    assert_eq!(
        aggregate.launch_timing_diagnostics()[0].diagnostic_last_issue_cycles(),
        diagnostic_last_issue_cycles
    );
    assert_eq!(
        aggregate.launch_timing_diagnostics()[0].terminal_modeled_event_time(),
        terminal_modeled_event_time
    );

    let predictions = emit_all(&aggregate).expect("emit admitted aggregate");
    assert_eq!(
        predictions.launch_timing_diagnostics(),
        aggregate.launch_timing_diagnostics()
    );
    let cycles = predictions
        .metrics()
        .iter()
        .find(|prediction| prediction.key() == "exec_cycles_gpc_max")
        .expect("mandatory binding-cycle row");
    assert_eq!(
        cycles.outcome(),
        &EmitOutcome::RequiredUnimplemented {
            blocking: RequiredUnimplementedBlocker::AggregationSemanticsStudy,
        }
    );
}

#[test]
fn lossless_sampling_sidecar_is_admitted_and_raw_digest_bound() {
    let mut fixture = fixture();
    let sidecar_seal = add_lossless_sampling_sidecar(&mut fixture);
    let admitted =
        admit_launch_evidence(&fixture.manifest_path).expect("admit lossless canary manifest");
    let identity: Value = serde_json::from_str(admitted.identity().canonical_json())
        .expect("parse sealed launch identity");
    assert_eq!(identity["cell_id"], "canary-cell");
    assert_eq!(
        identity["kernel"]["args_hash"],
        "canary-args-sha256-unavailable-test-token"
    );
    assert_eq!(identity["coverage"]["type"], "lossless_stored");
    assert_eq!(identity["coverage"]["blocks_observed"], 1);
    assert_eq!(
        identity["coverage"]["sidecar_sha256"],
        sidecar_seal.to_string()
    );
    assert_eq!(
        identity["artifact_digests"]["trace_sampling_json_seal"],
        sidecar_seal.to_string()
    );

    let sidecar_path = fixture.root.join("trace/trace_sampling.json");
    let mut raw = fs::read(&sidecar_path).expect("read sampling mutation target");
    raw[0] ^= 1;
    fs::write(&sidecar_path, raw).expect("mutate sampling artifact");
    assert!(matches!(
        admit_launch_evidence(&fixture.manifest_path),
        Err(ManifestError::ArtifactDigestMismatch {
            artifact: "trace_sampling_json",
            ..
        })
    ));
}

#[test]
fn full_coverage_rejects_a_cta_with_a_missing_warp() {
    let mut fixture = fixture();
    set_fixture_block_dim(&mut fixture, (64, 1, 1));
    assert!(matches!(
        admit_launch_evidence(&fixture.manifest_path),
        Err(ManifestError::Coverage(
            ReplayError::WarpRosterCardinalityMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ))
    ));
}

#[test]
fn nonselected_root_payload_is_still_semantically_validated() {
    let mut fixture = fixture();
    let dynamic_path = fixture.root.join("trace/dynamic_trace.pb");
    let raw = fs::read(&dynamic_path).expect("read dynamic trace");
    let mut trace = sim_trace::pb::Trace::decode(raw.as_slice()).expect("decode dynamic trace");
    let stream = trace
        .gpu_device
        .get_mut(&0)
        .expect("fixture device")
        .streams
        .get_mut(&0)
        .expect("fixture stream");
    let mut second = stream.kernels[0].clone();
    second.id = 2;
    second.function_unique_id = 2;
    stream.kernels.push(second);
    fs::write(&dynamic_path, trace.encode_to_vec()).expect("rewrite dynamic trace");

    let first_payload = fixture
        .root
        .join("trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
    let raw = fs::read(&first_payload).expect("read first payload");
    let mut threadblock =
        sim_trace::pb::Threadblock::decode(raw.as_slice()).expect("decode first threadblock");
    for warp in threadblock.warps.values_mut() {
        for instruction in &mut warp.instructions {
            instruction.function_unique_id = 2;
        }
    }
    threadblock
        .warps
        .values_mut()
        .next()
        .expect("fixture warp")
        .instructions[0]
        .warpid_tb = None;
    let second_payload = fixture
        .root
        .join("trace/threadblocks/device_0/stream_0/kernel_2/d_0_s_0_k_2_0,0,0.pb");
    fs::create_dir_all(second_payload.parent().expect("second payload parent"))
        .expect("create second payload parent");
    fs::write(second_payload, threadblock.encode_to_vec()).expect("write second payload");
    refresh_trace_artifacts(&mut fixture);

    assert!(matches!(
        admit_launch_evidence(&fixture.manifest_path),
        Err(ManifestError::Trace(
            sim_trace::TraceError::WarpidTbMissing { .. }
        ))
    ));
}

#[test]
fn stale_manifest_rejects_single_byte_mutation_of_every_artifact_class() {
    for relative in [
        "trace/dynamic_trace.pb",
        "trace/extra_info/allocations.json",
        "trace/stats.csv",
        "trace/extra_info/enhanced_execution_info.json",
        "trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb",
        "control/canary_sm86.control_bits.json",
        "cards/sm_86.json",
        "cards/sm_86_static_card.json",
    ] {
        let fixture = fixture();
        let path = fixture.root.join(relative);
        let mut raw = fs::read(&path).expect("read mutation target");
        raw[0] ^= 1;
        fs::write(&path, raw).expect("mutate fixture artifact");
        let error = match admit_launch_evidence(&fixture.manifest_path) {
            Err(error) => error,
            Ok(_) => panic!("{relative}: stale digest unexpectedly admitted"),
        };
        assert!(
            matches!(error, ManifestError::ArtifactDigestMismatch { .. }),
            "{relative}: {error:?}"
        );
    }
}

#[test]
fn resealed_stats_must_be_semantically_valid() {
    let mut fixture = fixture();
    let stats_path = fixture.root.join("trace/stats.csv");
    fs::write(&stats_path, b"total_insts,total_reported_insts\n1,2\n")
        .expect("rewrite producer stats");
    fixture.manifest["artifacts"]["stats_csv"] = artifact(&fixture.root, "trace/stats.csv");
    write_json(&fixture.manifest_path, &fixture.manifest);

    assert!(matches!(
        admit_launch_evidence(&fixture.manifest_path),
        Err(ManifestError::Trace(
            sim_trace::TraceError::StatsCountersInvalid { .. }
        ))
    ));
}

#[test]
fn symlinked_and_hardlinked_bundle_artifacts_fail_closed() {
    let symlink_fixture = fixture();
    let outside = TempDir::new().expect("create outside artifact directory");
    let outside_card = outside.path().join("sm_86.json");
    copy_file(
        Path::new("../../arch_profiles/cards/sm_86.json"),
        &outside_card,
    );
    let card_path = symlink_fixture.root.join("cards/sm_86.json");
    fs::remove_file(&card_path).expect("remove copied card");
    symlink(&outside_card, &card_path).expect("symlink card outside bundle");
    assert!(matches!(
        admit_launch_evidence(&symlink_fixture.manifest_path),
        Err(ManifestError::Anchored(_))
    ));

    let parent_symlink_fixture = fixture();
    let cards_path = parent_symlink_fixture.root.join("cards");
    fs::remove_dir_all(&cards_path).expect("remove copied cards directory");
    symlink(outside.path(), &cards_path).expect("symlink card directory outside bundle");
    assert!(matches!(
        admit_launch_evidence(&parent_symlink_fixture.manifest_path),
        Err(ManifestError::Anchored(_))
    ));

    let payload_fixture = fixture();
    let payload_path = payload_fixture
        .root
        .join("trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb");
    let outside_payload = fs::canonicalize(
        "../trace/tests/fixtures/addr_carry_canary/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb",
    )
    .expect("canonical canary payload");
    fs::remove_file(&payload_path).expect("remove copied payload");
    symlink(&outside_payload, &payload_path).expect("symlink payload outside bundle");
    assert!(matches!(
        admit_launch_evidence(&payload_fixture.manifest_path),
        Err(ManifestError::Trace(
            sim_trace::TraceError::AnchoredSnapshot(
                sim_trace::AnchoredReadError::NotRegular { .. }
            )
        ))
    ));

    let mut hardlink_fixture = fixture();
    let dynamic_path = hardlink_fixture.root.join("trace/dynamic_trace.pb");
    let allocations_path = hardlink_fixture
        .root
        .join("trace/extra_info/allocations.json");
    fs::remove_file(&allocations_path).expect("remove copied allocations");
    fs::hard_link(&dynamic_path, &allocations_path).expect("hardlink trace artifacts");
    hardlink_fixture.manifest["artifacts"]["allocations_json"]["artifact"] =
        artifact(&hardlink_fixture.root, "trace/extra_info/allocations.json");
    write_json(&hardlink_fixture.manifest_path, &hardlink_fixture.manifest);
    assert!(matches!(
        admit_launch_evidence(&hardlink_fixture.manifest_path),
        Err(ManifestError::ArtifactFileAlias { .. })
    ));
}

#[test]
fn absent_sampling_declaration_and_binary_gap_fail_closed() {
    let sampling_fixture = fixture();
    fs::write(
        sampling_fixture.root.join("trace/trace_sampling.json"),
        b"{}",
    )
    .expect("add sidecar");
    assert!(matches!(
        admit_launch_evidence(&sampling_fixture.manifest_path),
        Err(ManifestError::ArtifactPresenceMismatch {
            artifact: "trace_sampling_json",
            expected: false,
            actual: true,
        })
    ));

    let binary_fixture = fixture();
    let mut manifest = binary_fixture.manifest.clone();
    manifest["binary_digest"] = json!({
        "status": "value",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    });
    write_json(&binary_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&binary_fixture.manifest_path),
        Err(ManifestError::EdgeUnverifiable {
            edge: "binary.digest",
            ..
        })
    ));
}

#[test]
fn unknown_fields_noncanonical_paths_and_digest_spelling_are_rejected() {
    let unknown_fixture = fixture();
    let mut manifest = unknown_fixture.manifest.clone();
    manifest["launch_selector"]["unexpected"] = json!(true);
    write_json(&unknown_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&unknown_fixture.manifest_path),
        Err(ManifestError::Parse { .. })
    ));

    let path_fixture = fixture();
    let mut manifest = path_fixture.manifest.clone();
    manifest["artifacts"]["arch_card"]["path"] = json!("cards/../cards/sm_86.json");
    write_json(&path_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&path_fixture.manifest_path),
        Err(ManifestError::ArtifactPathInvalid {
            artifact: "arch_card",
            ..
        })
    ));

    let dot_path_fixture = fixture();
    let mut manifest = dot_path_fixture.manifest.clone();
    manifest["artifacts"]["arch_card"]["path"] = json!("cards/./sm_86.json");
    write_json(&dot_path_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&dot_path_fixture.manifest_path),
        Err(ManifestError::ArtifactPathInvalid {
            artifact: "arch_card",
            ..
        })
    ));

    let digest_fixture = fixture();
    let mut manifest = digest_fixture.manifest.clone();
    manifest["artifacts"]["dynamic_trace_pb"]["sha256"] = json!("A".repeat(64));
    write_json(&digest_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&digest_fixture.manifest_path),
        Err(ManifestError::Parse { .. })
    ));
}

#[test]
fn unsupported_version_precedes_version_specific_shape() {
    let future_fixture = fixture();
    let mut manifest = future_fixture.manifest.clone();
    manifest["schema_version"] = json!(3);
    manifest["future_required_field"] = json!({"new_shape": true});
    write_json(&future_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&future_fixture.manifest_path),
        Err(ManifestError::SchemaVersionUnsupported {
            got: 3,
            expected: 2,
        })
    ));

    let historical_fixture = fixture();
    let mut manifest = historical_fixture.manifest.clone();
    manifest["schema_version"] = json!(1);
    manifest
        .as_object_mut()
        .expect("manifest object")
        .remove("artifacts");
    write_json(&historical_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&historical_fixture.manifest_path),
        Err(ManifestError::SchemaVersionUnsupported {
            got: 1,
            expected: 2,
        })
    ));
}

#[test]
fn declared_resource_costs_are_bounded_before_artifact_reads() {
    let artifact_fixture = fixture();
    let mut manifest = artifact_fixture.manifest.clone();
    manifest["artifacts"]["dynamic_trace_pb"]["size_bytes"] = json!(u64::MAX);
    write_json(&artifact_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&artifact_fixture.manifest_path),
        Err(ManifestError::ArtifactResourceLimitExceeded {
            artifact: "dynamic_trace_pb",
            resource: "bytes",
            ..
        })
    ));

    let file_fixture = fixture();
    let mut manifest = file_fixture.manifest.clone();
    manifest["artifacts"]["threadblock_payload"]["file_count"] = json!(u64::MAX);
    write_json(&file_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&file_fixture.manifest_path),
        Err(ManifestError::ArtifactResourceLimitExceeded {
            artifact: "threadblock_payload",
            resource: "files",
            ..
        })
    ));

    let aggregate_fixture = fixture();
    let mut manifest = aggregate_fixture.manifest.clone();
    manifest["artifacts"]["threadblock_payload"]["raw_bytes"] = json!(768_u64 * 1024 * 1024);
    for name in [
        "dynamic_trace_pb",
        "stats_csv",
        "enhanced_execution_info_json",
        "control_bits_sidecar",
        "arch_card",
        "pchase_card",
    ] {
        manifest["artifacts"][name]["size_bytes"] = json!(64_u64 * 1024 * 1024);
    }
    manifest["artifacts"]["allocations_json"]["artifact"]["size_bytes"] =
        json!(64_u64 * 1024 * 1024);
    write_json(&aggregate_fixture.manifest_path, &manifest);
    assert!(matches!(
        admit_launch_evidence(&aggregate_fixture.manifest_path),
        Err(ManifestError::SnapshotResourceLimitExceeded { .. })
    ));
}
