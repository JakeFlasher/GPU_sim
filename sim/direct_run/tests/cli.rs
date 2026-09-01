use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::{Command, Output};

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../run/tests/fixtures/addr_carry_canary_bundle")
}

fn base_command() -> Command {
    let root = fixture_root();
    let mut command = Command::new(env!("CARGO_BIN_EXE_sim-run"));
    command
        .arg("--trace")
        .arg(root.join("trace"))
        .arg("--control")
        .arg(root.join("control/canary_sm86.control_bits.json"))
        .arg("--arch-card")
        .arg(root.join("cards/sm_86.json"));
    command
}

fn run(command: &mut Command) -> Output {
    command.output().expect("sim-run must execute")
}

#[test]
fn flat_cli_is_deterministic_and_emits_exact_root_and_launch_identity() {
    let first = run(&mut base_command());
    let second = run(&mut base_command());
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(first.stderr, b"");
    assert_eq!(first.stdout, second.stdout);

    let report: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    let root = report.as_object().unwrap();
    assert_eq!(
        root.keys().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "architecture".to_owned(),
            "cache_scope".to_owned(),
            "carded_dram_bandwidth".to_owned(),
            "launches".to_owned(),
            "memory_admission_candidate_count".to_owned(),
            "memory_mode".to_owned(),
            "memory_service_candidate_count".to_owned(),
            "schema_version".to_owned(),
            "simulator_version".to_owned(),
        ])
    );
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["simulator_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(report["architecture"], "sm_86");
    assert_eq!(report["memory_mode"], "flat");
    assert_eq!(report["cache_scope"], "launch");
    assert_eq!(report["carded_dram_bandwidth"], false);
    assert_eq!(report["memory_service_candidate_count"], 0);
    assert_eq!(report["memory_admission_candidate_count"], 0);
    assert!(
        report
            .get("memory_admission_feedback_candidate_id")
            .is_none()
    );
    assert!(report.get("cache_session").is_none());

    let launches = report["launches"].as_array().unwrap();
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0]["device_id"], 0);
    assert_eq!(launches[0]["stream_id"], 0);
    assert_eq!(launches[0]["kernel_id"], 1);
    assert_eq!(launches[0]["function_unique_id"], 1);
    assert_eq!(launches[0]["grid"], serde_json::json!([1, 1, 1]));
    assert_eq!(launches[0]["block"], serde_json::json!([32, 1, 1]));
    assert_eq!(launches[0]["cluster"], serde_json::json!([1, 1, 1]));
    assert_eq!(launches[0]["coverage"]["kind"], "full");
    assert_eq!(launches[0]["coverage"]["blocks_observed"], 1);
    assert!(launches[0].get("cache_handoff").is_none());
}

#[test]
fn mutually_exclusive_candidate_flags_fail_as_cli_usage_before_file_access() {
    let output = run(base_command()
        .arg("--memory-service-candidates")
        .arg("does-not-exist-service.json")
        .arg("--memory-admission-feedback")
        .arg("does-not-exist-feedback.json"));
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(output.stdout, b"");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("mutually exclusive"), "{stderr}");
    assert!(!stderr.contains("No such file"), "{stderr}");
}

#[test]
fn feedback_cli_reports_the_exact_one_contract_with_its_own_stage() {
    let temporary = tempfile::tempdir().unwrap();
    let feedback = temporary.path().join("feedback.json");
    std::fs::write(&feedback, br#"{"schema_version":1,"candidates":[]}"#).unwrap();
    let root = fixture_root();
    let output = run(base_command()
        .arg("--pchase-card")
        .arg(root.join("cards/sm_86_static_card.json"))
        .arg("--mode")
        .arg("l1-l2")
        .arg("--memory-admission-feedback")
        .arg(&feedback));
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stdout, b"");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.starts_with("memory-admission feedback:"), "{stderr}");
    assert!(stderr.contains("exactly one candidate, got 0"), "{stderr}");
}

#[test]
fn feedback_cli_serializes_the_timing_active_configuration_identity() {
    let temporary = tempfile::tempdir().unwrap();
    let feedback = temporary.path().join("feedback.json");
    let candidate = serde_json::json!({
        "candidate_id": 7,
        "scope": "smsp",
        "operation": "store",
        "queue_capacity_instructions": 16,
        "admission_latency_cycles": 8,
        "l2_hit_sector_units": 1,
        "l2_hit_sector_cycles": 1,
        "l2_miss_sector_units": 1,
        "l2_miss_sector_cycles": 10,
    });
    std::fs::write(
        &feedback,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "candidates": [&candidate],
        }))
        .unwrap(),
    )
    .unwrap();
    let root = fixture_root();
    let output = run(base_command()
        .arg("--pchase-card")
        .arg(root.join("cards/sm_86_static_card.json"))
        .arg("--mode")
        .arg("l1-l2")
        .arg("--memory-admission-feedback")
        .arg(&feedback));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stderr, b"");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["memory_admission_feedback_candidate_id"], 7);
    assert_eq!(
        report["launches"][0]["timing"]["readiness"]["schema_version"],
        24
    );
    assert_eq!(
        report["launches"][0]["timing"]["readiness"]["memory_admission_feedback"]["candidate"],
        candidate
    );
}

#[test]
fn invalid_mode_is_a_usage_error() {
    let output = run(base_command().arg("--mode").arg("shared"));
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(output.stdout, b"");
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("unknown memory mode")
    );
}

#[cfg(feature = "performance-counters")]
#[test]
fn performance_feature_requires_runtime_opt_in_and_preserves_json() {
    let output = run(base_command().env("SIM_RUN_PERFORMANCE_COUNTERS", "1"));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("SIM_PHASE_PERF setup_ns="), "{stderr}");
    assert!(
        stderr.contains("SIM_PHASE_PERF serialization_ns="),
        "{stderr}"
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["launches"][0]["coverage"]["kind"], "full");
}

#[cfg(feature = "performance-counters")]
#[test]
fn streaming_trace_loader_preserves_exact_normal_json() {
    let full = run(&mut base_command());
    let streaming = run(base_command().env("SIM_RUN_STREAM_TRACE_PAYLOAD", "1"));
    assert!(full.status.success());
    assert!(streaming.status.success());
    assert_eq!(full.stderr, b"");
    assert_eq!(streaming.stderr, b"");
    assert_eq!(streaming.stdout, full.stdout);
}
