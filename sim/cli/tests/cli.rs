#![allow(clippy::unwrap_used)] // integration-test assertions

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{Value, json};
use sim_run::RunError;
use sim_trace::{Sha256Digest, canonical_json};
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_sim-cli")
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("run/tests/fixtures/addr_carry_canary_bundle")
}

fn fixture_manifest() -> PathBuf {
    fixture_root().join("run_manifest.json")
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

fn assert_error_document(raw: &[u8], code: &str, exit_code: u64) {
    assert_eq!(raw.last(), Some(&b'\n'));
    let mut error: Value = serde_json::from_slice(raw).unwrap();
    let canonical = format!("{}\n", canonical_json(&error).unwrap());
    assert_eq!(raw, canonical.as_bytes());
    assert_eq!(error["schema_version"], "sim_error.v1");
    assert_eq!(error["code"], code);
    assert_eq!(error["exit_code"], exit_code);
    let seal = error
        .as_object_mut()
        .unwrap()
        .remove("error_sha256")
        .unwrap();
    let unsigned = canonical_json(&error).unwrap();
    assert_eq!(seal, Sha256Digest::of(unsigned.as_bytes()).to_string());
}

#[test]
fn cli_emits_only_the_canonical_result_on_success() {
    let output = Command::new(binary())
        .arg(fixture_manifest())
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let expected = format!(
        "{}\n",
        include_str!("../../run/tests/fixtures/addr_carry_canary_bundle/result.golden.json")
    );
    assert_eq!(output.stdout, expected.as_bytes());
    let parsed: Value = serde_json::from_slice(&output.stdout).unwrap();
    let cycles = parsed["emissions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["key"] == "exec_cycles_gpc_max")
        .unwrap();
    assert_eq!(cycles["outcome"]["status"], "required_unimplemented");
}

#[test]
fn ambient_mode_variables_do_not_change_manifest_selected_replay() {
    let (_temp, manifest_path) = copied_fixture();
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["replay_mode"] = json!({"mode": "l1_shared_l2", "dram_bandwidth": "unmodeled"});
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let mode_variables = [
        "SIM_L1_DISABLE",
        "SIM_L2_DISABLE",
        "SIM_DRAM_BW_DISABLE",
        "SIM_REPLAY_MODE",
        "SIM_MEM_MODE",
        "SIM_MODE",
    ];
    let mut baseline_command = Command::new(binary());
    baseline_command.arg(&manifest_path);
    for name in mode_variables {
        baseline_command.env_remove(name);
    }
    let baseline = baseline_command.output().unwrap();

    let contaminated = Command::new(binary())
        .arg(&manifest_path)
        .env("SIM_L1_DISABLE", "1")
        .env("SIM_L2_DISABLE", "1")
        .env("SIM_DRAM_BW_DISABLE", "1")
        .env("SIM_REPLAY_MODE", "flat")
        .env("SIM_MEM_MODE", "flat")
        .env("SIM_MODE", "flat")
        .output()
        .unwrap();
    assert!(baseline.status.success());
    assert!(contaminated.status.success());
    assert_eq!(baseline.stdout, contaminated.stdout);
    assert_eq!(baseline.stderr, contaminated.stderr);
    let result: Value = serde_json::from_slice(&contaminated.stdout).unwrap();
    assert_eq!(
        result["resolved_configuration"]["replay_mode"]["mode"],
        "l1_shared_l2"
    );
}

#[test]
fn cli_usage_is_a_canonical_prerequisite_error() {
    for args in [vec![], vec![""], vec!["one", "two"]] {
        let output = Command::new(binary()).args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert_error_document(&output.stderr, "cli_usage", 2);
    }
}

#[test]
fn noncanonical_manifest_path_spellings_are_rejected_literally() {
    let canonical = fixture_manifest().to_string_lossy().into_owned();
    let aliases = [
        format!("{canonical}/."),
        format!("{canonical}//"),
        canonical.replacen("/run/", "/run/./", 1),
    ];
    for alias in aliases {
        let output = Command::new(binary()).arg(alias).output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert_error_document(&output.stderr, "run_manifest_rejected", 1);
    }
}

#[test]
fn missing_and_malformed_manifests_have_distinct_typed_exits() {
    let temp = TempDir::new().unwrap();
    let missing = Command::new(binary())
        .arg(temp.path().join("missing.json"))
        .output()
        .unwrap();
    assert_eq!(missing.status.code(), Some(2));
    assert!(missing.stdout.is_empty());
    assert_error_document(&missing.stderr, "run_manifest_read", 2);

    let malformed_path = temp.path().join("malformed.json");
    fs::write(&malformed_path, b"{}").unwrap();
    let malformed = Command::new(binary()).arg(malformed_path).output().unwrap();
    assert_eq!(malformed.status.code(), Some(1));
    assert!(malformed.stdout.is_empty());
    assert_error_document(&malformed.stderr, "run_manifest_rejected", 1);
}

#[test]
fn serialization_fallback_matches_the_error_schema_and_seal() {
    let document = RunError::canonical_serialization_fallback();
    let framed = format!("{document}\n");
    assert_error_document(framed.as_bytes(), "output_serialization_failed", 1);
    let parsed: Value = serde_json::from_str(&document).unwrap();
    let object = parsed.as_object().unwrap();
    assert_eq!(object.len(), 6);
    assert_eq!(parsed["category"], "failure");
    assert!(
        parsed["message"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
}

#[test]
fn closed_stdout_is_an_output_write_failure() {
    let mut child = Command::new(binary())
        .arg(fixture_manifest())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_error_document(&output.stderr, "output_write_failed", 1);
}
