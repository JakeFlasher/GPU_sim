#![allow(clippy::unwrap_used)] // integration-test assertions

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};
use sim_run::RunError;
use sim_trace::{Sha256Digest, canonical_json_utf8};
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_sim-run-campaign")
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/addr_carry_canary_bundle")
}

fn fixture_manifest() -> PathBuf {
    fixture_root().join("run_manifest.v2.json")
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

fn retained_token(manifest_fd: &str, evidence_base_fd: &str, manifest_name: &str) -> String {
    let name_hex = manifest_name
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("--retained-manifest-fds-v1:{manifest_fd}:{evidence_base_fd}:{name_hex}")
}

fn retained_output_token(manifest: &Path, evidence_base: &Path, token: &str) -> Output {
    Command::new("sh")
        .arg("-c")
        .arg(
            r#"
exec 7<"$1" || exit 90
exec 8<"$2" || exit 91
exec 9<"$1" || exit 92
exec 10<"$2" || exit 93
exec "$3" "$4"
"#,
        )
        .arg("sim-run-retained-test")
        .arg(manifest)
        .arg(evidence_base)
        .arg(binary())
        .arg(token)
        .output()
        .unwrap()
}

fn retained_output(
    manifest: &Path,
    evidence_base: &Path,
    manifest_fd: &str,
    evidence_base_fd: &str,
    manifest_name: &str,
) -> Output {
    retained_output_token(
        manifest,
        evidence_base,
        &retained_token(manifest_fd, evidence_base_fd, manifest_name),
    )
}

fn unframe_one_document(raw: &[u8]) -> &[u8] {
    assert_eq!(raw.last(), Some(&b'\n'));
    let document = &raw[..raw.len() - 1];
    assert!(!document.contains(&b'\n'));
    document
}

fn assert_error_document(raw: &[u8], failure_type: &str, stage: &str, exit_code: u64) -> Value {
    let document = unframe_one_document(raw);
    let mut error: Value = serde_json::from_slice(document).unwrap();
    assert_eq!(canonical_json_utf8(&error).unwrap().as_bytes(), document);
    assert_eq!(error["schema_version"], "sim_error.v2");
    assert_eq!(error["error"]["failure_type"], failure_type);
    assert_eq!(error["error"]["stage"], stage);
    assert_eq!(error["exit_status"]["status"], "failure");
    assert_eq!(error["exit_status"]["exit_code"], exit_code);

    let seal = error
        .as_object_mut()
        .unwrap()
        .remove("error_sha256")
        .unwrap();
    let unsigned = canonical_json_utf8(&error).unwrap();
    assert_eq!(seal, Sha256Digest::of(unsigned.as_bytes()).to_string());
    error
}

fn assert_cli_error(output: &Output, failure_type: &str, stage: &str, exit_code: i32) -> Value {
    assert_eq!(output.status.code(), Some(exit_code));
    assert!(output.stdout.is_empty());
    assert_error_document(
        &output.stderr,
        failure_type,
        stage,
        u64::try_from(exit_code).unwrap(),
    )
}

fn assert_success_document(raw: &[u8]) -> Value {
    let document = unframe_one_document(raw);
    let mut result: Value = serde_json::from_slice(document).unwrap();
    assert_eq!(canonical_json_utf8(&result).unwrap().as_bytes(), document);
    assert_eq!(result["schema_version"], "sim_result.v2");
    assert_eq!(
        result["exit_status"],
        json!({"status": "success", "exit_code": 0})
    );
    let seal = result
        .as_object_mut()
        .unwrap()
        .remove("result_sha256")
        .unwrap();
    let unsigned = canonical_json_utf8(&result).unwrap();
    assert_eq!(seal, Sha256Digest::of(unsigned.as_bytes()).to_string());
    result
}

#[test]
fn cli_emits_only_one_canonical_v2_result_on_success() {
    let output = Command::new(binary())
        .arg(fixture_manifest())
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let parsed = assert_success_document(&output.stdout);
    let cycle_metric = &parsed["aggregate"]["metrics"]["exec_cycles_gpc_max"];
    assert!(matches!(
        cycle_metric["status"].as_str(),
        Some("predicted") | Some("required_unimplemented")
    ));
    if cycle_metric["status"] == "predicted" {
        assert_eq!(cycle_metric["value"]["kind"], "binding_cycles");
    }
}

#[test]
fn retained_descriptor_mode_matches_the_compatibility_path_result() {
    let (_temp, manifest_path) = copied_fixture();
    let base = manifest_path.parent().unwrap();
    let path_output = Command::new(binary()).arg(&manifest_path).output().unwrap();
    let retained = retained_output(
        &manifest_path,
        base,
        "7",
        "8",
        manifest_path.file_name().unwrap().to_str().unwrap(),
    );

    assert!(path_output.status.success());
    assert!(retained.status.success());
    assert!(retained.stderr.is_empty());
    assert_eq!(retained.stdout, path_output.stdout);
    assert_success_document(&retained.stdout);
}

#[test]
fn retained_mode_survives_named_manifest_permission_removal() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let token = retained_token(
        "7",
        "8",
        manifest_path.file_name().unwrap().to_str().unwrap(),
    );
    let output = Command::new("sh")
        .arg("-c")
        .arg(
            r#"
exec 7<"$1" || exit 90
exec 8<"$2" || exit 91
chmod 000 "$1" || exit 92
"$3" "$4"
status=$?
chmod 600 "$1" || exit 93
exit "$status"
"#,
        )
        .arg("sim-run-retained-permission-test")
        .arg(&manifest_path)
        .arg(root)
        .arg(binary())
        .arg(&token)
        .output()
        .unwrap();

    assert!(output.status.success(), "{:?}", output.stderr);
    assert!(output.stderr.is_empty());
    assert_success_document(&output.stdout);
}

#[test]
fn retained_mode_preserves_closed_low_base_descriptor_failure() {
    let (_temp, manifest_path) = copied_fixture();
    let token = retained_token(
        "7",
        "3",
        manifest_path.file_name().unwrap().to_str().unwrap(),
    );
    let output = Command::new("sh")
        .arg("-c")
        .arg(
            r#"
exec 7<"$1" || exit 90
exec 3<&-
exec "$2" "$3"
"#,
        )
        .arg("sim-run-retained-closed-base-test")
        .arg(&manifest_path)
        .arg(binary())
        .arg(&token)
        .output()
        .unwrap();

    let error = assert_cli_error(
        &output,
        "MANIFEST_ADMISSION_FAILED",
        "manifest_admission",
        2,
    );
    assert!(
        error["error"]["detail"]
            .as_str()
            .unwrap()
            .contains("retained descriptor is unavailable")
    );
}

#[test]
fn retained_mode_accepts_valid_descriptors_at_soft_limit() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let token = retained_token(
        "62",
        "63",
        manifest_path.file_name().unwrap().to_str().unwrap(),
    );
    let output = Command::new("sh")
        .arg("-c")
        .arg(
            r#"
ulimit -n 64 || exit 90
exec 62<"$1" || exit 91
exec 63<"$2" || exit 92
exec "$3" "$4"
"#,
        )
        .arg("sim-run-retained-high-fd-test")
        .arg(&manifest_path)
        .arg(root)
        .arg(binary())
        .arg(&token)
        .output()
        .unwrap();

    assert!(output.status.success(), "{:?}", output.stderr);
    assert!(output.stderr.is_empty());
    assert_success_document(&output.stdout);
}

#[test]
fn retained_mode_derives_nested_launch_parent_from_the_anchor() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let nested = root.join("nested/deeper");
    fs::create_dir_all(&nested).unwrap();
    for entry in ["launch_evidence.json", "trace", "control", "cards"] {
        fs::rename(root.join(entry), nested.join(entry)).unwrap();
    }
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["launches"][0]["launch_evidence_manifest_path"] =
        json!("nested/deeper/launch_evidence.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let path_output = Command::new(binary()).arg(&manifest_path).output().unwrap();
    let retained = retained_output(
        &manifest_path,
        root,
        "7",
        "8",
        manifest_path.file_name().unwrap().to_str().unwrap(),
    );
    assert!(path_output.status.success(), "{:?}", path_output.stderr);
    assert!(retained.status.success(), "{:?}", retained.stderr);
    assert_eq!(retained.stdout, path_output.stdout);
    assert_success_document(&retained.stdout);
}

#[test]
fn retained_mode_rejects_wrong_types_closed_fds_and_named_inode_mismatch() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let name = manifest_path.file_name().unwrap().to_str().unwrap();

    for (output, exit_code) in [
        (retained_output(&manifest_path, root, "8", "10", name), 1),
        (retained_output(&manifest_path, root, "7", "9", name), 1),
        (
            retained_output(&manifest_path, root, "1048575", "8", name),
            2,
        ),
    ] {
        let error = assert_cli_error(
            &output,
            "MANIFEST_ADMISSION_FAILED",
            "manifest_admission",
            exit_code,
        );
        assert!(error["error"].get("evidence").is_none());
    }

    let duplicate = retained_output(&manifest_path, root, "7", "7", name);
    assert_cli_error(&duplicate, "CLI_USAGE_FAILED", "cli_usage", 2);

    let alias_name = "same-bytes-different-inode.json";
    fs::copy(&manifest_path, root.join(alias_name)).unwrap();
    let mismatch = retained_output(&manifest_path, root, "7", "8", alias_name);
    let error = assert_cli_error(
        &mismatch,
        "MANIFEST_ADMISSION_FAILED",
        "manifest_admission",
        1,
    );
    assert!(
        error["error"]["detail"]
            .as_str()
            .unwrap()
            .contains("device/inode identity mismatch")
    );
}

#[test]
fn retained_mode_rejects_non_readonly_manifest_descriptors() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let token = retained_token(
        "7",
        "8",
        manifest_path.file_name().unwrap().to_str().unwrap(),
    );

    for script in [
        r#"
exec 7>>"$1" || exit 90
exec 8<"$2" || exit 91
exec "$3" "$4"
"#,
        r#"
exec 7<>"$1" || exit 90
exec 8<"$2" || exit 91
exec "$3" "$4"
"#,
    ] {
        let output = Command::new("sh")
            .arg("-c")
            .arg(script)
            .arg("sim-run-retained-access-mode-test")
            .arg(&manifest_path)
            .arg(root)
            .arg(binary())
            .arg(&token)
            .output()
            .unwrap();
        let error = assert_cli_error(
            &output,
            "MANIFEST_ADMISSION_FAILED",
            "manifest_admission",
            1,
        );
        assert!(
            error["error"]["detail"]
                .as_str()
                .unwrap()
                .contains("not an ordinary read-only descriptor")
        );
    }
}

#[test]
fn retained_token_rejects_noncanonical_fd_hex_and_name_forms() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let cases = [
        "--retained-manifest-fds-v1:0:8:72756e5f6d616e69666573742e76322e6a736f6e",
        "--retained-manifest-fds-v1:7:2:72756e5f6d616e69666573742e76322e6a736f6e",
        "--retained-manifest-fds-v1:07:8:72756e5f6d616e69666573742e76322e6a736f6e",
        "--retained-manifest-fds-v1:+7:8:72756e5f6d616e69666573742e76322e6a736f6e",
        "--retained-manifest-fds-v1:7:8:72756E",
        "--retained-manifest-fds-v1:7:8:0",
        "--retained-manifest-fds-v1:7:8:ff",
        "--retained-manifest-fds-v1:7:8:6e65737465642f72756e2e6a736f6e",
        "--retained-manifest-fds-v1:7:8:2e2e",
        "--retained-manifest-fds-v1:7:8:72756e:00",
    ];
    for token in cases {
        let output = retained_output_token(&manifest_path, root, token);
        let error = assert_cli_error(&output, "CLI_USAGE_FAILED", "cli_usage", 2);
        assert_eq!(
            error["error"]["evidence"]["provided_positional_argument_count"],
            1
        );
        assert_eq!(
            error["error"]["evidence"]["expected_positional_argument_count"],
            1
        );
    }

    for args in [
        vec!["--retained-manifest-fds", "7", "8", "run_manifest.v2.json"],
        vec!["--retained-manifest-fds-v1"],
        vec!["manifest.json", "--retained-manifest-fds-v1:7:8:6e616d65"],
    ] {
        let output = Command::new(binary()).args(args).output().unwrap();
        assert_cli_error(&output, "CLI_USAGE_FAILED", "cli_usage", 2);
    }
}

#[test]
fn retained_manifest_reads_ignore_shared_inherited_offset_and_repeat() {
    let (_temp, manifest_path) = copied_fixture();
    let root = manifest_path.parent().unwrap();
    let token = retained_token(
        "7",
        "8",
        manifest_path.file_name().unwrap().to_str().unwrap(),
    );
    let repeated = Command::new("sh")
        .arg("-c")
        .arg(
            r#"
exec 7<"$1" || exit 90
exec 8<"$2" || exit 91
while IFS= read -r ignored <&7; do :; done
"$3" "$4" || exit $?
"$3" "$4"
"#,
        )
        .arg("sim-run-retained-repeat-test")
        .arg(&manifest_path)
        .arg(root)
        .arg(binary())
        .arg(token)
        .output()
        .unwrap();
    assert!(repeated.status.success(), "{:?}", repeated.stderr);
    assert!(repeated.stderr.is_empty());

    let baseline = Command::new(binary()).arg(&manifest_path).output().unwrap();
    assert!(baseline.status.success());
    let mut expected = baseline.stdout.clone();
    expected.extend_from_slice(&baseline.stdout);
    assert_eq!(repeated.stdout, expected);
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
    let result = assert_success_document(&contaminated.stdout);
    assert_eq!(
        result["resolved_configuration"]["replay_mode"]["mode"],
        "l1_shared_l2"
    );
}

#[test]
fn cli_usage_counts_every_positional_argument_exactly() {
    for (args, expected_count) in [
        (Vec::<&str>::new(), 0),
        (vec!["one", "two"], 2),
        (vec!["one", "two", "three"], 3),
    ] {
        let output = Command::new(binary()).args(args).output().unwrap();
        let error = assert_cli_error(&output, "CLI_USAGE_FAILED", "cli_usage", 2);
        assert_eq!(
            error["error"]["evidence"]["provided_positional_argument_count"],
            expected_count
        );
        assert_eq!(
            error["error"]["evidence"]["expected_positional_argument_count"],
            1
        );
    }
}

#[test]
fn empty_path_is_one_argument_owned_by_manifest_admission() {
    let output = Command::new(binary()).arg("").output().unwrap();
    let error = assert_cli_error(
        &output,
        "MANIFEST_ADMISSION_FAILED",
        "manifest_admission",
        1,
    );
    assert!(error["error"].get("evidence").is_none());
}

#[test]
fn production_cli_rejects_v1_without_fallback() {
    let output = Command::new(binary())
        .arg(fixture_root().join("run_manifest.json"))
        .output()
        .unwrap();
    assert_cli_error(
        &output,
        "MANIFEST_ADMISSION_FAILED",
        "manifest_admission",
        1,
    );
}

#[test]
fn duplicate_json_members_are_manifest_admission_failures() {
    let (_temp, manifest_path) = copied_fixture();
    let raw = fs::read_to_string(&manifest_path).unwrap();
    let duplicate = raw.replacen(
        "\"schema_version\": 2,",
        "\"schema_version\": 2,\n  \"schema_version\": 2,",
        1,
    );
    assert_ne!(duplicate, raw);
    fs::write(&manifest_path, duplicate).unwrap();

    let output = Command::new(binary()).arg(manifest_path).output().unwrap();
    assert_cli_error(
        &output,
        "MANIFEST_ADMISSION_FAILED",
        "manifest_admission",
        1,
    );
}

#[test]
fn artifact_role_path_duplicates_keep_subprocess_error_boundaries() {
    for (artifact, duplicate_of, failure_type, stage) in [
        (
            "arch_card",
            "control_bits_sidecar",
            "ARCH_CARD_ADMISSION_FAILED",
            "arch_card_admission",
        ),
        (
            "pchase_card",
            "arch_card",
            "CONFIGURATION_RESOLUTION_FAILED",
            "configuration_resolution",
        ),
        (
            "stats_csv",
            "dynamic_trace_pb",
            "TRACE_ADMISSION_FAILED",
            "trace_admission",
        ),
    ] {
        let (_temp, manifest_path) = copied_fixture();
        let launch_manifest_path = manifest_path.parent().unwrap().join("launch_evidence.json");
        let mut launch_manifest: Value =
            serde_json::from_slice(&fs::read(&launch_manifest_path).unwrap()).unwrap();
        launch_manifest["artifacts"][artifact]["path"] =
            launch_manifest["artifacts"][duplicate_of]["path"].clone();
        fs::write(
            &launch_manifest_path,
            serde_json::to_vec_pretty(&launch_manifest).unwrap(),
        )
        .unwrap();

        let output = Command::new(binary()).arg(&manifest_path).output().unwrap();
        assert_cli_error(&output, failure_type, stage, 1);
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
        assert_cli_error(
            &output,
            "MANIFEST_ADMISSION_FAILED",
            "manifest_admission",
            1,
        );
    }
}

#[test]
fn missing_and_malformed_manifests_have_distinct_typed_exits() {
    let temp = TempDir::new().unwrap();
    let missing = Command::new(binary())
        .arg(temp.path().join("missing.json"))
        .output()
        .unwrap();
    assert_cli_error(
        &missing,
        "MANIFEST_ADMISSION_FAILED",
        "manifest_admission",
        2,
    );

    let malformed_path = temp.path().join("malformed.json");
    fs::write(&malformed_path, b"{}").unwrap();
    let malformed = Command::new(binary()).arg(malformed_path).output().unwrap();
    assert_cli_error(
        &malformed,
        "MANIFEST_ADMISSION_FAILED",
        "manifest_admission",
        1,
    );
}

#[test]
fn serialization_fallback_matches_the_v2_error_schema_and_seal() {
    let document = RunError::canonical_serialization_fallback();
    let framed = format!("{document}\n");
    let parsed = assert_error_document(
        framed.as_bytes(),
        "RESULT_SERIALIZATION_FAILED",
        "result_serialization",
        1,
    );
    assert_eq!(parsed.as_object().unwrap().len(), 3);
    assert!(parsed.get("build_identity").is_none());
    assert!(parsed["error"].get("evidence").is_none());
    assert!(
        parsed["error"]["detail"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
}

#[test]
fn closed_stdout_becomes_one_stderr_output_write_failure() {
    let mut child = Command::new(binary())
        .arg(fixture_manifest())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert_cli_error(&output, "OUTPUT_WRITE_FAILED", "output_write", 1);
}
