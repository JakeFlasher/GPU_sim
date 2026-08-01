use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};

const SOURCE_INPUTS: &[&str] = &[
    "rust-toolchain.toml",
    "sim/Cargo.toml",
    "sim/Cargo.lock",
    "sim/arch/Cargo.toml",
    "sim/arch/src",
    "sim/engine/Cargo.toml",
    "sim/engine/src",
    "sim/mem/Cargo.toml",
    "sim/mem/src",
    "sim/sm/Cargo.toml",
    "sim/sm/src",
    "sim/trace/Cargo.toml",
    "sim/trace/src",
    "sim/run/Cargo.toml",
    "sim/run/build.rs",
    "sim/run/schema",
    "sim/run/src",
];

const GIT_REDIRECT_ENV: &[&str] = &[
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_WORK_TREE",
];

fn main() {
    let manifest_dir = required_path_env("CARGO_MANIFEST_DIR");
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| {
            panic!("CARGO_MANIFEST_DIR has no repository ancestor: {manifest_dir:?}")
        })
        .to_path_buf();
    verify_repository_root(&repo_root);
    emit_rerun_directives(&repo_root);

    let sim_git_commit = git_commit(&repo_root);
    let sim_git_dirty = simulator_source_dirty(&repo_root);
    let rustc_version = rustc_version();
    let cargo_profile = cargo_profile();

    let generated = format!(
        "pub(crate) const SIM_GIT_COMMIT: &str = {sim_git_commit:?};\n\
         pub(crate) const SIM_GIT_DIRTY: bool = {sim_git_dirty};\n\
         pub(crate) const RUSTC_VERSION: &str = {rustc_version:?};\n\
         pub(crate) const CARGO_PROFILE: &str = {cargo_profile:?};\n"
    );
    let out_dir = required_path_env("OUT_DIR");
    fs::write(out_dir.join("build_provenance.rs"), generated)
        .unwrap_or_else(|error| panic!("could not write embedded build provenance: {error}"));
}

fn required_path_env(name: &'static str) -> PathBuf {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("required Cargo environment variable {name} is unavailable"))
}

fn verify_repository_root(repo_root: &Path) {
    let actual = git_output(repo_root, ["rev-parse", "--show-toplevel"]);
    let actual = one_line_utf8("git repository root", &actual.stdout);
    let expected = repo_root
        .canonicalize()
        .unwrap_or_else(|error| panic!("could not canonicalize repository root: {error}"));
    let actual = PathBuf::from(actual)
        .canonicalize()
        .unwrap_or_else(|error| panic!("could not canonicalize Git repository root: {error}"));
    assert_eq!(
        actual, expected,
        "sim-run build provenance resolved the wrong Git repository"
    );
}

fn emit_rerun_directives(repo_root: &Path) {
    for relative in SOURCE_INPUTS {
        println!(
            "cargo:rerun-if-changed={}",
            repo_root.join(relative).display()
        );
    }
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=PROFILE");

    for git_path in ["HEAD", "index", "packed-refs"] {
        let output = git_output(
            repo_root,
            [
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                git_path,
            ],
        );
        let path = one_line_utf8("Git metadata path", &output.stdout);
        if Path::new(path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }

    let mut symbolic_ref = git_command(repo_root);
    symbolic_ref.args(["symbolic-ref", "-q", "HEAD"]);
    let output = symbolic_ref
        .output()
        .unwrap_or_else(|error| panic!("could not resolve symbolic Git HEAD: {error}"));
    match output.status.code() {
        Some(0) => {
            let reference = one_line_utf8("symbolic Git HEAD", &output.stdout);
            let path = git_output(
                repo_root,
                [
                    "rev-parse",
                    "--path-format=absolute",
                    "--git-path",
                    reference,
                ],
            );
            let path = one_line_utf8("Git reference path", &path.stdout);
            emit_rerun_path_or_existing_parent(Path::new(path));
        }
        Some(1) => {
            if !output.stdout.is_empty() {
                panic!("detached Git HEAD probe produced unexpected output");
            }
        }
        _ => fail_command("git symbolic-ref -q HEAD", output.status, &output.stderr),
    }
}

fn emit_rerun_path_or_existing_parent(path: &Path) {
    let watched = path
        .ancestors()
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| panic!("Git reference has no existing ancestor: {path:?}"));
    println!("cargo:rerun-if-changed={}", watched.display());
}

fn git_commit(repo_root: &Path) -> String {
    let output = git_output(repo_root, ["rev-parse", "--verify", "HEAD^{commit}"]);
    let commit = one_line_utf8("Git commit", &output.stdout);
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        panic!("Git commit is not exactly 40 lowercase hexadecimal characters");
    }
    commit.to_owned()
}

fn simulator_source_dirty(repo_root: &Path) -> bool {
    let mut diff = git_command(repo_root);
    diff.args([
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--quiet",
        "HEAD",
        "--",
    ]);
    diff.args(SOURCE_INPUTS);
    let status = diff
        .status()
        .unwrap_or_else(|error| panic!("could not inspect simulator source status: {error}"));
    let tracked_dirty = match status.code() {
        Some(0) => false,
        Some(1) => true,
        _ => fail_command("git diff --quiet HEAD", status, &[]),
    };

    let mut index_flags = git_command(repo_root);
    index_flags.args(["ls-files", "-v", "-z", "--"]);
    index_flags.args(SOURCE_INPUTS);
    let output = index_flags
        .output()
        .unwrap_or_else(|error| panic!("could not inspect simulator index flags: {error}"));
    if !output.status.success() {
        fail_command("git ls-files -v", output.status, &output.stderr);
    }
    let hidden_index_state = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
        .any(|entry| !matches!(entry, [b'H', b' ', ..]));

    let mut untracked = git_command(repo_root);
    untracked.args(["ls-files", "--others", "--exclude-standard", "-z", "--"]);
    untracked.args(SOURCE_INPUTS);
    let output = untracked
        .output()
        .unwrap_or_else(|error| panic!("could not inspect untracked simulator sources: {error}"));
    if !output.status.success() {
        fail_command("git ls-files --others", output.status, &output.stderr);
    }
    tracked_dirty || hidden_index_state || !output.stdout.is_empty()
}

fn rustc_version() -> String {
    let rustc = env::var_os("RUSTC")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("Cargo did not identify the rustc executable"));
    let output = Command::new(&rustc)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| panic!("could not query Cargo's rustc executable: {error}"));
    if !output.status.success() {
        fail_command("rustc --version", output.status, &output.stderr);
    }
    let version = one_line_utf8("rustc version", &output.stdout);
    if !version.strip_prefix("rustc ").is_some_and(|suffix| {
        !suffix.is_empty()
            && suffix
                .bytes()
                .all(|byte| byte == b' ' || byte.is_ascii_graphic())
    }) {
        panic!("rustc --version returned an unknown spelling");
    }
    version.to_owned()
}

fn cargo_profile() -> String {
    let profile =
        env::var("PROFILE").unwrap_or_else(|_| panic!("Cargo did not identify the active profile"));
    if profile.is_empty()
        || !profile
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        panic!("Cargo profile has an unknown spelling: {profile:?}");
    }
    profile
}

fn git_output<I, S>(repo_root: &Path, args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = git_command(repo_root);
    command.args(args);
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("could not execute Git provenance probe: {error}"));
    if !output.status.success() {
        fail_command("git provenance probe", output.status, &output.stderr);
    }
    output
}

fn git_command(repo_root: &Path) -> Command {
    let mut command = Command::new("git");
    command.current_dir(repo_root);
    for name in GIT_REDIRECT_ENV {
        command.env_remove(name);
    }
    command
}

fn one_line_utf8<'a>(label: &'static str, raw: &'a [u8]) -> &'a str {
    let text =
        std::str::from_utf8(raw).unwrap_or_else(|error| panic!("{label} is not UTF-8: {error}"));
    let text = text.strip_suffix('\n').unwrap_or(text);
    let text = text.strip_suffix('\r').unwrap_or(text);
    if text.is_empty() || text.contains(['\r', '\n']) {
        panic!("{label} is empty or has multiple lines");
    }
    text
}

fn fail_command(context: &'static str, status: ExitStatus, stderr: &[u8]) -> ! {
    let stderr = String::from_utf8_lossy(stderr);
    panic!("{context} failed with {status}: {stderr}");
}
