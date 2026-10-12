//! Regenerates `sim/trace/src/pb/dynamic_trace.rs` from the tracer's
//! authoritative .proto schema at
//! `tools/nvbit_tracer/traces_enhanced/dynamic_trace/`.
//!
//! Run via `sim/trace/regen_pb.sh`, which pins PROTOC to the repo's
//! `.toolchain/pb/bin/protoc` (34.1), generates into an external temporary
//! directory, and diffs those bytes against the committed binding. The generator
//! refuses repository-local outputs so verification cannot overwrite its own
//! authority. The generated file is COMMITTED, so the sim workspace never needs
//! protoc at build time; prost implements the wire format itself, so the C++ exact
//! gencode==runtime rule does not apply here.

use std::ffi::OsString;
use std::path::PathBuf;

fn parse_paths(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<(PathBuf, PathBuf), String> {
    let mut arguments = arguments.into_iter();
    let repo_root = arguments
        .next()
        .ok_or_else(|| "missing runtime repository root".to_string())?;
    let out_dir = arguments
        .next()
        .ok_or_else(|| "missing runtime output directory".to_string())?;
    if arguments.next().is_some() {
        return Err("unexpected extra generator argument".to_string());
    }
    Ok((PathBuf::from(repo_root), PathBuf::from(out_dir)))
}

fn validate_external_output(
    repo_root: &std::path::Path,
    out_dir: &std::path::Path,
) -> Result<(), String> {
    if out_dir.starts_with(repo_root) {
        return Err(format!(
            "generator output must be outside the repository: {}",
            out_dir.display()
        ));
    }
    Ok(())
}

fn canonical_paths(
    repo_root: PathBuf,
    out_dir: PathBuf,
) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    if !repo_root.is_absolute() || !out_dir.is_absolute() {
        return Err("generator paths must be absolute".to_string());
    }
    let repo_root = repo_root
        .canonicalize()
        .map_err(|error| format!("canonicalize repository root: {error}"))?;
    let out_dir = out_dir
        .canonicalize()
        .map_err(|error| format!("canonicalize output directory: {error}"))?;
    validate_external_output(&repo_root, &out_dir)?;
    if !out_dir.is_dir() {
        return Err(format!(
            "generator output path is not a directory: {}",
            out_dir.display()
        ));
    }
    let proto_dir = repo_root
        .join("tools/nvbit_tracer/traces_enhanced/dynamic_trace")
        .canonicalize()
        .map_err(|error| format!("canonicalize proto schema directory: {error}"))?;
    if !proto_dir.is_dir() {
        return Err(format!(
            "proto schema path is not a directory: {}",
            proto_dir.display()
        ));
    }
    Ok((repo_root, proto_dir, out_dir))
}

fn main() {
    let (repo_root, out_dir) =
        parse_paths(std::env::args_os().skip(1)).unwrap_or_else(|error| panic!("{error}"));
    let (_repo_root, proto_dir, out_dir) =
        canonical_paths(repo_root, out_dir).unwrap_or_else(|error| panic!("{error}"));

    // Enumerate the schema dir instead of hardcoding a list: a new .proto
    // added upstream must be picked up, never silently omitted.
    let mut protos: Vec<PathBuf> = std::fs::read_dir(&proto_dir)
        .expect("read proto schema dir")
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()
        .expect("read every proto schema directory entry");
    protos.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "proto")
    });
    protos.sort();
    assert!(
        !protos.is_empty(),
        "no .proto files found in {}",
        proto_dir.display()
    );

    let mut config = prost_build::Config::new();
    config
        .out_dir(&out_dir)
        // Deterministic iteration order for device/stream/warp maps: the
        // simulator's event schedule must not depend on hash seeds.
        .btree_map(["."]);
    config
        .compile_protos(&protos, &[proto_dir.clone()])
        .expect("prost codegen failed");
    println!(
        "generated {} from {}",
        out_dir.join("dynamic_trace.rs").display(),
        proto_dir.display()
    );
}
