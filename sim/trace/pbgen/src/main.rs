//! Regenerates `sim/trace/src/pb/dynamic_trace.rs` from the tracer's
//! authoritative .proto schema at
//! `tools/nvbit_tracer/traces_enhanced/dynamic_trace/`.
//!
//! Run via `sim/trace/regen_pb.sh`, which pins PROTOC to the repo's
//! `.toolchain/pb/bin/protoc` (34.1). The generated file is COMMITTED so the
//! sim workspace never needs protoc at build time; prost implements the wire
//! format itself, so the C++ exact gencode==runtime rule does not apply here.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .ancestors()
        .nth(3)
        .expect("pbgen sits at <repo>/sim/trace/pbgen");
    let proto_dir = repo_root.join("tools/nvbit_tracer/traces_enhanced/dynamic_trace");
    let out_dir = manifest_dir
        .parent()
        .expect("pbgen parent is sim/trace")
        .join("src/pb");

    // Enumerate the schema dir instead of hardcoding a list: a new .proto
    // added upstream must be picked up, never silently omitted.
    let mut protos: Vec<PathBuf> = std::fs::read_dir(&proto_dir)
        .expect("read proto schema dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "proto"))
        .collect();
    protos.sort();
    assert!(!protos.is_empty(), "no .proto files found in {}", proto_dir.display());

    std::fs::create_dir_all(&out_dir).expect("create sim/trace/src/pb");
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
