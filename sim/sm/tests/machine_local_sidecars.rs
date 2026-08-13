#![allow(clippy::unwrap_used)] // test binary: unwrap is an assertion
//! Machine-local smoke: load every real control-bits sidecar available for
//! the sm_86 roster. The path comes from the environment:
//!
//! ```sh
//! SIM_CONTROL_BITS_DIR=<dir with *.control_bits.json> \
//!   cargo test --release -p sim-sm --test machine_local_sidecars \
//!   real_sidecars_all_load -- --ignored --exact --nocapture
//! ```

use sim_sm::ControlSidecar;

#[test]
#[ignore = "requires machine-local control-bit sidecars"]
fn real_sidecars_all_load() {
    let dir = std::env::var("SIM_CONTROL_BITS_DIR")
        .expect("SIM_CONTROL_BITS_DIR must point to control-bit sidecars");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read sidecar dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with(".control_bits.json"))
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no sidecars found in {dir}");
    let mut total_insts = 0usize;
    for f in &files {
        let sc =
            ControlSidecar::load(f, "sm_86").unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        let n: usize = sc.functions.values().map(|fc| fc.by_pc.len()).sum();
        total_insts += n;
        eprintln!(
            "{}: {} function(s), {} decoded instruction(s)",
            sc.binary,
            sc.functions.len(),
            n
        );
    }
    eprintln!(
        "smoke: {} sidecar(s), {} decoded instruction(s) total",
        files.len(),
        total_insts
    );
}
