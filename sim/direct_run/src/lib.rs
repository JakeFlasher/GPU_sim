//! Direct composition of trace loading, scheduling, and GPU replay.
//!
//! The runner accepts ordinary input paths. It deliberately has no admission
//! manifest, artifact digest, receipt, publication, or approval layer.

// Child modules declared by the original public surface.
pub mod emit;
mod metrics;
pub mod ncu;
mod prepared;
mod resources;

// Lexical sections share this parent module scope, preserving private fields and impl visibility.
include!("lib/reports.rs");
include!("lib/core.rs");
