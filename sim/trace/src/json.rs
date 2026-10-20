//! Shared strict JSON decoding helper for simulator input contracts.

pub use sim_arch::strict_json_from_slice;

// Public strict JSON decoding entry point shared by trace metadata readers.
// This documents the boundary without adding a second parser.
