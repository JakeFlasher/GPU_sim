//! Typed boundary state shared by the detailed and accelerated executors.
//!
//! These records are deliberately model-relative.  They describe the state
//! that the simulator can transform and compare; they do not claim that a
//! virtual trace address is a physical cache address or that activity counts
//! are measured watts.

// Child modules declared by the original public surface.

// Lexical sections share this parent module scope, preserving private fields and impl visibility.
include!("state/cache_boundary.rs");
include!("state/typed_events.rs");
include!("state/core_boundary.rs");
