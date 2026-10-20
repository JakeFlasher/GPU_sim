//! Timing-neutral native-replay observations. Counts are not profiler samples.

// Child modules declared by the original public surface.

// Lexical sections share this parent module scope, preserving private fields and impl visibility.
include!("metrics/locality_mrc.rs");
include!("metrics/pipeline_activity.rs");
include!("metrics/tests.rs");
