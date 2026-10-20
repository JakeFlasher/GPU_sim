//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

#[path = "execution_completion.rs"]
mod execution_completion;
#[path = "execution_loop.rs"]
mod execution_loop;
pub use self::execution_completion::*;
pub use self::execution_loop::*;
