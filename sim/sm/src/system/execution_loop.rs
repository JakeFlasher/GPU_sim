//! Interleaved event-loop facade; child modules mutate event ordering.

use super::*;

#[path = "execution_loop_a.rs"]
mod execution_loop_a;
#[path = "execution_loop_b.rs"]
mod execution_loop_b;
pub use self::execution_loop_a::*;
pub use self::execution_loop_b::*;
