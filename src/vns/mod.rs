//! Variable Neighborhood Search (VNS).
//!
//! A single-solution metaheuristic that systematically changes
//! neighborhood structures during the search. At each step, a random
//! perturbation (shaking) in the current neighborhood is followed by
//! local search. If improvement is found, the search resets to the
//! first (smallest) neighborhood; otherwise, it moves to the next
//! (larger) neighborhood.
//!
//! # References
//!
//! - Mladenović, N. & Hansen, P. (1997). "Variable neighborhood search",
//!   *Computers & Operations Research* 24(11), 1097-1100.
//! - Hansen, P. & Mladenović, N. (2001). "Variable neighborhood search:
//!   Principles and applications", *European Journal of Operational Research* 130(3), 449-467.

mod config;
mod runner;
mod types;

pub use config::VnsConfig;
pub use runner::{VnsResult, VnsRunner};
pub use types::VnsProblem;
