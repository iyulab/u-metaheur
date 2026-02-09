//! Tabu Search (TS).
//!
//! A single-solution trajectory metaheuristic that uses memory structures
//! (the tabu list) to forbid recently visited moves, preventing cycling
//! and encouraging exploration of new regions of the search space.
//!
//! # References
//!
//! - Glover, F. (1989). "Tabu Search—Part I", *ORSA Journal on Computing* 1(3), 190-206.
//! - Glover, F. (1990). "Tabu Search—Part II", *ORSA Journal on Computing* 2(1), 4-32.

mod config;
mod runner;
mod types;

pub use config::TabuConfig;
pub use runner::{TabuResult, TabuRunner};
pub use types::{TabuMove, TabuProblem};
