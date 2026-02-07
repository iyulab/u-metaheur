//! Adaptive Large Neighborhood Search (ALNS) framework.
//!
//! ALNS iteratively destroys and repairs solutions using a portfolio of
//! operators whose selection probabilities adapt based on past performance.
//!
//! # References
//!
//! Ropke & Pisinger (2006), "An Adaptive Large Neighborhood Search Heuristic
//! for the Pickup and Delivery Problem with Time Windows"

mod config;
mod runner;
mod types;

pub use config::AlnsConfig;
pub use runner::{AlnsResult, AlnsRunner};
pub use types::{AlnsProblem, DestroyOperator, RepairOperator};
