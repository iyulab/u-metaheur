//! Simulated Annealing (SA).
//!
//! A single-solution trajectory metaheuristic inspired by the physical
//! annealing process. Accepts worsening moves with a probability that
//! decreases over time (temperature), allowing the search to escape
//! local optima.
//!
//! # References
//!
//! - Kirkpatrick, Gelatt & Vecchi (1983), "Optimization by Simulated Annealing"
//! - Cerny (1985), "Thermodynamical Approach to the Travelling Salesman Problem"
//! - Lundy & Mees (1986), "Convergence of an Annealing Algorithm"

mod config;
mod runner;
mod types;

pub use config::{CoolingSchedule, SaConfig};
pub use runner::{SaResult, SaRunner};
pub use types::SaProblem;
