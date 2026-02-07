//! Biased Random-Key Genetic Algorithm (BRKGA).
//!
//! BRKGA separates the evolutionary engine from the problem by using a
//! random-key representation: chromosomes are vectors of `f64` in `[0, 1)`,
//! and a user-provided **decoder** maps keys to a solution and its cost.
//!
//! The engine handles population management (elite copy, mutant injection,
//! biased crossover) entirely — the user implements only [`BrkgaDecoder`].
//!
//! # References
//!
//! - Bean (1994), "Genetic algorithms and random keys for sequencing and optimization"
//! - Goncalves & Resende (2011), "Biased random-key genetic algorithms for
//!   combinatorial optimization", *J. Heuristics* 17(5), 487–525

mod config;
mod runner;
mod types;

pub use config::BrkgaConfig;
pub use runner::{BrkgaResult, BrkgaRunner};
pub use types::BrkgaDecoder;
