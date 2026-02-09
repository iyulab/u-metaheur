//! Domain-agnostic metaheuristic optimization framework.
//!
//! Provides generic implementations of common metaheuristic algorithms:
//!
//! - **Genetic Algorithm (GA)**: Population-based evolutionary optimization
//!   with pluggable selection, crossover, and mutation operators.
//! - **BRKGA**: Biased Random-Key Genetic Algorithm — the user implements
//!   only a decoder; all evolutionary mechanics are handled generically.
//! - **Simulated Annealing (SA)**: Single-solution trajectory optimization
//!   with pluggable cooling schedules.
//! - **ALNS**: Adaptive Large Neighborhood Search — destroy/repair operators
//!   with adaptive weight selection.
//! - **Dispatching**: Generic priority rule composition engine for
//!   multi-rule item ranking.
//! - **CP (Constraint Programming)**: Domain-agnostic modeling layer for
//!   constrained optimization with interval, integer, and boolean variables.
//! - **Tabu Search (TS)**: Single-solution trajectory optimization using
//!   short-term memory (tabu list) to escape local optima.
//! - **Variable Neighborhood Search (VNS)**: Systematic neighborhood
//!   switching for escaping local optima via diversified perturbation.
//!
//! # Architecture
//!
//! This crate sits at Layer 2 (Algorithms) in the U-Engine ecosystem,
//! depending only on `u-numerics` (Layer 1: Foundation). It contains no
//! domain-specific concepts — scheduling, nesting, routing, etc. are
//! all defined by consumers at higher layers.

pub mod alns;
pub mod brkga;
pub mod cp;
pub mod dispatching;
pub mod ga;
pub mod sa;
pub mod tabu;
pub mod vns;
