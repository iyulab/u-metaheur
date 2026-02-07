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
//!
//! # Architecture
//!
//! This crate sits at Layer 2 (Algorithms) in the U-Engine ecosystem,
//! depending only on `u-optim` (Layer 1: Foundation). It contains no
//! domain-specific concepts — scheduling, nesting, routing, etc. are
//! all defined by consumers at higher layers.

pub mod brkga;
pub mod ga;
pub mod sa;
