//! Domain-agnostic metaheuristic optimization framework.
//!
//! Provides generic implementations of common metaheuristic algorithms:
//!
//! - **Genetic Algorithm (GA)**: Population-based evolutionary optimization
//!   with pluggable selection, crossover, and mutation operators.
//!
//! # Architecture
//!
//! This crate sits at Layer 2 (Algorithms) in the U-Engine ecosystem,
//! depending only on `u-optim` (Layer 1: Foundation). It contains no
//! domain-specific concepts — scheduling, nesting, routing, etc. are
//! all defined by consumers at higher layers.
//!
//! # Usage
//!
//! Define your problem by implementing the [`ga::GaProblem`] trait,
//! then run optimization with [`ga::GaRunner`].

pub mod ga;
