//! Genetic Algorithm framework.
//!
//! A generic, domain-agnostic GA framework built on trait-based abstractions.
//! Users define their problem by implementing [`GaProblem`], which specifies
//! how to create, evaluate, crossover, and mutate individuals.
//!
//! # Core Traits
//!
//! - [`Individual`]: A candidate solution with associated fitness type
//! - [`GaProblem`]: Problem definition — initialization, evaluation, operators
//!
//! # Key Types
//!
//! - [`GaConfig`]: Algorithm parameters (population size, selection, presets)
//! - [`GaRunner`]: Executes the evolutionary loop
//! - [`GaResult`]: Final optimization result with statistics
//!
//! # Submodules
//!
//! - [`operators`]: Generic permutation crossover (OX, PMX) and mutation operators
//! - [`multi_objective`]: Pareto non-dominated sorting and crowding distance (NSGA-II utilities)
//!
//! # References
//!
//! - Holland (1975), *Adaptation in Natural and Artificial Systems*
//! - Goldberg (1989), *Genetic Algorithms in Search, Optimization, and Machine Learning*
//! - De Jong (2006), *Evolutionary Computation: A Unified Approach*
//! - Deb et al. (2002), *A Fast and Elitist Multiobjective GA: NSGA-II*

mod config;
pub mod multi_objective;
pub mod operators;
mod runner;
mod selection;
mod types;

pub use config::GaConfig;
pub use runner::{GaResult, GaRunner, GenerationStats};
pub use selection::Selection;
pub use types::{Fitness, GaProblem, Individual};
