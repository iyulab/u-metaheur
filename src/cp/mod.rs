//! Constraint Programming (CP) framework.
//!
//! Provides a domain-agnostic model for expressing constrained optimization
//! problems using interval, integer, and boolean variables with constraints.
//!
//! # Key Components
//!
//! - **Variables**: [`IntervalVar`], [`IntVar`], [`BoolVar`] — decision variables
//! - **Constraints**: [`Constraint`] — NoOverlap, Cumulative, Precedence, etc.
//! - **Model**: [`CpModel`] — container for variables, constraints, objective
//! - **Solver**: [`CpSolver`] trait — interface for solver implementations
//!
//! # Design
//!
//! This module defines the modeling layer only. It does NOT include a full
//! constraint propagation engine. The [`CpSolver`] trait allows plugging in
//! external solvers (OR-Tools, CPLEX) or custom heuristics.
//!
//! Domain-specific objectives (e.g., makespan, tardiness) belong in consumer
//! layers. This module provides only generic `Minimize`/`Maximize` objectives.
//!
//! # References
//!
//! Rossi, van Beek & Walsh (2006), "Handbook of Constraint Programming"

mod model;
mod solver;
mod variables;

pub use model::{Constraint, CpModel, Objective};
pub use solver::{
    CpSolution, CpSolver, IntervalSolution, SimpleCpSolver, SolverConfig, SolverStatus,
};
pub use variables::{BoolVar, DurationVar, IntVar, IntervalVar, TimeVar};
