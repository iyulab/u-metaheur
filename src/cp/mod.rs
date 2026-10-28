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
//! - **Solver**: [`CpSolver`] trait — interface for solver implementations;
//!   [`BranchAndBoundSolver`] solves disjunctive scheduling models exactly,
//!   [`SimpleCpSolver`] is one greedy pass
//! - **Check**: [`CpModel::check`] — every [`CpViolation`] of a solution against its model
//!
//! # Design
//!
//! The model is general; the solvers here are not -- each states the class of
//! model it searches and reports any other as not searched. There is no
//! general constraint propagation engine: the [`CpSolver`] trait allows
//! plugging in external solvers (OR-Tools, CPLEX) or custom heuristics.
//!
//! Domain-specific objectives (e.g., makespan, tardiness) belong in consumer
//! layers. This module provides only generic `Minimize`/`Maximize` objectives.
//!
//! # References
//!
//! Rossi, van Beek & Walsh (2006), "Handbook of Constraint Programming"

mod branch_bound;
mod check;
mod model;
mod solver;
mod variables;

pub use branch_bound::BranchAndBoundSolver;
pub use check::CpViolation;
pub use model::{Constraint, CpModel, Objective};
pub use solver::{
    CpSolution, CpSolver, IntervalSolution, SimpleCpSolver, SolverConfig, SolverStatus,
};
pub use variables::{BoolVar, DurationVar, IntVar, IntervalVar, TimeVar};
