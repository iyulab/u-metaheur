//! Generic priority rule composition framework.
//!
//! Provides a domain-agnostic engine for evaluating and ranking items
//! using multiple scoring rules. The framework supports:
//!
//! - **Sequential evaluation**: Rules are applied in order; later rules
//!   act as tie-breakers when earlier rules cannot differentiate.
//! - **Weighted evaluation**: All rules contribute simultaneously via
//!   weighted sum.
//!
//! # Design
//!
//! This module contains NO domain-specific concepts. Scheduling rules
//! (SPT, EDD, etc.) and nesting rules (utilization, area) are defined
//! by consumers at higher layers using the generic `PriorityRule` trait.
//!
//! # References
//!
//! Dispatching rule composition: Pinedo (2016), "Scheduling: Theory,
//! Algorithms, and Systems"

mod engine;
mod types;

pub use engine::{EvaluationMode, RuleEngine, TieBreaker};
pub use types::PriorityRule;
