//! CP model definition.

use super::variables::{BoolVar, IntVar, IntervalVar};
use std::collections::HashMap;

/// A constraint in the CP model.
///
/// These are domain-agnostic scheduling/resource constraints commonly
/// used in constraint programming. Domain-specific constraints
/// (e.g., transition matrices) should be added at the consumer layer.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Non-overlapping intervals on a shared resource.
    ///
    /// No two intervals in the set may overlap in time.
    NoOverlap {
        /// Names of interval variables that must not overlap.
        intervals: Vec<String>,
    },

    /// Cumulative resource constraint.
    ///
    /// At any point in time, the sum of demands of active intervals
    /// must not exceed the capacity.
    Cumulative {
        /// Names of interval variables.
        intervals: Vec<String>,
        /// Demand of each interval (parallel to `intervals`).
        demands: Vec<i64>,
        /// Maximum capacity.
        capacity: i64,
    },

    /// Precedence constraint: `before` must end before `after` starts.
    ///
    /// More precisely: `end(before) + min_delay <= start(after)`.
    Precedence {
        /// Interval that must come first.
        before: String,
        /// Interval that must come after.
        after: String,
        /// Minimum delay between end of `before` and start of `after`.
        min_delay: i64,
    },

    /// Same-start constraint: two intervals must start at the same time.
    SameStart {
        interval1: String,
        interval2: String,
    },

    /// Same-end constraint: two intervals must end at the same time.
    SameEnd {
        interval1: String,
        interval2: String,
    },

    /// Alternative constraint: exactly one of the alternatives is present.
    ///
    /// The `main` interval is performed if and only if exactly one
    /// of the `alternatives` is present.
    Alternative {
        /// The main interval.
        main: String,
        /// Alternative implementations.
        alternatives: Vec<String>,
    },
}

/// Objective function for the CP model.
#[derive(Debug, Clone)]
pub enum Objective {
    /// Minimize a linear combination of integer variables.
    Minimize {
        /// (variable_name, coefficient) pairs.
        terms: Vec<(String, f64)>,
    },

    /// Maximize a linear combination of integer variables.
    Maximize {
        /// (variable_name, coefficient) pairs.
        terms: Vec<(String, f64)>,
    },

    /// Minimize the maximum end time across all intervals.
    MinimizeMaxEnd,

    /// Hierarchical (lexicographic) multi-objective.
    Hierarchical { objectives: Vec<Objective> },
}

/// A constraint programming model.
///
/// Contains variables, constraints, and an optional objective function.
///
/// # Examples
///
/// ```
/// use u_metaheur::cp::{CpModel, IntervalVar, Constraint, Objective};
///
/// let mut model = CpModel::new("example", 1000);
/// model.add_interval(IntervalVar::new("op1", 0, 100, 50, 200));
/// model.add_interval(IntervalVar::new("op2", 0, 100, 30, 200));
/// model.add_no_overlap(vec!["op1".into(), "op2".into()]);
/// model.set_objective(Objective::MinimizeMaxEnd);
/// assert!(model.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct CpModel {
    /// Model name.
    pub name: String,
    /// Interval variables.
    pub intervals: HashMap<String, IntervalVar>,
    /// Integer variables.
    pub int_vars: HashMap<String, IntVar>,
    /// Boolean variables.
    pub bool_vars: HashMap<String, BoolVar>,
    /// Constraints.
    pub constraints: Vec<Constraint>,
    /// Objective function.
    pub objective: Option<Objective>,
    /// Planning horizon (maximum time).
    pub horizon: i64,
}

impl CpModel {
    /// Creates a new empty model.
    pub fn new(name: impl Into<String>, horizon: i64) -> Self {
        Self {
            name: name.into(),
            intervals: HashMap::new(),
            int_vars: HashMap::new(),
            bool_vars: HashMap::new(),
            constraints: Vec::new(),
            objective: None,
            horizon,
        }
    }

    /// Adds an interval variable.
    pub fn add_interval(&mut self, var: IntervalVar) {
        self.intervals.insert(var.name.clone(), var);
    }

    /// Adds an integer variable.
    pub fn add_int_var(&mut self, var: IntVar) {
        self.int_vars.insert(var.name.clone(), var);
    }

    /// Adds a boolean variable.
    pub fn add_bool_var(&mut self, var: BoolVar) {
        self.bool_vars.insert(var.name.clone(), var);
    }

    /// Adds a constraint.
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Convenience: add a no-overlap constraint.
    pub fn add_no_overlap(&mut self, intervals: Vec<String>) {
        self.constraints.push(Constraint::NoOverlap { intervals });
    }

    /// Convenience: add a cumulative constraint.
    pub fn add_cumulative(&mut self, intervals: Vec<String>, demands: Vec<i64>, capacity: i64) {
        self.constraints.push(Constraint::Cumulative {
            intervals,
            demands,
            capacity,
        });
    }

    /// Convenience: add a precedence constraint.
    pub fn add_precedence(&mut self, before: String, after: String, min_delay: i64) {
        self.constraints.push(Constraint::Precedence {
            before,
            after,
            min_delay,
        });
    }

    /// Sets the objective function.
    pub fn set_objective(&mut self, objective: Objective) {
        self.objective = Some(objective);
    }

    /// Validates the model for consistency.
    ///
    /// Checks that all referenced interval/variable names exist.
    pub fn validate(&self) -> Result<(), String> {
        for constraint in &self.constraints {
            match constraint {
                Constraint::NoOverlap { intervals } => {
                    for name in intervals {
                        if !self.intervals.contains_key(name) {
                            return Err(format!("undefined interval: {name}"));
                        }
                    }
                }
                Constraint::Cumulative {
                    intervals, demands, ..
                } => {
                    if intervals.len() != demands.len() {
                        return Err("cumulative: intervals and demands length mismatch".into());
                    }
                    for name in intervals {
                        if !self.intervals.contains_key(name) {
                            return Err(format!("undefined interval: {name}"));
                        }
                    }
                }
                Constraint::Precedence { before, after, .. } => {
                    if !self.intervals.contains_key(before) {
                        return Err(format!("undefined interval: {before}"));
                    }
                    if !self.intervals.contains_key(after) {
                        return Err(format!("undefined interval: {after}"));
                    }
                }
                Constraint::SameStart {
                    interval1,
                    interval2,
                }
                | Constraint::SameEnd {
                    interval1,
                    interval2,
                } => {
                    if !self.intervals.contains_key(interval1) {
                        return Err(format!("undefined interval: {interval1}"));
                    }
                    if !self.intervals.contains_key(interval2) {
                        return Err(format!("undefined interval: {interval2}"));
                    }
                }
                Constraint::Alternative {
                    main, alternatives, ..
                } => {
                    if !self.intervals.contains_key(main) {
                        return Err(format!("undefined interval: {main}"));
                    }
                    for name in alternatives {
                        if !self.intervals.contains_key(name) {
                            return Err(format!("undefined interval: {name}"));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns the number of interval variables.
    pub fn interval_count(&self) -> usize {
        self.intervals.len()
    }

    /// Returns the number of constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let mut model = CpModel::new("test", 1_000_000);
        model.add_interval(IntervalVar::new("op1", 0, 100_000, 50_000, 200_000));
        model.add_interval(IntervalVar::new("op2", 0, 100_000, 30_000, 200_000));
        model.add_no_overlap(vec!["op1".into(), "op2".into()]);
        model.set_objective(Objective::MinimizeMaxEnd);

        assert_eq!(model.interval_count(), 2);
        assert_eq!(model.constraint_count(), 1);
        assert!(model.objective.is_some());
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_precedence() {
        let mut model = CpModel::new("test", 1000);
        model.add_interval(IntervalVar::new("a", 0, 100, 50, 200));
        model.add_interval(IntervalVar::new("b", 0, 100, 30, 200));
        model.add_precedence("a".into(), "b".into(), 10);

        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_cumulative() {
        let mut model = CpModel::new("test", 1000);
        model.add_interval(IntervalVar::new("a", 0, 100, 50, 200));
        model.add_interval(IntervalVar::new("b", 0, 100, 30, 200));
        model.add_cumulative(vec!["a".into(), "b".into()], vec![2, 3], 5);

        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_cumulative_mismatch() {
        let mut model = CpModel::new("test", 1000);
        model.add_interval(IntervalVar::new("a", 0, 100, 50, 200));
        model.add_cumulative(vec!["a".into()], vec![2, 3], 5);

        assert!(model.validate().is_err());
    }

    #[test]
    fn test_alternative() {
        let mut model = CpModel::new("test", 1000);
        model.add_interval(IntervalVar::new("main", 0, 100, 50, 200));
        model.add_interval(IntervalVar::new("alt1", 0, 100, 50, 200).as_optional("alt1_p"));
        model.add_interval(IntervalVar::new("alt2", 0, 100, 50, 200).as_optional("alt2_p"));
        model.add_constraint(Constraint::Alternative {
            main: "main".into(),
            alternatives: vec!["alt1".into(), "alt2".into()],
        });

        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_undefined_interval() {
        let mut model = CpModel::new("test", 1000);
        model.add_no_overlap(vec!["nonexistent".into()]);

        assert!(model.validate().is_err());
    }

    #[test]
    fn test_same_start_end() {
        let mut model = CpModel::new("test", 1000);
        model.add_interval(IntervalVar::new("a", 0, 100, 50, 200));
        model.add_interval(IntervalVar::new("b", 0, 100, 30, 200));
        model.add_constraint(Constraint::SameStart {
            interval1: "a".into(),
            interval2: "b".into(),
        });
        model.add_constraint(Constraint::SameEnd {
            interval1: "a".into(),
            interval2: "b".into(),
        });

        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_minimize_objective() {
        let mut model = CpModel::new("test", 1000);
        model.add_int_var(IntVar::new("cost", 0, 1000));
        model.set_objective(Objective::Minimize {
            terms: vec![("cost".into(), 1.0)],
        });

        assert!(model.objective.is_some());
    }
}
