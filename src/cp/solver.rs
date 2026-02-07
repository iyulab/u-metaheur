//! CP solver interface and basic implementation.

use super::model::{CpModel, Objective};
use std::collections::HashMap;

/// Status of the solver after execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStatus {
    /// Proven optimal solution found.
    Optimal,
    /// Feasible (but not necessarily optimal) solution found.
    Feasible,
    /// No feasible solution exists.
    Infeasible,
    /// Model is invalid or malformed.
    ModelInvalid,
    /// Solver exceeded time limit.
    Timeout,
    /// No solution found for unknown reasons.
    Unknown,
}

/// Solution for an interval variable.
#[derive(Debug, Clone)]
pub struct IntervalSolution {
    /// Assigned start time.
    pub start: i64,
    /// Assigned end time.
    pub end: i64,
    /// Assigned duration.
    pub duration: i64,
    /// Whether this interval is present (for optional intervals).
    pub is_present: bool,
}

/// Solution from a CP solver.
#[derive(Debug, Clone)]
pub struct CpSolution {
    /// Solver status.
    pub status: SolverStatus,
    /// Objective function value (if any).
    pub objective_value: Option<f64>,
    /// Interval variable assignments.
    pub intervals: HashMap<String, IntervalSolution>,
    /// Integer variable assignments.
    pub int_vars: HashMap<String, i64>,
    /// Boolean variable assignments.
    pub bool_vars: HashMap<String, bool>,
    /// Solve time in milliseconds.
    pub solve_time_ms: i64,
}

impl CpSolution {
    /// Creates an empty solution with the given status.
    pub fn empty(status: SolverStatus) -> Self {
        Self {
            status,
            objective_value: None,
            intervals: HashMap::new(),
            int_vars: HashMap::new(),
            bool_vars: HashMap::new(),
            solve_time_ms: 0,
        }
    }

    /// Whether a feasible solution was found.
    pub fn is_solution_found(&self) -> bool {
        matches!(self.status, SolverStatus::Optimal | SolverStatus::Feasible)
    }

    /// Returns the maximum end time across all present intervals.
    pub fn max_end(&self) -> i64 {
        self.intervals
            .values()
            .filter(|s| s.is_present)
            .map(|s| s.end)
            .max()
            .unwrap_or(0)
    }
}

/// Solver configuration.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum solve time in milliseconds.
    pub time_limit_ms: i64,
    /// Number of parallel workers.
    pub num_workers: usize,
    /// Stop after finding the first feasible solution.
    pub stop_after_first: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            time_limit_ms: 60_000,
            num_workers: 1,
            stop_after_first: false,
        }
    }
}

/// Trait for CP solver implementations.
///
/// Implementors provide the actual constraint solving logic.
/// This can wrap external solvers (e.g., OR-Tools CP-SAT) or
/// provide custom heuristics.
pub trait CpSolver {
    /// Solves the model and returns a solution.
    fn solve(&self, model: &CpModel, config: &SolverConfig) -> CpSolution;
}

/// A simple greedy CP solver for testing.
///
/// Places intervals sequentially respecting no-overlap and precedence
/// constraints. This is a trivial heuristic, not a real CP solver.
///
/// # Limitations
///
/// - Only handles NoOverlap and Precedence constraints
/// - Does not optimize: just finds a feasible solution
/// - Does not support cumulative or alternative constraints
pub struct SimpleCpSolver;

impl SimpleCpSolver {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimpleCpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl CpSolver for SimpleCpSolver {
    fn solve(&self, model: &CpModel, _config: &SolverConfig) -> CpSolution {
        if model.validate().is_err() {
            return CpSolution::empty(SolverStatus::ModelInvalid);
        }

        let start_time = std::time::Instant::now();

        // Collect all interval names, sorted for determinism
        let mut names: Vec<&String> = model.intervals.keys().collect();
        names.sort();

        // Build precedence graph: after -> set of (before, min_delay)
        let mut prec: HashMap<&str, Vec<(&str, i64)>> = HashMap::new();
        for c in &model.constraints {
            if let super::model::Constraint::Precedence {
                before,
                after,
                min_delay,
            } = c
            {
                prec.entry(after.as_str())
                    .or_default()
                    .push((before.as_str(), *min_delay));
            }
        }

        // Build no-overlap groups
        let mut overlap_groups: Vec<Vec<&str>> = Vec::new();
        for c in &model.constraints {
            if let super::model::Constraint::NoOverlap { intervals } = c {
                overlap_groups.push(intervals.iter().map(|s| s.as_str()).collect());
            }
        }

        // Greedy placement: process in name order, respecting precedence
        let mut assignments: HashMap<String, IntervalSolution> = HashMap::new();

        // Simple topological-ish ordering: repeat until all placed
        let mut placed: HashMap<&str, i64> = HashMap::new(); // name -> end_time
        let mut remaining: Vec<&String> = names.clone();

        for _ in 0..model.intervals.len() + 1 {
            if remaining.is_empty() {
                break;
            }

            let mut next_remaining = Vec::new();

            for name in &remaining {
                let interval = &model.intervals[name.as_str()];

                // Check precedence constraints
                let mut earliest_start = interval.start.min;
                let mut blocked = false;

                if let Some(preds) = prec.get(name.as_str()) {
                    for &(before, delay) in preds {
                        if let Some(&end) = placed.get(before) {
                            earliest_start = earliest_start.max(end + delay);
                        } else {
                            blocked = true;
                            break;
                        }
                    }
                }

                if blocked {
                    next_remaining.push(*name);
                    continue;
                }

                // Check no-overlap: find latest end in same group
                for group in &overlap_groups {
                    if group.contains(&name.as_str()) {
                        for &member in group {
                            if let Some(&end) = placed.get(member) {
                                earliest_start = earliest_start.max(end);
                            }
                        }
                    }
                }

                let duration = interval.duration.fixed.unwrap_or(interval.duration.min);
                let start = earliest_start;
                let end = start + duration;

                assignments.insert(
                    name.to_string(),
                    IntervalSolution {
                        start,
                        end,
                        duration,
                        is_present: true,
                    },
                );

                placed.insert(name.as_str(), end);
            }

            remaining = next_remaining;
        }

        let mut solution = CpSolution {
            status: if assignments.len() == model.intervals.len() {
                SolverStatus::Feasible
            } else {
                SolverStatus::Unknown
            },
            objective_value: None,
            intervals: assignments,
            int_vars: HashMap::new(),
            bool_vars: HashMap::new(),
            solve_time_ms: start_time.elapsed().as_millis() as i64,
        };

        // Compute objective
        if let Some(Objective::MinimizeMaxEnd) = &model.objective {
            solution.objective_value = Some(solution.max_end() as f64);
        }

        solution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cp::{IntervalVar, Objective};

    #[test]
    fn test_simple_solver_basic() {
        let mut model = CpModel::new("test", 1_000_000);
        model.add_interval(IntervalVar::new("op1", 0, 100_000, 50_000, 200_000));
        model.add_interval(IntervalVar::new("op2", 0, 100_000, 30_000, 200_000));
        model.set_objective(Objective::MinimizeMaxEnd);

        let solver = SimpleCpSolver::new();
        let solution = solver.solve(&model, &SolverConfig::default());

        assert!(solution.is_solution_found());
        assert_eq!(solution.intervals.len(), 2);
        assert!(solution.objective_value.is_some());
    }

    #[test]
    fn test_no_overlap() {
        let mut model = CpModel::new("test", 1000);
        model.add_interval(IntervalVar::new("a", 0, 100, 50, 200));
        model.add_interval(IntervalVar::new("b", 0, 100, 30, 200));
        model.add_no_overlap(vec!["a".into(), "b".into()]);
        model.set_objective(Objective::MinimizeMaxEnd);

        let solver = SimpleCpSolver::new();
        let solution = solver.solve(&model, &SolverConfig::default());

        assert!(solution.is_solution_found());

        // Intervals should not overlap
        let a = &solution.intervals["a"];
        let b = &solution.intervals["b"];
        assert!(a.end <= b.start || b.end <= a.start);
    }

    #[test]
    fn test_precedence() {
        let mut model = CpModel::new("test", 1000);
        model.add_interval(IntervalVar::new("first", 0, 100, 50, 200));
        model.add_interval(IntervalVar::new("second", 0, 200, 30, 300));
        model.add_precedence("first".into(), "second".into(), 10);

        let solver = SimpleCpSolver::new();
        let solution = solver.solve(&model, &SolverConfig::default());

        assert!(solution.is_solution_found());

        let first = &solution.intervals["first"];
        let second = &solution.intervals["second"];
        assert!(
            first.end + 10 <= second.start,
            "precedence violated: first.end={} + 10 > second.start={}",
            first.end,
            second.start
        );
    }

    #[test]
    fn test_invalid_model() {
        let mut model = CpModel::new("test", 1000);
        model.add_no_overlap(vec!["nonexistent".into()]);

        let solver = SimpleCpSolver::new();
        let solution = solver.solve(&model, &SolverConfig::default());

        assert_eq!(solution.status, SolverStatus::ModelInvalid);
    }

    #[test]
    fn test_max_end() {
        let mut solution = CpSolution::empty(SolverStatus::Feasible);
        solution.intervals.insert(
            "a".into(),
            IntervalSolution { start: 0, end: 50, duration: 50, is_present: true },
        );
        solution.intervals.insert(
            "b".into(),
            IntervalSolution { start: 10, end: 80, duration: 70, is_present: true },
        );
        solution.intervals.insert(
            "c".into(),
            IntervalSolution { start: 0, end: 100, duration: 100, is_present: false },
        );

        assert_eq!(solution.max_end(), 80); // c is not present
    }

    #[test]
    fn test_chain_precedence() {
        let mut model = CpModel::new("test", 1000);
        model.add_interval(IntervalVar::new("a", 0, 100, 10, 200));
        model.add_interval(IntervalVar::new("b", 0, 200, 20, 300));
        model.add_interval(IntervalVar::new("c", 0, 300, 30, 400));
        model.add_precedence("a".into(), "b".into(), 0);
        model.add_precedence("b".into(), "c".into(), 0);

        let solver = SimpleCpSolver::new();
        let solution = solver.solve(&model, &SolverConfig::default());

        assert!(solution.is_solution_found());

        let a = &solution.intervals["a"];
        let b = &solution.intervals["b"];
        let c = &solution.intervals["c"];

        assert!(a.end <= b.start);
        assert!(b.end <= c.start);
    }

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.time_limit_ms, 60_000);
        assert_eq!(config.num_workers, 1);
        assert!(!config.stop_after_first);
    }
}
