//! CP solver interface and basic implementation.

use super::model::{CpModel, Objective};
use std::collections::HashMap;

/// Status of the solver after execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Number of search nodes explored.
    pub num_nodes: u64,
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
            num_nodes: 0,
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
/// An interval with alternatives is placed as the alternative that would end
/// first; that one is present, with the interval's start and end, and the
/// others absent.
///
/// The placement is then [checked](CpModel::check) against the whole model:
/// it is reported `Feasible` only when it breaks nothing, and `Unknown`
/// otherwise -- a start or end outside its domain, past the horizon, or a
/// constraint the greedy pass does not place by (cumulative, same start/end)
/// that it happens to break. The intervals are returned either way, so a
/// caller can see what was tried.
///
/// # Limitations
///
/// - Places by NoOverlap, Precedence and Alternative only; other constraints
///   are checked, not satisfied
/// - Does not optimize: one greedy pass
/// - Reads no [`SolverConfig`] setting: a single pass has no time to limit,
///   no workers to spread and no second solution to stop before
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

        let start_time = web_time::Instant::now();

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

        // The alternatives of each interval: placed with it, not on their own.
        let mut alternatives: HashMap<&str, &[String]> = HashMap::new();
        for c in &model.constraints {
            if let super::model::Constraint::Alternative {
                main,
                alternatives: alts,
            } = c
            {
                alternatives.insert(main.as_str(), alts.as_slice());
            }
        }
        let is_alternative: std::collections::HashSet<&str> = alternatives
            .values()
            .flat_map(|alts| alts.iter().map(String::as_str))
            .collect();
        // When `name` may start on its resources, at `earliest` or after.
        let free = |name: &str, earliest: i64, placed: &HashMap<&str, i64>| {
            overlap_groups
                .iter()
                .filter(|group| group.contains(&name))
                .flat_map(|group| group.iter().filter_map(|m| placed.get(m)))
                .fold(earliest, |at, &end| at.max(end))
        };

        // Greedy placement: process in name order, respecting precedence
        let mut assignments: HashMap<String, IntervalSolution> = HashMap::new();

        // Simple topological-ish ordering: repeat until all placed
        let mut placed: HashMap<&str, i64> = HashMap::new(); // name -> end_time
        let mut remaining: Vec<&String> = names
            .iter()
            .copied()
            .filter(|n| !is_alternative.contains(n.as_str()))
            .collect();

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

                if let Some(alts) = alternatives.get(name.as_str()) {
                    // The alternative that ends first, ties to the first listed.
                    let chosen = alts
                        .iter()
                        .map(|alt| {
                            let v = &model.intervals[alt];
                            let start = free(alt, earliest_start.max(v.start.min), &placed);
                            (start + v.duration.min, start, alt)
                        })
                        .min_by_key(|&(end, _, _)| end);
                    if let Some((end, start, chosen)) = chosen {
                        for alt in alts.iter() {
                            assignments.insert(
                                alt.clone(),
                                IntervalSolution {
                                    start,
                                    end,
                                    duration: end - start,
                                    is_present: alt == chosen,
                                },
                            );
                        }
                        placed.insert(chosen.as_str(), end);
                        assignments.insert(
                            name.to_string(),
                            IntervalSolution {
                                start,
                                end,
                                duration: end - start,
                                is_present: true,
                            },
                        );
                        placed.insert(name.as_str(), end);
                        continue;
                    }
                }

                // Check no-overlap: find latest end in same group
                let earliest_start = free(name, earliest_start, &placed);

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
            num_nodes: 0,
        };

        // Compute objective
        if let Some(Objective::MinimizeMaxEnd) = &model.objective {
            solution.objective_value = Some(solution.max_end() as f64);
        }

        // The pass places by precedence and no-overlap only, from each start's
        // lower bound; whatever else the model asks is checked, not assumed.
        if solution.status == SolverStatus::Feasible && !model.check(&solution).is_empty() {
            solution.status = SolverStatus::Unknown;
        }

        solution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cp::{IntervalVar, Objective};

    #[test]
    fn an_interval_with_alternatives_takes_the_one_that_ends_first() {
        // "a" holds M1 until 5; "b" ends sooner on M2 (at 4) than on M1 (5 + 2).
        let mut model = CpModel::new("alternatives", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 5, 100));
        model.add_interval(IntervalVar::new("b", 0, 100, 2, 100).with_variable_duration(2, 4));
        model.add_interval(IntervalVar::new("b@M1", 0, 100, 2, 100).as_optional("b@M1?"));
        model.add_interval(IntervalVar::new("b@M2", 0, 100, 4, 100).as_optional("b@M2?"));
        model.add_constraint(crate::cp::Constraint::Alternative {
            main: "b".into(),
            alternatives: vec!["b@M1".into(), "b@M2".into()],
        });
        model.add_no_overlap(vec!["a".into(), "b@M1".into()]);
        model.add_no_overlap(vec!["b@M2".into()]);
        model.add_precedence("a".into(), "c".into(), 0);
        model.add_interval(IntervalVar::new("c", 0, 100, 1, 100));

        let solution = SimpleCpSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Feasible);
        assert!(solution.intervals["b@M2"].is_present);
        assert!(!solution.intervals["b@M1"].is_present);
        assert_eq!(
            (solution.intervals["b"].start, solution.intervals["b"].end),
            (0, 4)
        );
        assert!(model.check(&solution).is_empty());
    }

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
            IntervalSolution {
                start: 0,
                end: 50,
                duration: 50,
                is_present: true,
            },
        );
        solution.intervals.insert(
            "b".into(),
            IntervalSolution {
                start: 10,
                end: 80,
                duration: 70,
                is_present: true,
            },
        );
        solution.intervals.insert(
            "c".into(),
            IntervalSolution {
                start: 0,
                end: 100,
                duration: 100,
                is_present: false,
            },
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
