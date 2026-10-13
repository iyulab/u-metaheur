//! Checking a solution against its model.
//!
//! [`CpModel::check`] lists every way a [`CpSolution`] breaks the model it
//! answers -- variable domains, the horizon, and each constraint -- so a
//! solver's status can be verified rather than taken on trust.

use super::model::{Constraint, CpModel};
use super::solver::{CpSolution, IntervalSolution};

/// One way a solution breaks the model it answers. [`CpModel::check`] lists them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpViolation {
    /// A mandatory interval has no assignment, or is marked absent.
    Missing {
        /// The interval.
        interval: String,
    },
    /// A start, end or duration outside the variable's domain.
    OutOfDomain {
        /// The interval.
        interval: String,
        /// `"start"`, `"end"` or `"duration"`.
        variable: &'static str,
        /// The assigned value.
        value: i64,
        /// The domain's lower bound.
        min: i64,
        /// The domain's upper bound.
        max: i64,
    },
    /// `end` is not `start + duration`.
    Inconsistent {
        /// The interval.
        interval: String,
    },
    /// An interval ends after the model's horizon.
    BeyondHorizon {
        /// The interval.
        interval: String,
        /// Its end.
        end: i64,
        /// The horizon.
        horizon: i64,
    },
    /// Two intervals of a no-overlap set overlap.
    Overlap {
        /// One interval.
        first: String,
        /// The other.
        second: String,
    },
    /// `after` starts sooner than `min_delay` after `before` ends.
    Precedence {
        /// The interval that must come first.
        before: String,
        /// The interval that must come after.
        after: String,
        /// `start(after) - end(before)`.
        gap: i64,
        /// The required minimum.
        min_delay: i64,
    },
    /// The demand of the intervals running at `at` exceeds the capacity.
    OverCapacity {
        /// A time at which the load is exceeded (a start time).
        at: i64,
        /// The load there.
        load: i64,
        /// The capacity.
        capacity: i64,
    },
    /// Two intervals required to start (or end) together do not.
    NotSimultaneous {
        /// One interval.
        first: String,
        /// The other.
        second: String,
        /// `"start"` or `"end"`.
        at: &'static str,
    },
    /// A performed `main` interval without exactly one present alternative, or an
    /// absent one with any.
    Alternative {
        /// The main interval.
        main: String,
        /// How many alternatives are present.
        present: usize,
    },
}

impl CpModel {
    /// Every way `solution` breaks this model: interval domains and the horizon,
    /// `end = start + duration`, and each constraint. Empty means the solution
    /// satisfies the model. Interval findings come first (by name), then each
    /// constraint's, in the model's order. An absent optional interval is exempt from the
    /// constraints it appears in, as in constraint programming generally.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_metaheur::cp::{CpModel, CpSolver, IntervalVar, SimpleCpSolver, SolverConfig};
    ///
    /// let mut model = CpModel::new("two jobs", 100);
    /// model.add_interval(IntervalVar::new("a", 0, 100, 60, 100));
    /// model.add_interval(IntervalVar::new("b", 0, 100, 60, 100));
    /// model.add_no_overlap(vec!["a".into(), "b".into()]);
    ///
    /// // 120 units of work in a horizon of 100: the greedy pass cannot vouch for it.
    /// let solution = SimpleCpSolver::new().solve(&model, &SolverConfig::default());
    /// assert!(!model.check(&solution).is_empty());
    /// ```
    pub fn check(&self, solution: &CpSolution) -> Vec<CpViolation> {
        let mut found = Vec::new();
        let mut names: Vec<&String> = self.intervals.keys().collect();
        names.sort();
        let present = |name: &str| solution.intervals.get(name).filter(|s| s.is_present);
        for name in names {
            let var = &self.intervals[name];
            let Some(s) = present(name) else {
                if !var.is_optional {
                    found.push(CpViolation::Missing {
                        interval: name.clone(),
                    });
                }
                continue;
            };
            for (variable, value, min, max) in [
                ("start", s.start, var.start.min, var.start.max),
                ("end", s.end, var.end.min, var.end.max),
                ("duration", s.duration, var.duration.min, var.duration.max),
            ] {
                if value < min || value > max {
                    found.push(CpViolation::OutOfDomain {
                        interval: name.clone(),
                        variable,
                        value,
                        min,
                        max,
                    });
                }
            }
            if s.end != s.start + s.duration {
                found.push(CpViolation::Inconsistent {
                    interval: name.clone(),
                });
            }
            if s.end > self.horizon {
                found.push(CpViolation::BeyondHorizon {
                    interval: name.clone(),
                    end: s.end,
                    horizon: self.horizon,
                });
            }
        }
        for constraint in &self.constraints {
            match constraint {
                Constraint::NoOverlap { intervals } => {
                    for (i, a) in intervals.iter().enumerate() {
                        for b in &intervals[i + 1..] {
                            if let (Some(x), Some(y)) = (present(a), present(b)) {
                                if x.start < y.end && y.start < x.end {
                                    found.push(CpViolation::Overlap {
                                        first: a.clone(),
                                        second: b.clone(),
                                    });
                                }
                            }
                        }
                    }
                }
                Constraint::Cumulative {
                    intervals,
                    demands,
                    capacity,
                } => {
                    let running: Vec<(&IntervalSolution, i64)> = intervals
                        .iter()
                        .zip(demands)
                        .filter_map(|(n, d)| present(n).map(|s| (s, *d)))
                        .collect();
                    // The load can only rise at a start, so the starts are where to look.
                    let mut starts: Vec<i64> = running.iter().map(|(s, _)| s.start).collect();
                    starts.sort_unstable();
                    starts.dedup();
                    if let Some((at, load)) = starts
                        .into_iter()
                        .map(|t| {
                            let load = running
                                .iter()
                                .filter(|(s, _)| s.start <= t && t < s.end)
                                .map(|(_, d)| d)
                                .sum::<i64>();
                            (t, load)
                        })
                        .find(|&(_, load)| load > *capacity)
                    {
                        found.push(CpViolation::OverCapacity {
                            at,
                            load,
                            capacity: *capacity,
                        });
                    }
                }
                Constraint::Precedence {
                    before,
                    after,
                    min_delay,
                } => {
                    if let (Some(x), Some(y)) = (present(before), present(after)) {
                        let gap = y.start - x.end;
                        if gap < *min_delay {
                            found.push(CpViolation::Precedence {
                                before: before.clone(),
                                after: after.clone(),
                                gap,
                                min_delay: *min_delay,
                            });
                        }
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
                    let start = matches!(constraint, Constraint::SameStart { .. });
                    if let (Some(x), Some(y)) = (present(interval1), present(interval2)) {
                        let (a, b) = if start {
                            (x.start, y.start)
                        } else {
                            (x.end, y.end)
                        };
                        if a != b {
                            found.push(CpViolation::NotSimultaneous {
                                first: interval1.clone(),
                                second: interval2.clone(),
                                at: if start { "start" } else { "end" },
                            });
                        }
                    }
                }
                Constraint::Alternative { main, alternatives } => {
                    let count = alternatives.iter().filter(|a| present(a).is_some()).count();
                    let expected = usize::from(present(main).is_some());
                    if count != expected {
                        found.push(CpViolation::Alternative {
                            main: main.clone(),
                            present: count,
                        });
                    }
                }
            }
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cp::{CpSolver, IntervalVar, SimpleCpSolver, SolverConfig, SolverStatus};
    use std::collections::HashMap;

    fn solve(model: &CpModel) -> CpSolution {
        SimpleCpSolver::new().solve(model, &SolverConfig::default())
    }

    fn placed(entries: &[(&str, i64, i64)]) -> CpSolution {
        let mut solution = CpSolution::empty(SolverStatus::Feasible);
        solution.intervals = entries
            .iter()
            .map(|&(name, start, end)| {
                let at = IntervalSolution {
                    start,
                    end,
                    duration: end - start,
                    is_present: true,
                };
                (name.to_string(), at)
            })
            .collect::<HashMap<_, _>>();
        solution
    }

    #[test]
    fn a_solution_that_keeps_every_constraint_has_no_violation() {
        let mut model = CpModel::new("ok", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 10, 100));
        model.add_interval(IntervalVar::new("b", 0, 100, 10, 100));
        model.add_no_overlap(vec!["a".into(), "b".into()]);
        model.add_precedence("a".into(), "b".into(), 5);
        let solution = solve(&model);
        assert_eq!(solution.status, SolverStatus::Feasible);
        assert_eq!(model.check(&solution), vec![]);
    }

    #[test]
    fn past_the_horizon_is_not_feasible() {
        let mut model = CpModel::new("tight", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 60, 1000));
        model.add_interval(IntervalVar::new("b", 0, 100, 60, 1000));
        model.add_no_overlap(vec!["a".into(), "b".into()]);
        let solution = solve(&model);
        assert_eq!(solution.status, SolverStatus::Unknown);
        assert!(model
            .check(&solution)
            .contains(&CpViolation::BeyondHorizon {
                interval: "b".into(),
                end: 120,
                horizon: 100
            }));
    }

    #[test]
    fn a_start_pushed_past_its_domain_is_reported() {
        let mut model = CpModel::new("late", 1000);
        model.add_interval(IntervalVar::new("a", 0, 100, 50, 1000));
        model.add_interval(IntervalVar::new("b", 0, 10, 5, 1000));
        model.add_precedence("a".into(), "b".into(), 0);
        let solution = solve(&model);
        assert_eq!(solution.status, SolverStatus::Unknown);
        let found = model.check(&solution);
        assert!(
            found.contains(&CpViolation::OutOfDomain {
                interval: "b".into(),
                variable: "start",
                value: 50,
                min: 0,
                max: 10
            }),
            "{found:?}"
        );
    }

    #[test]
    fn a_cumulative_constraint_the_pass_does_not_place_by_is_still_checked() {
        let mut model = CpModel::new("capacity", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 10, 100));
        model.add_interval(IntervalVar::new("b", 0, 100, 10, 100));
        model.add_cumulative(vec!["a".into(), "b".into()], vec![1, 1], 1);
        let solution = solve(&model);
        assert_eq!(solution.status, SolverStatus::Unknown);
        assert_eq!(
            model.check(&solution),
            vec![CpViolation::OverCapacity {
                at: 0,
                load: 2,
                capacity: 1
            }]
        );
    }

    #[test]
    fn precedence_overlap_and_simultaneity_are_measured() {
        let mut model = CpModel::new("hand", 100);
        for name in ["a", "b", "c"] {
            model.add_interval(IntervalVar::new(name, 0, 100, 10, 100));
        }
        model.add_precedence("a".into(), "b".into(), 3);
        model.add_no_overlap(vec!["a".into(), "c".into()]);
        model.add_constraint(Constraint::SameStart {
            interval1: "b".into(),
            interval2: "c".into(),
        });
        let solution = placed(&[("a", 0, 10), ("b", 11, 21), ("c", 5, 15)]);
        assert_eq!(
            model.check(&solution),
            vec![
                CpViolation::Precedence {
                    before: "a".into(),
                    after: "b".into(),
                    gap: 1,
                    min_delay: 3
                },
                CpViolation::Overlap {
                    first: "a".into(),
                    second: "c".into()
                },
                CpViolation::NotSimultaneous {
                    first: "b".into(),
                    second: "c".into(),
                    at: "start"
                },
            ]
        );
    }

    #[test]
    fn a_missing_mandatory_interval_counts_and_an_absent_optional_one_does_not() {
        let mut model = CpModel::new("presence", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 10, 100));
        model.add_interval(IntervalVar::new("o", 0, 100, 10, 100).as_optional("o_on"));
        model.add_no_overlap(vec!["a".into(), "o".into()]);
        let solution = placed(&[]);
        assert_eq!(
            model.check(&solution),
            vec![CpViolation::Missing {
                interval: "a".into()
            }]
        );
    }

    #[test]
    fn an_alternative_needs_exactly_one_present_alternative() {
        let mut model = CpModel::new("alt", 100);
        for name in ["m", "x", "y"] {
            model.add_interval(
                IntervalVar::new(name, 0, 100, 10, 100).as_optional(format!("{name}_on")),
            );
        }
        model.add_constraint(Constraint::Alternative {
            main: "m".into(),
            alternatives: vec!["x".into(), "y".into()],
        });
        let both = placed(&[("m", 0, 10), ("x", 0, 10), ("y", 0, 10)]);
        assert_eq!(
            model.check(&both),
            vec![CpViolation::Alternative {
                main: "m".into(),
                present: 2
            }]
        );
        assert_eq!(model.check(&placed(&[("m", 0, 10), ("y", 0, 10)])), vec![]);
    }
}
