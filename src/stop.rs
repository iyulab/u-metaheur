//! Why a run stopped.
//!
//! A best-so-far solution cut short by the time limit and one the search
//! finished look alike; only the run knows which it is. Every runner in this
//! crate reports one [`StopReason`], set where it stops.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Why a run stopped: the run's own account of the solution it returns.
///
/// The names follow the termination statuses of mathematical-programming
/// solvers (Gurobi `OPTIMAL` · `ITERATION_LIMIT` · `TIME_LIMIT` ·
/// `USER_OBJ_LIMIT` · `INTERRUPTED`) and MATLAB `ga`'s stall generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum StopReason {
    /// The method ran to its own end: a schedule that cooled to its final
    /// temperature, a construction that tried every element.
    Completed,
    /// An exact method proved the solution optimal.
    Optimal,
    /// The solution met the target the caller set, so the search stopped
    /// there.
    TargetReached,
    /// The search spent its generation or iteration budget.
    IterationLimit,
    /// The search went its stall budget without improving its best
    /// solution, and stopped there.
    NoImprovement,
    /// The time limit ran out; the solution is the best found by then.
    TimeLimit,
    /// The caller cancelled the run; the solution is the best found by then.
    Cancelled,
}

impl StopReason {
    /// Every reason, in declaration order.
    pub const ALL: [Self; 7] = [
        Self::Completed,
        Self::Optimal,
        Self::TargetReached,
        Self::IterationLimit,
        Self::NoImprovement,
        Self::TimeLimit,
        Self::Cancelled,
    ];

    /// The wire name (`"time_limit"`, `"no_improvement"`, ...).
    pub fn name(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Optimal => "optimal",
            Self::TargetReached => "target_reached",
            Self::IterationLimit => "iteration_limit",
            Self::NoImprovement => "no_improvement",
            Self::TimeLimit => "time_limit",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether the run ended on something other than its own course: the
    /// caller cancelled it or its time limit ran out.
    pub fn cut_short(self) -> bool {
        matches!(self, Self::TimeLimit | Self::Cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_distinct_snake_case() {
        let names: std::collections::HashSet<&str> =
            StopReason::ALL.iter().map(|r| r.name()).collect();
        assert_eq!(names.len(), StopReason::ALL.len());
        for name in names {
            assert!(name.chars().all(|c| c.is_ascii_lowercase() || c == '_'));
        }
    }
}
