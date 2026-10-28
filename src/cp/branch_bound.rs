//! An exact solver for disjunctive scheduling models.
//!
//! [`BranchAndBoundSolver`] enumerates *active* schedules by the branching of
//! Giffler & Thompson (1960) and bounds each node from below, so a search that
//! runs to the end proves its best schedule optimal (or the model infeasible).

use super::model::{Constraint, CpModel, Objective};
use super::solver::{CpSolution, CpSolver, IntervalSolution, SolverConfig, SolverStatus};
use std::collections::HashMap;

/// Exact makespan minimization over a disjunctive scheduling model, by
/// depth-first branch and bound over active schedules.
///
/// # Models it solves
///
/// Every interval mandatory with a fixed duration; constraints
/// [`Constraint::NoOverlap`] (each interval in at most one group -- a
/// unary resource) and [`Constraint::Precedence`] (with a `min_delay` of at
/// least 0);
/// the bounds of each interval's start and end, and the horizon, as
/// deadlines; objective [`Objective::MinimizeMaxEnd`], or none (any feasible
/// schedule). This is the job shop with release dates, deadlines and time
/// lags, and any precedence graph rather than chains.
///
/// A model outside that class -- an optional or variable-length interval, an
/// interval in two no-overlap groups, a negative delay, any other constraint
/// or objective --
/// is not searched: the solution is empty with [`SolverStatus::Unknown`].
/// An invalid model is [`SolverStatus::ModelInvalid`].
///
/// # Algorithm
///
/// A node fixes the start of some intervals. An interval with no unary
/// resource is started as soon as its predecessors allow -- that is never
/// worse. Otherwise, of the intervals whose predecessors are all fixed, the
/// one that can finish first, at `c`, names a resource; the branches start,
/// on that resource, each such interval that could start before `c`
/// (Giffler & Thompson's conflict set). Every active schedule is reached,
/// and some active schedule is optimal: shifting intervals earlier never
/// breaks a deadline. An interval whose predecessors are not all fixed
/// cannot start before that finish -- its unfixed predecessors end no
/// earlier, and delays are not negative -- so the conflict set misses none. Branches are taken earliest start first, so the first
/// leaf is the Giffler-Thompson schedule of the earliest-start rule.
///
/// A node is pruned when an interval can no longer meet its deadline, or
/// when its lower bound reaches the best makespan found: the larger of each
/// unfixed interval's earliest start + duration + the longest chain of
/// successors after it, and, on each resource, the bound of Jackson's
/// preemptive schedule of its unfixed intervals (Carlier 1982) -- their
/// earliest starts as release dates, those chains as tails.
///
/// # Status
///
/// From what the search proved: [`SolverStatus::Optimal`] (or, with no
/// objective, [`SolverStatus::Feasible`]) and [`SolverStatus::Infeasible`]
/// only when it ran to the end; cut off by
/// [`SolverConfig::time_limit_ms`] it reports the best schedule found as
/// [`SolverStatus::Feasible`], or [`SolverStatus::Timeout`] when none was.
/// [`SolverConfig::stop_after_first`] stops at the first schedule
/// ([`SolverStatus::Feasible`]). [`SolverConfig::num_workers`] is not read:
/// the search is sequential.
///
/// # Starting from a known schedule
///
/// [`with_incumbent`](Self::with_incumbent) gives the search a schedule to
/// beat -- from a heuristic, or an earlier run. Its makespan bounds the
/// search from the first node, which prunes far more than the search's own
/// first schedule does, and it is the answer unless a shorter one is found:
/// a search that runs to the end proves it [`SolverStatus::Optimal`], one cut
/// off returns it as [`SolverStatus::Feasible`]. An incumbent the model's
/// [check](CpModel::check) finds fault with is not used.
///
/// # Complexity
///
/// Exponential in the worst case (the job shop is strongly NP-hard); each
/// node costs O(n + m) for n intervals and m precedence arcs.
///
/// # References
///
/// - Giffler, B. & Thompson, G. L. (1960). "Algorithms for solving
///   production-scheduling problems". *Operations Research*, 8(4), 487–503.
/// - Brucker, P., Jurisch, B. & Sievers, B. (1994). "A branch and bound
///   algorithm for the job-shop scheduling problem". *Discrete Applied
///   Mathematics*, 49, 107–127.
/// - Carlier, J. (1982). "The one-machine sequencing problem". *European
///   Journal of Operational Research*, 11(1), 42–47.
/// - Carlier, J. & Pinson, E. (1989). "An algorithm for solving the job-shop
///   problem". *Management Science*, 35(2), 164–176.
///
/// # Examples
///
/// ```
/// use u_metaheur::cp::{BranchAndBoundSolver, CpModel, CpSolver, IntervalVar, Objective,
///     SolverConfig, SolverStatus};
///
/// // Two jobs of two operations on two machines.
/// let mut model = CpModel::new("2x2", 100);
/// for (name, d) in [("a1", 3), ("a2", 2), ("b1", 2), ("b2", 4)] {
///     model.add_interval(IntervalVar::new(name, 0, 100, d, 100));
/// }
/// model.add_precedence("a1".into(), "a2".into(), 0);
/// model.add_precedence("b1".into(), "b2".into(), 0);
/// model.add_no_overlap(vec!["a1".into(), "b2".into()]); // machine 0
/// model.add_no_overlap(vec!["a2".into(), "b1".into()]); // machine 1
/// model.set_objective(Objective::MinimizeMaxEnd);
///
/// let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
/// assert_eq!(solution.status, SolverStatus::Optimal);
/// assert_eq!(solution.objective_value, Some(7.0));
/// assert!(model.check(&solution).is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct BranchAndBoundSolver {
    /// The schedule to beat, if one was given.
    incumbent: Option<CpSolution>,
}

impl BranchAndBoundSolver {
    /// The solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts the search from `incumbent`, a schedule of the model to be
    /// solved (see [Starting from a known schedule](Self#starting-from-a-known-schedule)).
    pub fn with_incumbent(mut self, incumbent: CpSolution) -> Self {
        self.incumbent = Some(incumbent);
        self
    }
}

impl CpSolver for BranchAndBoundSolver {
    fn solve(&self, model: &CpModel, config: &SolverConfig) -> CpSolution {
        let started = web_time::Instant::now();
        if model.validate().is_err() {
            return CpSolution::empty(SolverStatus::ModelInvalid);
        }
        let Some(problem) = Problem::from_model(model) else {
            return CpSolution::empty(SolverStatus::Unknown);
        };
        let Some(order) = problem.topological_order() else {
            // A cycle of precedences: no start satisfies it.
            let mut solution = CpSolution::empty(SolverStatus::Infeasible);
            solution.solve_time_ms = started.elapsed().as_millis() as i64;
            return solution;
        };

        let deadline =
            started + std::time::Duration::from_millis(config.time_limit_ms.max(0) as u64);
        // A given schedule that keeps the model, as starts by index.
        let incumbent = self
            .incumbent
            .as_ref()
            .filter(|s| model.check(s).is_empty())
            .and_then(|s| {
                problem
                    .names
                    .iter()
                    .map(|name| s.intervals.get(name).map(|i| i.start))
                    .collect::<Option<Vec<i64>>>()
            });
        let incumbent_makespan = incumbent.as_ref().map_or(i64::MAX, |starts| {
            (0..problem.len())
                .map(|i| starts[i] + problem.duration[i])
                .max()
                .unwrap_or(0)
        });
        let mut search = Search {
            problem: &problem,
            order,
            tails: problem.tails(),
            start: vec![None; problem.len()],
            free_at: vec![i64::MIN; problem.groups],
            best: incumbent,
            best_makespan: incumbent_makespan,
            nodes: 0,
            deadline,
            cut_off: false,
            stop_after_first: config.stop_after_first || !problem.minimize,
        };
        search.descend(0);

        let status = match (&search.best, search.cut_off) {
            (Some(_), false) if problem.minimize && !config.stop_after_first => {
                SolverStatus::Optimal
            }
            (Some(_), _) => SolverStatus::Feasible,
            (None, false) => SolverStatus::Infeasible,
            (None, true) => SolverStatus::Timeout,
        };
        let mut solution = CpSolution::empty(status);
        if let Some(starts) = &search.best {
            for (i, &s) in starts.iter().enumerate() {
                let d = problem.duration[i];
                solution.intervals.insert(
                    problem.names[i].clone(),
                    IntervalSolution {
                        start: s,
                        end: s + d,
                        duration: d,
                        is_present: true,
                    },
                );
            }
            if problem.minimize {
                solution.objective_value = Some(search.best_makespan as f64);
            }
        }
        solution.num_nodes = search.nodes;
        solution.solve_time_ms = started.elapsed().as_millis() as i64;
        solution
    }
}

/// A model in the solver's class, indexed: intervals by name order.
struct Problem {
    names: Vec<String>,
    duration: Vec<i64>,
    release: Vec<i64>,
    latest_start: Vec<i64>,
    latest_end: Vec<i64>,
    /// The unary resource of each interval.
    group: Vec<Option<usize>>,
    groups: usize,
    /// `(predecessor, min_delay)` of each interval.
    preds: Vec<Vec<(usize, i64)>>,
    /// `(successor, min_delay)` of each interval.
    succs: Vec<Vec<(usize, i64)>>,
    minimize: bool,
}

impl Problem {
    fn from_model(model: &CpModel) -> Option<Self> {
        let minimize = match &model.objective {
            None => false,
            Some(Objective::MinimizeMaxEnd) => true,
            Some(_) => return None,
        };
        let mut names: Vec<String> = model.intervals.keys().cloned().collect();
        names.sort();
        let index: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let n = names.len();
        let (mut duration, mut release, mut latest_start, mut latest_end) =
            (vec![0; n], vec![0; n], vec![0; n], vec![0; n]);
        for (i, name) in names.iter().enumerate() {
            let v = &model.intervals[name];
            if v.is_optional || v.duration.min != v.duration.max {
                return None;
            }
            duration[i] = v.duration.min;
            release[i] = v.start.min.max(v.end.min - v.duration.min);
            latest_start[i] = v.start.max.min(v.end.max - v.duration.min);
            latest_end[i] = v.end.max.min(model.horizon);
        }
        let mut group = vec![None; n];
        let mut groups = 0;
        let (mut preds, mut succs) = (vec![Vec::new(); n], vec![Vec::new(); n]);
        for constraint in &model.constraints {
            match constraint {
                Constraint::NoOverlap { intervals } => {
                    for name in intervals {
                        let i = index[name.as_str()];
                        if group[i].is_some() {
                            return None;
                        }
                        group[i] = Some(groups);
                    }
                    groups += 1;
                }
                Constraint::Precedence {
                    before,
                    after,
                    min_delay,
                } => {
                    // A negative delay lets the successor start before its
                    // predecessor ends, which the branching's completeness
                    // does not cover.
                    if *min_delay < 0 {
                        return None;
                    }
                    let (b, a) = (index[before.as_str()], index[after.as_str()]);
                    preds[a].push((b, *min_delay));
                    succs[b].push((a, *min_delay));
                }
                _ => return None,
            }
        }
        Some(Self {
            names,
            duration,
            release,
            latest_start,
            latest_end,
            group,
            groups,
            preds,
            succs,
            minimize,
        })
    }

    fn len(&self) -> usize {
        self.names.len()
    }

    /// The intervals in an order every precedence respects (Kahn), or `None`
    /// on a cycle.
    fn topological_order(&self) -> Option<Vec<usize>> {
        let mut indegree: Vec<usize> = self.preds.iter().map(Vec::len).collect();
        let mut ready: Vec<usize> = (0..self.len()).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(self.len());
        while let Some(i) = ready.pop() {
            order.push(i);
            for &(j, _) in &self.succs[i] {
                indegree[j] -= 1;
                if indegree[j] == 0 {
                    ready.push(j);
                }
            }
        }
        (order.len() == self.len()).then_some(order)
    }

    /// The longest chain of delays and durations after each interval ends.
    fn tails(&self) -> Vec<i64> {
        let order = self
            .topological_order()
            .expect("tails are computed for an acyclic model");
        let mut tail = vec![0; self.len()];
        for &i in order.iter().rev() {
            tail[i] = self.succs[i]
                .iter()
                .map(|&(j, delay)| delay + self.duration[j] + tail[j])
                .max()
                .unwrap_or(0);
        }
        tail
    }
}

/// The one-machine bound of tasks `(head, duration, tail)`: the largest
/// completion + tail of Jackson's preemptive schedule -- at every moment the
/// released task with the largest tail runs, a newly released one preempting
/// it. Optimal for the preemptive problem, so a lower bound for the
/// non-preemptive one, and never below `min head + total duration + min
/// tail`. O(k log k) for k tasks (Carlier 1982).
fn jackson_preemptive(mut tasks: Vec<(i64, i64, i64)>) -> i64 {
    use std::collections::BinaryHeap;
    tasks.sort_unstable_by_key(|&(head, _, _)| head);
    // Released tasks by tail: (tail, remaining duration).
    let mut released: BinaryHeap<(i64, i64)> = BinaryHeap::new();
    let (mut next, mut time, mut bound) = (0, i64::MIN, 0);
    while next < tasks.len() || !released.is_empty() {
        if released.is_empty() {
            time = time.max(tasks[next].0);
        }
        while next < tasks.len() && tasks[next].0 <= time {
            released.push((tasks[next].2, tasks[next].1));
            next += 1;
        }
        let (tail, remaining) = released.pop().expect("a task was released");
        let until = tasks.get(next).map_or(i64::MAX, |t| t.0);
        let run = remaining.min(until - time);
        time += run;
        if run == remaining {
            bound = bound.max(time + tail);
        } else {
            released.push((tail, remaining - run));
        }
    }
    bound
}

struct Search<'p> {
    problem: &'p Problem,
    order: Vec<usize>,
    tails: Vec<i64>,
    start: Vec<Option<i64>>,
    /// When each unary resource is next free.
    free_at: Vec<i64>,
    best: Option<Vec<i64>>,
    best_makespan: i64,
    nodes: u64,
    deadline: web_time::Instant,
    cut_off: bool,
    stop_after_first: bool,
}

impl Search<'_> {
    /// The earliest start of every unfixed interval from the fixed ones and
    /// the precedences (resources only through `free_at`), or `None` when an
    /// interval can no longer meet its latest start.
    fn heads(&self) -> Option<Vec<i64>> {
        let p = self.problem;
        let mut head = vec![0; p.len()];
        for &i in &self.order {
            head[i] = match self.start[i] {
                Some(s) => s,
                None => {
                    let mut h = p.release[i];
                    if let Some(g) = p.group[i] {
                        h = h.max(self.free_at[g]);
                    }
                    for &(j, delay) in &p.preds[i] {
                        h = h.max(head[j] + p.duration[j] + delay);
                    }
                    if h > p.latest_start[i] || h + p.duration[i] > p.latest_end[i] {
                        return None;
                    }
                    h
                }
            };
        }
        Some(head)
    }

    /// A lower bound on the makespan of every completion of this node.
    fn lower_bound(&self, head: &[i64]) -> i64 {
        let p = self.problem;
        let mut bound = 0;
        let mut on_group: Vec<Vec<(i64, i64, i64)>> = vec![Vec::new(); p.groups];
        for (i, &h) in head.iter().enumerate() {
            let end = h + p.duration[i];
            bound = bound.max(
                end + if self.start[i].is_some() {
                    0
                } else {
                    self.tails[i]
                },
            );
            if let (None, Some(g)) = (self.start[i], p.group[i]) {
                on_group[g].push((h, p.duration[i], self.tails[i]));
            }
        }
        for tasks in on_group {
            bound = bound.max(jackson_preemptive(tasks));
        }
        bound
    }

    fn descend(&mut self, fixed: usize) {
        if self.cut_off || (self.stop_after_first && self.best.is_some()) {
            return;
        }
        self.nodes += 1;
        if self.nodes.is_multiple_of(1024) && web_time::Instant::now() >= self.deadline {
            self.cut_off = true;
            return;
        }
        let p = self.problem;
        if fixed == p.len() {
            let starts: Vec<i64> = self
                .start
                .iter()
                .map(|s| s.expect("every interval is fixed at a leaf"))
                .collect();
            let makespan = (0..p.len())
                .map(|i| starts[i] + p.duration[i])
                .max()
                .unwrap_or(0);
            if makespan < self.best_makespan || self.best.is_none() {
                self.best_makespan = makespan;
                self.best = Some(starts);
            }
            return;
        }
        let Some(head) = self.heads() else {
            return;
        };
        if self.best.is_some() && self.lower_bound(&head) >= self.best_makespan {
            return;
        }

        let ready = |i: usize| {
            self.start[i].is_none() && p.preds[i].iter().all(|&(j, _)| self.start[j].is_some())
        };
        // An interval with no resource starts as soon as it may.
        if let Some(i) = (0..p.len()).find(|&i| p.group[i].is_none() && ready(i)) {
            self.fix(i, head[i], fixed);
            return;
        }
        // Giffler-Thompson: the ready interval that can finish first names
        // the resource; branch on each ready interval there that could start
        // before that finish.
        let (first, finish) = (0..p.len())
            .filter(|&i| ready(i))
            .map(|i| (i, head[i] + p.duration[i]))
            .min_by_key(|&(i, c)| (c, i))
            .expect("an unfixed acyclic model has a ready interval");
        let g = p.group[first].expect("ready intervals without a resource are fixed first");
        let mut conflict: Vec<usize> = (0..p.len())
            .filter(|&i| p.group[i] == Some(g) && ready(i) && head[i] < finish)
            .collect();
        conflict.sort_by_key(|&i| (head[i], i));
        for i in conflict {
            self.fix(i, head[i], fixed);
        }
    }

    /// Starts interval `i` at `s`, searches below, and undoes it.
    fn fix(&mut self, i: usize, s: i64, fixed: usize) {
        let p = self.problem;
        self.start[i] = Some(s);
        let saved = p.group[i].map(|g| {
            let before = self.free_at[g];
            self.free_at[g] = s + p.duration[i];
            (g, before)
        });
        self.descend(fixed + 1);
        if let Some((g, before)) = saved {
            self.free_at[g] = before;
        }
        self.start[i] = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cp::IntervalVar;

    /// A job shop: each job a chain of `(machine, duration)` operations.
    fn job_shop(jobs: &[&[(usize, i64)]], horizon: i64) -> CpModel {
        let mut model = CpModel::new("job shop", horizon);
        let machines = jobs
            .iter()
            .flat_map(|j| j.iter().map(|o| o.0))
            .max()
            .unwrap()
            + 1;
        let mut on_machine = vec![Vec::new(); machines];
        for (j, ops) in jobs.iter().enumerate() {
            for (k, &(m, d)) in ops.iter().enumerate() {
                let name = format!("j{j}o{k}");
                model.add_interval(IntervalVar::new(&name, 0, horizon, d, horizon));
                on_machine[m].push(name.clone());
                if k > 0 {
                    model.add_precedence(format!("j{j}o{}", k - 1), name, 0);
                }
            }
        }
        for names in on_machine {
            model.add_no_overlap(names);
        }
        model.set_objective(Objective::MinimizeMaxEnd);
        model
    }

    /// Fisher & Thompson's 6 × 6 instance (ft06), optimum 55.
    const FT06: [&[(usize, i64)]; 6] = [
        &[(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
        &[(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
        &[(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
        &[(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
        &[(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
        &[(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
    ];

    #[test]
    fn proves_the_optimum_of_ft06() {
        let model = job_shop(&FT06, 1000);
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Optimal);
        assert_eq!(solution.objective_value, Some(55.0));
        assert_eq!(solution.max_end(), 55);
        assert!(model.check(&solution).is_empty());
        // 663 nodes when written: the bound keeps the proof small.
        assert!(solution.num_nodes < 5_000);
    }

    #[test]
    fn a_deadline_below_the_optimum_is_infeasible() {
        let model = job_shop(&FT06, 54);
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Infeasible);
        assert!(solution.intervals.is_empty());
    }

    /// A 10 × 10 job shop from a fixed generator -- too large to finish
    /// within the first check of the time limit.
    fn ten_by_ten() -> CpModel {
        let mut x: u64 = 12345;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x >> 33
        };
        let jobs: Vec<Vec<(usize, i64)>> = (0..10)
            .map(|_| {
                let mut machines: Vec<usize> = (0..10).collect();
                for k in (1..10).rev() {
                    machines.swap(k, next() as usize % (k + 1));
                }
                machines
                    .into_iter()
                    .map(|m| (m, 1 + (next() % 99) as i64))
                    .collect()
            })
            .collect();
        let jobs: Vec<&[(usize, i64)]> = jobs.iter().map(Vec::as_slice).collect();
        job_shop(&jobs, 100_000)
    }

    #[test]
    fn a_cut_off_search_reports_what_it_has() {
        let model = ten_by_ten();
        let config = SolverConfig {
            time_limit_ms: 0,
            ..SolverConfig::default()
        };
        let solution = BranchAndBoundSolver::new().solve(&model, &config);
        // The limit is checked every 1024 nodes; the first descent reaches a
        // leaf in 100 fixes, so a schedule is found before the cut.
        assert_eq!(solution.status, SolverStatus::Feasible);
        assert_eq!(solution.num_nodes, 1024);
        assert!(model.check(&solution).is_empty());
    }

    #[test]
    fn stop_after_first_returns_the_giffler_thompson_schedule() {
        let model = job_shop(&FT06, 1000);
        let config = SolverConfig {
            stop_after_first: true,
            ..SolverConfig::default()
        };
        let solution = BranchAndBoundSolver::new().solve(&model, &config);
        assert_eq!(solution.status, SolverStatus::Feasible);
        assert_eq!(solution.num_nodes, 37);
        assert!(model.check(&solution).is_empty());
    }

    #[test]
    fn no_objective_is_any_feasible_schedule() {
        let mut model = job_shop(&FT06, 1000);
        model.objective = None;
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Feasible);
        assert_eq!(solution.objective_value, None);
        assert!(model.check(&solution).is_empty());
    }

    #[test]
    fn release_dates_delays_and_free_intervals() {
        let mut model = CpModel::new("lags", 100);
        model.add_interval(IntervalVar::new("a", 5, 100, 3, 100));
        model.add_interval(IntervalVar::new("b", 0, 100, 4, 100));
        model.add_interval(IntervalVar::new("c", 0, 100, 2, 100)); // no resource
        model.add_no_overlap(vec!["a".into(), "b".into()]);
        model.add_precedence("a".into(), "c".into(), 10);
        model.set_objective(Objective::MinimizeMaxEnd);
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Optimal);
        // b then a (a is released at 5 anyway): a ends 8, c starts 18.
        assert_eq!(solution.intervals["a"].start, 5);
        assert_eq!(solution.intervals["c"].start, 18);
        assert_eq!(solution.objective_value, Some(20.0));
        assert!(model.check(&solution).is_empty());
    }

    #[test]
    fn a_precedence_cycle_is_infeasible() {
        let mut model = CpModel::new("cycle", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 1, 100));
        model.add_interval(IntervalVar::new("b", 0, 100, 1, 100));
        model.add_precedence("a".into(), "b".into(), 0);
        model.add_precedence("b".into(), "a".into(), 0);
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Infeasible);
    }

    #[test]
    fn a_model_outside_its_class_is_not_searched() {
        let mut model = CpModel::new("cumulative", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 1, 100));
        model.add_cumulative(vec!["a".into()], vec![1], 1);
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Unknown);

        let mut model = CpModel::new("negative delay", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 5, 100));
        model.add_interval(IntervalVar::new("b", 0, 100, 5, 100));
        model.add_precedence("a".into(), "b".into(), -2);
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Unknown);

        let mut model = CpModel::new("two groups", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 1, 100));
        model.add_no_overlap(vec!["a".into()]);
        model.add_no_overlap(vec!["a".into()]);
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Unknown);
    }

    #[test]
    fn an_incumbent_bounds_the_search_and_is_kept_when_nothing_beats_it() {
        let model = job_shop(&FT06, 1000);
        let optimum = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        let plain_nodes = optimum.num_nodes;

        // Given the optimum, the search only has to prove it.
        let warm = BranchAndBoundSolver::new()
            .with_incumbent(optimum.clone())
            .solve(&model, &SolverConfig::default());
        assert_eq!(warm.status, SolverStatus::Optimal);
        assert_eq!(warm.objective_value, Some(55.0));
        assert!(
            warm.num_nodes < plain_nodes,
            "{} vs {plain_nodes}",
            warm.num_nodes
        );
        assert!(model.check(&warm).is_empty());

        // Cut off at once, the incumbent is the answer -- or better.
        let model = ten_by_ten();
        let first = BranchAndBoundSolver::new().solve(
            &model,
            &SolverConfig {
                stop_after_first: true,
                ..SolverConfig::default()
            },
        );
        let cut = BranchAndBoundSolver::new()
            .with_incumbent(first.clone())
            .solve(
                &model,
                &SolverConfig {
                    time_limit_ms: 0,
                    ..SolverConfig::default()
                },
            );
        assert_eq!(cut.status, SolverStatus::Feasible);
        assert!(cut.max_end() <= first.max_end());
        assert!(model.check(&cut).is_empty());
    }

    #[test]
    fn an_incumbent_that_breaks_the_model_is_not_used() {
        let model = job_shop(&FT06, 1000);
        let mut broken = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        broken.intervals.get_mut("j0o0").expect("an interval").start += 1;
        let solution = BranchAndBoundSolver::new()
            .with_incumbent(broken)
            .solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Optimal);
        assert!(model.check(&solution).is_empty());
    }

    #[test]
    fn jackson_preemptive_bound() {
        // One task: head + duration + tail.
        assert_eq!(jackson_preemptive(vec![(2, 3, 4)]), 9);
        // B (tail 10) is released at 1 and preempts A (tail 0): B ends at 3,
        // A at 4 -> max(3 + 10, 4 + 0) = 13, above min head + work + min
        // tail = 0 + 4 + 0.
        assert_eq!(jackson_preemptive(vec![(0, 2, 0), (1, 2, 10)]), 13);
        // Idle until the first release.
        assert_eq!(jackson_preemptive(vec![(5, 1, 0), (7, 1, 0)]), 8);
        assert_eq!(jackson_preemptive(Vec::new()), 0);
    }
}
