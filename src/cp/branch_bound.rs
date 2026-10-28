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
/// Mandatory intervals -- the operations -- each with a fixed duration, or
/// performed as exactly one of its alternatives: a
/// [`Constraint::Alternative`] over optional fixed-duration intervals, each
/// on its own resource and for its own duration (the operation's duration
/// range keeps out an alternative it cannot take). Constraints
/// [`Constraint::NoOverlap`] (each interval in at most one group -- a unary
/// resource; an operation with alternatives is on theirs, not in a group)
/// and [`Constraint::Precedence`] between operations (with a `min_delay` of
/// at least 0); the bounds of each interval's start and end, and the horizon,
/// as deadlines; objective [`Objective::MinimizeMaxEnd`], or none (any
/// feasible schedule). This is the job shop -- flexible, when operations
/// have alternatives -- with release dates, deadlines and time lags, and any
/// precedence graph rather than chains.
///
/// A model outside that class -- a variable-length interval without
/// alternatives, an optional interval outside every alternative or in two,
/// an interval in two no-overlap groups, a precedence on an alternative, a
/// negative delay, any other constraint or objective -- is not searched: the
/// solution is empty with [`SolverStatus::Unknown`]. An invalid model is
/// [`SolverStatus::ModelInvalid`].
///
/// # Algorithm
///
/// A node fixes the start and mode -- the alternative taken, if any -- of
/// some operations. An operation left with one mode and no unary resource is
/// started as soon as its predecessors allow -- that is never worse.
/// Otherwise, of the open (operation, mode) pairs whose operation's
/// predecessors are all fixed, the one that can finish first, at `c`, names
/// a resource; the branches start, on that resource, each such pair that
/// could start before `c` (Giffler & Thompson's conflict set) -- and, when
/// the first pair's operation has another open mode, one more branch closes
/// that mode to it. Every active schedule is reached, and some active
/// schedule is optimal: shifting operations earlier never breaks a deadline.
/// If the first pair's operation takes that mode, the first operation on that
/// resource starts before `c` (else it could start earlier); an operation
/// whose predecessors are not all fixed cannot start before `c` -- its
/// unfixed predecessors end no earlier, and delays are not negative -- so the
/// conflict set misses none; if it does not take that mode, the last branch
/// covers it. Branches are taken earliest start first
/// and, among equal starts, the longest chain of successors first -- the most
/// work remaining behind it -- so the first leaf is the Giffler-Thompson
/// schedule of that rule.
///
/// Each node first narrows the window of every unfixed operation to what a
/// schedule shorter than the best one found must keep: precedences carry
/// earliest starts forward and latest ends back (an operation with open modes
/// counting its shortest), and on each resource edge finding (Vilím's
/// Θ-Λ-tree), over the operations left with that resource only, moves an
/// operation after a set it cannot precede and, mirrored, before a set it
/// cannot follow, until nothing moves. A pair that must start later than it
/// could start now is not branched on.
///
/// A node is pruned when a window empties -- no shorter schedule, or an
/// interval can no longer meet its deadline -- or when its lower bound reaches
/// the best makespan found: the larger of each unfixed interval's earliest
/// start + shortest duration + the longest chain of successors after it,
/// and, on each resource, the bound of Jackson's preemptive schedule of the
/// unfixed operations left with it only (Carlier 1982) -- those earliest
/// starts as release dates, those chains as tails.
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
/// round of a node's propagation costs O(n log n + m) for n intervals and m
/// precedence arcs.
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
/// - Vilím, P. (2008). "Filtering algorithms for the unary resource
///   constraint". *Archives of Control Sciences*, 18(2), 159–202.
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
        // A given schedule that keeps the model, as (start, mode) by index.
        let incumbent = self
            .incumbent
            .as_ref()
            .filter(|s| model.check(s).is_empty())
            .and_then(|s| problem.read(s));
        let incumbent_makespan = incumbent
            .as_ref()
            .map_or(i64::MAX, |plan| problem.makespan(plan));
        let mut search = Search {
            problem: &problem,
            order,
            tails: problem.tails(),
            start: vec![None; problem.len()],
            allowed: problem.modes.iter().map(|m| vec![true; m.len()]).collect(),
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
        if let Some(plan) = &search.best {
            problem.write(plan, &mut solution);
            if problem.minimize {
                solution.objective_value = Some(search.best_makespan as f64);
            }
        }
        solution.num_nodes = search.nodes;
        solution.solve_time_ms = started.elapsed().as_millis() as i64;
        solution
    }
}

/// One way to perform an operation: on a unary resource or none, for a
/// duration, within a window -- and, for an operation with alternatives, the
/// optional interval that is present when it is chosen.
#[derive(Debug, Clone)]
struct Mode {
    group: Option<usize>,
    duration: i64,
    release: i64,
    latest_start: i64,
    latest_end: i64,
    alternative: Option<String>,
}

/// A model in the solver's class, indexed: its mandatory intervals -- the
/// operations -- by name order, each with the modes it may be performed in.
struct Problem {
    names: Vec<String>,
    modes: Vec<Vec<Mode>>,
    /// The alternatives of each operation that no mode can take: always absent.
    never: Vec<Vec<String>>,
    groups: usize,
    /// `(predecessor, min_delay)` of each operation.
    preds: Vec<Vec<(usize, i64)>>,
    /// `(successor, min_delay)` of each operation.
    succs: Vec<Vec<(usize, i64)>>,
    minimize: bool,
}

/// The start and the chosen mode of every operation.
type Plan = Vec<(i64, usize)>;

impl Problem {
    fn from_model(model: &CpModel) -> Option<Self> {
        let minimize = match &model.objective {
            None => false,
            Some(Objective::MinimizeMaxEnd) => true,
            Some(_) => return None,
        };
        let window = |name: &str, duration: i64| {
            let v = &model.intervals[name];
            (
                v.start.min.max(v.end.min - duration),
                v.start.max.min(v.end.max - duration),
                v.end.max.min(model.horizon),
            )
        };
        let mut names: Vec<String> = model
            .intervals
            .values()
            .filter(|v| !v.is_optional)
            .map(|v| v.name.clone())
            .collect();
        names.sort();
        let index: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let n = names.len();

        // The optional intervals each operation may be performed as.
        let mut alternatives: Vec<Option<Vec<String>>> = vec![None; n];
        let mut taken: HashMap<&str, ()> = HashMap::new();
        for constraint in &model.constraints {
            if let Constraint::Alternative {
                main,
                alternatives: alts,
            } = constraint
            {
                let &i = index.get(main.as_str())?;
                if alternatives[i].is_some() {
                    return None;
                }
                for alt in alts {
                    let v = &model.intervals[alt];
                    if !v.is_optional
                        || v.duration.min != v.duration.max
                        || taken.insert(alt.as_str(), ()).is_some()
                    {
                        return None;
                    }
                }
                alternatives[i] = Some(alts.clone());
            }
        }
        // An optional interval outside every alternative is not in the class.
        if model
            .intervals
            .values()
            .any(|v| v.is_optional && !taken.contains_key(v.name.as_str()))
        {
            return None;
        }

        // The unary resource of each interval, by name.
        let mut group_of: HashMap<&str, usize> = HashMap::new();
        let mut groups = 0;
        let (mut preds, mut succs) = (vec![Vec::new(); n], vec![Vec::new(); n]);
        for constraint in &model.constraints {
            match constraint {
                Constraint::NoOverlap { intervals } => {
                    for name in intervals {
                        if group_of.insert(name.as_str(), groups).is_some() {
                            return None;
                        }
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
                    // does not cover; a precedence on an alternative is
                    // outside the class.
                    if *min_delay < 0 {
                        return None;
                    }
                    let (&b, &a) = (index.get(before.as_str())?, index.get(after.as_str())?);
                    preds[a].push((b, *min_delay));
                    succs[b].push((a, *min_delay));
                }
                Constraint::Alternative { .. } => {}
                _ => return None,
            }
        }

        let mut modes = Vec::with_capacity(n);
        let mut never = vec![Vec::new(); n];
        for (i, name) in names.iter().enumerate() {
            let v = &model.intervals[name];
            match &alternatives[i] {
                None => {
                    if v.duration.min != v.duration.max {
                        return None;
                    }
                    let d = v.duration.min;
                    let (release, latest_start, latest_end) = window(name, d);
                    modes.push(vec![Mode {
                        group: group_of.get(name.as_str()).copied(),
                        duration: d,
                        release,
                        latest_start,
                        latest_end,
                        alternative: None,
                    }]);
                }
                Some(alts) => {
                    // The operation's resource is its alternative's.
                    if group_of.contains_key(name.as_str()) {
                        return None;
                    }
                    // An alternative whose length the operation cannot take
                    // is never chosen; with none left, no schedule exists.
                    let own: Vec<Mode> = alts
                        .iter()
                        .filter_map(|alt| {
                            let d = model.intervals[alt].duration.min;
                            if d < v.duration.min || d > v.duration.max {
                                never[i].push(alt.clone());
                                return None;
                            }
                            let (r1, s1, e1) = window(name, d);
                            let (r2, s2, e2) = window(alt, d);
                            Some(Mode {
                                group: group_of.get(alt.as_str()).copied(),
                                duration: d,
                                release: r1.max(r2),
                                latest_start: s1.min(s2),
                                latest_end: e1.min(e2),
                                alternative: Some(alt.clone()),
                            })
                        })
                        .collect();
                    modes.push(own);
                }
            }
        }
        Some(Self {
            names,
            modes,
            never,
            groups,
            preds,
            succs,
            minimize,
        })
    }

    fn len(&self) -> usize {
        self.names.len()
    }

    /// The shortest duration among an operation's modes (0 with none).
    fn min_duration(&self, i: usize) -> i64 {
        self.modes[i].iter().map(|m| m.duration).min().unwrap_or(0)
    }

    fn makespan(&self, plan: &Plan) -> i64 {
        plan.iter()
            .enumerate()
            .map(|(i, &(s, m))| s + self.modes[i][m].duration)
            .max()
            .unwrap_or(0)
    }

    /// A solution of the model as a plan, or `None` when it lacks an
    /// operation or the mode an operation was performed in.
    fn read(&self, solution: &CpSolution) -> Option<Plan> {
        let present = |name: &str| solution.intervals.get(name).filter(|s| s.is_present);
        (0..self.len())
            .map(|i| {
                let start = present(&self.names[i])?.start;
                let mode = self.modes[i].iter().position(|m| match &m.alternative {
                    None => true,
                    Some(alt) => present(alt).is_some(),
                })?;
                Some((start, mode))
            })
            .collect()
    }

    /// Writes a plan into a solution: every operation, its chosen
    /// alternative present with it and the others absent.
    fn write(&self, plan: &Plan, solution: &mut CpSolution) {
        for (i, &(start, chosen)) in plan.iter().enumerate() {
            let d = self.modes[i][chosen].duration;
            let present = IntervalSolution {
                start,
                end: start + d,
                duration: d,
                is_present: true,
            };
            solution
                .intervals
                .insert(self.names[i].clone(), present.clone());
            for (m, mode) in self.modes[i].iter().enumerate() {
                if let Some(alt) = &mode.alternative {
                    let solved = if m == chosen {
                        present.clone()
                    } else {
                        IntervalSolution {
                            is_present: false,
                            ..present.clone()
                        }
                    };
                    solution.intervals.insert(alt.clone(), solved);
                }
            }
            for alt in &self.never[i] {
                let absent = IntervalSolution {
                    is_present: false,
                    ..present.clone()
                };
                solution.intervals.insert(alt.clone(), absent);
            }
        }
    }

    /// The operations in an order every precedence respects (Kahn), or
    /// `None` on a cycle.
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

    /// The longest chain of delays and shortest durations after each
    /// operation ends.
    fn tails(&self) -> Vec<i64> {
        let order = self
            .topological_order()
            .expect("tails are computed for an acyclic model");
        let mut tail = vec![0; self.len()];
        for &i in order.iter().rev() {
            tail[i] = self.succs[i]
                .iter()
                .map(|&(j, delay)| delay + self.min_duration(j) + tail[j])
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

/// A node of the Θ-Λ-tree: the tasks of its subtree split into Θ (white)
/// and Λ (gray), sorted by earliest start at the leaves.
#[derive(Clone, Copy)]
struct ThetaLambda {
    /// Total duration of the white tasks.
    work: i64,
    /// Earliest completion of the white tasks.
    ect: i64,
    /// Largest total duration with at most one gray task added.
    work_gray: i64,
    /// Largest earliest completion with at most one gray task added.
    ect_gray: i64,
    /// The gray task behind `work_gray`, if one is.
    work_gray_task: Option<usize>,
    /// The gray task behind `ect_gray`, if one is.
    ect_gray_task: Option<usize>,
}

impl ThetaLambda {
    /// Below any earliest completion; sums of durations stay far from overflow.
    const NONE: i64 = i64::MIN / 4;
    const EMPTY: Self = Self {
        work: 0,
        ect: Self::NONE,
        work_gray: 0,
        ect_gray: Self::NONE,
        work_gray_task: None,
        ect_gray_task: None,
    };

    fn white(est: i64, duration: i64) -> Self {
        Self {
            work: duration,
            ect: est + duration,
            work_gray: duration,
            ect_gray: est + duration,
            work_gray_task: None,
            ect_gray_task: None,
        }
    }

    fn gray(task: usize, est: i64, duration: i64) -> Self {
        Self {
            work_gray: duration,
            ect_gray: est + duration,
            work_gray_task: Some(task),
            ect_gray_task: Some(task),
            ..Self::EMPTY
        }
    }

    fn join(l: Self, r: Self) -> Self {
        // Of equal values, the one a gray task is behind.
        fn pick(a: (i64, Option<usize>), b: (i64, Option<usize>)) -> (i64, Option<usize>) {
            if a.0 > b.0 || (a.0 == b.0 && a.1.is_some()) {
                a
            } else {
                b
            }
        }
        let (work_gray, work_gray_task) = pick(
            (l.work_gray + r.work, l.work_gray_task),
            (l.work + r.work_gray, r.work_gray_task),
        );
        let (ect_gray, ect_gray_task) = pick(
            pick(
                (r.ect_gray, r.ect_gray_task),
                (l.ect + r.work_gray, r.work_gray_task),
            ),
            (l.ect_gray + r.work, l.ect_gray_task),
        );
        Self {
            work: l.work + r.work,
            ect: r.ect.max(l.ect + r.work),
            work_gray,
            ect_gray,
            work_gray_task,
            ect_gray_task,
        }
    }
}

/// Edge finding on one unary resource, by the Θ-Λ-tree of Vilím (2008):
/// when a task cannot end before a set of others -- their earliest start,
/// with it, plus their total duration overruns their latest end -- it starts
/// after the set's earliest completion. `tasks` are `(earliest start, latest
/// end, duration)`; the result is each task's raised earliest start, or
/// `None` when no order of the tasks meets every latest end. O(k log k).
fn edge_finding(tasks: &[(i64, i64, i64)]) -> Option<Vec<i64>> {
    let k = tasks.len();
    let mut by_est: Vec<usize> = (0..k).collect();
    by_est.sort_unstable_by_key(|&t| tasks[t].0);
    let mut leaf = vec![0; k];
    let size = k.next_power_of_two();
    let mut tree = vec![ThetaLambda::EMPTY; 2 * size];
    for (rank, &t) in by_est.iter().enumerate() {
        leaf[t] = size + rank;
        tree[size + rank] = ThetaLambda::white(tasks[t].0, tasks[t].2);
    }
    for node in (1..size).rev() {
        tree[node] = ThetaLambda::join(tree[2 * node], tree[2 * node + 1]);
    }
    let set = |tree: &mut Vec<ThetaLambda>, at: usize, value: ThetaLambda| {
        tree[at] = value;
        let mut node = at / 2;
        while node >= 1 {
            tree[node] = ThetaLambda::join(tree[2 * node], tree[2 * node + 1]);
            node /= 2;
        }
    };

    let mut by_lct: Vec<usize> = (0..k).collect();
    by_lct.sort_unstable_by_key(|&t| std::cmp::Reverse(tasks[t].1));
    let mut est: Vec<i64> = tasks.iter().map(|t| t.0).collect();
    for (n, &j) in by_lct.iter().enumerate() {
        // The white tasks are those with latest end up to j's: they overrun it.
        if tree[1].ect > tasks[j].1 {
            return None;
        }
        let Some(&next) = by_lct.get(n + 1) else {
            break;
        };
        set(
            &mut tree,
            leaf[j],
            ThetaLambda::gray(j, tasks[j].0, tasks[j].2),
        );
        // A gray task that cannot join the white ones and still meet the
        // largest latest end among them ends after all of them.
        while tree[1].ect_gray > tasks[next].1 {
            let Some(i) = tree[1].ect_gray_task else {
                break;
            };
            est[i] = est[i].max(tree[1].ect);
            set(&mut tree, leaf[i], ThetaLambda::EMPTY);
        }
    }
    Some(est)
}

struct Search<'p> {
    problem: &'p Problem,
    order: Vec<usize>,
    tails: Vec<i64>,
    /// The start and mode of each fixed operation.
    start: Vec<Option<(i64, usize)>>,
    /// Which modes of each operation are still open.
    allowed: Vec<Vec<bool>>,
    /// When each unary resource is next free.
    free_at: Vec<i64>,
    best: Option<Plan>,
    best_makespan: i64,
    nodes: u64,
    deadline: web_time::Instant,
    cut_off: bool,
    stop_after_first: bool,
}

impl Search<'_> {
    /// The duration operation `i` takes for certain: its mode's when fixed,
    /// else the shortest of its open modes.
    fn duration(&self, i: usize) -> i64 {
        let modes = &self.problem.modes[i];
        match self.start[i] {
            Some((_, m)) => modes[m].duration,
            None => modes
                .iter()
                .zip(&self.allowed[i])
                .filter(|(_, &open)| open)
                .map(|(m, _)| m.duration)
                .min()
                .unwrap_or(0),
        }
    }

    /// The single open mode of an unfixed operation, if it has one.
    fn sole_mode(&self, i: usize) -> Option<usize> {
        let mut open = (0..self.allowed[i].len()).filter(|&m| self.allowed[i][m]);
        let m = open.next()?;
        open.next().is_none().then_some(m)
    }

    /// The earliest start of operation `i` in mode `m` from the fixed
    /// operations, the precedences (at `ends` of the predecessors) and the
    /// resource, or `None` when the mode can no longer meet its window.
    fn mode_start(&self, i: usize, m: usize, after: i64) -> Option<i64> {
        let mode = &self.problem.modes[i][m];
        let mut h = after.max(mode.release);
        if let Some(g) = mode.group {
            h = h.max(self.free_at[g]);
        }
        (h <= mode.latest_start && h + mode.duration <= mode.latest_end).then_some(h)
    }

    /// The earliest start of every operation from the fixed ones and the
    /// precedences -- the least over its open modes -- or `None` when an
    /// operation has no open mode left that meets its window.
    fn heads(&self) -> Option<Vec<i64>> {
        let p = self.problem;
        let mut head = vec![0; p.len()];
        for &i in &self.order {
            head[i] = match self.start[i] {
                Some((s, _)) => s,
                None => {
                    let after = self.after_preds(i, &head);
                    (0..p.modes[i].len())
                        .filter(|&m| self.allowed[i][m])
                        .filter_map(|m| self.mode_start(i, m, after))
                        .min()?
                }
            };
        }
        Some(head)
    }

    /// When the predecessors of `i` let it start, from their `head`s.
    fn after_preds(&self, i: usize, head: &[i64]) -> i64 {
        self.problem.preds[i]
            .iter()
            .map(|&(j, delay)| head[j] + self.duration(j) + delay)
            .max()
            .unwrap_or(i64::MIN)
    }

    /// Tightens the window `[earliest start, latest end]` of every operation
    /// that a completion of this node beating the best makespan must keep,
    /// or `None` when no such completion exists. Precedences carry earliest
    /// starts forward and latest ends back; on each resource, edge finding
    /// over the operations bound to it raises earliest starts and, on the
    /// mirrored problem, lowers latest ends; repeated until nothing moves.
    fn propagate(&self, head: &[i64]) -> Option<(Vec<i64>, Vec<i64>)> {
        let p = self.problem;
        let bound = if p.minimize && self.best.is_some() {
            self.best_makespan - 1
        } else {
            i64::MAX / 4
        };
        let duration: Vec<i64> = (0..p.len()).map(|i| self.duration(i)).collect();
        let mut est = head.to_vec();
        let mut lct: Vec<i64> = (0..p.len())
            .map(|i| match self.start[i] {
                Some((s, _)) => s + duration[i],
                None => p.modes[i]
                    .iter()
                    .zip(&self.allowed[i])
                    .filter(|(_, &open)| open)
                    .map(|(m, _)| m.latest_end.min(m.latest_start + m.duration))
                    .max()
                    .unwrap_or(i64::MIN / 4)
                    .min(bound),
            })
            .collect();
        // The unfixed operations bound to each resource.
        let mut on_group: Vec<Vec<usize>> = vec![Vec::new(); p.groups];
        for i in 0..p.len() {
            if self.start[i].is_none() {
                if let Some(g) = self.sole_mode(i).and_then(|m| p.modes[i][m].group) {
                    on_group[g].push(i);
                }
            }
        }
        loop {
            for &i in &self.order {
                if self.start[i].is_none() {
                    for &(j, delay) in &p.preds[i] {
                        est[i] = est[i].max(est[j] + duration[j] + delay);
                    }
                }
            }
            for &i in self.order.iter().rev() {
                for &(j, delay) in &p.succs[i] {
                    lct[i] = lct[i].min(lct[j] - duration[j] - delay);
                }
            }
            if (0..p.len()).any(|i| est[i] + duration[i] > lct[i]) {
                return None;
            }
            let mut moved = false;
            for members in on_group.iter().filter(|m| m.len() > 1) {
                let tasks: Vec<(i64, i64, i64)> = members
                    .iter()
                    .map(|&i| (est[i], lct[i], duration[i]))
                    .collect();
                let raised = edge_finding(&tasks)?;
                let mirrored: Vec<(i64, i64, i64)> =
                    tasks.iter().map(|&(e, l, d)| (-l, -e, d)).collect();
                let lowered = edge_finding(&mirrored)?;
                for (n, &i) in members.iter().enumerate() {
                    if raised[n] > est[i] || -lowered[n] < lct[i] {
                        est[i] = est[i].max(raised[n]);
                        lct[i] = lct[i].min(-lowered[n]);
                        moved = true;
                    }
                }
            }
            if !moved {
                return Some((est, lct));
            }
        }
    }

    /// A lower bound on the makespan of every completion of this node.
    fn lower_bound(&self, head: &[i64]) -> i64 {
        let p = self.problem;
        let mut bound = 0;
        let mut on_group: Vec<Vec<(i64, i64, i64)>> = vec![Vec::new(); p.groups];
        for (i, &h) in head.iter().enumerate() {
            let d = self.duration(i);
            bound = bound.max(
                h + d
                    + if self.start[i].is_some() {
                        0
                    } else {
                        self.tails[i]
                    },
            );
            if self.start[i].is_none() {
                if let Some(g) = self.sole_mode(i).and_then(|m| p.modes[i][m].group) {
                    on_group[g].push((h, d, self.tails[i]));
                }
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
            let plan: Plan = self
                .start
                .iter()
                .map(|s| s.expect("every operation is fixed at a leaf"))
                .collect();
            let makespan = p.makespan(&plan);
            if makespan < self.best_makespan || self.best.is_none() {
                self.best_makespan = makespan;
                self.best = Some(plan);
            }
            return;
        }
        let Some(head) = self.heads() else {
            return;
        };
        let Some((est, _)) = self.propagate(&head) else {
            return;
        };
        if self.best.is_some() && self.lower_bound(&est) >= self.best_makespan {
            return;
        }

        let this = &*self;
        let ready = |i: usize| {
            this.start[i].is_none() && p.preds[i].iter().all(|&(j, _)| this.start[j].is_some())
        };
        // Each open (operation, mode) of a ready operation, with its
        // earliest start.
        let pairs: Vec<(usize, usize, i64)> = (0..p.len())
            .filter(|&i| ready(i))
            .flat_map(|i| {
                let after = this.after_preds(i, &head);
                (0..p.modes[i].len())
                    .filter(move |&m| this.allowed[i][m])
                    .filter_map(move |m| this.mode_start(i, m, after).map(|h| (i, m, h)))
            })
            .collect();
        // An operation with one mode and no resource starts as soon as it may.
        if let Some(&(i, m, h)) = pairs
            .iter()
            .find(|&&(i, m, _)| p.modes[i][m].group.is_none() && self.sole_mode(i).is_some())
        {
            self.fix(i, m, h, fixed);
            return;
        }
        // Giffler-Thompson over modes: the pair that can finish first names
        // the resource; branch on each pair there that could start before
        // that finish -- and, when the first pair's operation has another
        // mode, on that operation not taking this one.
        let Some(&(first, first_mode, first_start)) = pairs
            .iter()
            .min_by_key(|&&(i, m, h)| (h + p.modes[i][m].duration, i, m))
        else {
            return;
        };
        let finish = first_start + p.modes[first][first_mode].duration;
        let mut branches: Vec<(usize, usize, i64)> = match p.modes[first][first_mode].group {
            // No resource to contend for: in that mode it starts at once.
            None => vec![(first, first_mode, first_start)],
            Some(g) => pairs
                .iter()
                .copied()
                .filter(|&(i, m, h)| p.modes[i][m].group == Some(g) && h < finish)
                .collect(),
        };
        // Started next in a mode, an operation starts at that mode's earliest
        // start; one that must start later to beat the best makespan is no
        // branch.
        branches.retain(|&(i, _, h)| est[i] <= h);
        branches.sort_by_key(|&(i, m, h)| (h, std::cmp::Reverse(self.tails[i]), i, m));
        for (i, m, h) in branches {
            self.fix(i, m, h, fixed);
        }
        if self.sole_mode(first).is_none() {
            self.allowed[first][first_mode] = false;
            self.descend(fixed);
            self.allowed[first][first_mode] = true;
        }
    }

    /// Starts operation `i` in mode `m` at `s`, searches below, and undoes it.
    fn fix(&mut self, i: usize, m: usize, s: i64, fixed: usize) {
        let p = self.problem;
        self.start[i] = Some((s, m));
        let saved = p.modes[i][m].group.map(|g| {
            let before = self.free_at[g];
            self.free_at[g] = s + p.modes[i][m].duration;
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
    use crate::cp::{DurationVar, IntervalVar};

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
        // 140 nodes when written (663 before edge finding and the tie-break):
        // the bound and the propagation keep the proof small.
        assert!(solution.num_nodes < 5_000, "{} nodes", solution.num_nodes);
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
    fn edge_finding_orders_what_cannot_come_first() {
        // y (3) cannot fit before x (4) and keep x's latest end 4: y ends
        // after x, so it starts at 4. x is not moved.
        assert_eq!(edge_finding(&[(0, 4, 4), (0, 10, 3)]), Some(vec![0, 4]));
        // z must follow both x and y, whose latest ends are 7: it starts once
        // they are done, at 0 + 3 + 4.
        assert_eq!(
            edge_finding(&[(0, 7, 3), (0, 7, 4), (0, 20, 2)]),
            Some(vec![0, 0, 7])
        );
        // Two tasks of 2 within [0, 3] overrun it.
        assert_eq!(edge_finding(&[(0, 3, 2), (0, 3, 2)]), None);
        // Nothing to infer.
        assert_eq!(edge_finding(&[(0, 100, 5), (0, 100, 5)]), Some(vec![0, 0]));
    }

    /// The optimal makespan of a job shop by enumerating every order of the
    /// operations on every machine.
    fn brute_force(jobs: &[Vec<(usize, i64)>], machines: usize) -> i64 {
        let ops: Vec<(usize, usize)> = jobs
            .iter()
            .enumerate()
            .flat_map(|(j, o)| (0..o.len()).map(move |k| (j, k)))
            .collect();
        let on: Vec<Vec<(usize, usize)>> = (0..machines)
            .map(|m| {
                ops.iter()
                    .copied()
                    .filter(|&(j, k)| jobs[j][k].0 == m)
                    .collect()
            })
            .collect();
        fn permutations(v: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
            if v.len() <= 1 {
                return vec![v.to_vec()];
            }
            let mut out = Vec::new();
            for i in 0..v.len() {
                let mut rest = v.to_vec();
                let x = rest.remove(i);
                for mut p in permutations(&rest) {
                    p.insert(0, x);
                    out.push(p);
                }
            }
            out
        }
        let choices: Vec<Vec<Vec<(usize, usize)>>> = on.iter().map(|o| permutations(o)).collect();
        let mut best = i64::MAX;
        let mut pick = vec![0; machines];
        loop {
            // Earliest starts by relaxation; a cycle never settles.
            let mut start: HashMap<(usize, usize), i64> = ops.iter().map(|&o| (o, 0)).collect();
            let mut settled = false;
            for _ in 0..=ops.len() {
                let mut moved = false;
                for &(j, k) in &ops {
                    let mut s = 0;
                    if k > 0 {
                        s = s.max(start[&(j, k - 1)] + jobs[j][k - 1].1);
                    }
                    let seq = &choices[jobs[j][k].0][pick[jobs[j][k].0]];
                    let at = seq.iter().position(|&o| o == (j, k)).unwrap();
                    if at > 0 {
                        let (pj, pk) = seq[at - 1];
                        s = s.max(start[&(pj, pk)] + jobs[pj][pk].1);
                    }
                    if s != start[&(j, k)] {
                        start.insert((j, k), s);
                        moved = true;
                    }
                }
                if !moved {
                    settled = true;
                    break;
                }
            }
            if settled {
                let makespan = ops
                    .iter()
                    .map(|&(j, k)| start[&(j, k)] + jobs[j][k].1)
                    .max()
                    .unwrap_or(0);
                best = best.min(makespan);
            }
            // Next combination of machine orders.
            let mut m = 0;
            while m < machines {
                pick[m] += 1;
                if pick[m] < choices[m].len() {
                    break;
                }
                pick[m] = 0;
                m += 1;
            }
            if m == machines {
                return best;
            }
        }
    }

    #[test]
    fn proves_the_brute_force_optimum_on_random_job_shops() {
        let mut x: u64 = 7;
        let mut next = move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x >> 33
        };
        for _ in 0..40 {
            let (jobs_n, machines) = (3, 3 + next() as usize % 2);
            let jobs: Vec<Vec<(usize, i64)>> = (0..jobs_n)
                .map(|_| {
                    let mut order: Vec<usize> = (0..machines).collect();
                    for k in (1..machines).rev() {
                        order.swap(k, next() as usize % (k + 1));
                    }
                    order
                        .into_iter()
                        .map(|m| (m, 1 + (next() % 9) as i64))
                        .collect()
                })
                .collect();
            let slices: Vec<&[(usize, i64)]> = jobs.iter().map(Vec::as_slice).collect();
            let model = job_shop(&slices, 1000);
            let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
            assert_eq!(solution.status, SolverStatus::Optimal);
            assert!(model.check(&solution).is_empty());
            assert_eq!(
                solution.objective_value,
                Some(brute_force(&jobs, machines) as f64),
                "{jobs:?}"
            );
        }
    }

    /// A flexible job shop: each operation a list of `(machine, duration)`
    /// it may run as -- an operation with one choice is a plain interval on
    /// that machine, one with several an `Alternative` of optional intervals.
    fn flexible_job_shop(jobs: &[Vec<Vec<(usize, i64)>>], horizon: i64) -> CpModel {
        let mut model = CpModel::new("flexible job shop", horizon);
        let mut on_machine: Vec<Vec<String>> = Vec::new();
        let mut put = |m: usize, name: String| {
            if on_machine.len() <= m {
                on_machine.resize(m + 1, Vec::new());
            }
            on_machine[m].push(name);
        };
        for (j, ops) in jobs.iter().enumerate() {
            for (k, choices) in ops.iter().enumerate() {
                let name = format!("j{j}o{k}");
                if let [(m, d)] = choices.as_slice() {
                    model.add_interval(IntervalVar::new(&name, 0, horizon, *d, horizon));
                    put(*m, name.clone());
                } else {
                    let lo = choices.iter().map(|c| c.1).min().unwrap();
                    let hi = choices.iter().map(|c| c.1).max().unwrap();
                    model.add_interval(
                        IntervalVar::new(&name, 0, horizon, lo, horizon)
                            .with_variable_duration(lo, hi),
                    );
                    let alternatives: Vec<String> = choices
                        .iter()
                        .map(|&(m, d)| {
                            let alt = format!("{name}@{m}");
                            model.add_interval(
                                IntervalVar::new(&alt, 0, horizon, d, horizon)
                                    .as_optional(format!("{alt}?")),
                            );
                            put(m, alt.clone());
                            alt
                        })
                        .collect();
                    model.add_constraint(Constraint::Alternative {
                        main: name.clone(),
                        alternatives,
                    });
                }
                if k > 0 {
                    model.add_precedence(format!("j{j}o{}", k - 1), name, 0);
                }
            }
        }
        for names in on_machine.into_iter().filter(|n| !n.is_empty()) {
            model.add_no_overlap(names);
        }
        model.set_objective(Objective::MinimizeMaxEnd);
        model
    }

    #[test]
    fn proves_the_best_assignment_on_random_flexible_job_shops() {
        let mut x: u64 = 11;
        let mut next = move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x >> 33
        };
        for _ in 0..25 {
            let machines = 3;
            let jobs: Vec<Vec<Vec<(usize, i64)>>> = (0..3)
                .map(|_| {
                    (0..3)
                        .map(|_| {
                            let first = next() as usize % machines;
                            let mut choices = vec![(first, 1 + (next() % 9) as i64)];
                            if next() % 2 == 0 {
                                let other = (first + 1 + next() as usize % 2) % machines;
                                choices.push((other, 1 + (next() % 9) as i64));
                            }
                            choices
                        })
                        .collect()
                })
                .collect();
            let model = flexible_job_shop(&jobs, 1000);
            let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
            assert_eq!(solution.status, SolverStatus::Optimal, "{jobs:?}");
            assert!(model.check(&solution).is_empty(), "{jobs:?}");

            // Every assignment of machines, each a job shop solved exactly.
            let flexible: Vec<(usize, usize)> = jobs
                .iter()
                .enumerate()
                .flat_map(|(j, ops)| {
                    ops.iter()
                        .enumerate()
                        .filter(|(_, c)| c.len() > 1)
                        .map(move |(k, _)| (j, k))
                })
                .collect();
            let best = (0..1u32 << flexible.len())
                .map(|mask| {
                    let mut fixed = jobs.clone();
                    for (bit, &(j, k)) in flexible.iter().enumerate() {
                        let pick = fixed[j][k][((mask >> bit) & 1) as usize];
                        fixed[j][k] = vec![pick];
                    }
                    let shop = flexible_job_shop(&fixed, 1000);
                    BranchAndBoundSolver::new()
                        .solve(&shop, &SolverConfig::default())
                        .objective_value
                        .expect("a job shop has a schedule")
                })
                .fold(f64::INFINITY, f64::min);
            assert_eq!(solution.objective_value, Some(best), "{jobs:?}");
            // The chosen alternative is present, the others absent.
            for (j, ops) in jobs.iter().enumerate() {
                for (k, choices) in ops.iter().enumerate().filter(|(_, c)| c.len() > 1) {
                    let present = choices
                        .iter()
                        .filter(|(m, _)| solution.intervals[&format!("j{j}o{k}@{m}")].is_present)
                        .count();
                    assert_eq!(present, 1);
                }
            }
        }
    }

    #[test]
    fn an_alternative_the_operation_cannot_take_is_never_chosen() {
        let mut model = CpModel::new("lengths", 100);
        model.add_interval(IntervalVar::new("op", 0, 100, 2, 100).with_variable_duration(2, 5));
        model.add_interval(IntervalVar::new("fast", 0, 100, 1, 100).as_optional("fast?"));
        model.add_interval(IntervalVar::new("slow", 0, 100, 4, 100).as_optional("slow?"));
        model.add_constraint(Constraint::Alternative {
            main: "op".into(),
            alternatives: vec!["fast".into(), "slow".into()],
        });
        model.set_objective(Objective::MinimizeMaxEnd);
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Optimal);
        assert!(solution.intervals["slow"].is_present);
        assert!(!solution.intervals["fast"].is_present);
        assert_eq!(solution.objective_value, Some(4.0));
        assert!(model.check(&solution).is_empty());

        // With no length it can take, no schedule exists.
        model.intervals.get_mut("slow").unwrap().duration = DurationVar::fixed(9);
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Infeasible);
    }

    #[test]
    fn a_flexible_incumbent_is_read_back() {
        let jobs = vec![
            vec![vec![(0, 3), (1, 5)], vec![(1, 2)]],
            vec![vec![(0, 4)], vec![(0, 2), (1, 3)]],
        ];
        let model = flexible_job_shop(&jobs, 100);
        let optimum = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        let warm = BranchAndBoundSolver::new()
            .with_incumbent(optimum.clone())
            .solve(&model, &SolverConfig::default());
        assert_eq!(warm.status, SolverStatus::Optimal);
        assert_eq!(warm.objective_value, optimum.objective_value);
        assert!(warm.num_nodes <= optimum.num_nodes);
        assert!(model.check(&warm).is_empty());
    }

    #[test]
    fn an_optional_interval_outside_an_alternative_is_not_searched() {
        let mut model = CpModel::new("loose", 100);
        model.add_interval(IntervalVar::new("a", 0, 100, 1, 100).as_optional("a?"));
        let solution = BranchAndBoundSolver::new().solve(&model, &SolverConfig::default());
        assert_eq!(solution.status, SolverStatus::Unknown);
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
