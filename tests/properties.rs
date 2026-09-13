//! Property tests: what every metaheuristic runner and operator must satisfy
//! on generated problems, not only on the hand-picked instances the unit
//! tests use.
//!
//! - Permutation operators return permutations: crossover children and
//!   mutated tours contain every index exactly once.
//! - Every runner reports a best it actually holds — `best_cost` is the
//!   cost of `best` re-evaluated through the problem — and a best-so-far
//!   history that never rises and ends at that best.
//! - A fixed seed reproduces the run exactly.
//! - Non-dominated sorting partitions the population into Pareto layers,
//!   the first of which is the brute-force non-dominated set.

use proptest::prelude::*;
use rand::{Rng, RngExt};
use u_metaheur::alns::{AlnsConfig, AlnsProblem, AlnsRunner, DestroyOperator, RepairOperator};
use u_metaheur::brkga::{BrkgaConfig, BrkgaDecoder, BrkgaRunner};
use u_metaheur::ga::multi_objective::{crowding_distance, non_dominated_sort};
use u_metaheur::ga::operators::{
    insert_mutation, invert_mutation, order_crossover, pmx_crossover, swap_mutation,
};
use u_metaheur::ga::{GaConfig, GaProblem, GaRunner, Individual, Selection};
use u_metaheur::sa::{CoolingSchedule, SaConfig, SaProblem, SaRunner};
use u_metaheur::tabu::{TabuConfig, TabuMove, TabuProblem, TabuRunner};
use u_metaheur::vns::{VnsConfig, VnsProblem, VnsRunner};
use u_numflow::random::{create_rng, shuffled_indices};

// ── A random symmetric TSP shared by every runner ────────────────────

#[derive(Clone, Debug)]
struct Tsp {
    dist: Vec<Vec<f64>>,
}

impl Tsp {
    fn n(&self) -> usize {
        self.dist.len()
    }

    fn tour_cost(&self, tour: &[usize]) -> f64 {
        tour.iter()
            .zip(tour.iter().cycle().skip(1))
            .map(|(&a, &b)| self.dist[a][b])
            .sum()
    }

    fn random_tour<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
        shuffled_indices(self.n(), rng)
    }
}

/// `n` cities with symmetric distances in [1, 100].
fn tsp(n: impl Strategy<Value = usize>) -> impl Strategy<Value = Tsp> {
    n.prop_flat_map(|n| {
        prop::collection::vec(1.0f64..100.0, n * n).prop_map(move |flat| {
            let mut dist = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in (i + 1)..n {
                    dist[i][j] = flat[i * n + j];
                    dist[j][i] = flat[i * n + j];
                }
            }
            Tsp { dist }
        })
    })
}

fn is_permutation(perm: &[usize], n: usize) -> bool {
    let mut seen = vec![false; n];
    perm.len() == n
        && perm
            .iter()
            .all(|&i| i < n && !std::mem::replace(&mut seen[i], true))
}

fn check_non_increasing(history: &[f64], what: &str) -> Result<(), TestCaseError> {
    for (i, w) in history.windows(2).enumerate() {
        prop_assert!(
            w[1] <= w[0] + 1e-9,
            "{what} rose from {} to {} at step {}",
            w[0],
            w[1],
            i + 1
        );
    }
    Ok(())
}

// ── GA ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Tour {
    perm: Vec<usize>,
    fitness: f64,
}

impl Individual for Tour {
    type Fitness = f64;
    fn fitness(&self) -> f64 {
        self.fitness
    }
    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }
}

impl GaProblem for Tsp {
    type Individual = Tour;

    fn create_individual<R: Rng>(&self, rng: &mut R) -> Tour {
        Tour {
            perm: self.random_tour(rng),
            fitness: f64::INFINITY,
        }
    }

    fn evaluate(&self, individual: &Tour) -> f64 {
        self.tour_cost(&individual.perm)
    }

    fn crossover<R: Rng>(&self, p1: &Tour, p2: &Tour, rng: &mut R) -> Vec<Tour> {
        let (a, b) = if rng.random_bool(0.5) {
            order_crossover(&p1.perm, &p2.perm, rng)
        } else {
            pmx_crossover(&p1.perm, &p2.perm, rng)
        };
        vec![
            Tour {
                perm: a,
                fitness: f64::INFINITY,
            },
            Tour {
                perm: b,
                fitness: f64::INFINITY,
            },
        ]
    }

    fn mutate<R: Rng>(&self, individual: &mut Tour, rng: &mut R) {
        match rng.random_range(0..3) {
            0 => swap_mutation(&mut individual.perm, rng),
            1 => insert_mutation(&mut individual.perm, rng),
            _ => invert_mutation(&mut individual.perm, rng),
        }
    }
}

// ── BRKGA ────────────────────────────────────────────────────────────

impl BrkgaDecoder for Tsp {
    fn decode(&self, keys: &[f64]) -> f64 {
        let mut order: Vec<usize> = (0..keys.len()).collect();
        order.sort_by(|&a, &b| keys[a].partial_cmp(&keys[b]).expect("keys are finite"));
        self.tour_cost(&order)
    }
}

// ── SA · Tabu · VNS · ALNS on a swap neighbourhood ──────────────────

fn swapped(tour: &[usize], i: usize, j: usize) -> Vec<usize> {
    let mut t = tour.to_vec();
    t.swap(i, j);
    t
}

impl SaProblem for Tsp {
    type Solution = Vec<usize>;
    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
        self.random_tour(rng)
    }
    fn cost(&self, solution: &Vec<usize>) -> f64 {
        self.tour_cost(solution)
    }
    fn neighbor<R: Rng>(&self, solution: &Vec<usize>, rng: &mut R) -> Vec<usize> {
        let n = self.n();
        swapped(solution, rng.random_range(0..n), rng.random_range(0..n))
    }
}

impl TabuProblem for Tsp {
    type Solution = Vec<usize>;
    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
        self.random_tour(rng)
    }
    fn cost(&self, solution: &Vec<usize>) -> f64 {
        self.tour_cost(solution)
    }
    fn neighbors<R: Rng>(&self, solution: &Vec<usize>, _rng: &mut R) -> Vec<TabuMove<Vec<usize>>> {
        let n = self.n();
        let mut moves = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let s = swapped(solution, i, j);
                let cost = self.tour_cost(&s);
                moves.push(TabuMove {
                    solution: s,
                    key: format!("{i}-{j}"),
                    cost,
                });
            }
        }
        moves
    }
}

impl VnsProblem for Tsp {
    type Solution = Vec<usize>;
    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
        self.random_tour(rng)
    }
    fn cost(&self, solution: &Vec<usize>) -> f64 {
        self.tour_cost(solution)
    }
    fn neighborhood_count(&self) -> usize {
        3
    }
    fn shake<R: Rng>(&self, solution: &Vec<usize>, k: usize, rng: &mut R) -> Vec<usize> {
        let n = self.n();
        let mut s = solution.clone();
        for _ in 0..=k {
            s.swap(rng.random_range(0..n), rng.random_range(0..n));
        }
        s
    }
    fn local_search(&self, solution: &Vec<usize>) -> Vec<usize> {
        // First-improvement swap descent.
        let n = self.n();
        let mut current = solution.clone();
        let mut cost = self.tour_cost(&current);
        let mut improved = true;
        while improved {
            improved = false;
            'outer: for i in 0..n {
                for j in (i + 1)..n {
                    let s = swapped(&current, i, j);
                    let c = self.tour_cost(&s);
                    if c < cost - 1e-12 {
                        current = s;
                        cost = c;
                        improved = true;
                        break 'outer;
                    }
                }
            }
        }
        current
    }
}

impl AlnsProblem for Tsp {
    type Solution = Vec<usize>;
    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
        self.random_tour(rng)
    }
    fn cost(&self, solution: &Vec<usize>) -> f64 {
        self.tour_cost(solution)
    }
}

/// Reverses a random segment whose length follows the destroy degree.
struct SegmentReverse;

impl DestroyOperator<Vec<usize>> for SegmentReverse {
    fn name(&self) -> &str {
        "segment-reverse"
    }
    fn destroy<R: Rng>(&self, solution: &Vec<usize>, degree: f64, rng: &mut R) -> Vec<usize> {
        let n = solution.len();
        let len = ((n as f64 * degree) as usize).clamp(2, n);
        let start = rng.random_range(0..n);
        let mut s = solution.clone();
        let idx: Vec<usize> = (0..len).map(|k| (start + k) % n).collect();
        let vals: Vec<usize> = idx.iter().rev().map(|&i| solution[i]).collect();
        for (i, v) in idx.into_iter().zip(vals) {
            s[i] = v;
        }
        s
    }
}

/// Applies one pass of first-improvement swaps.
struct SwapRepair(Tsp);

impl RepairOperator<Vec<usize>> for SwapRepair {
    fn name(&self) -> &str {
        "swap-repair"
    }
    fn repair<R: Rng>(&self, solution: &Vec<usize>, _rng: &mut R) -> Vec<usize> {
        let n = solution.len();
        let mut current = solution.clone();
        let mut cost = self.0.tour_cost(&current);
        for i in 0..n {
            for j in (i + 1)..n {
                let s = swapped(&current, i, j);
                let c = self.0.tour_cost(&s);
                if c < cost {
                    current = s;
                    cost = c;
                }
            }
        }
        current
    }
}

// ── Brute-force Pareto dominance ────────────────────────────────────

/// Indices of the points with the smallest and largest value of objective `k`.
fn extremes(objectives: &[Vec<f64>], k: usize) -> (usize, usize) {
    let by_k = |&a: &usize, &b: &usize| objectives[a][k].partial_cmp(&objectives[b][k]).unwrap();
    let lo = (0..objectives.len()).min_by(by_k).expect("non-empty");
    let hi = (0..objectives.len()).max_by(by_k).expect("non-empty");
    (lo, hi)
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Both crossover children and every mutation output are permutations
    /// of `0..n`, for any pair of parent permutations.
    #[test]
    fn permutation_operators_return_permutations(
        n in 2..=12usize,
        seed in any::<u64>(),
    ) {
        let mut rng = create_rng(seed);
        let p1 = shuffled_indices(n, &mut rng);
        let p2 = shuffled_indices(n, &mut rng);

        let (a, b) = order_crossover(&p1, &p2, &mut rng);
        prop_assert!(is_permutation(&a, n), "OX child 1 {:?}", a);
        prop_assert!(is_permutation(&b, n), "OX child 2 {:?}", b);
        let (a, b) = pmx_crossover(&p1, &p2, &mut rng);
        prop_assert!(is_permutation(&a, n), "PMX child 1 {:?}", a);
        prop_assert!(is_permutation(&b, n), "PMX child 2 {:?}", b);

        let mut m = p1.clone();
        swap_mutation(&mut m, &mut rng);
        prop_assert!(is_permutation(&m, n), "swap {:?}", m);
        insert_mutation(&mut m, &mut rng);
        prop_assert!(is_permutation(&m, n), "insert {:?}", m);
        invert_mutation(&mut m, &mut rng);
        prop_assert!(is_permutation(&m, n), "invert {:?}", m);
    }

    /// The GA's best-so-far never rises, ends at `best_fitness`, which is
    /// the cost of `best` re-evaluated; with elitism the population best
    /// equals the best-so-far every generation; the per-generation stats
    /// are ordered; and a fixed seed reproduces the run.
    #[test]
    fn ga_best_never_worsens_and_is_what_it_reports(
        problem in tsp(3..=8usize),
        population_size in 4..=12usize,
        max_generations in 1..=15usize,
        selection_idx in 0..5usize,
        elite_ratio in 0.05f64..0.5,
        seed in any::<u64>(),
    ) {
        let selection = [
            Selection::Tournament(2),
            Selection::Tournament(3),
            Selection::Tournament(4),
            Selection::Roulette,
            Selection::Rank,
        ][selection_idx];
        // At least one elite, never the whole population.
        let elite_ratio = elite_ratio.max(1.0 / population_size as f64 + 1e-9);
        let config = GaConfig::default()
            .with_population_size(population_size)
            .with_max_generations(max_generations)
            .with_selection(selection)
            .with_elite_ratio(elite_ratio)
            .with_stagnation_limit(0)
            .with_seed(seed);
        let result = GaRunner::run(&problem, &config).unwrap();
        let again = GaRunner::run(&problem, &config).unwrap();

        prop_assert!(is_permutation(&result.best.perm, problem.n()), "best {:?}", result.best.perm);
        let cost = problem.tour_cost(&result.best.perm);
        prop_assert!((cost - result.best_fitness).abs() < 1e-9, "best_fitness {} vs cost {cost}", result.best_fitness);
        prop_assert!((result.best.fitness() - result.best_fitness).abs() < 1e-9);

        prop_assert_eq!(result.generations, max_generations);
        prop_assert_eq!(result.fitness_history.len(), max_generations + 1);
        prop_assert_eq!(result.generation_stats.len(), max_generations + 1);
        check_non_increasing(&result.fitness_history, "GA best-so-far")?;
        prop_assert_eq!(*result.fitness_history.last().unwrap(), result.best_fitness);

        for (g, stats) in result.generation_stats.iter().enumerate() {
            prop_assert_eq!(stats.generation, g);
            prop_assert!(
                (stats.best_fitness - result.fitness_history[g]).abs() < 1e-9,
                "generation {g}: population best {} but best-so-far {} — an elite was lost",
                stats.best_fitness, result.fitness_history[g]
            );
            prop_assert!(stats.best_fitness <= stats.mean_fitness + 1e-9);
            prop_assert!(stats.mean_fitness <= stats.worst_fitness + 1e-9);
            prop_assert!(stats.std_dev >= 0.0);
        }

        prop_assert_eq!(&again.fitness_history, &result.fitness_history, "same seed, different run");
        prop_assert_eq!(&again.best.perm, &result.best.perm);
    }

    /// BRKGA keeps every key of its best chromosome in [0, 1), reports the
    /// cost that chromosome decodes to, never lets the best-so-far rise,
    /// and reproduces under a fixed seed.
    #[test]
    fn brkga_keys_stay_in_the_unit_interval_and_cost_never_rises(
        problem in tsp(3..=8usize),
        population_size in 4..=12usize,
        max_generations in 1..=15usize,
        seed in any::<u64>(),
    ) {
        // At least one elite (0.3 × 4 = 1) and never a whole population of them.
        let config = BrkgaConfig::new(problem.n())
            .with_population_size(population_size)
            .with_elite_fraction(0.3)
            .with_mutant_fraction(0.2)
            .with_max_generations(max_generations)
            .with_stagnation_limit(0)
            .with_seed(seed);
        let result = BrkgaRunner::run(&problem, &config).unwrap();
        let again = BrkgaRunner::run(&problem, &config).unwrap();

        prop_assert_eq!(result.best_keys.len(), problem.n());
        for (i, &k) in result.best_keys.iter().enumerate() {
            prop_assert!((0.0..1.0).contains(&k), "key {i} = {k} outside [0, 1)");
        }
        let decoded = problem.decode(&result.best_keys);
        prop_assert!((decoded - result.best_cost).abs() < 1e-9, "best_cost {} vs decoded {decoded}", result.best_cost);
        prop_assert!(!result.cost_history.is_empty());
        check_non_increasing(&result.cost_history, "BRKGA best-so-far")?;
        prop_assert_eq!(*result.cost_history.last().unwrap(), result.best_cost);
        prop_assert!(result.generations <= max_generations);

        prop_assert_eq!(&again.cost_history, &result.cost_history, "same seed, different run");
        prop_assert_eq!(&again.best_keys, &result.best_keys);
    }

    /// SA, tabu search and VNS each report a best they actually hold: the
    /// cost of `best` re-evaluated, a best-so-far history that never rises
    /// and ends at it, a `best_iteration` whose history entry is it, and
    /// the same run under the same seed.
    #[test]
    fn trajectory_searches_report_a_best_they_actually_hold(
        problem in tsp(3..=8usize),
        max_iterations in 1..=40usize,
        cooling_idx in 0..3usize,
        tenure in 1..=6usize,
        seed in any::<u64>(),
    ) {
        let n = problem.n();

        let cooling = [
            CoolingSchedule::Geometric { alpha: 0.9 },
            CoolingSchedule::Linear,
            CoolingSchedule::LundyMees { beta: 0.01 },
        ][cooling_idx];
        let sa_config = SaConfig::default()
            .with_initial_temperature(50.0)
            .with_min_temperature(0.1)
            .with_cooling(cooling)
            .with_iterations_per_temperature(3)
            .with_max_iterations(max_iterations)
            .with_seed(seed);
        let sa = SaRunner::run(&problem, &sa_config);
        let sa_again = SaRunner::run(&problem, &sa_config);
        prop_assert!(is_permutation(&sa.best, n), "SA best {:?}", sa.best);
        prop_assert!((SaProblem::cost(&problem, &sa.best) - sa.best_cost).abs() < 1e-9, "SA best_cost");
        check_non_increasing(&sa.cost_history, "SA best-so-far")?;
        prop_assert_eq!(*sa.cost_history.last().unwrap(), sa.best_cost);
        prop_assert!(sa.iterations <= max_iterations);
        prop_assert!(sa.improving_moves <= sa.accepted_moves, "improving {} > accepted {}", sa.improving_moves, sa.accepted_moves);
        prop_assert!(sa.accepted_moves <= sa.iterations);
        prop_assert!(sa.final_temperature <= 50.0 + 1e-9 && sa.final_temperature > 0.0);
        prop_assert_eq!(&sa_again.cost_history, &sa.cost_history, "SA: same seed, different run");
        prop_assert_eq!(&sa_again.best, &sa.best);

        let tabu_config = TabuConfig::default()
            .with_max_iterations(max_iterations)
            .with_tabu_tenure(tenure)
            .with_max_no_improve(max_iterations)
            .with_seed(seed);
        let tabu = TabuRunner::run(&problem, &tabu_config);
        let tabu_again = TabuRunner::run(&problem, &tabu_config);
        prop_assert!(is_permutation(&tabu.best, n), "tabu best {:?}", tabu.best);
        prop_assert!((TabuProblem::cost(&problem, &tabu.best) - tabu.best_cost).abs() < 1e-9, "tabu best_cost");
        check_non_increasing(&tabu.cost_history, "tabu best-so-far")?;
        prop_assert_eq!(tabu.iterations, tabu.cost_history.len());
        prop_assert!(tabu.iterations >= 1 && tabu.iterations <= max_iterations);
        prop_assert!(tabu.best_iteration < tabu.iterations);
        prop_assert_eq!(tabu.cost_history[tabu.best_iteration], tabu.best_cost);
        prop_assert_eq!(*tabu.cost_history.last().unwrap(), tabu.best_cost);
        prop_assert_eq!(&tabu_again.cost_history, &tabu.cost_history, "tabu: same seed, different run");
        prop_assert_eq!(&tabu_again.best, &tabu.best);

        let vns_config = VnsConfig::default()
            .with_max_iterations(max_iterations)
            .with_max_no_improve(max_iterations)
            .with_seed(seed);
        let vns = VnsRunner::run(&problem, &vns_config);
        let vns_again = VnsRunner::run(&problem, &vns_config);
        prop_assert!(is_permutation(&vns.best, n), "VNS best {:?}", vns.best);
        prop_assert!((VnsProblem::cost(&problem, &vns.best) - vns.best_cost).abs() < 1e-9, "VNS best_cost");
        check_non_increasing(&vns.cost_history, "VNS best-so-far")?;
        prop_assert!(!vns.cost_history.is_empty() && vns.cost_history.len() <= max_iterations);
        prop_assert!(vns.best_iteration < vns.cost_history.len());
        prop_assert_eq!(vns.cost_history[vns.best_iteration], vns.best_cost);
        prop_assert_eq!(*vns.cost_history.last().unwrap(), vns.best_cost);
        // VNS starts from a local optimum: no single swap improves `best`.
        for i in 0..n {
            for j in (i + 1)..n {
                let c = problem.tour_cost(&swapped(&vns.best, i, j));
                prop_assert!(c >= vns.best_cost - 1e-9, "VNS best is not swap-optimal: {i}<->{j} gives {c} < {}", vns.best_cost);
            }
        }
        prop_assert_eq!(&vns_again.cost_history, &vns.cost_history, "VNS: same seed, different run");
        prop_assert_eq!(&vns_again.best, &vns.best);
    }

    /// ALNS keeps one weight per operator, none below the floor, reports the
    /// cost of the best it holds, never lets the best-so-far rise, counts no
    /// more improvements than iterations, and reproduces under a fixed seed.
    #[test]
    fn alns_keeps_weights_above_the_floor_and_best_monotone(
        problem in tsp(3..=8usize),
        max_iterations in 1..=40usize,
        segment_length in 1..=10usize,
        seed in any::<u64>(),
    ) {
        let n = problem.n();
        let destroy = [SegmentReverse, SegmentReverse];
        let repair = [SwapRepair(problem.clone())];
        let config = AlnsConfig::default()
            .with_max_iterations(max_iterations)
            .with_segment_length(segment_length)
            .with_seed(seed);
        let result = AlnsRunner::run(&problem, &destroy, &repair, &config).unwrap();
        let again = AlnsRunner::run(&problem, &destroy, &repair, &config).unwrap();

        prop_assert!(is_permutation(&result.best, n), "ALNS best {:?}", result.best);
        prop_assert!((AlnsProblem::cost(&problem, &result.best) - result.best_cost).abs() < 1e-9, "ALNS best_cost");
        check_non_increasing(&result.cost_history, "ALNS best-so-far")?;
        prop_assert_eq!(*result.cost_history.last().unwrap(), result.best_cost);
        prop_assert!(result.iterations <= max_iterations);
        prop_assert!(result.improvements <= result.iterations, "improvements {} > iterations {}", result.improvements, result.iterations);
        prop_assert_eq!(result.destroy_weights.len(), destroy.len());
        prop_assert_eq!(result.repair_weights.len(), repair.len());
        for w in result.destroy_weights.iter().chain(&result.repair_weights) {
            prop_assert!(w.is_finite() && *w >= config.min_weight - 1e-12, "weight {w} below floor {}", config.min_weight);
        }
        prop_assert!(result.final_temperature <= config.initial_temperature + 1e-9);
        prop_assert!(result.final_temperature >= config.min_temperature - 1e-9);

        prop_assert_eq!(&again.cost_history, &result.cost_history, "ALNS: same seed, different run");
        prop_assert_eq!(&again.best, &result.best);
    }

    /// Non-dominated sorting: every point is in exactly one front and its
    /// rank names it; front 0 is the brute-force non-dominated set; a point
    /// in front r is dominated by nothing in front r and by something in
    /// front r − 1. Crowding distance: with three or more points every
    /// objective's extremes are infinite and the rest are finite and ≥ 0.
    #[test]
    fn non_dominated_sort_layers_the_population_by_pareto_dominance(
        objectives in prop::collection::vec(
            prop::collection::vec(0.0f64..10.0, 2), 1..=20
        ),
        m in 2..=3usize,
    ) {
        let objectives: Vec<Vec<f64>> = objectives
            .iter()
            .map(|o| (0..m).map(|k| o[k % 2] * (k as f64 + 1.0)).collect())
            .collect();
        let n = objectives.len();
        let sorted = non_dominated_sort(&objectives);

        prop_assert_eq!(sorted.ranks.len(), n);
        let mut seen = vec![false; n];
        for (r, front) in sorted.fronts.iter().enumerate() {
            prop_assert!(!front.is_empty(), "front {r} is empty");
            for &i in front {
                prop_assert!(i < n);
                prop_assert!(!std::mem::replace(&mut seen[i], true), "point {i} in two fronts");
                prop_assert_eq!(sorted.ranks[i], r);
            }
        }
        prop_assert!(seen.iter().all(|&s| s), "a point is in no front");

        for i in 0..n {
            let r = sorted.ranks[i];
            for j in 0..n {
                if dominates(&objectives[j], &objectives[i]) {
                    prop_assert!(sorted.ranks[j] < r, "{j} dominates {i} but rank {} ≥ {r}", sorted.ranks[j]);
                }
            }
            if r > 0 {
                let dominated_by_previous = sorted.fronts[r - 1]
                    .iter()
                    .any(|&j| dominates(&objectives[j], &objectives[i]));
                prop_assert!(dominated_by_previous, "point {i} in front {r} is dominated by nothing in front {}", r - 1);
            }
        }

        let distances = crowding_distance(&objectives);
        prop_assert_eq!(distances.len(), n);
        for (i, &d) in distances.iter().enumerate() {
            prop_assert!(d >= 0.0, "crowding distance {d} of point {i}");
        }
        if n <= 2 {
            prop_assert!(distances.iter().all(|d| d.is_infinite()));
        } else {
            for k in 0..m {
                let (lo, hi) = extremes(&objectives, k);
                prop_assert!(distances[lo].is_infinite(), "minimum of objective {k} (point {lo}) is not infinite");
                prop_assert!(distances[hi].is_infinite(), "maximum of objective {k} (point {hi}) is not infinite");
            }
        }
    }
}
