//! ALNS execution loop.

use super::config::AlnsConfig;
use super::types::{AlnsProblem, DestroyOperator, RepairOperator};
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_numerics::random::create_rng;

/// Result of an ALNS optimization run.
#[derive(Debug, Clone)]
pub struct AlnsResult<S: Clone> {
    /// The best solution found.
    pub best: S,

    /// Cost of the best solution.
    pub best_cost: f64,

    /// Total number of iterations.
    pub iterations: usize,

    /// Number of improvements found.
    pub improvements: usize,

    /// Final temperature.
    pub final_temperature: f64,

    /// Whether cancelled externally.
    pub cancelled: bool,

    /// Final destroy operator weights.
    pub destroy_weights: Vec<f64>,

    /// Final repair operator weights.
    pub repair_weights: Vec<f64>,

    /// Best cost sampled at regular intervals.
    pub cost_history: Vec<f64>,
}

/// Tracks per-operator statistics for adaptive weight updates.
#[derive(Debug, Clone)]
struct OperatorStats {
    weight: f64,
    segment_score: f64,
    segment_uses: usize,
}

impl OperatorStats {
    fn new() -> Self {
        Self {
            weight: 1.0,
            segment_score: 0.0,
            segment_uses: 0,
        }
    }

    fn record(&mut self, score: f64) {
        self.segment_score += score;
        self.segment_uses += 1;
    }

    /// Update weight using exponential smoothing at end of segment.
    ///
    /// w_new = w * (1 - rho) + rho * (pi_j / theta_j)
    ///
    /// where pi_j = accumulated score, theta_j = times used in segment.
    ///
    /// Reference: Ropke & Pisinger (2006), Equation (1)
    fn update_weight(&mut self, reaction_factor: f64, min_weight: f64) {
        if self.segment_uses > 0 {
            let avg_score = self.segment_score / self.segment_uses as f64;
            self.weight = self.weight * (1.0 - reaction_factor) + avg_score * reaction_factor;
            self.weight = self.weight.max(min_weight);
        }
        self.segment_score = 0.0;
        self.segment_uses = 0;
    }
}

/// Select an operator index using roulette wheel selection on weights.
fn roulette_select<R: Rng>(weights: &[OperatorStats], rng: &mut R) -> usize {
    let total: f64 = weights.iter().map(|s| s.weight).sum();
    if total <= 0.0 || weights.is_empty() {
        return 0;
    }

    let mut roll = rng.random_range(0.0..total);
    for (i, stat) in weights.iter().enumerate() {
        roll -= stat.weight;
        if roll <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

/// Executes the ALNS algorithm.
pub struct AlnsRunner;

impl AlnsRunner {
    /// Runs ALNS optimization.
    ///
    /// # Arguments
    /// * `problem` - The problem definition (initial solution + cost)
    /// * `destroy_ops` - Slice of destroy operators
    /// * `repair_ops` - Slice of repair operators
    /// * `config` - Algorithm configuration
    pub fn run<P, D, R>(
        problem: &P,
        destroy_ops: &[D],
        repair_ops: &[R],
        config: &AlnsConfig,
    ) -> AlnsResult<P::Solution>
    where
        P: AlnsProblem,
        D: DestroyOperator<P::Solution>,
        R: RepairOperator<P::Solution>,
    {
        Self::run_with_cancel(problem, destroy_ops, repair_ops, config, None)
    }

    /// Runs ALNS with an optional cancellation token.
    pub fn run_with_cancel<P, D, RP>(
        problem: &P,
        destroy_ops: &[D],
        repair_ops: &[RP],
        config: &AlnsConfig,
        cancel: Option<Arc<AtomicBool>>,
    ) -> AlnsResult<P::Solution>
    where
        P: AlnsProblem,
        D: DestroyOperator<P::Solution>,
        RP: RepairOperator<P::Solution>,
    {
        config.validate().expect("invalid AlnsConfig");
        assert!(
            !destroy_ops.is_empty(),
            "at least one destroy operator required"
        );
        assert!(
            !repair_ops.is_empty(),
            "at least one repair operator required"
        );

        let mut rng = match config.seed {
            Some(seed) => create_rng(seed),
            None => create_rng(rand::random()),
        };

        // Initialize
        let mut current = problem.initial_solution(&mut rng);
        let mut current_cost = problem.cost(&current);
        let mut best = current.clone();
        let mut best_cost = current_cost;

        let mut destroy_stats: Vec<OperatorStats> =
            destroy_ops.iter().map(|_| OperatorStats::new()).collect();
        let mut repair_stats: Vec<OperatorStats> =
            repair_ops.iter().map(|_| OperatorStats::new()).collect();

        let mut temperature = config.initial_temperature;
        let mut improvements = 0usize;
        let mut cancelled = false;

        // Cost history
        let history_interval = config.segment_length.max(1);
        let mut cost_history = Vec::new();
        cost_history.push(best_cost);

        for iteration in 0..config.max_iterations {
            if let Some(ref flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    cancelled = true;
                    break;
                }
            }

            // Select operators via roulette wheel
            let d_idx = roulette_select(&destroy_stats, &mut rng);
            let r_idx = roulette_select(&repair_stats, &mut rng);

            // Determine destroy degree
            let degree = rng.random_range(config.min_destroy_degree..config.max_destroy_degree);

            // Destroy then repair
            let destroyed = destroy_ops[d_idx].destroy(&current, degree, &mut rng);
            let candidate = repair_ops[r_idx].repair(&destroyed, &mut rng);
            let candidate_cost = problem.cost(&candidate);

            // Determine score and acceptance
            let (accepted, score) = if candidate_cost < best_cost {
                // New global best (sigma_1)
                best = candidate.clone();
                best_cost = candidate_cost;
                improvements += 1;
                (true, config.score_new_best)
            } else if candidate_cost < current_cost {
                // Better than current (sigma_2)
                (true, config.score_improved)
            } else {
                // SA acceptance criterion
                let delta = candidate_cost - current_cost;
                let accept_prob = if temperature > 0.0 {
                    (-delta / temperature).exp()
                } else {
                    0.0
                };
                if rng.random_range(0.0..1.0) < accept_prob {
                    (true, config.score_accepted)
                } else {
                    (false, 0.0)
                }
            };

            if accepted {
                current = candidate;
                current_cost = candidate_cost;
            }

            // Record operator usage
            destroy_stats[d_idx].record(score);
            repair_stats[r_idx].record(score);

            // Cool down
            temperature = (temperature * config.cooling_rate).max(config.min_temperature);

            // End-of-segment weight update
            if (iteration + 1) % config.segment_length == 0 {
                for stat in &mut destroy_stats {
                    stat.update_weight(config.reaction_factor, config.min_weight);
                }
                for stat in &mut repair_stats {
                    stat.update_weight(config.reaction_factor, config.min_weight);
                }
            }

            // Record history
            if (iteration + 1).is_multiple_of(history_interval) {
                cost_history.push(best_cost);
            }
        }

        // Final history entry
        if cost_history
            .last()
            .is_none_or(|&last| (last - best_cost).abs() > 1e-15)
        {
            cost_history.push(best_cost);
        }

        AlnsResult {
            best,
            best_cost,
            iterations: if cancelled {
                cost_history.len().saturating_sub(1) * history_interval
            } else {
                config.max_iterations
            },
            improvements,
            final_temperature: temperature,
            cancelled,
            destroy_weights: destroy_stats.iter().map(|s| s.weight).collect(),
            repair_weights: repair_stats.iter().map(|s| s.weight).collect(),
            cost_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alns::AlnsConfig;

    // ---- Set Cover Minimization ----
    // Solution: Vec<bool> (which elements to include)
    // Cost: negative count of true bits (minimize = maximize count)

    struct SubsetProblem {
        n: usize,
    }

    impl AlnsProblem for SubsetProblem {
        type Solution = Vec<bool>;

        fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<bool> {
            (0..self.n).map(|_| rng.random_bool(0.5)).collect()
        }

        fn cost(&self, solution: &Vec<bool>) -> f64 {
            let count = solution.iter().filter(|&&b| b).count();
            -(count as f64)
        }
    }

    // Destroy: randomly flip some true bits to false
    struct RandomDestroy;

    impl DestroyOperator<Vec<bool>> for RandomDestroy {
        fn name(&self) -> &str {
            "random"
        }

        fn destroy<R: Rng>(&self, solution: &Vec<bool>, degree: f64, rng: &mut R) -> Vec<bool> {
            let mut result = solution.clone();
            for bit in &mut result {
                if *bit && rng.random_range(0.0..1.0) < degree {
                    *bit = false;
                }
            }
            result
        }
    }

    // Destroy: flip the "worst" (biased toward more removals) bits
    struct WorstDestroy;

    impl DestroyOperator<Vec<bool>> for WorstDestroy {
        fn name(&self) -> &str {
            "worst"
        }

        fn destroy<R: Rng>(&self, solution: &Vec<bool>, degree: f64, rng: &mut R) -> Vec<bool> {
            let mut result = solution.clone();
            let n_remove = (result.len() as f64 * degree).ceil() as usize;
            let mut removed = 0;
            for bit in &mut result {
                if *bit && removed < n_remove && rng.random_range(0.0..1.0) < 0.7 {
                    *bit = false;
                    removed += 1;
                }
            }
            result
        }
    }

    // Repair: randomly flip some false bits to true
    struct GreedyRepair;

    impl RepairOperator<Vec<bool>> for GreedyRepair {
        fn name(&self) -> &str {
            "greedy"
        }

        fn repair<R: Rng>(&self, solution: &Vec<bool>, rng: &mut R) -> Vec<bool> {
            let mut result = solution.clone();
            for bit in &mut result {
                if !*bit && rng.random_range(0.0..1.0) < 0.6 {
                    *bit = true;
                }
            }
            result
        }
    }

    // Repair: flip all false to true
    struct FullRepair;

    impl RepairOperator<Vec<bool>> for FullRepair {
        fn name(&self) -> &str {
            "full"
        }

        fn repair<R: Rng>(&self, solution: &Vec<bool>, _rng: &mut R) -> Vec<bool> {
            let mut result = solution.clone();
            result.fill(true);
            result
        }
    }

    // Enum dispatch for multiple destroy operators
    enum TestDestroy {
        Random(RandomDestroy),
        Worst(WorstDestroy),
    }

    impl DestroyOperator<Vec<bool>> for TestDestroy {
        fn name(&self) -> &str {
            match self {
                TestDestroy::Random(d) => d.name(),
                TestDestroy::Worst(d) => d.name(),
            }
        }

        fn destroy<R: Rng>(&self, solution: &Vec<bool>, degree: f64, rng: &mut R) -> Vec<bool> {
            match self {
                TestDestroy::Random(d) => d.destroy(solution, degree, rng),
                TestDestroy::Worst(d) => d.destroy(solution, degree, rng),
            }
        }
    }

    // Enum dispatch for multiple repair operators
    enum TestRepair {
        Greedy(GreedyRepair),
        Full(FullRepair),
    }

    impl RepairOperator<Vec<bool>> for TestRepair {
        fn name(&self) -> &str {
            match self {
                TestRepair::Greedy(r) => r.name(),
                TestRepair::Full(r) => r.name(),
            }
        }

        fn repair<R: Rng>(&self, solution: &Vec<bool>, rng: &mut R) -> Vec<bool> {
            match self {
                TestRepair::Greedy(r) => r.repair(solution, rng),
                TestRepair::Full(r) => r.repair(solution, rng),
            }
        }
    }

    #[test]
    fn test_alns_basic() {
        let problem = SubsetProblem { n: 20 };
        let destroy_ops = [
            TestDestroy::Random(RandomDestroy),
            TestDestroy::Worst(WorstDestroy),
        ];
        let repair_ops = [
            TestRepair::Greedy(GreedyRepair),
            TestRepair::Full(FullRepair),
        ];

        let config = AlnsConfig::default().with_max_iterations(500).with_seed(42);

        let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &config);

        // With FullRepair available, should find all-true (cost = -20)
        assert!(
            result.best_cost <= -15.0,
            "expected cost <= -15, got {}",
            result.best_cost
        );
        assert_eq!(result.iterations, 500);
    }

    #[test]
    fn test_alns_weight_adaptation() {
        let problem = SubsetProblem { n: 20 };
        let destroy_ops = [
            TestDestroy::Random(RandomDestroy),
            TestDestroy::Worst(WorstDestroy),
        ];
        let repair_ops = [
            TestRepair::Greedy(GreedyRepair),
            TestRepair::Full(FullRepair),
        ];

        let config = AlnsConfig::default()
            .with_max_iterations(500)
            .with_segment_length(50)
            .with_reaction_factor(0.5)
            .with_seed(42);

        let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &config);

        assert_eq!(result.destroy_weights.len(), 2);
        assert_eq!(result.repair_weights.len(), 2);

        for &w in &result.destroy_weights {
            assert!(w >= config.min_weight, "weight {w} below min");
        }
        for &w in &result.repair_weights {
            assert!(w >= config.min_weight, "weight {w} below min");
        }
    }

    #[test]
    fn test_alns_cancellation() {
        let problem = SubsetProblem { n: 20 };
        let destroy_ops = [RandomDestroy];
        let repair_ops = [GreedyRepair];

        let config = AlnsConfig::default()
            .with_max_iterations(1_000_000)
            .with_seed(42);

        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            cancel_clone.store(true, Ordering::Relaxed);
        });

        let result =
            AlnsRunner::run_with_cancel(&problem, &destroy_ops, &repair_ops, &config, Some(cancel));
        assert!(result.cancelled);
    }

    #[test]
    fn test_alns_cost_history_non_increasing() {
        let problem = SubsetProblem { n: 10 };
        let destroy_ops = [RandomDestroy];
        let repair_ops = [GreedyRepair];

        let config = AlnsConfig::default()
            .with_max_iterations(500)
            .with_segment_length(50)
            .with_seed(42);

        let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &config);

        for window in result.cost_history.windows(2) {
            assert!(
                window[1] <= window[0] + 1e-10,
                "best cost history should be non-increasing: {} > {}",
                window[1],
                window[0]
            );
        }
    }

    #[test]
    fn test_alns_improvements_counted() {
        let problem = SubsetProblem { n: 10 };
        let destroy_ops = [RandomDestroy];
        let repair_ops = [FullRepair];

        let config = AlnsConfig::default().with_max_iterations(200).with_seed(42);

        let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &config);

        assert!(result.improvements > 0, "expected at least one improvement");
    }

    // ---- Continuous minimization: f(x) = sum(x_i^2) ----

    struct ContinuousProblem {
        n: usize,
    }

    impl AlnsProblem for ContinuousProblem {
        type Solution = Vec<f64>;

        fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
            (0..self.n).map(|_| rng.random_range(-10.0..10.0)).collect()
        }

        fn cost(&self, x: &Vec<f64>) -> f64 {
            x.iter().map(|v| v * v).sum()
        }
    }

    struct PerturbDestroy;

    impl DestroyOperator<Vec<f64>> for PerturbDestroy {
        fn name(&self) -> &str {
            "perturb"
        }

        fn destroy<R: Rng>(&self, solution: &Vec<f64>, degree: f64, rng: &mut R) -> Vec<f64> {
            solution
                .iter()
                .map(|&v| {
                    if rng.random_range(0.0..1.0) < degree {
                        v + rng.random_range(-3.0..3.0)
                    } else {
                        v
                    }
                })
                .collect()
        }
    }

    struct IdentityRepair;

    impl RepairOperator<Vec<f64>> for IdentityRepair {
        fn name(&self) -> &str {
            "identity"
        }

        fn repair<R: Rng>(&self, solution: &Vec<f64>, _rng: &mut R) -> Vec<f64> {
            solution.clone()
        }
    }

    #[test]
    fn test_alns_continuous() {
        let problem = ContinuousProblem { n: 5 };
        let destroy_ops = [PerturbDestroy];
        let repair_ops = [IdentityRepair];

        let config = AlnsConfig::default()
            .with_max_iterations(5000)
            .with_temperature(100.0, 0.999, 0.001)
            .with_destroy_degree(0.3, 0.8)
            .with_seed(42);

        let result = AlnsRunner::run(&problem, &destroy_ops, &repair_ops, &config);

        assert!(
            result.best_cost < 10.0,
            "expected cost < 10, got {}",
            result.best_cost
        );
    }
}
