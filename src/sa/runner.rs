//! SA execution loop.

use super::config::{CoolingSchedule, SaConfig};
use super::types::SaProblem;
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_numflow::random::create_rng;

/// Result of a Simulated Annealing run.
#[derive(Debug, Clone)]
pub struct SaResult<S: Clone> {
    /// The best solution found.
    pub best: S,

    /// Cost of the best solution.
    pub best_cost: f64,

    /// Total number of iterations (neighbor evaluations).
    pub iterations: usize,

    /// Final temperature when the algorithm stopped.
    pub final_temperature: f64,

    /// Number of accepted moves (including improvements).
    pub accepted_moves: usize,

    /// Number of improving moves.
    pub improving_moves: usize,

    /// Whether cancelled externally.
    pub cancelled: bool,

    /// Best cost sampled at regular intervals for history tracking.
    pub cost_history: Vec<f64>,
}

/// Executes the Simulated Annealing algorithm.
pub struct SaRunner;

impl SaRunner {
    /// Runs SA optimization.
    pub fn run<P: SaProblem>(problem: &P, config: &SaConfig) -> SaResult<P::Solution> {
        Self::run_with_cancel(problem, config, None)
    }

    /// Runs SA with an optional cancellation token.
    pub fn run_with_cancel<P: SaProblem>(
        problem: &P,
        config: &SaConfig,
        cancel: Option<Arc<AtomicBool>>,
    ) -> SaResult<P::Solution> {
        config.validate().expect("invalid SaConfig");

        let mut rng = match config.seed {
            Some(seed) => create_rng(seed),
            None => create_rng(rand::random()),
        };

        // Initialize
        let mut current = problem.initial_solution(&mut rng);
        let mut current_cost = problem.cost(&current);
        let mut best = current.clone();
        let mut best_cost = current_cost;

        let mut temperature = config.initial_temperature;
        let mut total_iterations = 0usize;
        let mut accepted_moves = 0usize;
        let mut improving_moves = 0usize;
        let mut cancelled = false;

        // For linear cooling: compute step count
        let linear_max_steps = compute_linear_steps(config);

        // Cost history: sample every N iterations
        let history_interval = 100.max(config.iterations_per_temperature);
        let mut cost_history = Vec::new();
        cost_history.push(best_cost);

        let mut step = 0usize; // temperature step counter

        while temperature > config.min_temperature {
            if let Some(ref flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    cancelled = true;
                    break;
                }
            }

            let inner_iters = match config.cooling {
                CoolingSchedule::LundyMees { .. } => 1,
                _ => config.iterations_per_temperature,
            };

            for _ in 0..inner_iters {
                if config.max_iterations > 0 && total_iterations >= config.max_iterations {
                    break;
                }

                let neighbor = problem.neighbor(&current, &mut rng);
                let neighbor_cost = problem.cost(&neighbor);
                let delta = neighbor_cost - current_cost;

                // Metropolis acceptance criterion
                let accept = if delta < 0.0 {
                    improving_moves += 1;
                    true
                } else if temperature > 0.0 {
                    let probability = (-delta / temperature).exp();
                    rng.random_range(0.0..1.0) < probability
                } else {
                    false
                };

                if accept {
                    current = neighbor;
                    current_cost = neighbor_cost;
                    accepted_moves += 1;

                    if current_cost < best_cost {
                        best = current.clone();
                        best_cost = current_cost;
                    }
                }

                total_iterations += 1;

                // Record history
                if total_iterations.is_multiple_of(history_interval) {
                    cost_history.push(best_cost);
                }
            }

            // Check hard iteration limit
            if config.max_iterations > 0 && total_iterations >= config.max_iterations {
                break;
            }

            // Cool down
            temperature = cool(temperature, config, step, linear_max_steps);
            step += 1;
        }

        // Final history entry
        if cost_history
            .last()
            .is_none_or(|&last| (last - best_cost).abs() > 1e-15)
        {
            cost_history.push(best_cost);
        }

        SaResult {
            best,
            best_cost,
            iterations: total_iterations,
            final_temperature: temperature,
            accepted_moves,
            improving_moves,
            cancelled,
            cost_history,
        }
    }
}

/// Apply the cooling schedule to compute the next temperature.
fn cool(temperature: f64, config: &SaConfig, step: usize, linear_max_steps: usize) -> f64 {
    match config.cooling {
        CoolingSchedule::Geometric { alpha } => temperature * alpha,

        CoolingSchedule::Linear => {
            if linear_max_steps == 0 {
                config.min_temperature
            } else {
                let t = config.initial_temperature
                    - (step + 1) as f64 * (config.initial_temperature - config.min_temperature)
                        / linear_max_steps as f64;
                t.max(config.min_temperature)
            }
        }

        CoolingSchedule::LundyMees { beta } => temperature / (1.0 + beta * temperature),
    }
}

/// Estimate the number of temperature steps for linear cooling.
fn compute_linear_steps(config: &SaConfig) -> usize {
    match config.cooling {
        CoolingSchedule::Linear => {
            if config.max_iterations > 0 && config.iterations_per_temperature > 0 {
                config.max_iterations / config.iterations_per_temperature
            } else {
                1000 // reasonable default
            }
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sa::{CoolingSchedule, SaConfig};

    // ---- Quadratic minimization: f(x) = x^2, minimum at 0 ----

    struct QuadraticProblem;

    impl SaProblem for QuadraticProblem {
        type Solution = f64;

        fn initial_solution<R: Rng>(&self, rng: &mut R) -> f64 {
            rng.random_range(-10.0..10.0)
        }

        fn cost(&self, x: &f64) -> f64 {
            x * x
        }

        fn neighbor<R: Rng>(&self, x: &f64, rng: &mut R) -> f64 {
            x + rng.random_range(-1.0..1.0)
        }
    }

    #[test]
    fn test_sa_quadratic_geometric() {
        let problem = QuadraticProblem;
        let config = SaConfig::default()
            .with_initial_temperature(100.0)
            .with_min_temperature(0.001)
            .with_cooling(CoolingSchedule::Geometric { alpha: 0.95 })
            .with_iterations_per_temperature(50)
            .with_seed(42);

        let result = SaRunner::run(&problem, &config);

        assert!(
            result.best_cost < 1.0,
            "expected near-zero cost, got {}",
            result.best_cost
        );
        assert!(result.improving_moves > 0);
        assert!(result.accepted_moves > result.improving_moves);
    }

    #[test]
    fn test_sa_quadratic_linear() {
        let problem = QuadraticProblem;
        let config = SaConfig::default()
            .with_initial_temperature(100.0)
            .with_min_temperature(0.001)
            .with_cooling(CoolingSchedule::Linear)
            .with_iterations_per_temperature(50)
            .with_max_iterations(50000)
            .with_seed(42);

        let result = SaRunner::run(&problem, &config);

        assert!(
            result.best_cost < 1.0,
            "expected near-zero cost, got {}",
            result.best_cost
        );
    }

    #[test]
    fn test_sa_quadratic_lundy_mees() {
        let t0 = 100.0;
        let t_min = 0.001;
        let max_iter = 50000;
        let beta = (t0 - t_min) / (max_iter as f64 * t0 * t_min);

        let problem = QuadraticProblem;
        let config = SaConfig::default()
            .with_initial_temperature(t0)
            .with_min_temperature(t_min)
            .with_cooling(CoolingSchedule::LundyMees { beta })
            .with_max_iterations(max_iter)
            .with_seed(42);

        let result = SaRunner::run(&problem, &config);

        assert!(
            result.best_cost < 1.0,
            "expected near-zero cost, got {}",
            result.best_cost
        );
    }

    #[test]
    fn test_sa_max_iterations_limit() {
        let problem = QuadraticProblem;
        let config = SaConfig::default()
            .with_initial_temperature(1e10)
            .with_min_temperature(1e-15)
            .with_iterations_per_temperature(10)
            .with_max_iterations(100)
            .with_seed(42);

        let result = SaRunner::run(&problem, &config);

        assert!(
            result.iterations <= 100,
            "expected <= 100 iterations, got {}",
            result.iterations
        );
    }

    #[test]
    fn test_sa_cancellation() {
        let problem = QuadraticProblem;
        let config = SaConfig::default()
            .with_initial_temperature(1e10)
            .with_min_temperature(1e-15)
            .with_iterations_per_temperature(100)
            .with_seed(42);

        // Set cancel flag before running — ensures deterministic cancellation
        // regardless of how fast the solver completes.
        let cancel = Arc::new(AtomicBool::new(true));

        let result = SaRunner::run_with_cancel(&problem, &config, Some(cancel));
        assert!(result.cancelled);
    }

    #[test]
    fn test_sa_cost_history_non_increasing() {
        let problem = QuadraticProblem;
        let config = SaConfig::default()
            .with_initial_temperature(50.0)
            .with_min_temperature(0.01)
            .with_cooling(CoolingSchedule::Geometric { alpha: 0.95 })
            .with_iterations_per_temperature(100)
            .with_seed(42);

        let result = SaRunner::run(&problem, &config);

        for window in result.cost_history.windows(2) {
            assert!(
                window[1] <= window[0] + 1e-10,
                "best cost history should be non-increasing: {} > {}",
                window[1],
                window[0]
            );
        }
    }

    // ---- Discrete: permutation sorting ----

    struct PermSortProblem {
        n: usize,
    }

    impl SaProblem for PermSortProblem {
        type Solution = Vec<usize>;

        fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
            let mut perm: Vec<usize> = (0..self.n).collect();
            u_numflow::random::shuffle(&mut perm, rng);
            perm
        }

        fn cost(&self, perm: &Vec<usize>) -> f64 {
            // Number of elements not in their correct position
            perm.iter().enumerate().filter(|&(i, &v)| i != v).count() as f64
        }

        fn neighbor<R: Rng>(&self, perm: &Vec<usize>, rng: &mut R) -> Vec<usize> {
            let mut new = perm.clone();
            let i = rng.random_range(0..self.n);
            let j = rng.random_range(0..self.n);
            new.swap(i, j);
            new
        }
    }

    #[test]
    fn test_sa_permutation_sort() {
        let problem = PermSortProblem { n: 10 };
        let config = SaConfig::default()
            .with_initial_temperature(50.0)
            .with_min_temperature(0.01)
            .with_cooling(CoolingSchedule::Geometric { alpha: 0.98 })
            .with_iterations_per_temperature(200)
            .with_seed(42);

        let result = SaRunner::run(&problem, &config);

        assert!(
            result.best_cost <= 4.0,
            "expected near-sorted permutation, got cost {}",
            result.best_cost
        );
    }

    #[test]
    fn test_sa_metropolis_accepts_uphill() {
        // At very high temperature, almost all moves should be accepted
        let problem = QuadraticProblem;
        let config = SaConfig::default()
            .with_initial_temperature(1e8)
            .with_min_temperature(1e7) // stay at very high temp
            .with_cooling(CoolingSchedule::Geometric { alpha: 0.99 })
            .with_iterations_per_temperature(1000)
            .with_seed(42);

        let result = SaRunner::run(&problem, &config);

        // At extreme temperature, acceptance ratio should be very high
        let acceptance_ratio = result.accepted_moves as f64 / result.iterations as f64;
        assert!(
            acceptance_ratio > 0.8,
            "expected high acceptance at high temp, got {acceptance_ratio}"
        );
    }
}
