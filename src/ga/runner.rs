//! GA evolutionary loop execution.
//!
//! [`GaRunner`] orchestrates the complete evolutionary process:
//! initialization → evaluation → selection → crossover → mutation → repeat.

use super::config::GaConfig;
use super::types::{Fitness, GaProblem, Individual};
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_optim::random::create_rng;

/// Result of a GA optimization run.
///
/// Contains the best solution found, along with statistics about the
/// evolutionary process.
#[derive(Debug, Clone)]
pub struct GaResult<I: Individual> {
    /// The best individual found during the entire run.
    pub best: I,

    /// Best fitness value (same as `best.fitness()`).
    pub best_fitness: I::Fitness,

    /// Total number of generations executed.
    pub generations: usize,

    /// Whether the run was terminated due to stagnation.
    pub stagnated: bool,

    /// Whether the run was cancelled externally.
    pub cancelled: bool,

    /// Best fitness at the end of each generation.
    pub fitness_history: Vec<f64>,
}

/// Executes the GA evolutionary loop.
///
/// # Usage
///
/// ```ignore
/// let problem = MyProblem::new();
/// let config = GaConfig::default().with_seed(42);
/// let result = GaRunner::run(&problem, &config);
/// println!("Best fitness: {:?}", result.best_fitness);
/// ```
pub struct GaRunner;

impl GaRunner {
    /// Runs the GA optimization.
    ///
    /// # Panics
    /// Panics if the configuration is invalid (call [`GaConfig::validate`] first
    /// to get a descriptive error).
    pub fn run<P: GaProblem>(problem: &P, config: &GaConfig) -> GaResult<P::Individual> {
        Self::run_with_cancel(problem, config, None)
    }

    /// Runs the GA with an optional cancellation token.
    ///
    /// If `cancel` is `Some` and the flag is set to `true`, the GA will
    /// stop at the end of the current generation and return the best
    /// solution found so far.
    pub fn run_with_cancel<P: GaProblem>(
        problem: &P,
        config: &GaConfig,
        cancel: Option<Arc<AtomicBool>>,
    ) -> GaResult<P::Individual> {
        config.validate().expect("invalid GaConfig");

        let mut rng = match config.seed {
            Some(seed) => create_rng(seed),
            None => create_rng(rand::random()),
        };

        // 1. Initialize population
        let mut population: Vec<P::Individual> = (0..config.population_size)
            .map(|_| problem.create_individual(&mut rng))
            .collect();

        // 2. Evaluate initial population
        evaluate_population(problem, &mut population, config.parallel);

        // 3. Track best
        let mut best = find_best(&population).clone();
        let mut fitness_history = Vec::with_capacity(config.max_generations);
        fitness_history.push(best.fitness().to_f64());

        let mut stagnation_counter = 0usize;
        let mut cancelled = false;

        // 4. Evolutionary loop
        for gen in 0..config.max_generations {
            // Check cancellation
            if let Some(ref flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    cancelled = true;
                    break;
                }
            }

            // Sort population by fitness (ascending = best first)
            population.sort_by(|a, b| {
                a.fitness()
                    .partial_cmp(&b.fitness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Elite preservation
            let elite_count =
                (config.population_size as f64 * config.elite_ratio) as usize;
            let mut next_gen: Vec<P::Individual> =
                population[..elite_count].to_vec();

            // Generate offspring
            while next_gen.len() < config.population_size {
                // Selection
                let p1_idx = config.selection.select(&population, &mut rng);
                let p2_idx = config.selection.select(&population, &mut rng);

                // Crossover
                let children = if rng.random_range(0.0..1.0) < config.crossover_rate {
                    problem.crossover(&population[p1_idx], &population[p2_idx], &mut rng)
                } else {
                    vec![population[p1_idx].clone()]
                };

                for mut child in children {
                    if next_gen.len() >= config.population_size {
                        break;
                    }

                    // Mutation
                    if rng.random_range(0.0..1.0) < config.mutation_rate {
                        problem.mutate(&mut child, &mut rng);
                    }

                    next_gen.push(child);
                }
            }

            // Evaluate new individuals (skip elites, they're already evaluated)
            if config.parallel {
                next_gen[elite_count..]
                    .par_iter_mut()
                    .for_each(|ind| {
                        let f = problem.evaluate(ind);
                        ind.set_fitness(f);
                    });
            } else {
                for ind in &mut next_gen[elite_count..] {
                    let f = problem.evaluate(ind);
                    ind.set_fitness(f);
                }
            }

            population = next_gen;

            // Update best
            let gen_best = find_best(&population);
            if gen_best.fitness() < best.fitness() {
                best = gen_best.clone();
                stagnation_counter = 0;
            } else {
                stagnation_counter += 1;
            }

            fitness_history.push(best.fitness().to_f64());

            // Callback
            problem.on_generation(gen + 1, best.fitness());

            // Stagnation check
            if config.stagnation_limit > 0 && stagnation_counter >= config.stagnation_limit {
                return GaResult {
                    best_fitness: best.fitness(),
                    best,
                    generations: gen + 1,
                    stagnated: true,
                    cancelled: false,
                    fitness_history,
                };
            }
        }

        GaResult {
            best_fitness: best.fitness(),
            best,
            generations: if cancelled {
                fitness_history.len().saturating_sub(1)
            } else {
                config.max_generations
            },
            stagnated: false,
            cancelled,
            fitness_history,
        }
    }
}

/// Evaluate all individuals in the population.
fn evaluate_population<P: GaProblem>(
    problem: &P,
    population: &mut [P::Individual],
    parallel: bool,
) {
    if parallel {
        population.par_iter_mut().for_each(|ind| {
            let f = problem.evaluate(ind);
            ind.set_fitness(f);
        });
    } else {
        for ind in population.iter_mut() {
            let f = problem.evaluate(ind);
            ind.set_fitness(f);
        }
    }
}

/// Find the individual with the best (lowest) fitness.
fn find_best<I: Individual>(population: &[I]) -> &I {
    population
        .iter()
        .min_by(|a, b| {
            a.fitness()
                .partial_cmp(&b.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("population must not be empty")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ga::{GaConfig, Selection};

    // ---- OneMax problem: maximize sum of bits (minimize negative sum) ----

    #[derive(Clone, Debug)]
    struct BitString {
        bits: Vec<bool>,
        fitness: f64,
    }

    impl Individual for BitString {
        type Fitness = f64;
        fn fitness(&self) -> f64 {
            self.fitness
        }
        fn set_fitness(&mut self, f: f64) {
            self.fitness = f;
        }
    }

    struct OneMaxProblem {
        n: usize,
    }

    impl GaProblem for OneMaxProblem {
        type Individual = BitString;

        fn create_individual<R: Rng>(&self, rng: &mut R) -> BitString {
            let bits: Vec<bool> = (0..self.n).map(|_| rng.random_bool(0.5)).collect();
            BitString {
                bits,
                fitness: f64::INFINITY,
            }
        }

        fn evaluate(&self, ind: &BitString) -> f64 {
            // Minimize negative count of true bits
            -(ind.bits.iter().filter(|&&b| b).count() as f64)
        }

        fn crossover<R: Rng>(
            &self,
            p1: &BitString,
            p2: &BitString,
            rng: &mut R,
        ) -> Vec<BitString> {
            // Single-point crossover
            let point = rng.random_range(0..self.n);
            let mut c1_bits = Vec::with_capacity(self.n);
            let mut c2_bits = Vec::with_capacity(self.n);
            for i in 0..self.n {
                if i < point {
                    c1_bits.push(p1.bits[i]);
                    c2_bits.push(p2.bits[i]);
                } else {
                    c1_bits.push(p2.bits[i]);
                    c2_bits.push(p1.bits[i]);
                }
            }
            vec![
                BitString {
                    bits: c1_bits,
                    fitness: f64::INFINITY,
                },
                BitString {
                    bits: c2_bits,
                    fitness: f64::INFINITY,
                },
            ]
        }

        fn mutate<R: Rng>(&self, ind: &mut BitString, rng: &mut R) {
            // Flip one random bit
            let idx = rng.random_range(0..self.n);
            ind.bits[idx] = !ind.bits[idx];
        }
    }

    #[test]
    fn test_onemax_convergence() {
        let problem = OneMaxProblem { n: 20 };
        let config = GaConfig::default()
            .with_population_size(50)
            .with_max_generations(200)
            .with_mutation_rate(0.3)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config);

        // Should find near-optimal solution (all true = fitness -20)
        assert!(
            result.best_fitness <= -15.0,
            "expected fitness <= -15.0 for 20-bit OneMax, got {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_stagnation_termination() {
        let problem = OneMaxProblem { n: 5 };
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(1000)
            .with_stagnation_limit(10)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config);

        // Should stop early due to convergence
        assert!(
            result.stagnated || result.generations < 1000,
            "expected stagnation or early stop"
        );
    }

    #[test]
    fn test_cancellation() {
        let problem = OneMaxProblem { n: 20 };
        let config = GaConfig::default()
            .with_population_size(50)
            .with_max_generations(10000)
            .with_stagnation_limit(0) // disable stagnation
            .with_seed(42)
            .with_parallel(false);

        let cancel = Arc::new(AtomicBool::new(false));

        // Cancel after a few generations
        let cancel_clone = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            cancel_clone.store(true, Ordering::Relaxed);
        });

        let result = GaRunner::run_with_cancel(&problem, &config, Some(cancel));

        assert!(result.cancelled, "expected cancelled result");
        assert!(result.generations < 10000, "should have stopped early");
    }

    #[test]
    fn test_elite_preservation() {
        let problem = OneMaxProblem { n: 10 };
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(50)
            .with_elite_ratio(0.2)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config);

        // Fitness should never get worse across generations
        for window in result.fitness_history.windows(2) {
            assert!(
                window[1] <= window[0],
                "fitness should be monotonically non-increasing with elitism: {} > {}",
                window[1],
                window[0]
            );
        }
    }

    #[test]
    fn test_fitness_history() {
        let problem = OneMaxProblem { n: 10 };
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(30)
            .with_stagnation_limit(0)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config);

        // History should have max_generations + 1 entries (initial + each gen)
        assert_eq!(result.fitness_history.len(), 31);
    }

    #[test]
    fn test_all_selection_strategies() {
        let problem = OneMaxProblem { n: 10 };

        for selection in [
            Selection::Tournament(3),
            Selection::Roulette,
            Selection::Rank,
        ] {
            let config = GaConfig::default()
                .with_population_size(30)
                .with_max_generations(50)
                .with_selection(selection)
                .with_seed(42)
                .with_parallel(false);

            let result = GaRunner::run(&problem, &config);

            assert!(
                result.best_fitness < 0.0,
                "selection {:?} should find some true bits, got fitness {}",
                selection,
                result.best_fitness
            );
        }
    }

    #[test]
    fn test_parallel_gives_same_quality() {
        let problem = OneMaxProblem { n: 20 };

        // Note: parallel results may differ from sequential due to
        // evaluation order, but quality should be similar.
        let config = GaConfig::default()
            .with_population_size(50)
            .with_max_generations(100)
            .with_seed(42)
            .with_parallel(true);

        let result = GaRunner::run(&problem, &config);

        assert!(
            result.best_fitness <= -10.0,
            "parallel should find reasonable solution, got {}",
            result.best_fitness
        );
    }

    // ---- Continuous optimization: sphere function ----

    #[derive(Clone, Debug)]
    struct RealVector {
        genes: Vec<f64>,
        fitness: f64,
    }

    impl Individual for RealVector {
        type Fitness = f64;
        fn fitness(&self) -> f64 {
            self.fitness
        }
        fn set_fitness(&mut self, f: f64) {
            self.fitness = f;
        }
    }

    struct SphereProblem {
        dim: usize,
    }

    impl GaProblem for SphereProblem {
        type Individual = RealVector;

        fn create_individual<R: Rng>(&self, rng: &mut R) -> RealVector {
            let genes: Vec<f64> = (0..self.dim)
                .map(|_| rng.random_range(-5.0..5.0))
                .collect();
            RealVector {
                genes,
                fitness: f64::INFINITY,
            }
        }

        fn evaluate(&self, ind: &RealVector) -> f64 {
            // f(x) = sum(x_i^2), minimum at origin
            ind.genes.iter().map(|x| x * x).sum()
        }

        fn crossover<R: Rng>(
            &self,
            p1: &RealVector,
            p2: &RealVector,
            rng: &mut R,
        ) -> Vec<RealVector> {
            // BLX-alpha crossover
            let alpha = 0.5;
            let genes: Vec<f64> = p1
                .genes
                .iter()
                .zip(p2.genes.iter())
                .map(|(&a, &b)| {
                    let lo = a.min(b);
                    let hi = a.max(b);
                    let range = hi - lo;
                    if range < 1e-15 {
                        lo // genes are identical, no crossover needed
                    } else {
                        rng.random_range((lo - alpha * range)..(hi + alpha * range))
                    }
                })
                .collect();
            vec![RealVector {
                genes,
                fitness: f64::INFINITY,
            }]
        }

        fn mutate<R: Rng>(&self, ind: &mut RealVector, rng: &mut R) {
            // Gaussian mutation on one gene
            let idx = rng.random_range(0..self.dim);
            let perturbation: f64 = rng.random_range(-0.5..0.5);
            ind.genes[idx] += perturbation;
        }
    }

    #[test]
    fn test_sphere_optimization() {
        let problem = SphereProblem { dim: 5 };
        let config = GaConfig::default()
            .with_population_size(100)
            .with_max_generations(300)
            .with_mutation_rate(0.3)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config);

        // Should get close to 0 (the global minimum)
        assert!(
            result.best_fitness < 1.0,
            "expected fitness < 1.0 for 5D sphere, got {}",
            result.best_fitness
        );
    }

    // ---- Default crossover/mutate (no-op) ----

    struct NoOpProblem;

    impl GaProblem for NoOpProblem {
        type Individual = RealVector;

        fn create_individual<R: Rng>(&self, rng: &mut R) -> RealVector {
            let genes = vec![rng.random_range(-10.0..10.0)];
            RealVector {
                genes,
                fitness: f64::INFINITY,
            }
        }

        fn evaluate(&self, ind: &RealVector) -> f64 {
            ind.genes[0].abs()
        }
        // Uses default crossover (clone) and mutate (no-op)
    }

    #[test]
    fn test_default_operators() {
        let problem = NoOpProblem;
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config);

        // Should complete without error
        assert!(result.generations > 0);
        assert!(!result.fitness_history.is_empty());
    }
}
