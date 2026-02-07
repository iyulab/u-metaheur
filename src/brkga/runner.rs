//! BRKGA evolutionary loop.

use super::config::BrkgaConfig;
use super::types::BrkgaDecoder;
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_optim::random::create_rng;

/// A chromosome in the BRKGA population.
#[derive(Debug, Clone)]
struct Chromosome {
    keys: Vec<f64>,
    cost: f64,
}

/// Result of a BRKGA optimization run.
#[derive(Debug, Clone)]
pub struct BrkgaResult {
    /// The best random-key chromosome found.
    pub best_keys: Vec<f64>,

    /// Cost of the best solution.
    pub best_cost: f64,

    /// Number of generations executed.
    pub generations: usize,

    /// Whether terminated due to stagnation.
    pub stagnated: bool,

    /// Whether cancelled externally.
    pub cancelled: bool,

    /// Best cost at the end of each generation.
    pub cost_history: Vec<f64>,
}

/// Executes the BRKGA algorithm.
pub struct BrkgaRunner;

impl BrkgaRunner {
    /// Runs BRKGA optimization.
    pub fn run<D: BrkgaDecoder>(decoder: &D, config: &BrkgaConfig) -> BrkgaResult {
        Self::run_with_cancel(decoder, config, None)
    }

    /// Runs BRKGA with an optional cancellation token.
    pub fn run_with_cancel<D: BrkgaDecoder>(
        decoder: &D,
        config: &BrkgaConfig,
        cancel: Option<Arc<AtomicBool>>,
    ) -> BrkgaResult {
        config.validate().expect("invalid BrkgaConfig");

        let mut rng = match config.seed {
            Some(seed) => create_rng(seed),
            None => create_rng(rand::random()),
        };

        let n = config.chromosome_length;
        let pop_size = config.population_size;
        let elite_count = (pop_size as f64 * config.elite_fraction) as usize;
        let mutant_count = (pop_size as f64 * config.mutant_fraction) as usize;
        let crossover_count = pop_size - elite_count - mutant_count;

        // Initialize population
        let mut population: Vec<Chromosome> = (0..pop_size)
            .map(|_| {
                let keys = match decoder.seed_chromosome(&mut rng) {
                    Some(k) if k.len() == n => k,
                    _ => (0..n).map(|_| rng.random_range(0.0..1.0)).collect(),
                };
                Chromosome {
                    keys,
                    cost: f64::INFINITY,
                }
            })
            .collect();

        // Evaluate initial population
        decode_population(decoder, &mut population, config.parallel);

        // Sort by cost (ascending)
        population.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));

        let mut best = population[0].clone();
        let mut cost_history = Vec::with_capacity(config.max_generations);
        cost_history.push(best.cost);

        let mut stagnation_counter = 0usize;
        let mut cancelled = false;

        // Evolutionary loop
        for _gen in 0..config.max_generations {
            if let Some(ref flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    cancelled = true;
                    break;
                }
            }

            let mut next_gen: Vec<Chromosome> = Vec::with_capacity(pop_size);

            // Phase 1: Elite copy
            for chr in population.iter().take(elite_count) {
                next_gen.push(chr.clone());
            }

            // Phase 2: Mutant injection
            for _ in 0..mutant_count {
                let keys: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..1.0)).collect();
                next_gen.push(Chromosome {
                    keys,
                    cost: f64::INFINITY,
                });
            }

            // Phase 3: Biased uniform crossover
            for _ in 0..crossover_count {
                // One parent from elite, one from non-elite
                let elite_idx = rng.random_range(0..elite_count);
                let nonelite_idx = rng.random_range(elite_count..pop_size);

                let keys: Vec<f64> = (0..n)
                    .map(|j| {
                        if rng.random_range(0.0..1.0) < config.elite_inheritance_prob {
                            population[elite_idx].keys[j]
                        } else {
                            population[nonelite_idx].keys[j]
                        }
                    })
                    .collect();

                next_gen.push(Chromosome {
                    keys,
                    cost: f64::INFINITY,
                });
            }

            // Decode non-elite individuals
            let non_elite_slice = &mut next_gen[elite_count..];
            if config.parallel {
                non_elite_slice.par_iter_mut().for_each(|chr| {
                    chr.cost = decoder.decode(&chr.keys);
                });
            } else {
                for chr in non_elite_slice.iter_mut() {
                    chr.cost = decoder.decode(&chr.keys);
                }
            }

            // Sort
            next_gen.sort_by(|a, b| {
                a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal)
            });

            population = next_gen;

            // Update best
            if population[0].cost < best.cost {
                best = population[0].clone();
                stagnation_counter = 0;
            } else {
                stagnation_counter += 1;
            }

            cost_history.push(best.cost);

            // Stagnation check
            if config.stagnation_limit > 0 && stagnation_counter >= config.stagnation_limit {
                return BrkgaResult {
                    best_keys: best.keys,
                    best_cost: best.cost,
                    generations: cost_history.len() - 1,
                    stagnated: true,
                    cancelled: false,
                    cost_history,
                };
            }
        }

        BrkgaResult {
            best_keys: best.keys,
            best_cost: best.cost,
            generations: if cancelled {
                cost_history.len().saturating_sub(1)
            } else {
                config.max_generations
            },
            stagnated: false,
            cancelled,
            cost_history,
        }
    }
}

fn decode_population<D: BrkgaDecoder>(
    decoder: &D,
    population: &mut [Chromosome],
    parallel: bool,
) {
    if parallel {
        population.par_iter_mut().for_each(|chr| {
            chr.cost = decoder.decode(&chr.keys);
        });
    } else {
        for chr in population.iter_mut() {
            chr.cost = decoder.decode(&chr.keys);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brkga::BrkgaConfig;

    // ---- Permutation sorting: sort keys, cost = number of inversions ----

    struct SortingDecoder {
        target: Vec<usize>,
    }

    impl BrkgaDecoder for SortingDecoder {
        fn decode(&self, keys: &[f64]) -> f64 {
            // Decode keys as permutation: sort indices by key value
            let mut indexed: Vec<(usize, f64)> =
                keys.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let perm: Vec<usize> = indexed.iter().map(|&(i, _)| i).collect();

            // Cost: number of positions where perm differs from target
            perm.iter()
                .zip(self.target.iter())
                .filter(|(&a, &b)| a != b)
                .count() as f64
        }
    }

    #[test]
    fn test_brkga_sorting() {
        let decoder = SortingDecoder {
            target: vec![0, 1, 2, 3, 4],
        };
        let config = BrkgaConfig::new(5)
            .with_population_size(50)
            .with_max_generations(200)
            .with_seed(42)
            .with_parallel(false);

        let result = BrkgaRunner::run(&decoder, &config);

        assert!(
            result.best_cost <= 2.0,
            "expected near-optimal permutation, got cost {}",
            result.best_cost
        );
    }

    // ---- OneMax via threshold: keys > 0.5 = 1, minimize negative count ----

    struct OneMaxDecoder;

    impl BrkgaDecoder for OneMaxDecoder {
        fn decode(&self, keys: &[f64]) -> f64 {
            let count = keys.iter().filter(|&&k| k > 0.5).count();
            -(count as f64) // minimize negative = maximize count
        }
    }

    #[test]
    fn test_brkga_onemax() {
        let decoder = OneMaxDecoder;
        let config = BrkgaConfig::new(20)
            .with_population_size(100)
            .with_max_generations(200)
            .with_elite_fraction(0.20)
            .with_mutant_fraction(0.15)
            .with_elite_inheritance_prob(0.70)
            .with_seed(42)
            .with_parallel(false);

        let result = BrkgaRunner::run(&decoder, &config);

        assert!(
            result.best_cost <= -15.0,
            "expected cost <= -15.0, got {}",
            result.best_cost
        );
    }

    #[test]
    fn test_brkga_stagnation() {
        let decoder = OneMaxDecoder;
        let config = BrkgaConfig::new(5)
            .with_population_size(30)
            .with_max_generations(1000)
            .with_stagnation_limit(10)
            .with_seed(42)
            .with_parallel(false);

        let result = BrkgaRunner::run(&decoder, &config);

        assert!(
            result.stagnated || result.generations < 1000,
            "expected early termination"
        );
    }

    #[test]
    fn test_brkga_cancellation() {
        let decoder = OneMaxDecoder;
        let config = BrkgaConfig::new(20)
            .with_population_size(50)
            .with_max_generations(100000)
            .with_stagnation_limit(0)
            .with_seed(42)
            .with_parallel(false);

        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            cancel_clone.store(true, Ordering::Relaxed);
        });

        let result = BrkgaRunner::run_with_cancel(&decoder, &config, Some(cancel));
        assert!(result.cancelled);
    }

    #[test]
    fn test_brkga_cost_monotonic() {
        let decoder = OneMaxDecoder;
        let config = BrkgaConfig::new(10)
            .with_population_size(30)
            .with_max_generations(50)
            .with_stagnation_limit(0)
            .with_seed(42)
            .with_parallel(false);

        let result = BrkgaRunner::run(&decoder, &config);

        for window in result.cost_history.windows(2) {
            assert!(
                window[1] <= window[0],
                "cost should be monotonically non-increasing: {} > {}",
                window[1],
                window[0]
            );
        }
    }

    #[test]
    fn test_brkga_parallel() {
        let decoder = OneMaxDecoder;
        let config = BrkgaConfig::new(20)
            .with_population_size(50)
            .with_max_generations(100)
            .with_seed(42)
            .with_parallel(true);

        let result = BrkgaRunner::run(&decoder, &config);

        assert!(
            result.best_cost <= -10.0,
            "parallel should find reasonable solution, got {}",
            result.best_cost
        );
    }

    // ---- Seed chromosome test ----

    struct SeededDecoder;

    impl BrkgaDecoder for SeededDecoder {
        fn decode(&self, keys: &[f64]) -> f64 {
            // Cost = distance from all-0.9 vector
            keys.iter().map(|k| (k - 0.9).powi(2)).sum()
        }

        fn seed_chromosome<R: Rng>(&self, _rng: &mut R) -> Option<Vec<f64>> {
            Some(vec![0.9; 5]) // optimal seed
        }
    }

    #[test]
    fn test_brkga_seeded() {
        let decoder = SeededDecoder;
        let config = BrkgaConfig::new(5)
            .with_population_size(20)
            .with_max_generations(10)
            .with_seed(42)
            .with_parallel(false);

        let result = BrkgaRunner::run(&decoder, &config);

        // With seeded population, should find near-optimal quickly
        assert!(
            result.best_cost < 0.1,
            "expected cost < 0.1 with seed, got {}",
            result.best_cost
        );
    }
}
