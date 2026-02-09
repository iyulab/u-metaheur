//! Selection strategies for the GA.
//!
//! Selection determines which individuals are chosen as parents for
//! crossover. Different strategies provide different selection pressure.
//!
//! # References
//!
//! - Blickle & Thiele (1996), "A Comparison of Selection Schemes used in
//!   Evolutionary Algorithms"
//! - Goldberg & Deb (1991), "A Comparative Analysis of Selection Schemes
//!   Used in Genetic Algorithms"

use super::types::{Fitness, Individual};
use rand::Rng;

/// Selection strategy for choosing parents.
///
/// All strategies assume **minimization** (lower fitness = better).
///
/// # Examples
///
/// ```
/// use u_metaheur::ga::Selection;
///
/// // Tournament with size 3 (moderate selection pressure)
/// let sel = Selection::Tournament(3);
///
/// // Roulette wheel (fitness-proportionate)
/// let sel = Selection::Roulette;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// Tournament selection: pick `k` individuals at random, select the best.
    ///
    /// Higher `k` = stronger selection pressure.
    /// - k=2: light pressure (good for diversity)
    /// - k=3-5: moderate pressure (typical default)
    /// - k>5: strong pressure (risk of premature convergence)
    ///
    /// # Complexity
    /// O(k) per selection
    Tournament(usize),

    /// Fitness-proportionate (roulette wheel) selection.
    ///
    /// Probability of selection is proportional to fitness quality.
    /// Since we minimize, uses inverse fitness transformation.
    ///
    /// **Warning**: Susceptible to super-individual dominance when
    /// fitness variance is high.
    ///
    /// # Complexity
    /// O(n) per selection (linear scan)
    Roulette,

    /// Rank-based selection.
    ///
    /// Individuals are sorted by fitness and selection probability is
    /// proportional to rank position, not raw fitness value. This avoids
    /// the scaling problems of roulette wheel selection.
    ///
    /// Uses linear ranking: P(i) = (2 - s)/n + 2·i·(s - 1)/(n·(n - 1))
    /// where s is selection pressure (typically 1.5–2.0), i is rank.
    ///
    /// Reference: Baker (1985), "Adaptive Selection Methods for Genetic
    /// Algorithms"
    ///
    /// # Complexity
    /// O(n log n) per generation (sort), O(n) per selection
    Rank,
}

impl Default for Selection {
    fn default() -> Self {
        Selection::Tournament(3)
    }
}

impl Selection {
    /// Select a parent index from the population.
    ///
    /// # Panics
    /// Panics if `population` is empty.
    pub fn select<I: Individual, R: Rng>(&self, population: &[I], rng: &mut R) -> usize {
        assert!(
            !population.is_empty(),
            "cannot select from empty population"
        );

        match self {
            Selection::Tournament(k) => tournament(population, *k, rng),
            Selection::Roulette => roulette(population, rng),
            Selection::Rank => rank(population, rng),
        }
    }
}

/// Tournament selection: pick k random individuals, return best.
fn tournament<I: Individual, R: Rng>(population: &[I], k: usize, rng: &mut R) -> usize {
    let k = k.max(1);
    let n = population.len();

    let mut best_idx = rng.random_range(0..n);
    for _ in 1..k {
        let idx = rng.random_range(0..n);
        if population[idx].fitness() < population[best_idx].fitness() {
            best_idx = idx;
        }
    }
    best_idx
}

/// Roulette wheel selection using inverse fitness transformation.
///
/// For minimization: weight_i = max_fitness - fitness_i + epsilon
/// This ensures the best (lowest fitness) individual gets the highest weight.
fn roulette<I: Individual, R: Rng>(population: &[I], rng: &mut R) -> usize {
    let n = population.len();
    if n == 1 {
        return 0;
    }

    let fitnesses: Vec<f64> = population
        .iter()
        .map(|ind| ind.fitness().to_f64())
        .collect();

    // Find max fitness for inversion
    let max_fitness = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let epsilon = 1e-10;

    // Invert: lower fitness -> higher weight
    let weights: Vec<f64> = fitnesses
        .iter()
        .map(|&f| {
            let w = max_fitness - f + epsilon;
            if w > 0.0 {
                w
            } else {
                epsilon
            }
        })
        .collect();

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return rng.random_range(0..n);
    }

    let threshold = rng.random_range(0.0..total);
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if cumulative > threshold {
            return i;
        }
    }

    n - 1 // floating-point fallback
}

/// Rank-based selection using linear ranking.
///
/// Individuals are sorted by fitness (best first), then selection
/// probability is proportional to rank.
fn rank<I: Individual, R: Rng>(population: &[I], rng: &mut R) -> usize {
    let n = population.len();
    if n == 1 {
        return 0;
    }

    // Build (index, fitness) pairs and sort by fitness ascending (best first)
    let mut indexed: Vec<(usize, f64)> = population
        .iter()
        .enumerate()
        .map(|(i, ind)| (i, ind.fitness().to_f64()))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Linear ranking: rank 0 (best) gets highest weight
    // weight_i = n - rank_i
    let total: f64 = (n * (n + 1)) as f64 / 2.0;
    let threshold = rng.random_range(0.0..total);
    let mut cumulative = 0.0;

    for (rank, &(original_idx, _)) in indexed.iter().enumerate() {
        let weight = (n - rank) as f64;
        cumulative += weight;
        if cumulative > threshold {
            return original_idx;
        }
    }

    indexed.last().expect("population has n >= 2 elements").0 // fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestInd {
        fit: f64,
    }

    impl Individual for TestInd {
        type Fitness = f64;
        fn fitness(&self) -> f64 {
            self.fit
        }
        fn set_fitness(&mut self, f: f64) {
            self.fit = f;
        }
    }

    fn make_population(fitnesses: &[f64]) -> Vec<TestInd> {
        fitnesses.iter().map(|&f| TestInd { fit: f }).collect()
    }

    #[test]
    fn test_tournament_favors_best() {
        let pop = make_population(&[10.0, 5.0, 1.0, 8.0]);
        let mut rng = u_optim::random::create_rng(42);

        // With tournament size = population size, best should be selected
        // most often (though not always due to with-replacement sampling)
        let mut counts = [0u32; 4];
        let n = 10000;
        for _ in 0..n {
            let idx = Selection::Tournament(4).select(&pop, &mut rng);
            counts[idx] += 1;
        }
        // Index 2 (fitness=1.0) should dominate
        let best_count = counts[2];
        assert!(
            best_count > 6000,
            "expected best to be selected >60% of the time, got {best_count}/{n}"
        );
    }

    #[test]
    fn test_tournament_size_1_is_random() {
        let pop = make_population(&[10.0, 5.0, 1.0, 8.0]);
        let mut rng = u_optim::random::create_rng(42);

        let mut counts = [0u32; 4];
        let n = 10000;
        for _ in 0..n {
            let idx = Selection::Tournament(1).select(&pop, &mut rng);
            counts[idx] += 1;
        }
        // All should be selected roughly equally
        for &c in &counts {
            assert!(c > 1500, "expected uniform, got counts: {counts:?}");
        }
    }

    #[test]
    fn test_roulette_favors_best() {
        let pop = make_population(&[100.0, 50.0, 1.0, 80.0]);
        let mut rng = u_optim::random::create_rng(42);

        let mut counts = [0u32; 4];
        let n = 10000;
        for _ in 0..n {
            let idx = Selection::Roulette.select(&pop, &mut rng);
            counts[idx] += 1;
        }
        // Index 2 (fitness=1.0, lowest) should be selected most often
        let best_count = counts[2];
        let worst_count = counts[0];
        assert!(
            best_count > worst_count,
            "best should be selected more often: best={best_count}, worst={worst_count}"
        );
    }

    #[test]
    fn test_rank_favors_best() {
        let pop = make_population(&[100.0, 50.0, 1.0, 80.0]);
        let mut rng = u_optim::random::create_rng(42);

        let mut counts = [0u32; 4];
        let n = 10000;
        for _ in 0..n {
            let idx = Selection::Rank.select(&pop, &mut rng);
            counts[idx] += 1;
        }
        // Index 2 (fitness=1.0, best) should be selected most
        let best_count = counts[2];
        let worst_count = counts[0];
        assert!(
            best_count > worst_count,
            "best should be selected more: best={best_count}, worst={worst_count}"
        );
    }

    #[test]
    fn test_single_individual() {
        let pop = make_population(&[5.0]);
        let mut rng = u_optim::random::create_rng(42);

        assert_eq!(Selection::Tournament(3).select(&pop, &mut rng), 0);
        assert_eq!(Selection::Roulette.select(&pop, &mut rng), 0);
        assert_eq!(Selection::Rank.select(&pop, &mut rng), 0);
    }

    #[test]
    fn test_equal_fitness() {
        let pop = make_population(&[5.0, 5.0, 5.0, 5.0]);
        let mut rng = u_optim::random::create_rng(42);

        // With equal fitness, all methods should select roughly uniformly
        let mut counts = [0u32; 4];
        let n = 10000;
        for _ in 0..n {
            let idx = Selection::Tournament(2).select(&pop, &mut rng);
            counts[idx] += 1;
        }
        for &c in &counts {
            assert!(
                c > 1500,
                "expected roughly uniform with equal fitness, got {counts:?}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "cannot select from empty population")]
    fn test_empty_population_panics() {
        let pop: Vec<TestInd> = vec![];
        let mut rng = u_optim::random::create_rng(42);
        Selection::Tournament(3).select(&pop, &mut rng);
    }
}
