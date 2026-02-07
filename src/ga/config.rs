//! GA configuration.
//!
//! [`GaConfig`] holds all parameters that control the evolutionary loop.

use super::selection::Selection;

/// Configuration for the Genetic Algorithm.
///
/// Controls population size, selection strategy, operator rates,
/// termination conditions, and parallelism.
///
/// # Defaults
///
/// ```
/// use u_metaheur::ga::GaConfig;
///
/// let config = GaConfig::default();
/// assert_eq!(config.population_size, 100);
/// assert_eq!(config.max_generations, 500);
/// ```
///
/// # Builder Pattern
///
/// ```
/// use u_metaheur::ga::{GaConfig, Selection};
///
/// let config = GaConfig::default()
///     .with_population_size(200)
///     .with_selection(Selection::Tournament(5))
///     .with_elite_ratio(0.1)
///     .with_mutation_rate(0.1);
/// ```
#[derive(Debug, Clone)]
pub struct GaConfig {
    /// Number of individuals in the population.
    ///
    /// Larger populations increase diversity but slow down each generation.
    /// Typical range: 50–500.
    pub population_size: usize,

    /// Maximum number of generations before termination.
    pub max_generations: usize,

    /// Selection strategy for choosing parents.
    pub selection: Selection,

    /// Fraction of the population preserved as elites (0.0–1.0).
    ///
    /// Elite individuals are copied unchanged to the next generation.
    /// Typical range: 0.05–0.2.
    pub elite_ratio: f64,

    /// Probability of applying crossover to a pair of parents (0.0–1.0).
    ///
    /// When crossover is not applied, a clone of one parent is used.
    pub crossover_rate: f64,

    /// Probability of applying mutation to an offspring (0.0–1.0).
    pub mutation_rate: f64,

    /// Number of generations with no improvement before stopping.
    ///
    /// Set to 0 to disable stagnation-based termination.
    pub stagnation_limit: usize,

    /// Whether to evaluate individuals in parallel using rayon.
    pub parallel: bool,

    /// Random seed for reproducibility.
    ///
    /// `None` uses a random seed.
    pub seed: Option<u64>,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 500,
            selection: Selection::default(),
            elite_ratio: 0.1,
            crossover_rate: 0.9,
            mutation_rate: 0.1,
            stagnation_limit: 50,
            parallel: true,
            seed: None,
        }
    }
}

impl GaConfig {
    /// Sets the population size.
    pub fn with_population_size(mut self, n: usize) -> Self {
        self.population_size = n;
        self
    }

    /// Sets the maximum number of generations.
    pub fn with_max_generations(mut self, n: usize) -> Self {
        self.max_generations = n;
        self
    }

    /// Sets the selection strategy.
    pub fn with_selection(mut self, sel: Selection) -> Self {
        self.selection = sel;
        self
    }

    /// Sets the elite ratio.
    pub fn with_elite_ratio(mut self, ratio: f64) -> Self {
        self.elite_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Sets the crossover rate.
    pub fn with_crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Sets the mutation rate.
    pub fn with_mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Sets the stagnation limit (0 to disable).
    pub fn with_stagnation_limit(mut self, limit: usize) -> Self {
        self.stagnation_limit = limit;
        self
    }

    /// Enables or disables parallel evaluation.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Sets the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the configuration.
    ///
    /// Returns `Err` with a description if any parameter is invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.population_size < 2 {
            return Err("population_size must be at least 2".into());
        }
        if self.max_generations == 0 {
            return Err("max_generations must be at least 1".into());
        }
        let elite_count = (self.population_size as f64 * self.elite_ratio) as usize;
        if elite_count >= self.population_size {
            return Err("elite_ratio too high: elites fill entire population".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GaConfig::default();
        assert_eq!(config.population_size, 100);
        assert_eq!(config.max_generations, 500);
        assert_eq!(config.selection, Selection::Tournament(3));
        assert!((config.elite_ratio - 0.1).abs() < 1e-10);
        assert!((config.crossover_rate - 0.9).abs() < 1e-10);
        assert!((config.mutation_rate - 0.1).abs() < 1e-10);
        assert_eq!(config.stagnation_limit, 50);
        assert!(config.parallel);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_builder_pattern() {
        let config = GaConfig::default()
            .with_population_size(200)
            .with_max_generations(1000)
            .with_selection(Selection::Rank)
            .with_elite_ratio(0.2)
            .with_crossover_rate(0.8)
            .with_mutation_rate(0.05)
            .with_stagnation_limit(100)
            .with_parallel(false)
            .with_seed(42);

        assert_eq!(config.population_size, 200);
        assert_eq!(config.max_generations, 1000);
        assert_eq!(config.selection, Selection::Rank);
        assert!((config.elite_ratio - 0.2).abs() < 1e-10);
        assert!((config.crossover_rate - 0.8).abs() < 1e-10);
        assert!((config.mutation_rate - 0.05).abs() < 1e-10);
        assert_eq!(config.stagnation_limit, 100);
        assert!(!config.parallel);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_validate_ok() {
        assert!(GaConfig::default().validate().is_ok());
    }

    #[test]
    fn test_validate_population_too_small() {
        let config = GaConfig::default().with_population_size(1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_generations() {
        let config = GaConfig::default().with_max_generations(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_elite_too_high() {
        let config = GaConfig::default()
            .with_population_size(10)
            .with_elite_ratio(1.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_clamp_rates() {
        let config = GaConfig::default()
            .with_elite_ratio(1.5)
            .with_crossover_rate(-0.5)
            .with_mutation_rate(2.0);

        assert!((config.elite_ratio - 1.0).abs() < 1e-10);
        assert!((config.crossover_rate - 0.0).abs() < 1e-10);
        assert!((config.mutation_rate - 1.0).abs() < 1e-10);
    }
}
