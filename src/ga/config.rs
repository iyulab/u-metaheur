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

    /// Number of generations with no significant improvement before stopping.
    ///
    /// Set to 0 to disable stagnation-based termination.
    pub stagnation_limit: usize,

    /// Minimum relative improvement to reset the stagnation counter.
    ///
    /// When a new best fitness is found, the improvement ratio is computed as
    /// `|old - new| / |old|`. If this ratio is below `convergence_threshold`,
    /// the generation is still counted as stagnating.
    ///
    /// Set to 0.0 to count any improvement (the default).
    /// A typical value for production scheduling is 0.001 (0.1%).
    pub convergence_threshold: f64,

    /// Whether to evaluate individuals in parallel using rayon.
    pub parallel: bool,

    /// Random seed for reproducibility.
    ///
    /// `None` uses a random seed.
    pub seed: Option<u64>,

    /// Optional wall-clock time limit in milliseconds.
    ///
    /// When set, the GA will stop after approximately this many milliseconds
    /// have elapsed, returning the best solution found so far.
    /// The check happens at the start of each generation, so the actual
    /// runtime may slightly exceed this limit by one generation's worth of work.
    ///
    /// `None` disables time-based termination (the default).
    pub time_limit_ms: Option<u64>,
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
            convergence_threshold: 0.0,
            parallel: true,
            seed: None,
            time_limit_ms: None,
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

    /// Sets the convergence threshold.
    ///
    /// The stagnation counter is only reset when the relative improvement
    /// exceeds this threshold: `|old - new| / |old| >= threshold`.
    ///
    /// Set to 0.0 to count any improvement (default).
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold.max(0.0);
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

    /// Sets the wall-clock time limit in milliseconds.
    ///
    /// The GA will stop after approximately this many milliseconds,
    /// returning the best solution found so far.
    pub fn with_time_limit_ms(mut self, ms: u64) -> Self {
        self.time_limit_ms = Some(ms);
        self
    }

    /// Preset for fast optimization: small population, few generations.
    ///
    /// Suitable for quick feasibility checks or real-time applications.
    ///
    /// - Population: 50, Generations: 100, Time limit: 10s
    /// - Stagnation limit: 20, Convergence threshold: 0.001
    pub fn fast() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            stagnation_limit: 20,
            convergence_threshold: 0.001,
            time_limit_ms: Some(10_000),
            ..Self::default()
        }
    }

    /// Preset for balanced optimization: moderate population and generations.
    ///
    /// Good trade-off between solution quality and computation time.
    ///
    /// - Population: 100, Generations: 300, Time limit: 30s
    /// - Stagnation limit: 50, Convergence threshold: 0.001
    pub fn balanced() -> Self {
        Self {
            population_size: 100,
            max_generations: 300,
            stagnation_limit: 50,
            convergence_threshold: 0.001,
            time_limit_ms: Some(30_000),
            ..Self::default()
        }
    }

    /// Preset for quality optimization: large population, many generations.
    ///
    /// Maximizes solution quality at the cost of longer computation.
    ///
    /// - Population: 150, Generations: 500, Time limit: 60s
    /// - Stagnation limit: 80, Convergence threshold: 0.0005
    pub fn quality() -> Self {
        Self {
            population_size: 150,
            max_generations: 500,
            stagnation_limit: 80,
            convergence_threshold: 0.0005,
            time_limit_ms: Some(60_000),
            ..Self::default()
        }
    }

    /// Automatically selects a preset based on problem size.
    ///
    /// - `item_count < 50` → [`fast()`](Self::fast)
    /// - `50 ≤ item_count < 200` → [`balanced()`](Self::balanced)
    /// - `item_count ≥ 200` → [`quality()`](Self::quality)
    ///
    /// The `item_count` is a domain-specific measure of problem size
    /// (e.g., number of operations, genes, or decision variables).
    pub fn auto_select(item_count: usize) -> Self {
        if item_count < 50 {
            Self::fast()
        } else if item_count < 200 {
            Self::balanced()
        } else {
            Self::quality()
        }
    }

    /// Convenience builder for setting tournament size.
    ///
    /// Equivalent to `.with_selection(Selection::Tournament(k))`.
    pub fn with_tournament_size(self, k: usize) -> Self {
        self.with_selection(Selection::Tournament(k))
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
        if self.convergence_threshold < 0.0 {
            return Err("convergence_threshold must be non-negative".into());
        }
        if self.time_limit_ms == Some(0) {
            return Err("time_limit_ms must be positive or None".into());
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
        assert!((config.convergence_threshold - 0.0).abs() < 1e-15);
        assert!(config.parallel);
        assert!(config.seed.is_none());
        assert!(config.time_limit_ms.is_none());
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

    #[test]
    fn test_time_limit_builder() {
        let config = GaConfig::default().with_time_limit_ms(5000);
        assert_eq!(config.time_limit_ms, Some(5000));
    }

    #[test]
    fn test_validate_zero_time_limit() {
        let config = GaConfig::default().with_time_limit_ms(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_positive_time_limit() {
        let config = GaConfig::default().with_time_limit_ms(1);
        assert!(config.validate().is_ok());
    }

    // ---- Convergence threshold ----

    #[test]
    fn test_convergence_threshold_builder() {
        let config = GaConfig::default().with_convergence_threshold(0.001);
        assert!((config.convergence_threshold - 0.001).abs() < 1e-15);
    }

    #[test]
    fn test_convergence_threshold_clamps_negative() {
        let config = GaConfig::default().with_convergence_threshold(-0.5);
        assert!((config.convergence_threshold - 0.0).abs() < 1e-15);
    }

    // ---- Presets ----

    #[test]
    fn test_preset_fast() {
        let config = GaConfig::fast();
        assert_eq!(config.population_size, 50);
        assert_eq!(config.max_generations, 100);
        assert_eq!(config.stagnation_limit, 20);
        assert!((config.convergence_threshold - 0.001).abs() < 1e-15);
        assert_eq!(config.time_limit_ms, Some(10_000));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_preset_balanced() {
        let config = GaConfig::balanced();
        assert_eq!(config.population_size, 100);
        assert_eq!(config.max_generations, 300);
        assert_eq!(config.stagnation_limit, 50);
        assert!((config.convergence_threshold - 0.001).abs() < 1e-15);
        assert_eq!(config.time_limit_ms, Some(30_000));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_preset_quality() {
        let config = GaConfig::quality();
        assert_eq!(config.population_size, 150);
        assert_eq!(config.max_generations, 500);
        assert_eq!(config.stagnation_limit, 80);
        assert!((config.convergence_threshold - 0.0005).abs() < 1e-15);
        assert_eq!(config.time_limit_ms, Some(60_000));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_preset_chainable() {
        let config = GaConfig::fast().with_population_size(75).with_seed(42);
        assert_eq!(config.population_size, 75);
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.time_limit_ms, Some(10_000));
    }

    // ---- auto_select ----

    #[test]
    fn test_auto_select_small() {
        let config = GaConfig::auto_select(10);
        assert_eq!(config.population_size, 50); // fast preset
        assert_eq!(config.max_generations, 100);
    }

    #[test]
    fn test_auto_select_medium() {
        let config = GaConfig::auto_select(100);
        assert_eq!(config.population_size, 100); // balanced preset
        assert_eq!(config.max_generations, 300);
    }

    #[test]
    fn test_auto_select_large() {
        let config = GaConfig::auto_select(500);
        assert_eq!(config.population_size, 150); // quality preset
        assert_eq!(config.max_generations, 500);
    }

    #[test]
    fn test_auto_select_boundaries() {
        // Exactly 50 → balanced
        let config = GaConfig::auto_select(50);
        assert_eq!(config.population_size, 100);

        // Exactly 200 → quality
        let config = GaConfig::auto_select(200);
        assert_eq!(config.population_size, 150);

        // 49 → fast
        let config = GaConfig::auto_select(49);
        assert_eq!(config.population_size, 50);
    }

    // ---- with_tournament_size ----

    #[test]
    fn test_with_tournament_size() {
        let config = GaConfig::default().with_tournament_size(5);
        assert_eq!(config.selection, Selection::Tournament(5));
    }

    #[test]
    fn test_with_tournament_size_chainable() {
        let config = GaConfig::auto_select(100)
            .with_tournament_size(4)
            .with_seed(42);
        assert_eq!(config.selection, Selection::Tournament(4));
        assert_eq!(config.seed, Some(42));
    }
}
