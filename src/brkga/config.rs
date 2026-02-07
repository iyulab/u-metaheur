//! BRKGA configuration.

/// Configuration for the BRKGA algorithm.
///
/// # Parameters
///
/// The three population fractions must satisfy:
/// `elite_fraction + mutant_fraction < 1.0`
///
/// The remaining fraction `1.0 - elite_fraction - mutant_fraction`
/// is filled by crossover offspring.
///
/// # Examples
///
/// ```
/// use u_metaheur::brkga::BrkgaConfig;
///
/// let config = BrkgaConfig::new(50) // 50 random keys
///     .with_population_size(200)
///     .with_elite_fraction(0.20)
///     .with_mutant_fraction(0.15)
///     .with_elite_inheritance_prob(0.70);
/// ```
#[derive(Debug, Clone)]
pub struct BrkgaConfig {
    /// Number of random keys per chromosome.
    pub chromosome_length: usize,

    /// Total population size.
    pub population_size: usize,

    /// Fraction of population preserved as elite (0.10–0.25 typical).
    pub elite_fraction: f64,

    /// Fraction of population replaced by random mutants (0.10–0.30 typical).
    pub mutant_fraction: f64,

    /// Probability that offspring inherits the elite parent's allele
    /// during biased uniform crossover (0.55–0.80 typical).
    ///
    /// Must be > 0.5 for the bias toward elite to be meaningful.
    pub elite_inheritance_prob: f64,

    /// Maximum number of generations.
    pub max_generations: usize,

    /// Generations with no improvement before stopping (0 to disable).
    pub stagnation_limit: usize,

    /// Whether to decode chromosomes in parallel using rayon.
    pub parallel: bool,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl BrkgaConfig {
    /// Creates a new configuration with the given chromosome length.
    pub fn new(chromosome_length: usize) -> Self {
        Self {
            chromosome_length,
            population_size: 100,
            elite_fraction: 0.20,
            mutant_fraction: 0.15,
            elite_inheritance_prob: 0.70,
            max_generations: 500,
            stagnation_limit: 50,
            parallel: true,
            seed: None,
        }
    }

    pub fn with_population_size(mut self, n: usize) -> Self {
        self.population_size = n;
        self
    }

    pub fn with_elite_fraction(mut self, f: f64) -> Self {
        self.elite_fraction = f.clamp(0.0, 1.0);
        self
    }

    pub fn with_mutant_fraction(mut self, f: f64) -> Self {
        self.mutant_fraction = f.clamp(0.0, 1.0);
        self
    }

    pub fn with_elite_inheritance_prob(mut self, p: f64) -> Self {
        self.elite_inheritance_prob = p.clamp(0.5, 1.0);
        self
    }

    pub fn with_max_generations(mut self, n: usize) -> Self {
        self.max_generations = n;
        self
    }

    pub fn with_stagnation_limit(mut self, n: usize) -> Self {
        self.stagnation_limit = n;
        self
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.chromosome_length == 0 {
            return Err("chromosome_length must be at least 1".into());
        }
        if self.population_size < 3 {
            return Err("population_size must be at least 3".into());
        }
        if self.elite_fraction + self.mutant_fraction >= 1.0 {
            return Err(format!(
                "elite_fraction ({}) + mutant_fraction ({}) must be < 1.0",
                self.elite_fraction, self.mutant_fraction
            ));
        }
        let elite_count = (self.population_size as f64 * self.elite_fraction) as usize;
        if elite_count == 0 {
            return Err("elite_fraction too small: no elite individuals".into());
        }
        if self.elite_inheritance_prob <= 0.5 {
            return Err("elite_inheritance_prob must be > 0.5".into());
        }
        if self.max_generations == 0 {
            return Err("max_generations must be at least 1".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BrkgaConfig::new(20);
        assert_eq!(config.chromosome_length, 20);
        assert_eq!(config.population_size, 100);
        assert!((config.elite_fraction - 0.20).abs() < 1e-10);
        assert!((config.mutant_fraction - 0.15).abs() < 1e-10);
        assert!((config.elite_inheritance_prob - 0.70).abs() < 1e-10);
    }

    #[test]
    fn test_validate_ok() {
        assert!(BrkgaConfig::new(10).validate().is_ok());
    }

    #[test]
    fn test_validate_fractions_sum() {
        let config = BrkgaConfig::new(10)
            .with_elite_fraction(0.6)
            .with_mutant_fraction(0.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_chromosome() {
        let config = BrkgaConfig::new(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_clamp_inheritance() {
        let config = BrkgaConfig::new(10).with_elite_inheritance_prob(0.3);
        assert!((config.elite_inheritance_prob - 0.5).abs() < 1e-10);
    }
}
