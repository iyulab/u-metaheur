//! ALNS configuration.

/// Configuration for the ALNS algorithm.
///
/// # Scoring
///
/// At each iteration the selected destroy/repair operator pair receives a score:
/// - `score_new_best` (sigma_1): found a new global best solution
/// - `score_improved` (sigma_2): improved the current solution
/// - `score_accepted` (sigma_3): accepted a worse solution (via SA criterion)
///
/// These scores are accumulated over a *segment* of `segment_length` iterations,
/// then used to update operator weights via exponential smoothing with
/// `reaction_factor` (rho).
///
/// # Acceptance Criterion
///
/// Uses Simulated Annealing: worse solutions are accepted with probability
/// `exp(-delta / temperature)`. Temperature starts at `initial_temperature`
/// and decays geometrically by `cooling_rate` each iteration.
///
/// # References
///
/// Ropke & Pisinger (2006), Section 3
///
/// # Examples
///
/// ```
/// use u_metaheur::alns::AlnsConfig;
///
/// let config = AlnsConfig::default()
///     .with_max_iterations(5000)
///     .with_segment_length(100)
///     .with_scores(33.0, 9.0, 3.0)
///     .with_seed(42);
/// ```
#[derive(Debug, Clone)]
pub struct AlnsConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,

    /// Segment length for weight updates.
    ///
    /// Operator weights are updated every `segment_length` iterations
    /// based on accumulated scores.
    pub segment_length: usize,

    /// Score for finding a new global best (sigma_1).
    pub score_new_best: f64,

    /// Score for improving the current solution (sigma_2).
    pub score_improved: f64,

    /// Score for accepting a worse solution (sigma_3).
    pub score_accepted: f64,

    /// Reaction factor (rho) for weight updates, in (0, 1].
    ///
    /// Controls how quickly weights adapt: higher = faster adaptation.
    /// Ropke & Pisinger suggest 0.1.
    pub reaction_factor: f64,

    /// Minimum operator weight (prevents operators from becoming unused).
    pub min_weight: f64,

    /// Minimum destroy degree (fraction of solution to destroy).
    pub min_destroy_degree: f64,

    /// Maximum destroy degree.
    pub max_destroy_degree: f64,

    /// Initial temperature for SA acceptance.
    pub initial_temperature: f64,

    /// Cooling rate for SA acceptance (geometric), in (0, 1).
    pub cooling_rate: f64,

    /// Minimum temperature (stops cooling below this).
    pub min_temperature: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for AlnsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            segment_length: 100,
            score_new_best: 33.0,
            score_improved: 9.0,
            score_accepted: 3.0,
            reaction_factor: 0.1,
            min_weight: 0.01,
            min_destroy_degree: 0.1,
            max_destroy_degree: 0.4,
            initial_temperature: 100.0,
            cooling_rate: 0.9995,
            min_temperature: 0.01,
            seed: None,
        }
    }
}

impl AlnsConfig {
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_segment_length(mut self, n: usize) -> Self {
        self.segment_length = n.max(1);
        self
    }

    pub fn with_scores(mut self, new_best: f64, improved: f64, accepted: f64) -> Self {
        self.score_new_best = new_best;
        self.score_improved = improved;
        self.score_accepted = accepted;
        self
    }

    pub fn with_reaction_factor(mut self, rho: f64) -> Self {
        self.reaction_factor = rho;
        self
    }

    pub fn with_destroy_degree(mut self, min: f64, max: f64) -> Self {
        self.min_destroy_degree = min.clamp(0.0, 1.0);
        self.max_destroy_degree = max.clamp(self.min_destroy_degree, 1.0);
        self
    }

    pub fn with_temperature(mut self, initial: f64, cooling_rate: f64, min: f64) -> Self {
        self.initial_temperature = initial;
        self.cooling_rate = cooling_rate;
        self.min_temperature = min;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_iterations == 0 {
            return Err("max_iterations must be positive".into());
        }
        if self.reaction_factor <= 0.0 || self.reaction_factor > 1.0 {
            return Err(format!(
                "reaction_factor must be in (0, 1], got {}",
                self.reaction_factor
            ));
        }
        if self.cooling_rate <= 0.0 || self.cooling_rate >= 1.0 {
            return Err(format!(
                "cooling_rate must be in (0, 1), got {}",
                self.cooling_rate
            ));
        }
        if self.initial_temperature <= 0.0 {
            return Err("initial_temperature must be positive".into());
        }
        if self.min_temperature <= 0.0 {
            return Err("min_temperature must be positive".into());
        }
        if self.min_destroy_degree > self.max_destroy_degree {
            return Err("min_destroy_degree must be <= max_destroy_degree".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AlnsConfig::default();
        assert_eq!(config.max_iterations, 10000);
        assert_eq!(config.segment_length, 100);
        assert!((config.score_new_best - 33.0).abs() < 1e-10);
        assert!((config.reaction_factor - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_validate_ok() {
        assert!(AlnsConfig::default().validate().is_ok());
    }

    #[test]
    fn test_validate_bad_iterations() {
        let config = AlnsConfig {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_bad_reaction_factor() {
        let config = AlnsConfig::default().with_reaction_factor(0.0);
        assert!(config.validate().is_err());

        let config = AlnsConfig::default().with_reaction_factor(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_bad_cooling_rate() {
        let config = AlnsConfig::default().with_temperature(100.0, 0.0, 0.01);
        assert!(config.validate().is_err());

        let config = AlnsConfig::default().with_temperature(100.0, 1.0, 0.01);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_builder_chain() {
        let config = AlnsConfig::default()
            .with_max_iterations(500)
            .with_segment_length(50)
            .with_scores(10.0, 5.0, 1.0)
            .with_reaction_factor(0.2)
            .with_destroy_degree(0.2, 0.5)
            .with_temperature(50.0, 0.999, 0.001)
            .with_seed(42);

        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.segment_length, 50);
        assert!((config.score_new_best - 10.0).abs() < 1e-10);
        assert!((config.min_destroy_degree - 0.2).abs() < 1e-10);
        assert!((config.max_destroy_degree - 0.5).abs() < 1e-10);
        assert_eq!(config.seed, Some(42));
    }
}
