//! SA configuration and cooling schedules.

/// Cooling schedule for temperature reduction.
///
/// # References
///
/// - Geometric: standard textbook approach
/// - Linear: fixed-duration cooling
/// - LundyMees: Lundy & Mees (1986), with convergence proof
#[derive(Debug, Clone, Copy)]
pub enum CoolingSchedule {
    /// Geometric (exponential) cooling: `T_{k+1} = alpha * T_k`.
    ///
    /// Most widely used. Typical `alpha`: 0.95–0.99.
    Geometric {
        /// Cooling factor in (0, 1). Higher = slower cooling.
        alpha: f64,
    },

    /// Linear cooling: `T_k = T_0 - k * (T_0 - T_min) / max_steps`.
    ///
    /// Fixed total duration. Temperature decreases uniformly.
    Linear,

    /// Lundy-Mees cooling: `T_{k+1} = T_k / (1 + beta * T_k)`.
    ///
    /// One iteration per temperature step. Cools fast at high T,
    /// slow at low T. Has a convergence proof.
    ///
    /// Reference: Lundy & Mees (1986)
    LundyMees {
        /// Cooling parameter. Typically `(T_0 - T_min) / (max_iter * T_0 * T_min)`.
        beta: f64,
    },
}

impl Default for CoolingSchedule {
    fn default() -> Self {
        CoolingSchedule::Geometric { alpha: 0.95 }
    }
}

/// Configuration for the Simulated Annealing algorithm.
///
/// # Examples
///
/// ```
/// use u_metaheur::sa::{SaConfig, CoolingSchedule};
///
/// let config = SaConfig::default()
///     .with_initial_temperature(100.0)
///     .with_min_temperature(0.001)
///     .with_cooling(CoolingSchedule::Geometric { alpha: 0.98 })
///     .with_iterations_per_temperature(200);
/// ```
#[derive(Debug, Clone)]
pub struct SaConfig {
    /// Initial temperature. Higher values allow more exploration.
    pub initial_temperature: f64,

    /// Minimum temperature. The algorithm stops when T drops below this.
    pub min_temperature: f64,

    /// Cooling schedule.
    pub cooling: CoolingSchedule,

    /// Number of iterations at each temperature level.
    ///
    /// For `LundyMees`, this is ignored (1 iteration per temperature).
    pub iterations_per_temperature: usize,

    /// Maximum total iterations (hard budget). 0 = no limit.
    pub max_iterations: usize,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SaConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 100.0,
            min_temperature: 1e-6,
            cooling: CoolingSchedule::default(),
            iterations_per_temperature: 100,
            max_iterations: 0,
            seed: None,
        }
    }
}

impl SaConfig {
    pub fn with_initial_temperature(mut self, t: f64) -> Self {
        self.initial_temperature = t;
        self
    }

    pub fn with_min_temperature(mut self, t: f64) -> Self {
        self.min_temperature = t;
        self
    }

    pub fn with_cooling(mut self, cooling: CoolingSchedule) -> Self {
        self.cooling = cooling;
        self
    }

    pub fn with_iterations_per_temperature(mut self, n: usize) -> Self {
        self.iterations_per_temperature = n;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.initial_temperature <= 0.0 {
            return Err("initial_temperature must be positive".into());
        }
        if self.min_temperature <= 0.0 {
            return Err("min_temperature must be positive".into());
        }
        if self.min_temperature >= self.initial_temperature {
            return Err("min_temperature must be less than initial_temperature".into());
        }
        match self.cooling {
            CoolingSchedule::Geometric { alpha } => {
                if alpha <= 0.0 || alpha >= 1.0 {
                    return Err(format!("geometric alpha must be in (0, 1), got {alpha}"));
                }
            }
            CoolingSchedule::LundyMees { beta } => {
                if beta <= 0.0 {
                    return Err(format!("lundy-mees beta must be positive, got {beta}"));
                }
            }
            CoolingSchedule::Linear => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SaConfig::default();
        assert!((config.initial_temperature - 100.0).abs() < 1e-10);
        assert!((config.min_temperature - 1e-6).abs() < 1e-15);
        assert_eq!(config.iterations_per_temperature, 100);
    }

    #[test]
    fn test_validate_ok() {
        assert!(SaConfig::default().validate().is_ok());
    }

    #[test]
    fn test_validate_bad_temperature() {
        let config = SaConfig::default().with_initial_temperature(-1.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_min_ge_initial() {
        let config = SaConfig::default()
            .with_initial_temperature(10.0)
            .with_min_temperature(20.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_bad_alpha() {
        let config = SaConfig::default().with_cooling(CoolingSchedule::Geometric { alpha: 1.5 });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_bad_beta() {
        let config = SaConfig::default().with_cooling(CoolingSchedule::LundyMees { beta: -1.0 });
        assert!(config.validate().is_err());
    }
}
