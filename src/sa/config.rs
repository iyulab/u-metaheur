//! SA configuration and cooling schedules.

use crate::ConfigError;
use std::ops::Bound::{Excluded, Unbounded};

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
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(self.initial_temperature > 0.0 && self.initial_temperature.is_finite()) {
            return Err(ConfigError::out_of_range(
                "initial_temperature",
                (Excluded(0.0), Unbounded),
                self.initial_temperature,
                "positive and finite",
            ));
        }
        // The run cools from `initial_temperature` down to `min_temperature`,
        // so the floor lies strictly between 0 and the start.
        if !(self.min_temperature > 0.0 && self.min_temperature < self.initial_temperature) {
            return Err(ConfigError::out_of_range(
                "min_temperature",
                (Excluded(0.0), Excluded(self.initial_temperature)),
                self.min_temperature,
                "in (0, initial_temperature)",
            ));
        }
        match self.cooling {
            CoolingSchedule::Geometric { alpha } => {
                if !(alpha > 0.0 && alpha < 1.0) {
                    return Err(ConfigError::out_of_range(
                        "cooling.alpha",
                        (Excluded(0.0), Excluded(1.0)),
                        alpha,
                        "in (0, 1)",
                    ));
                }
            }
            CoolingSchedule::LundyMees { beta } => {
                if !(beta > 0.0 && beta.is_finite()) {
                    return Err(ConfigError::out_of_range(
                        "cooling.beta",
                        (Excluded(0.0), Unbounded),
                        beta,
                        "positive and finite",
                    ));
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
        // Positive: zero itself is refused, so the bound is open.
        assert!(matches!(
            config.validate(),
            Err(ConfigError::OutOfRange {
                parameter: "initial_temperature",
                min: None,
                exclusive_min: Some(0.0),
                max: None,
                exclusive_max: None,
                ..
            })
        ));
    }

    /// The floor lies strictly between 0 and the start: one range, whose open
    /// upper end is `initial_temperature`, for a floor too high or not positive.
    #[test]
    fn test_validate_min_ge_initial() {
        for min in [20.0, 10.0, 0.0, -1.0] {
            let config = SaConfig::default()
                .with_initial_temperature(10.0)
                .with_min_temperature(min);
            assert_eq!(
                config.validate(),
                Err(ConfigError::OutOfRange {
                    parameter: "min_temperature",
                    min: None,
                    max: None,
                    exclusive_min: Some(0.0),
                    exclusive_max: Some(10.0),
                    got: min,
                    range: "in (0, initial_temperature)",
                }),
                "min_temperature {min}"
            );
        }
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
