//! The unit of an annealing temperature, shared by SA and ALNS.

use crate::ConfigError;

/// What a configured temperature is measured in.
///
/// Metropolis acceptance is `exp(-delta / temperature)`, so a temperature only
/// means something next to the cost differences between neighbouring solutions.
/// A start far above them accepts nearly every move (a random walk that only
/// remembers the best); one far below them never leaves the first basin. The
/// right number therefore moves with the objective's scale, which is why the
/// default is not an absolute temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureScale {
    /// Temperatures are in the unit of the cost, used as configured.
    Absolute,

    /// Temperatures are multiples of the cost scale `sigma`: the mean amount by
    /// which a neighbouring solution is worse than the initial one, measured
    /// with `samples` neighbours before the run (they are not counted as
    /// iterations). `initial_temperature = 1` then accepts a typical worsening
    /// with probability `1/e`, whatever the objective's unit.
    ///
    /// When no sampled neighbour is worse, `sigma` is the mean absolute
    /// difference of the ones that differ, and `1` when all of them tie.
    ///
    /// Reference: Johnson, Aragon, McGeoch & Schevon (1989), "Optimization by
    /// simulated annealing: an experimental evaluation, part I" -- the start
    /// is chosen by the acceptance ratio it gives, not by an absolute number.
    Calibrated {
        /// Number of neighbours sampled to measure `sigma` (at least 1).
        samples: usize,
    },
}

impl TemperatureScale {
    /// The scale a configuration starts with: `Calibrated` over 50 neighbours.
    pub const DEFAULT_SAMPLES: usize = 50;

    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        if let TemperatureScale::Calibrated { samples: 0 } = self {
            return Err(ConfigError::out_of_range(
                "temperature_scale.samples",
                1.0..,
                0.0,
                "at least 1",
            ));
        }
        Ok(())
    }
}

impl Default for TemperatureScale {
    fn default() -> Self {
        TemperatureScale::Calibrated {
            samples: Self::DEFAULT_SAMPLES,
        }
    }
}

/// The cost scale `sigma` of a set of sampled cost differences (candidate cost
/// minus current cost); see [`TemperatureScale::Calibrated`].
pub(crate) fn cost_scale(deltas: impl IntoIterator<Item = f64>) -> f64 {
    let (mut worse_sum, mut worse_n) = (0.0, 0usize);
    let (mut abs_sum, mut abs_n) = (0.0, 0usize);
    for d in deltas {
        if !d.is_finite() || d == 0.0 {
            continue;
        }
        abs_sum += d.abs();
        abs_n += 1;
        if d > 0.0 {
            worse_sum += d;
            worse_n += 1;
        }
    }
    if worse_n > 0 {
        worse_sum / worse_n as f64
    } else if abs_n > 0 {
        abs_sum / abs_n as f64
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_is_the_mean_worsening() {
        assert_eq!(cost_scale([2.0, -5.0, 4.0, 0.0]), 3.0);
    }

    #[test]
    fn scale_falls_back_to_the_mean_difference_without_a_worsening() {
        assert_eq!(cost_scale([-2.0, -4.0, 0.0]), 3.0);
    }

    #[test]
    fn scale_of_a_plateau_is_one() {
        assert_eq!(cost_scale([0.0, 0.0]), 1.0);
        assert_eq!(cost_scale(std::iter::empty()), 1.0);
    }

    #[test]
    fn scale_ignores_non_finite_differences() {
        assert_eq!(cost_scale([f64::NAN, f64::INFINITY, 6.0]), 6.0);
    }

    #[test]
    fn zero_samples_are_refused() {
        assert!(TemperatureScale::Calibrated { samples: 0 }
            .validate()
            .is_err());
        assert!(TemperatureScale::default().validate().is_ok());
        assert!(TemperatureScale::Absolute.validate().is_ok());
    }
}
