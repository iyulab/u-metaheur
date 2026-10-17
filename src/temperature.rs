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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TemperatureScale {
    /// Temperatures are in the unit of the cost, used as configured.
    Absolute,

    /// Temperatures are multiples of the cost scale `sigma`: the median amount by
    /// which a neighbouring solution is worse than the initial one, measured
    /// with `samples` neighbours before the run (they are not counted as
    /// iterations). `initial_temperature = 1` then accepts a typical worsening
    /// with probability `1/e`, whatever the objective's unit.
    ///
    /// When no sampled neighbour is worse, `sigma` is the median absolute
    /// difference of the ones that differ, and `1` when all of them tie.
    ///
    /// The scale is read off neighbours of the start, so an objective whose
    /// neighbours differ on two very different scales (a count term and a fine
    /// tuning term, say) is read at the coarse one -- there an absolute
    /// temperature, chosen for the fine scale, is the right setting.
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

    /// The cost scale `sigma` of sampled cost differences, each the amount by
    /// which a neighbouring solution is **worse** than the one it came from
    /// (negative when better): their median over the worsening ones, the median
    /// absolute difference when none worsens, and `1` when all tie. This is
    /// what `Calibrated` measures; it is public for runners that sample their
    /// own neighbours.
    pub fn cost_scale(worsenings: impl IntoIterator<Item = f64>) -> f64 {
        cost_scale(worsenings)
    }

    /// Checks the scale against its own constraints.
    ///
    /// # Errors
    /// `samples` below 1 when `Calibrated`.
    pub fn validate(&self) -> Result<(), ConfigError> {
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
///
/// A median, not a mean: an objective with a plateau or a count in it (a piece
/// that does not fit costs a hundred, the fine tuning a tenth) has a few jumps
/// that dominate a mean and would set a start hot enough for a random walk.
pub(crate) fn cost_scale(deltas: impl IntoIterator<Item = f64>) -> f64 {
    let mut worse = Vec::new();
    let mut apart = Vec::new();
    for d in deltas {
        if !d.is_finite() || d == 0.0 {
            continue;
        }
        apart.push(d.abs());
        if d > 0.0 {
            worse.push(d);
        }
    }
    let sample = if worse.is_empty() { apart } else { worse };
    median(sample).unwrap_or(1.0)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_is_the_median_worsening() {
        assert_eq!(cost_scale([2.0, -5.0, 4.0, 0.0]), 3.0);
        assert_eq!(cost_scale([0.1, 0.3, 0.2, 400.0, 900.0]), 0.3);
    }

    #[test]
    fn scale_falls_back_to_the_median_difference_without_a_worsening() {
        assert_eq!(cost_scale([-2.0, -4.0, 0.0]), 3.0);
        assert_eq!(cost_scale([-1.0, -2.0, -90.0]), 2.0);
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
