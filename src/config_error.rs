//! The one error every runner's settings check returns.

use std::fmt;

/// A setting a runner refuses: which one, and what was wrong with it.
///
/// `parameter` is the setting's field name (`population_size`,
/// `mutation_rate`, `destroy_ops`, …), so a caller that exposes these settings
/// under its own names can say which of its inputs to change without reading
/// the message.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    /// A number outside the values the setting accepts. `min` and `max` are the
    /// bounds (`None` when the setting is unbounded on that side) and `range`
    /// says it in words, open or closed ends included (`"in (0, 1]"`,
    /// `"at least 2"`).
    OutOfRange {
        parameter: &'static str,
        min: Option<f64>,
        max: Option<f64>,
        got: f64,
        range: &'static str,
    },
    /// A setting that is wrong only beside the others (an elite ratio that
    /// leaves no elite in this population), or an argument the run cannot
    /// start without (no destroy operator).
    Invalid {
        parameter: &'static str,
        reason: String,
    },
}

impl ConfigError {
    /// The setting the error is about.
    pub fn parameter(&self) -> &'static str {
        match self {
            Self::OutOfRange { parameter, .. } | Self::Invalid { parameter, .. } => parameter,
        }
    }

    pub(crate) fn out_of_range(
        parameter: &'static str,
        min: Option<f64>,
        max: Option<f64>,
        got: f64,
        range: &'static str,
    ) -> Self {
        Self::OutOfRange {
            parameter,
            min,
            max,
            got,
            range,
        }
    }

    pub(crate) fn invalid(parameter: &'static str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            parameter,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange {
                parameter,
                got,
                range,
                ..
            } => write!(f, "{parameter} must be {range}, got {got}"),
            Self::Invalid { parameter, reason } => write!(f, "{parameter}: {reason}"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_setting_and_its_range() {
        let e = ConfigError::out_of_range("mutation_rate", Some(0.0), Some(1.0), 1.5, "in [0, 1]");
        assert_eq!(e.to_string(), "mutation_rate must be in [0, 1], got 1.5");
        assert_eq!(e.parameter(), "mutation_rate");
        let e = ConfigError::invalid("destroy_ops", "at least one destroy operator required");
        assert_eq!(
            e.to_string(),
            "destroy_ops: at least one destroy operator required"
        );
    }
}
