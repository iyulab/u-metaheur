//! The one error every runner's settings check returns.

use std::fmt;
use std::ops::{Bound, RangeBounds};

/// The ends of a range a setting accepts, split the way they are reported: a
/// closed (included) end as `min` / `max`, an open (excluded) end as
/// `exclusive_min` / `exclusive_max`. A side holds at most one of its two; an
/// unbounded side holds neither.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) min: Option<f64>,
    pub(crate) max: Option<f64>,
    pub(crate) exclusive_min: Option<f64>,
    pub(crate) exclusive_max: Option<f64>,
}

impl Bounds {
    /// Reads a range's ends: `1.0..` is "at least 1", `0.0..=1.0` is `[0, 1]`,
    /// `(Excluded(0.0), Included(1.0))` is `(0, 1]`.
    pub(crate) fn of(range: impl RangeBounds<f64>) -> Self {
        let split = |bound: Bound<&f64>| match bound {
            Bound::Included(&v) => (Some(v), None),
            Bound::Excluded(&v) => (None, Some(v)),
            Bound::Unbounded => (None, None),
        };
        let (min, exclusive_min) = split(range.start_bound());
        let (max, exclusive_max) = split(range.end_bound());
        Self {
            min,
            max,
            exclusive_min,
            exclusive_max,
        }
    }
}

/// A setting a runner refuses: which one, and what was wrong with it.
///
/// `parameter` is the setting's field name (`population_size`,
/// `mutation_rate`, `destroy_ops`, …), so a caller that exposes these settings
/// under its own names can say which of its inputs to change without reading
/// the message.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    /// A number outside the values the setting accepts. `min` and `max` are
    /// the closed (included) bounds, `exclusive_min` and `exclusive_max` the
    /// open (excluded) ones: a side carries at most one of its two, and neither
    /// when the setting is unbounded on that side — `(0, 1]` is
    /// `exclusive_min: Some(0.0)`, `max: Some(1.0)`. `range` says it in words
    /// (`"in (0, 1]"`, `"at least 2"`).
    OutOfRange {
        parameter: &'static str,
        min: Option<f64>,
        max: Option<f64>,
        exclusive_min: Option<f64>,
        exclusive_max: Option<f64>,
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

    /// `accepted` is the range the setting takes, each end open or closed as
    /// it is: `2.0..`, `0.0..=1.0`, `(Excluded(0.0), Unbounded)`.
    pub(crate) fn out_of_range(
        parameter: &'static str,
        accepted: impl RangeBounds<f64>,
        got: f64,
        range: &'static str,
    ) -> Self {
        let Bounds {
            min,
            max,
            exclusive_min,
            exclusive_max,
        } = Bounds::of(accepted);
        Self::OutOfRange {
            parameter,
            min,
            max,
            exclusive_min,
            exclusive_max,
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
        let e = ConfigError::out_of_range("mutation_rate", 0.0..=1.0, 1.5, "in [0, 1]");
        assert_eq!(e.to_string(), "mutation_rate must be in [0, 1], got 1.5");
        assert_eq!(e.parameter(), "mutation_rate");
        let e = ConfigError::invalid("destroy_ops", "at least one destroy operator required");
        assert_eq!(
            e.to_string(),
            "destroy_ops: at least one destroy operator required"
        );
    }

    /// An open end is reported as `exclusive_*`, never as `min` / `max`, and a
    /// closed end the other way round.
    #[test]
    fn open_ends_are_reported_apart_from_closed_ones() {
        let e = ConfigError::out_of_range(
            "reaction_factor",
            (Bound::Excluded(0.0), Bound::Included(1.0)),
            0.0,
            "in (0, 1]",
        );
        assert_eq!(
            e,
            ConfigError::OutOfRange {
                parameter: "reaction_factor",
                min: None,
                max: Some(1.0),
                exclusive_min: Some(0.0),
                exclusive_max: None,
                got: 0.0,
                range: "in (0, 1]",
            }
        );
        assert_eq!(e.to_string(), "reaction_factor must be in (0, 1], got 0");
        assert_eq!(
            Bounds::of(2.0..),
            Bounds {
                min: Some(2.0),
                max: None,
                exclusive_min: None,
                exclusive_max: None,
            }
        );
        assert_eq!(
            Bounds::of((Bound::Excluded(0.0), Bound::Excluded(1.0))),
            Bounds {
                min: None,
                max: None,
                exclusive_min: Some(0.0),
                exclusive_max: Some(1.0),
            }
        );
    }
}
