//! CP variable types.

/// An integer variable with a domain [min, max].
///
/// Represents a decision variable that takes integer values within
/// the specified bounds. Can be fixed to a single value.
#[derive(Debug, Clone)]
pub struct IntVar {
    /// Variable name (unique identifier within a model).
    pub name: String,
    /// Minimum value.
    pub min: i64,
    /// Maximum value.
    pub max: i64,
    /// Fixed value, if any.
    pub fixed: Option<i64>,
}

impl IntVar {
    /// Creates a new integer variable with the given bounds.
    pub fn new(name: impl Into<String>, min: i64, max: i64) -> Self {
        Self {
            name: name.into(),
            min,
            max,
            fixed: None,
        }
    }

    /// Creates a fixed integer variable.
    pub fn fixed(name: impl Into<String>, value: i64) -> Self {
        Self {
            name: name.into(),
            min: value,
            max: value,
            fixed: Some(value),
        }
    }

    /// Whether this variable is fixed to a single value.
    pub fn is_fixed(&self) -> bool {
        self.fixed.is_some()
    }

    /// Domain size (max - min + 1).
    pub fn domain_size(&self) -> i64 {
        self.max - self.min + 1
    }
}

/// A boolean variable (true/false decision).
#[derive(Debug, Clone)]
pub struct BoolVar {
    /// Variable name.
    pub name: String,
    /// Fixed value, if any.
    pub fixed: Option<bool>,
}

impl BoolVar {
    /// Creates a new boolean variable.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fixed: None,
        }
    }

    /// Creates a fixed boolean variable.
    pub fn fixed(name: impl Into<String>, value: bool) -> Self {
        Self {
            name: name.into(),
            fixed: Some(value),
        }
    }
}

/// A time variable representing a point in time.
///
/// Used for interval start/end times. Domain is [min, max].
#[derive(Debug, Clone)]
pub struct TimeVar {
    /// Minimum time.
    pub min: i64,
    /// Maximum time.
    pub max: i64,
    /// Fixed value, if any.
    pub fixed: Option<i64>,
}

impl TimeVar {
    /// Creates a new time variable.
    pub fn new(min: i64, max: i64) -> Self {
        Self {
            min,
            max,
            fixed: None,
        }
    }

    /// Creates a fixed time variable.
    pub fn fixed(value: i64) -> Self {
        Self {
            min: value,
            max: value,
            fixed: Some(value),
        }
    }

    /// Whether this variable is fixed.
    pub fn is_fixed(&self) -> bool {
        self.fixed.is_some()
    }
}

/// A duration variable representing a length of time.
#[derive(Debug, Clone)]
pub struct DurationVar {
    /// Minimum duration.
    pub min: i64,
    /// Maximum duration.
    pub max: i64,
    /// Fixed value, if any.
    pub fixed: Option<i64>,
}

impl DurationVar {
    /// Creates a duration variable with bounds.
    pub fn new(min: i64, max: i64) -> Self {
        Self {
            min,
            max,
            fixed: None,
        }
    }

    /// Creates a fixed duration.
    pub fn fixed(value: i64) -> Self {
        Self {
            min: value,
            max: value,
            fixed: Some(value),
        }
    }

    /// Whether this duration is fixed.
    pub fn is_fixed(&self) -> bool {
        self.fixed.is_some()
    }
}

/// An interval variable representing an activity with start, end, and duration.
///
/// The invariant `end = start + duration` is maintained by the solver.
/// Intervals can be optional (controlled by a presence literal).
///
/// # Examples
///
/// ```
/// use u_metaheur::cp::IntervalVar;
///
/// // Fixed-duration interval: start in [0, 100], duration = 50
/// let op = IntervalVar::new("op1", 0, 100, 50, 200);
/// assert_eq!(op.duration.fixed, Some(50));
///
/// // Optional interval
/// let opt = IntervalVar::new("op2", 0, 100, 30, 200).as_optional("op2_present");
/// assert!(opt.is_optional);
/// ```
#[derive(Debug, Clone)]
pub struct IntervalVar {
    /// Variable name.
    pub name: String,
    /// Start time variable.
    pub start: TimeVar,
    /// End time variable.
    pub end: TimeVar,
    /// Duration variable.
    pub duration: DurationVar,
    /// Whether this interval is optional.
    pub is_optional: bool,
    /// Presence literal (for optional intervals).
    pub presence: Option<BoolVar>,
}

impl IntervalVar {
    /// Creates a fixed-duration interval variable.
    ///
    /// # Arguments
    /// * `name` - Unique name
    /// * `start_min` - Earliest start time
    /// * `start_max` - Latest start time
    /// * `duration` - Fixed duration
    /// * `end_max` - Latest end time
    pub fn new(
        name: impl Into<String>,
        start_min: i64,
        start_max: i64,
        duration: i64,
        end_max: i64,
    ) -> Self {
        Self {
            name: name.into(),
            start: TimeVar::new(start_min, start_max),
            end: TimeVar::new(start_min + duration, end_max),
            duration: DurationVar::fixed(duration),
            is_optional: false,
            presence: None,
        }
    }

    /// Makes this interval optional with a presence literal.
    pub fn as_optional(mut self, presence_name: impl Into<String>) -> Self {
        self.is_optional = true;
        self.presence = Some(BoolVar::new(presence_name));
        self
    }

    /// Sets a variable duration instead of fixed.
    pub fn with_variable_duration(mut self, min: i64, max: i64) -> Self {
        self.duration = DurationVar::new(min, max);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_var() {
        let v = IntVar::new("x", 0, 10);
        assert_eq!(v.domain_size(), 11);
        assert!(!v.is_fixed());

        let f = IntVar::fixed("y", 5);
        assert!(f.is_fixed());
        assert_eq!(f.domain_size(), 1);
    }

    #[test]
    fn test_bool_var() {
        let b = BoolVar::new("flag");
        assert!(b.fixed.is_none());

        let f = BoolVar::fixed("flag2", true);
        assert_eq!(f.fixed, Some(true));
    }

    #[test]
    fn test_interval_var() {
        let iv = IntervalVar::new("op1", 0, 100, 50, 200);
        assert_eq!(iv.name, "op1");
        assert_eq!(iv.start.min, 0);
        assert_eq!(iv.start.max, 100);
        assert_eq!(iv.duration.fixed, Some(50));
        assert_eq!(iv.end.min, 50);
        assert_eq!(iv.end.max, 200);
        assert!(!iv.is_optional);
    }

    #[test]
    fn test_optional_interval() {
        let iv = IntervalVar::new("op1", 0, 100, 50, 200).as_optional("op1_present");
        assert!(iv.is_optional);
        assert_eq!(iv.presence.as_ref().unwrap().name, "op1_present");
    }

    #[test]
    fn test_variable_duration() {
        let iv = IntervalVar::new("op1", 0, 100, 50, 200).with_variable_duration(30, 70);
        assert!(!iv.duration.is_fixed());
        assert_eq!(iv.duration.min, 30);
        assert_eq!(iv.duration.max, 70);
    }
}
