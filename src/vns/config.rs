//! Variable Neighborhood Search configuration.

/// Configuration parameters for Variable Neighborhood Search.
///
/// # Examples
///
/// ```
/// use u_metaheur::vns::VnsConfig;
///
/// let config = VnsConfig::default()
///     .with_max_iterations(1000)
///     .with_max_no_improve(100);
/// assert_eq!(config.max_iterations, 1000);
/// assert_eq!(config.max_no_improve, 100);
/// ```
#[derive(Debug, Clone)]
pub struct VnsConfig {
    /// Maximum number of outer iterations (complete passes through
    /// all neighborhoods).
    pub max_iterations: usize,
    /// Maximum iterations without improvement before stopping.
    pub max_no_improve: usize,
    /// Random seed (None for default seed).
    pub seed: Option<u64>,
}

impl Default for VnsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            max_no_improve: 200,
            seed: None,
        }
    }
}

impl VnsConfig {
    /// Sets the maximum number of outer iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Sets the maximum iterations without improvement.
    pub fn with_max_no_improve(mut self, n: usize) -> Self {
        self.max_no_improve = n;
        self
    }

    /// Sets the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}
