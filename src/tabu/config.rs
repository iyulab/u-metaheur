//! Tabu Search configuration.

/// Configuration parameters for Tabu Search.
///
/// # Examples
///
/// ```
/// use u_metaheur::tabu::TabuConfig;
///
/// let config = TabuConfig::default()
///     .with_max_iterations(1000)
///     .with_tabu_tenure(7)
///     .with_aspiration(true);
/// assert_eq!(config.max_iterations, 1000);
/// assert_eq!(config.tabu_tenure, 7);
/// ```
#[derive(Debug, Clone)]
pub struct TabuConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// How many iterations a move stays in the tabu list.
    pub tabu_tenure: usize,
    /// Whether to use aspiration criterion (override tabu if the move
    /// produces a new global best).
    pub aspiration: bool,
    /// Maximum iterations without improvement before stopping.
    pub max_no_improve: usize,
    /// Random seed (None for random).
    pub seed: Option<u64>,
}

impl Default for TabuConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            tabu_tenure: 7,
            aspiration: true,
            max_no_improve: 200,
            seed: None,
        }
    }
}

impl TabuConfig {
    /// Sets the maximum number of iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Sets the tabu tenure (number of iterations a move remains tabu).
    pub fn with_tabu_tenure(mut self, tenure: usize) -> Self {
        self.tabu_tenure = tenure;
        self
    }

    /// Enables or disables aspiration criterion.
    pub fn with_aspiration(mut self, aspiration: bool) -> Self {
        self.aspiration = aspiration;
        self
    }

    /// Sets maximum iterations without improvement.
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
