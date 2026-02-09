//! Core traits for ALNS.

use rand::Rng;

/// A destroy operator removes elements from a solution.
///
/// Destroy operators partially disassemble a solution to create room
/// for improvement during the repair phase. The `degree` parameter
/// controls how much of the solution is destroyed (0.0 = none, 1.0 = all).
///
/// # References
///
/// Ropke & Pisinger (2006), Section 2
pub trait DestroyOperator<S>: Send + Sync {
    /// Returns a human-readable name for this operator.
    fn name(&self) -> &str;

    /// Destroys part of the solution.
    ///
    /// # Arguments
    /// * `solution` - The current solution to partially destroy
    /// * `degree` - Fraction of the solution to destroy, in [0, 1]
    /// * `rng` - Random number generator
    ///
    /// Returns the partially destroyed solution.
    fn destroy<R: Rng>(&self, solution: &S, degree: f64, rng: &mut R) -> S;
}

/// A repair operator reconstructs a (partially destroyed) solution.
///
/// Repair operators fill in the gaps left by destroy operators,
/// ideally producing an improved solution.
///
/// # References
///
/// Ropke & Pisinger (2006), Section 2
pub trait RepairOperator<S>: Send + Sync {
    /// Returns a human-readable name for this operator.
    fn name(&self) -> &str;

    /// Repairs a partially destroyed solution.
    ///
    /// # Arguments
    /// * `solution` - The partially destroyed solution
    /// * `rng` - Random number generator
    ///
    /// Returns a complete (repaired) solution.
    fn repair<R: Rng>(&self, solution: &S, rng: &mut R) -> S;
}

/// Defines an ALNS optimization problem.
///
/// The user implements initial solution generation and cost evaluation.
/// Destroy and repair operators are provided separately.
///
/// # Examples
///
/// ```ignore
/// struct TspProblem { distances: Vec<Vec<f64>> }
///
/// impl AlnsProblem for TspProblem {
///     type Solution = Vec<usize>;
///
///     fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
///         let mut tour: Vec<usize> = (0..self.distances.len()).collect();
///         u_numflow::random::shuffle(&mut tour, rng);
///         tour
///     }
///
///     fn cost(&self, tour: &Vec<usize>) -> f64 {
///         tour.windows(2).map(|w| self.distances[w[0]][w[1]]).sum()
///     }
/// }
/// ```
///
/// # References
///
/// Ropke & Pisinger (2006)
pub trait AlnsProblem: Send + Sync {
    /// The solution representation type.
    type Solution: Clone + Send;

    /// Creates a random initial solution.
    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Self::Solution;

    /// Computes the cost of a solution. Lower is better.
    fn cost(&self, solution: &Self::Solution) -> f64;
}
