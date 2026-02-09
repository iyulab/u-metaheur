//! Core trait for Variable Neighborhood Search.

use rand::Rng;

/// Defines a Variable Neighborhood Search problem.
///
/// The user supplies multiple neighborhood structures of increasing
/// "distance" from the current solution. VNS systematically switches
/// among these neighborhoods to escape local optima.
///
/// # Minimization
///
/// VNS minimizes the cost function. For maximization, negate the cost.
///
/// # References
///
/// Mladenović, N. & Hansen, P. (1997). "Variable neighborhood search",
/// *Computers & Operations Research* 24(11), 1097-1100.
pub trait VnsProblem: Send + Sync {
    /// The solution representation type.
    type Solution: Clone + Send;

    /// Creates an initial solution.
    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Self::Solution;

    /// Computes the cost of a solution. Lower is better.
    fn cost(&self, solution: &Self::Solution) -> f64;

    /// Returns the number of neighborhood structures (k_max).
    ///
    /// Neighborhoods are indexed from `0` to `neighborhood_count() - 1`,
    /// where lower indices correspond to smaller, more local perturbations
    /// and higher indices to larger, more diversified perturbations.
    fn neighborhood_count(&self) -> usize;

    /// Generates a random neighbor in the k-th neighborhood (shaking).
    ///
    /// The perturbation strength should increase with `k`. For example:
    /// - k=0: swap two adjacent elements
    /// - k=1: swap two random elements
    /// - k=2: reverse a random segment
    fn shake<R: Rng>(&self, solution: &Self::Solution, k: usize, rng: &mut R) -> Self::Solution;

    /// Performs local search starting from the given solution.
    ///
    /// Returns the locally optimal solution. This is the "improvement"
    /// step in VNS. A simple implementation can just return the input
    /// solution (making VNS degenerate to Variable Neighborhood Descent).
    fn local_search(&self, solution: &Self::Solution) -> Self::Solution;
}
