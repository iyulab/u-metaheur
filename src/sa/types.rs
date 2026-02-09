//! Core trait for Simulated Annealing.

use rand::Rng;

/// Defines a Simulated Annealing problem.
///
/// The user implements neighbor generation and cost evaluation.
/// The SA framework handles temperature management, acceptance
/// criterion, and cooling.
///
/// # Minimization
///
/// SA minimizes the cost function. For maximization, negate the cost.
///
/// # Examples
///
/// ```ignore
/// struct TspProblem { distances: Vec<Vec<f64>> }
///
/// impl SaProblem for TspProblem {
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
///
///     fn neighbor<R: Rng>(&self, tour: &Vec<usize>, rng: &mut R) -> Vec<usize> {
///         let mut new = tour.clone();
///         let i = rng.random_range(0..new.len());
///         let j = rng.random_range(0..new.len());
///         new.swap(i, j);
///         new
///     }
/// }
/// ```
///
/// # References
///
/// Kirkpatrick et al. (1983), Cerny (1985)
pub trait SaProblem: Send + Sync {
    /// The solution representation type.
    type Solution: Clone + Send;

    /// Creates a random initial solution.
    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Self::Solution;

    /// Computes the cost of a solution. Lower is better.
    fn cost(&self, solution: &Self::Solution) -> f64;

    /// Generates a neighbor of the current solution.
    ///
    /// The neighbor should be "close" to the current solution
    /// (small perturbation) but the neighborhood must be connected
    /// (any solution reachable from any other via a sequence of moves).
    fn neighbor<R: Rng>(&self, solution: &Self::Solution, rng: &mut R) -> Self::Solution;
}
