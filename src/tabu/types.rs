//! Core trait for Tabu Search problems.

use rand::Rng;

/// A move that transforms one solution into another.
///
/// The `key` is used to identify the move in the tabu list. Moves with the
/// same key are considered equivalent (e.g., "swap(3,7)" and "swap(7,3)").
#[derive(Debug, Clone)]
pub struct TabuMove<S: Clone> {
    /// The resulting solution after applying this move.
    pub solution: S,
    /// A string key identifying this move for tabu tracking.
    pub key: String,
    /// Cost of the resulting solution.
    pub cost: f64,
}

/// Defines a combinatorial optimization problem for Tabu Search.
///
/// Users implement this trait to specify:
/// - How to create an initial solution
/// - How to evaluate a solution's cost
/// - How to generate the neighborhood of a solution
///
/// # Type Parameters
///
/// * `Solution` — The solution representation (must be `Clone + Send`)
pub trait TabuProblem: Send + Sync {
    /// The solution type.
    type Solution: Clone + Send;

    /// Creates an initial solution.
    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Self::Solution;

    /// Evaluates the cost of a solution (lower is better).
    fn cost(&self, solution: &Self::Solution) -> f64;

    /// Generates neighboring solutions with their move keys.
    ///
    /// Each returned [`TabuMove`] includes the new solution, a move key
    /// for tabu tracking, and the solution cost.
    ///
    /// The neighborhood need not be exhaustive; a representative sample
    /// (e.g., random subset) is acceptable.
    fn neighbors<R: Rng>(
        &self,
        solution: &Self::Solution,
        rng: &mut R,
    ) -> Vec<TabuMove<Self::Solution>>;
}
