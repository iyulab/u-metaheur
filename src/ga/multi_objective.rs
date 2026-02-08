//! Multi-objective optimization utilities.
//!
//! Domain-agnostic algorithms for multi-objective optimization,
//! suitable for use with NSGA-II and other Pareto-based methods.
//!
//! # Algorithms
//!
//! - [`non_dominated_sort`]: Fast non-dominated sorting (Deb et al., 2002)
//! - [`crowding_distance`]: Crowding distance assignment for diversity preservation
//!
//! # References
//!
//! - Deb et al. (2002), "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
//! - IEEE Transactions on Evolutionary Computation, 6(2), 182-197

/// Result of non-dominated sorting.
///
/// Each element of `ranks` corresponds to the Pareto rank of the solution
/// at the same index. Rank 0 is the Pareto front (non-dominated solutions).
#[derive(Debug, Clone)]
pub struct NondominatedSortResult {
    /// Pareto rank for each solution (0 = front).
    pub ranks: Vec<usize>,

    /// Indices grouped by front: `fronts[0]` contains rank-0 indices, etc.
    pub fronts: Vec<Vec<usize>>,
}

/// Fast non-dominated sorting.
///
/// Assigns a Pareto rank to each solution based on dominance relationships.
/// All objectives are **minimized**: lower values are better.
///
/// # Algorithm (Deb et al., 2002)
///
/// 1. For each pair of solutions, determine dominance
/// 2. Solutions dominated by no other belong to front 0 (rank 0)
/// 3. Remove front 0, repeat to find subsequent fronts
///
/// # Complexity
///
/// O(m * n²) where m = number of objectives, n = number of solutions
///
/// # Arguments
///
/// - `objectives`: Slice of objective vectors. Each inner slice contains
///   the objective values for one solution. All inner slices must have
///   the same length (the number of objectives).
///
/// # Panics
///
/// Panics if `objectives` is empty or if inner slices have inconsistent lengths.
///
/// # Example
///
/// ```
/// use u_metaheur::ga::multi_objective::non_dominated_sort;
///
/// let objectives = vec![
///     vec![1.0, 5.0],  // Solution A
///     vec![3.0, 3.0],  // Solution B
///     vec![5.0, 1.0],  // Solution C
///     vec![4.0, 4.0],  // Solution D — dominated by B
/// ];
///
/// let result = non_dominated_sort(&objectives);
///
/// // A, B, C are non-dominated (rank 0)
/// assert_eq!(result.ranks[0], 0); // A
/// assert_eq!(result.ranks[1], 0); // B
/// assert_eq!(result.ranks[2], 0); // C
/// assert_eq!(result.ranks[3], 1); // D — dominated by B
/// ```
pub fn non_dominated_sort(objectives: &[Vec<f64>]) -> NondominatedSortResult {
    let n = objectives.len();
    assert!(n > 0, "objectives must not be empty");

    if n == 1 {
        return NondominatedSortResult {
            ranks: vec![0],
            fronts: vec![vec![0]],
        };
    }

    let m = objectives[0].len();
    assert!(m > 0, "each solution must have at least one objective");
    debug_assert!(
        objectives.iter().all(|o| o.len() == m),
        "all objective vectors must have the same length"
    );

    let mut domination_count = vec![0usize; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut ranks = vec![0usize; n];
    let mut front_0 = Vec::new();

    // Compute dominance relationships
    for i in 0..n {
        for j in (i + 1)..n {
            match dominance_cmp(&objectives[i], &objectives[j]) {
                Dominance::Left => {
                    // i dominates j
                    dominated_by[i].push(j);
                    domination_count[j] += 1;
                }
                Dominance::Right => {
                    // j dominates i
                    dominated_by[j].push(i);
                    domination_count[i] += 1;
                }
                Dominance::Neither => {}
            }
        }

        if domination_count[i] == 0 {
            ranks[i] = 0;
            front_0.push(i);
        }
    }

    // Build subsequent fronts
    let mut fronts = vec![front_0];
    loop {
        let current = fronts.last().expect("fronts is initialized with front_0; never empty");
        let mut next_front = Vec::new();

        for &i in current {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    ranks[j] = fronts.len();
                    next_front.push(j);
                }
            }
        }

        if next_front.is_empty() {
            break;
        }
        fronts.push(next_front);
    }

    NondominatedSortResult { ranks, fronts }
}

/// Dominance comparison result.
#[derive(Debug, PartialEq)]
enum Dominance {
    /// Left dominates right.
    Left,
    /// Right dominates left.
    Right,
    /// Neither dominates the other.
    Neither,
}

/// Compare two solutions for Pareto dominance (minimization).
fn dominance_cmp(a: &[f64], b: &[f64]) -> Dominance {
    let mut a_better_in_some = false;
    let mut b_better_in_some = false;

    for (&va, &vb) in a.iter().zip(b.iter()) {
        if va < vb {
            a_better_in_some = true;
        } else if vb < va {
            b_better_in_some = true;
        }
    }

    match (a_better_in_some, b_better_in_some) {
        (true, false) => Dominance::Left,
        (false, true) => Dominance::Right,
        _ => Dominance::Neither,
    }
}

/// Crowding distance assignment for diversity preservation.
///
/// Computes the crowding distance for each solution, measuring how
/// spread out the solutions are in objective space. Higher distance
/// means the solution is more isolated (more diverse).
///
/// Boundary solutions (min/max for any objective) receive `f64::INFINITY`.
///
/// # Algorithm (Deb et al., 2002)
///
/// For each objective:
/// 1. Sort solutions by objective value
/// 2. Assign infinity to boundary solutions
/// 3. For interior solutions, add normalized distance to neighbors
///
/// # Complexity
///
/// O(m * n * log n) where m = number of objectives, n = number of solutions
///
/// # Arguments
///
/// - `objectives`: Objective vectors for the solutions (same format as
///   [`non_dominated_sort`]).
///
/// # Returns
///
/// A vector of crowding distances, one per solution.
///
/// # Example
///
/// ```
/// use u_metaheur::ga::multi_objective::crowding_distance;
///
/// let objectives = vec![
///     vec![1.0, 5.0],
///     vec![3.0, 3.0],
///     vec![5.0, 1.0],
/// ];
///
/// let distances = crowding_distance(&objectives);
///
/// // Boundary solutions get infinity
/// assert!(distances[0].is_infinite());
/// assert!(distances[2].is_infinite());
/// // Interior solution gets finite distance
/// assert!(distances[1].is_finite());
/// ```
pub fn crowding_distance(objectives: &[Vec<f64>]) -> Vec<f64> {
    let n = objectives.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let m = objectives[0].len();
    let mut distances = vec![0.0f64; n];

    #[allow(clippy::needless_range_loop)] // obj_idx is a column index into 2D data
    for obj_idx in 0..m {
        // Sort indices by this objective
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            objectives[a][obj_idx]
                .partial_cmp(&objectives[b][obj_idx])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary solutions get infinity
        distances[indices[0]] = f64::INFINITY;
        distances[indices[n - 1]] = f64::INFINITY;

        // Objective range for normalization
        let min_val = objectives[indices[0]][obj_idx];
        let max_val = objectives[indices[n - 1]][obj_idx];
        let range = max_val - min_val;

        if range > 0.0 {
            for i in 1..(n - 1) {
                let prev = objectives[indices[i - 1]][obj_idx];
                let next = objectives[indices[i + 1]][obj_idx];
                distances[indices[i]] += (next - prev) / range;
            }
        }
    }

    distances
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Non-dominated sort ----

    #[test]
    fn test_single_solution() {
        let objs = vec![vec![1.0, 2.0]];
        let result = non_dominated_sort(&objs);
        assert_eq!(result.ranks, vec![0]);
        assert_eq!(result.fronts.len(), 1);
        assert_eq!(result.fronts[0], vec![0]);
    }

    #[test]
    fn test_two_non_dominated() {
        let objs = vec![
            vec![1.0, 3.0], // good in obj0, bad in obj1
            vec![3.0, 1.0], // bad in obj0, good in obj1
        ];
        let result = non_dominated_sort(&objs);
        assert_eq!(result.ranks[0], 0);
        assert_eq!(result.ranks[1], 0);
        assert_eq!(result.fronts.len(), 1);
    }

    #[test]
    fn test_clear_dominance() {
        let objs = vec![
            vec![1.0, 1.0], // dominates all
            vec![2.0, 2.0], // dominated by 0
            vec![3.0, 3.0], // dominated by 0 and 1
        ];
        let result = non_dominated_sort(&objs);
        assert_eq!(result.ranks[0], 0);
        assert_eq!(result.ranks[1], 1);
        assert_eq!(result.ranks[2], 2);
        assert_eq!(result.fronts.len(), 3);
    }

    #[test]
    fn test_mixed_fronts() {
        let objs = vec![
            vec![1.0, 5.0], // front 0
            vec![3.0, 3.0], // front 0
            vec![5.0, 1.0], // front 0
            vec![4.0, 4.0], // dominated by [1] → front 1
            vec![6.0, 6.0], // dominated by all front-0 → front 1 (dominated by [3] too? no, [3] is front 1)
        ];
        let result = non_dominated_sort(&objs);
        assert_eq!(result.ranks[0], 0); // (1,5) front 0
        assert_eq!(result.ranks[1], 0); // (3,3) front 0
        assert_eq!(result.ranks[2], 0); // (5,1) front 0
        assert_eq!(result.ranks[3], 1); // (4,4) front 1
        assert_eq!(result.ranks[4], 2); // (6,6) front 2
    }

    #[test]
    fn test_all_equal() {
        let objs = vec![
            vec![2.0, 2.0],
            vec![2.0, 2.0],
            vec![2.0, 2.0],
        ];
        let result = non_dominated_sort(&objs);
        // All are non-dominated (identical solutions don't dominate each other)
        assert!(result.ranks.iter().all(|&r| r == 0));
    }

    #[test]
    fn test_three_objectives() {
        let objs = vec![
            vec![1.0, 5.0, 3.0], // front 0
            vec![3.0, 1.0, 5.0], // front 0
            vec![5.0, 3.0, 1.0], // front 0
            vec![4.0, 4.0, 4.0], // dominated by none of the above individually? check:
                                  // vs [0]: 4>1, 4<5, 4>3 → neither
                                  // vs [1]: 4>3, 4>1, 4<5 → neither
                                  // vs [2]: 4<5, 4>3, 4>1 → neither
                                  // front 0 too!
        ];
        let result = non_dominated_sort(&objs);
        assert_eq!(result.ranks[0], 0);
        assert_eq!(result.ranks[1], 0);
        assert_eq!(result.ranks[2], 0);
        assert_eq!(result.ranks[3], 0);
    }

    // ---- Crowding distance ----

    #[test]
    fn test_crowding_single() {
        let objs = vec![vec![1.0, 2.0]];
        let dist = crowding_distance(&objs);
        assert_eq!(dist.len(), 1);
        assert!(dist[0].is_infinite());
    }

    #[test]
    fn test_crowding_two() {
        let objs = vec![vec![1.0, 3.0], vec![3.0, 1.0]];
        let dist = crowding_distance(&objs);
        assert!(dist[0].is_infinite());
        assert!(dist[1].is_infinite());
    }

    #[test]
    fn test_crowding_three_points() {
        let objs = vec![
            vec![1.0, 5.0], // boundary
            vec![3.0, 3.0], // interior
            vec![5.0, 1.0], // boundary
        ];
        let dist = crowding_distance(&objs);
        assert!(dist[0].is_infinite()); // boundary
        assert!(dist[2].is_infinite()); // boundary
        assert!(dist[1].is_finite());   // interior
        assert!(dist[1] > 0.0);
    }

    #[test]
    fn test_crowding_evenly_spaced() {
        // Evenly spaced solutions on a line
        let objs = vec![
            vec![0.0, 4.0],
            vec![1.0, 3.0],
            vec![2.0, 2.0],
            vec![3.0, 1.0],
            vec![4.0, 0.0],
        ];
        let dist = crowding_distance(&objs);

        // Boundaries
        assert!(dist[0].is_infinite());
        assert!(dist[4].is_infinite());

        // Interior points should have equal crowding distance
        let d1 = dist[1];
        let d2 = dist[2];
        let d3 = dist[3];
        assert!((d1 - d2).abs() < 1e-10, "expected equal: {d1} vs {d2}");
        assert!((d2 - d3).abs() < 1e-10, "expected equal: {d2} vs {d3}");
    }

    #[test]
    fn test_crowding_zero_range_objective() {
        // One objective has zero range — should not cause division by zero
        let objs = vec![
            vec![1.0, 5.0],
            vec![2.0, 5.0],
            vec![3.0, 5.0],
        ];
        let dist = crowding_distance(&objs);
        assert!(dist[0].is_infinite());
        assert!(dist[2].is_infinite());
        // Interior: only non-zero range objective contributes
        assert!(dist[1].is_finite());
    }

    // ---- Integration: sort + distance ----

    #[test]
    fn test_sort_then_distance() {
        let objs = vec![
            vec![1.0, 5.0],
            vec![3.0, 3.0],
            vec![5.0, 1.0],
            vec![4.0, 4.0], // dominated
            vec![6.0, 6.0], // dominated
        ];

        let sort_result = non_dominated_sort(&objs);

        // Compute crowding distance for front 0 only
        let front_0_objs: Vec<Vec<f64>> = sort_result.fronts[0]
            .iter()
            .map(|&i| objs[i].clone())
            .collect();
        let dist = crowding_distance(&front_0_objs);

        assert_eq!(dist.len(), 3);
        // All 3 are boundary in a 3-point front
        // Actually: with 3 points, boundaries get inf, middle gets finite
        // But since there are exactly 3 in front 0, the middle one is finite
    }
}
