//! Tabu Search execution engine.
//!
//! # Algorithm
//!
//! 1. Generate initial solution
//! 2. At each iteration:
//!    a. Generate neighborhood
//!    b. Select the best non-tabu move (or tabu move satisfying aspiration)
//!    c. Apply the move, add its key to the tabu list
//!    d. Update global best if improved
//! 3. Terminate after max iterations or stagnation
//!
//! # Reference
//!
//! Glover, F. (1989). "Tabu Search—Part I", *ORSA Journal on Computing* 1(3), 190-206.
//! Glover, F. (1990). "Tabu Search—Part II", *ORSA Journal on Computing* 2(1), 4-32.

use std::collections::HashSet;
use std::collections::VecDeque;

use super::config::TabuConfig;
use super::types::TabuProblem;

/// Result of a Tabu Search run.
#[derive(Debug, Clone)]
pub struct TabuResult<S: Clone> {
    /// Best solution found.
    pub best: S,
    /// Cost of the best solution.
    pub best_cost: f64,
    /// Total iterations executed.
    pub iterations: usize,
    /// Iteration at which the best solution was found.
    pub best_iteration: usize,
    /// Cost history (best cost at each iteration).
    pub cost_history: Vec<f64>,
}

/// Tabu Search runner.
pub struct TabuRunner;

impl TabuRunner {
    /// Executes Tabu Search on the given problem.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use u_metaheur::tabu::{TabuProblem, TabuConfig, TabuRunner, TabuMove};
    /// use rand::Rng;
    ///
    /// struct MyProblem;
    /// impl TabuProblem for MyProblem {
    ///     type Solution = Vec<usize>;
    ///     fn initial_solution<R: Rng>(&self, _rng: &mut R) -> Vec<usize> { vec![0, 1, 2] }
    ///     fn cost(&self, _sol: &Vec<usize>) -> f64 { 0.0 }
    ///     fn neighbors<R: Rng>(&self, _sol: &Vec<usize>, _rng: &mut R) -> Vec<TabuMove<Vec<usize>>> { vec![] }
    /// }
    /// ```
    pub fn run<P: TabuProblem>(problem: &P, config: &TabuConfig) -> TabuResult<P::Solution> {
        let mut rng = match config.seed {
            Some(s) => u_numflow::random::create_rng(s),
            None => u_numflow::random::create_rng(42),
        };

        // Initialize
        let mut current = problem.initial_solution(&mut rng);
        let mut best = current.clone();
        let mut best_cost = problem.cost(&current);
        let mut best_iteration = 0;

        // Tabu list: FIFO queue of move keys with set for O(1) lookup
        let mut tabu_queue: VecDeque<String> = VecDeque::new();
        let mut tabu_set: HashSet<String> = HashSet::new();

        let mut cost_history = Vec::with_capacity(config.max_iterations);
        let mut no_improve_count = 0;

        for iteration in 0..config.max_iterations {
            // Generate neighborhood
            let neighbors = problem.neighbors(&current, &mut rng);

            if neighbors.is_empty() {
                cost_history.push(best_cost);
                break;
            }

            // Find best admissible move
            let mut best_move = None;
            let mut best_move_cost = f64::INFINITY;

            for mv in &neighbors {
                let is_tabu = tabu_set.contains(&mv.key);

                if is_tabu {
                    // Aspiration: override tabu if this produces a new global best
                    if config.aspiration && mv.cost < best_cost {
                        // Aspiration criterion met
                    } else {
                        continue;
                    }
                }

                if mv.cost < best_move_cost {
                    best_move_cost = mv.cost;
                    best_move = Some(mv);
                }
            }

            // If no admissible move found, try to find any non-tabu move
            // (even if it worsens). If all are tabu, pick the best tabu move.
            if best_move.is_none() {
                // All moves are tabu and none meets aspiration — pick least bad
                let mut fallback_cost = f64::INFINITY;
                for mv in &neighbors {
                    if mv.cost < fallback_cost {
                        fallback_cost = mv.cost;
                        best_move = Some(mv);
                    }
                }
            }

            if let Some(mv) = best_move {
                // Update tabu list
                if tabu_queue.len() >= config.tabu_tenure {
                    if let Some(old_key) = tabu_queue.pop_front() {
                        tabu_set.remove(&old_key);
                    }
                }
                tabu_queue.push_back(mv.key.clone());
                tabu_set.insert(mv.key.clone());

                // Move to neighbor
                current = mv.solution.clone();

                // Update global best
                if mv.cost < best_cost {
                    best = current.clone();
                    best_cost = mv.cost;
                    best_iteration = iteration;
                    no_improve_count = 0;
                } else {
                    no_improve_count += 1;
                }
            } else {
                no_improve_count += 1;
            }

            cost_history.push(best_cost);

            // Stagnation check
            if no_improve_count >= config.max_no_improve {
                break;
            }
        }

        TabuResult {
            best,
            best_cost,
            iterations: cost_history.len(),
            best_iteration,
            cost_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tabu::{TabuConfig, TabuMove, TabuProblem};
    use rand::Rng;

    // ---- Quadratic minimization: f(x) = (x - 5)^2, minimum at x = 5 ----

    struct DiscretizedQuadratic;

    impl TabuProblem for DiscretizedQuadratic {
        type Solution = i32;

        fn initial_solution<R: Rng>(&self, rng: &mut R) -> i32 {
            rng.random_range(-50..50)
        }

        fn cost(&self, &x: &i32) -> f64 {
            let d = x as f64 - 5.0;
            d * d
        }

        fn neighbors<R: Rng>(&self, &x: &i32, _rng: &mut R) -> Vec<TabuMove<i32>> {
            vec![
                TabuMove {
                    solution: x - 1,
                    key: format!("to_{}", x - 1),
                    cost: {
                        let d = (x - 1) as f64 - 5.0;
                        d * d
                    },
                },
                TabuMove {
                    solution: x + 1,
                    key: format!("to_{}", x + 1),
                    cost: {
                        let d = (x + 1) as f64 - 5.0;
                        d * d
                    },
                },
            ]
        }
    }

    #[test]
    fn test_tabu_quadratic_finds_optimum() {
        let problem = DiscretizedQuadratic;
        let config = TabuConfig::default()
            .with_max_iterations(200)
            .with_tabu_tenure(3)
            .with_seed(42);

        let result = TabuRunner::run(&problem, &config);

        assert_eq!(
            result.best, 5,
            "expected optimum at x=5, got {}",
            result.best
        );
        assert!(
            result.best_cost < 1e-10,
            "expected zero cost, got {}",
            result.best_cost
        );
    }

    #[test]
    fn test_tabu_cost_history_non_increasing() {
        let problem = DiscretizedQuadratic;
        let config = TabuConfig::default()
            .with_max_iterations(100)
            .with_tabu_tenure(5)
            .with_seed(42);

        let result = TabuRunner::run(&problem, &config);

        for window in result.cost_history.windows(2) {
            assert!(
                window[1] <= window[0] + 1e-10,
                "best cost history should be non-increasing: {} > {}",
                window[1],
                window[0]
            );
        }
    }

    #[test]
    fn test_tabu_stagnation_termination() {
        let problem = DiscretizedQuadratic;
        let config = TabuConfig::default()
            .with_max_iterations(10_000)
            .with_max_no_improve(20)
            .with_tabu_tenure(3)
            .with_seed(42);

        let result = TabuRunner::run(&problem, &config);

        // Should converge and stop well before max_iterations
        assert!(
            result.iterations < 10_000,
            "expected early termination, ran {} iterations",
            result.iterations
        );
    }

    #[test]
    fn test_tabu_best_iteration_recorded() {
        let problem = DiscretizedQuadratic;
        let config = TabuConfig::default()
            .with_max_iterations(100)
            .with_tabu_tenure(3)
            .with_seed(42);

        let result = TabuRunner::run(&problem, &config);

        assert!(
            result.best_iteration < result.iterations,
            "best_iteration {} should be < total iterations {}",
            result.best_iteration,
            result.iterations
        );
    }

    // ---- Permutation sorting with swap neighborhoods ----

    struct PermSortTabu {
        n: usize,
    }

    impl TabuProblem for PermSortTabu {
        type Solution = Vec<usize>;

        fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
            let mut perm: Vec<usize> = (0..self.n).collect();
            u_numflow::random::shuffle(&mut perm, rng);
            perm
        }

        fn cost(&self, perm: &Vec<usize>) -> f64 {
            perm.iter().enumerate().filter(|&(i, &v)| i != v).count() as f64
        }

        fn neighbors<R: Rng>(&self, perm: &Vec<usize>, _rng: &mut R) -> Vec<TabuMove<Vec<usize>>> {
            let n = perm.len();
            let mut moves = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut new_perm = perm.clone();
                    new_perm.swap(i, j);
                    let c = new_perm
                        .iter()
                        .enumerate()
                        .filter(|&(k, &v)| k != v)
                        .count() as f64;
                    let key = if i < j {
                        format!("swap_{i}_{j}")
                    } else {
                        format!("swap_{j}_{i}")
                    };
                    moves.push(TabuMove {
                        solution: new_perm,
                        key,
                        cost: c,
                    });
                }
            }
            moves
        }
    }

    #[test]
    fn test_tabu_permutation_sort() {
        let problem = PermSortTabu { n: 8 };
        let config = TabuConfig::default()
            .with_max_iterations(500)
            .with_tabu_tenure(5)
            .with_max_no_improve(100)
            .with_seed(42);

        let result = TabuRunner::run(&problem, &config);

        assert!(
            result.best_cost < 1e-10,
            "expected sorted permutation (cost 0), got cost {}",
            result.best_cost
        );
    }

    #[test]
    fn test_tabu_aspiration_criterion() {
        // With a very high tenure, moves become tabu quickly.
        // Aspiration should still allow moves that improve the global best.
        let problem = DiscretizedQuadratic;
        let config_with_aspiration = TabuConfig::default()
            .with_max_iterations(200)
            .with_tabu_tenure(50) // very long tenure
            .with_aspiration(true)
            .with_seed(42);

        let result_aspiration = TabuRunner::run(&problem, &config_with_aspiration);

        let config_no_aspiration = TabuConfig::default()
            .with_max_iterations(200)
            .with_tabu_tenure(50)
            .with_aspiration(false)
            .with_seed(42);

        let result_no_aspiration = TabuRunner::run(&problem, &config_no_aspiration);

        // With aspiration, the search should find better or equal solutions
        assert!(
            result_aspiration.best_cost <= result_no_aspiration.best_cost,
            "aspiration should help: {} vs {}",
            result_aspiration.best_cost,
            result_no_aspiration.best_cost
        );
    }

    #[test]
    fn test_tabu_tenure_effect() {
        let problem = PermSortTabu { n: 6 };

        // Short tenure → faster convergence possible
        let config_short = TabuConfig::default()
            .with_max_iterations(300)
            .with_tabu_tenure(2)
            .with_seed(42);

        let result_short = TabuRunner::run(&problem, &config_short);

        // Long tenure → more exploration
        let config_long = TabuConfig::default()
            .with_max_iterations(300)
            .with_tabu_tenure(10)
            .with_seed(42);

        let result_long = TabuRunner::run(&problem, &config_long);

        // Both should find good solutions
        assert!(
            result_short.best_cost <= 2.0,
            "short tenure should find good solution, got {}",
            result_short.best_cost
        );
        assert!(
            result_long.best_cost <= 2.0,
            "long tenure should find good solution, got {}",
            result_long.best_cost
        );
    }

    #[test]
    fn test_tabu_empty_neighborhood() {
        struct EmptyNeighborhood;

        impl TabuProblem for EmptyNeighborhood {
            type Solution = i32;

            fn initial_solution<R: Rng>(&self, _rng: &mut R) -> i32 {
                0
            }

            fn cost(&self, &x: &i32) -> f64 {
                x as f64
            }

            fn neighbors<R: Rng>(&self, _sol: &i32, _rng: &mut R) -> Vec<TabuMove<i32>> {
                vec![]
            }
        }

        let problem = EmptyNeighborhood;
        let config = TabuConfig::default().with_seed(42);
        let result = TabuRunner::run(&problem, &config);

        // Should terminate immediately with initial solution
        assert_eq!(result.best, 0);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_tabu_config_defaults() {
        let config = TabuConfig::default();
        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.tabu_tenure, 7);
        assert!(config.aspiration);
        assert_eq!(config.max_no_improve, 200);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_tabu_config_builder() {
        let config = TabuConfig::default()
            .with_max_iterations(1000)
            .with_tabu_tenure(10)
            .with_aspiration(false)
            .with_max_no_improve(50)
            .with_seed(123);

        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tabu_tenure, 10);
        assert!(!config.aspiration);
        assert_eq!(config.max_no_improve, 50);
        assert_eq!(config.seed, Some(123));
    }
}
