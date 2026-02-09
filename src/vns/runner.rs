//! Variable Neighborhood Search execution engine.
//!
//! # Algorithm (Basic VNS)
//!
//! 1. Generate initial solution x, apply local search
//! 2. Set k = 0
//! 3. While stopping criterion not met:
//!    a. **Shaking**: Generate x' randomly in N_k(x)
//!    b. **Local search**: Apply local search to x' → x''
//!    c. **Move or not**: If f(x'') < f(x), set x = x'' and k = 0;
//!    otherwise k = k + 1
//!    d. If k = k_max, reset k = 0 (one full pass done)
//! 4. Return best solution found
//!
//! # Reference
//!
//! Mladenović, N. & Hansen, P. (1997). "Variable neighborhood search",
//! *Computers & Operations Research* 24(11), 1097-1100.

use super::config::VnsConfig;
use super::types::VnsProblem;

/// Result of a VNS run.
#[derive(Debug, Clone)]
pub struct VnsResult<S: Clone> {
    /// Best solution found.
    pub best: S,
    /// Cost of the best solution.
    pub best_cost: f64,
    /// Total iterations (neighborhood switches) executed.
    pub iterations: usize,
    /// Iteration at which the best solution was found.
    pub best_iteration: usize,
    /// Cost history (best cost at each outer iteration).
    pub cost_history: Vec<f64>,
}

/// Variable Neighborhood Search runner.
pub struct VnsRunner;

impl VnsRunner {
    /// Executes Basic VNS on the given problem.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use u_metaheur::vns::{VnsProblem, VnsConfig, VnsRunner};
    /// use rand::Rng;
    ///
    /// struct MyProblem;
    /// impl VnsProblem for MyProblem {
    ///     type Solution = Vec<usize>;
    ///     fn initial_solution<R: Rng>(&self, _rng: &mut R) -> Vec<usize> { vec![0, 1, 2] }
    ///     fn cost(&self, _sol: &Vec<usize>) -> f64 { 0.0 }
    ///     fn neighborhood_count(&self) -> usize { 2 }
    ///     fn shake<R: Rng>(&self, sol: &Vec<usize>, _k: usize, _rng: &mut R) -> Vec<usize> { sol.clone() }
    ///     fn local_search(&self, sol: &Vec<usize>) -> Vec<usize> { sol.clone() }
    /// }
    /// ```
    pub fn run<P: VnsProblem>(problem: &P, config: &VnsConfig) -> VnsResult<P::Solution> {
        let mut rng = match config.seed {
            Some(s) => u_numflow::random::create_rng(s),
            None => u_numflow::random::create_rng(42),
        };

        let k_max = problem.neighborhood_count();
        assert!(k_max > 0, "neighborhood_count must be at least 1");

        // Initialize with local search
        let initial = problem.initial_solution(&mut rng);
        let mut current = problem.local_search(&initial);
        let mut best = current.clone();
        let mut best_cost = problem.cost(&current);
        let mut best_iteration = 0;

        let mut cost_history = Vec::with_capacity(config.max_iterations);
        let mut no_improve_count = 0;
        let mut iteration = 0;

        for outer in 0..config.max_iterations {
            let mut k = 0;

            while k < k_max {
                // Shaking: random perturbation in neighborhood k
                let shaken = problem.shake(&current, k, &mut rng);

                // Local search on shaken solution
                let candidate = problem.local_search(&shaken);
                let candidate_cost = problem.cost(&candidate);

                if candidate_cost < best_cost - 1e-12 {
                    // Improvement found — accept and reset to first neighborhood
                    current = candidate;
                    best = current.clone();
                    best_cost = candidate_cost;
                    best_iteration = outer;
                    k = 0;
                    no_improve_count = 0;
                } else {
                    // No improvement — try next neighborhood
                    k += 1;
                    no_improve_count += 1;
                }

                iteration += 1;
            }

            cost_history.push(best_cost);

            // Stagnation check
            if no_improve_count >= config.max_no_improve {
                break;
            }
        }

        VnsResult {
            best,
            best_cost,
            iterations: iteration,
            best_iteration,
            cost_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vns::{VnsConfig, VnsProblem};
    use rand::Rng;

    // ---- Discretized quadratic: f(x) = (x - 10)^2, min at x = 10 ----

    struct DiscreteQuadratic;

    impl VnsProblem for DiscreteQuadratic {
        type Solution = i32;

        fn initial_solution<R: Rng>(&self, rng: &mut R) -> i32 {
            rng.random_range(-50..50)
        }

        fn cost(&self, &x: &i32) -> f64 {
            let d = x as f64 - 10.0;
            d * d
        }

        fn neighborhood_count(&self) -> usize {
            3
        }

        fn shake<R: Rng>(&self, &x: &i32, k: usize, rng: &mut R) -> i32 {
            let radius = (k as i32 + 1) * 2;
            x + rng.random_range(-radius..=radius)
        }

        fn local_search(&self, &x: &i32) -> i32 {
            // Simple hill-climbing: move toward 10
            let mut current = x;
            loop {
                let left = current - 1;
                let right = current + 1;
                let cost_current = {
                    let d = current as f64 - 10.0;
                    d * d
                };
                let cost_left = {
                    let d = left as f64 - 10.0;
                    d * d
                };
                let cost_right = {
                    let d = right as f64 - 10.0;
                    d * d
                };

                if cost_left < cost_current {
                    current = left;
                } else if cost_right < cost_current {
                    current = right;
                } else {
                    break;
                }
            }
            current
        }
    }

    #[test]
    fn test_vns_quadratic_finds_optimum() {
        let problem = DiscreteQuadratic;
        let config = VnsConfig::default().with_max_iterations(50).with_seed(42);

        let result = VnsRunner::run(&problem, &config);

        assert_eq!(
            result.best, 10,
            "expected optimum at x=10, got {}",
            result.best
        );
        assert!(
            result.best_cost < 1e-10,
            "expected zero cost, got {}",
            result.best_cost
        );
    }

    #[test]
    fn test_vns_cost_history_non_increasing() {
        let problem = DiscreteQuadratic;
        let config = VnsConfig::default().with_max_iterations(30).with_seed(42);

        let result = VnsRunner::run(&problem, &config);

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
    fn test_vns_stagnation_termination() {
        let problem = DiscreteQuadratic;
        let config = VnsConfig::default()
            .with_max_iterations(10_000)
            .with_max_no_improve(10)
            .with_seed(42);

        let result = VnsRunner::run(&problem, &config);

        // Should converge early (local search directly finds optimum)
        assert!(
            result.cost_history.len() < 10_000,
            "expected early termination, got {} outer iterations",
            result.cost_history.len()
        );
    }

    // ---- Permutation sorting with multiple neighborhoods ----

    struct PermSortVns {
        n: usize,
    }

    impl VnsProblem for PermSortVns {
        type Solution = Vec<usize>;

        fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
            let mut perm: Vec<usize> = (0..self.n).collect();
            u_numflow::random::shuffle(&mut perm, rng);
            perm
        }

        fn cost(&self, perm: &Vec<usize>) -> f64 {
            perm.iter().enumerate().filter(|&(i, &v)| i != v).count() as f64
        }

        fn neighborhood_count(&self) -> usize {
            3
        }

        fn shake<R: Rng>(&self, perm: &Vec<usize>, k: usize, rng: &mut R) -> Vec<usize> {
            let mut new = perm.clone();
            let swaps = k + 1; // k=0 → 1 swap, k=1 → 2 swaps, k=2 → 3 swaps
            for _ in 0..swaps {
                let i = rng.random_range(0..self.n);
                let j = rng.random_range(0..self.n);
                new.swap(i, j);
            }
            new
        }

        fn local_search(&self, perm: &Vec<usize>) -> Vec<usize> {
            // First-improvement swap local search
            let mut current = perm.clone();
            let n = current.len();
            let mut improved = true;
            while improved {
                improved = false;
                let current_cost: usize =
                    current.iter().enumerate().filter(|&(i, &v)| i != v).count();
                for i in 0..n {
                    for j in (i + 1)..n {
                        current.swap(i, j);
                        let new_cost: usize =
                            current.iter().enumerate().filter(|&(k, &v)| k != v).count();
                        if new_cost < current_cost {
                            improved = true;
                            break;
                        }
                        current.swap(i, j); // undo
                    }
                    if improved {
                        break;
                    }
                }
            }
            current
        }
    }

    #[test]
    fn test_vns_permutation_sort() {
        let problem = PermSortVns { n: 8 };
        let config = VnsConfig::default().with_max_iterations(100).with_seed(42);

        let result = VnsRunner::run(&problem, &config);

        assert!(
            result.best_cost < 1e-10,
            "expected sorted permutation (cost 0), got cost {}",
            result.best_cost
        );
    }

    #[test]
    fn test_vns_neighborhoods_explored() {
        let problem = DiscreteQuadratic;
        let config = VnsConfig::default().with_max_iterations(20).with_seed(42);

        let result = VnsRunner::run(&problem, &config);

        // iterations counts total neighborhood switches
        assert!(result.iterations > 0, "expected some iterations to execute");
    }

    #[test]
    fn test_vns_single_neighborhood() {
        // VNS with k_max = 1 reduces to repeated shake + local search
        struct SingleNeighborhood;

        impl VnsProblem for SingleNeighborhood {
            type Solution = i32;
            fn initial_solution<R: Rng>(&self, _rng: &mut R) -> i32 {
                20
            }
            fn cost(&self, &x: &i32) -> f64 {
                (x as f64).powi(2)
            }
            fn neighborhood_count(&self) -> usize {
                1
            }
            fn shake<R: Rng>(&self, &x: &i32, _k: usize, rng: &mut R) -> i32 {
                x + rng.random_range(-5..=5)
            }
            fn local_search(&self, &x: &i32) -> i32 {
                // Simple descent
                let mut c = x;
                loop {
                    let l = c - 1;
                    let r = c + 1;
                    let cc = (c as f64).powi(2);
                    let cl = (l as f64).powi(2);
                    let cr = (r as f64).powi(2);
                    if cl < cc {
                        c = l;
                    } else if cr < cc {
                        c = r;
                    } else {
                        break;
                    }
                }
                c
            }
        }

        let problem = SingleNeighborhood;
        let config = VnsConfig::default().with_max_iterations(50).with_seed(42);

        let result = VnsRunner::run(&problem, &config);

        assert_eq!(result.best, 0, "expected optimum at 0, got {}", result.best);
    }

    #[test]
    fn test_vns_config_defaults() {
        let config = VnsConfig::default();
        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.max_no_improve, 200);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_vns_config_builder() {
        let config = VnsConfig::default()
            .with_max_iterations(1000)
            .with_max_no_improve(50)
            .with_seed(123);

        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.max_no_improve, 50);
        assert_eq!(config.seed, Some(123));
    }

    #[test]
    fn test_vns_best_iteration_recorded() {
        let problem = DiscreteQuadratic;
        let config = VnsConfig::default().with_max_iterations(30).with_seed(42);

        let result = VnsRunner::run(&problem, &config);

        assert!(
            result.best_iteration < result.cost_history.len(),
            "best_iteration {} should be < cost_history length {}",
            result.best_iteration,
            result.cost_history.len()
        );
    }
}
