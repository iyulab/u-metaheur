//! WASM bindings for u-metaheur.
//!
//! Exposes TSP (Travelling Salesman Problem) solvers to JavaScript via
//! `wasm-bindgen`. Only compiled when the `wasm` feature is enabled.
//!
//! Both `run_ga` and `run_sa` accept a JSON config and return a JSON result.
//! They use self-contained TSP implementations to avoid WASM incompatibilities
//! in the generic runners (e.g. `std::time::Instant`).
//!
//! # Usage (JavaScript)
//! ```js
//! import init, { run_ga, run_sa } from '@iyulab/u-metaheur';
//! await init();
//!
//! const gaResult = run_ga({
//!   nodes: [[0, 0], [1, 2], [3, 1]],
//!   population_size: 100,
//!   generations: 200,
//!   mutation_rate: 0.05,
//! });
//! console.log(gaResult.best_distance, gaResult.best_tour);
//!
//! const saResult = run_sa({
//!   nodes: [[0, 0], [1, 2], [3, 1]],
//!   initial_temp: 1000.0,
//!   cooling_rate: 0.995,
//!   iterations: 5000,
//! });
//! console.log(saResult.best_distance, saResult.best_tour);
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ============================================================================
// Shared utilities
// ============================================================================

/// Euclidean distance between two 2-D nodes.
fn node_dist(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Total tour distance (closed loop: last node → first node).
fn tour_distance(tour: &[usize], nodes: &[[f64; 2]]) -> f64 {
    let n = tour.len();
    if n == 0 {
        return 0.0;
    }
    (0..n)
        .map(|i| node_dist(&nodes[tour[i]], &nodes[tour[(i + 1) % n]]))
        .sum()
}

/// Nearest-neighbour greedy tour starting from node 0.
fn nearest_neighbour_tour(nodes: &[[f64; 2]]) -> Vec<usize> {
    let n = nodes.len();
    let mut visited = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    let mut current = 0;
    visited[current] = true;
    tour.push(current);

    for _ in 1..n {
        let next = (0..n)
            .filter(|&j| !visited[j])
            .min_by(|&a, &b| {
                node_dist(&nodes[current], &nodes[a])
                    .partial_cmp(&node_dist(&nodes[current], &nodes[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("at least one unvisited node remains");
        visited[next] = true;
        tour.push(next);
        current = next;
    }
    tour
}

/// Simple linear congruential generator for WASM-safe randomness.
///
/// Uses `getrandom` (already a wasm_js dependency) via `rand` to seed,
/// then produces fast uniform `u64` values.
struct WasmRng {
    state: u64,
}

impl WasmRng {
    fn new() -> Self {
        // Use rand's thread_rng for seeding — rand uses getrandom under the
        // hood which is already configured for WASM via the wasm_js feature.
        use rand::RngCore;
        let seed = rand::rng().next_u64();
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Splitmix64
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform `usize` in `[0, n)`.
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Fisher-Yates shuffle using `WasmRng`.
fn shuffle(slice: &mut [usize], rng: &mut WasmRng) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        slice.swap(i, j);
    }
}

// ============================================================================
// GA — Genetic Algorithm for TSP
// ============================================================================

#[derive(Deserialize)]
struct GaConfig {
    nodes: Vec<[f64; 2]>,
    #[serde(default = "default_pop_size")]
    population_size: usize,
    #[serde(default = "default_generations")]
    generations: usize,
    #[serde(default = "default_mutation_rate")]
    mutation_rate: f64,
}

fn default_pop_size() -> usize {
    100
}
fn default_generations() -> usize {
    200
}
fn default_mutation_rate() -> f64 {
    0.05
}

#[derive(Serialize)]
struct GaResult {
    best_distance: f64,
    best_tour: Vec<usize>,
    generations_run: usize,
}

/// Order-crossover (OX1) operator for permutation chromosomes.
fn ox_crossover(p1: &[usize], p2: &[usize], rng: &mut WasmRng) -> Vec<usize> {
    use std::collections::HashSet;

    let n = p1.len();
    let a = rng.next_usize(n);
    let b = rng.next_usize(n);
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };

    let mut child = vec![usize::MAX; n];
    // Copy the segment [lo..=hi] from p1
    child[lo..=hi].copy_from_slice(&p1[lo..=hi]);
    // Build a set of genes already placed for O(1) membership tests
    let used: HashSet<usize> = child[lo..=hi].iter().copied().collect();
    // Fill remaining positions in p2 order
    let mut pos = (hi + 1) % n;
    for &gene in p2.iter().cycle().skip(hi + 1).take(n) {
        if used.contains(&gene) {
            continue;
        }
        child[pos] = gene;
        pos = (pos + 1) % n;
        if pos == lo {
            break;
        }
    }
    child
}

/// Swap mutation: swap two random positions.
fn swap_mutate(tour: &mut [usize], rng: &mut WasmRng) {
    let n = tour.len();
    let i = rng.next_usize(n);
    let j = rng.next_usize(n);
    tour.swap(i, j);
}

/// Runs a Genetic Algorithm for TSP.
///
/// # Arguments
/// `config_json` — JS object with fields:
/// - `nodes`: `[[x, y], ...]` (required)
/// - `population_size`: integer (default 100)
/// - `generations`: integer (default 200)
/// - `mutation_rate`: float 0–1 (default 0.05)
///
/// # Returns
/// JS object with `best_distance`, `best_tour`, `generations_run`.
#[wasm_bindgen]
pub fn run_ga(config_json: JsValue) -> Result<JsValue, JsValue> {
    let config: GaConfig = serde_wasm_bindgen::from_value(config_json)
        .map_err(|e| JsValue::from_str(&format!("invalid config: {e}")))?;

    let n = config.nodes.len();
    if n < 2 {
        return Err(JsValue::from_str("need at least 2 nodes"));
    }
    if config.population_size < 2 {
        return Err(JsValue::from_str("population_size must be at least 2"));
    }
    if config.generations == 0 {
        return Err(JsValue::from_str("generations must be at least 1"));
    }

    let mut rng = WasmRng::new();
    let nodes = &config.nodes;

    // Initialise population: first individual = nearest-neighbour, rest random.
    let mut population: Vec<Vec<usize>> = {
        let nn = nearest_neighbour_tour(nodes);
        let mut pop = vec![nn];
        while pop.len() < config.population_size {
            let mut tour: Vec<usize> = (0..n).collect();
            shuffle(&mut tour, &mut rng);
            pop.push(tour);
        }
        pop
    };

    let mut distances: Vec<f64> = population.iter().map(|t| tour_distance(t, nodes)).collect();

    let best_idx = distances
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .expect("population non-empty");

    let mut best_tour = population[best_idx].clone();
    let mut best_dist = distances[best_idx];

    let elite_count = (config.population_size / 10).max(1);

    for _ in 0..config.generations {
        // Sort by distance (ascending)
        let mut indexed: Vec<usize> = (0..config.population_size).collect();
        indexed.sort_by(|&a, &b| {
            distances[a]
                .partial_cmp(&distances[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut next_pop: Vec<Vec<usize>> = Vec::with_capacity(config.population_size);
        let mut next_dist: Vec<f64> = Vec::with_capacity(config.population_size);

        // Elitism: keep best individuals unchanged
        for &idx in indexed.iter().take(elite_count) {
            next_pop.push(population[idx].clone());
            next_dist.push(distances[idx]);
        }

        // Fill rest with tournament selection + OX crossover + swap mutation
        while next_pop.len() < config.population_size {
            // Tournament selection (size 3) for two parents
            let p1 = tournament_select(&distances, 3, &mut rng);
            let p2 = tournament_select(&distances, 3, &mut rng);

            let mut child = ox_crossover(&population[p1], &population[p2], &mut rng);

            if rng.next_f64() < config.mutation_rate {
                swap_mutate(&mut child, &mut rng);
            }

            let d = tour_distance(&child, nodes);
            if d < best_dist {
                best_dist = d;
                best_tour = child.clone();
            }
            next_pop.push(child);
            next_dist.push(d);
        }

        population = next_pop;
        distances = next_dist;
    }

    let result = GaResult {
        best_distance: best_dist,
        best_tour,
        generations_run: config.generations,
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("serialization error: {e}")))
}

/// Tournament selection: pick the best of `k` random individuals.
fn tournament_select(distances: &[f64], k: usize, rng: &mut WasmRng) -> usize {
    let n = distances.len();
    let mut best_idx = rng.next_usize(n);
    for _ in 1..k {
        let challenger = rng.next_usize(n);
        if distances[challenger] < distances[best_idx] {
            best_idx = challenger;
        }
    }
    best_idx
}

// ============================================================================
// SA — Simulated Annealing for TSP
// ============================================================================

#[derive(Deserialize)]
struct SaConfig {
    nodes: Vec<[f64; 2]>,
    #[serde(default = "default_temp")]
    initial_temp: f64,
    #[serde(default = "default_cooling")]
    cooling_rate: f64,
    #[serde(default = "default_iterations")]
    iterations: usize,
}

fn default_temp() -> f64 {
    1000.0
}
fn default_cooling() -> f64 {
    0.995
}
fn default_iterations() -> usize {
    5000
}

#[derive(Serialize)]
struct SaResult {
    best_distance: f64,
    best_tour: Vec<usize>,
    iterations_run: usize,
}

/// Runs Simulated Annealing for TSP.
///
/// Uses geometric cooling (`T *= cooling_rate` per iteration).
/// Neighbour move: 2-opt segment reversal between two random positions.
///
/// # Arguments
/// `config_json` — JS object with fields:
/// - `nodes`: `[[x, y], ...]` (required)
/// - `initial_temp`: float (default 1000.0)
/// - `cooling_rate`: float in (0, 1) (default 0.995)
/// - `iterations`: integer (default 5000)
///
/// # Returns
/// JS object with `best_distance`, `best_tour`, `iterations_run`.
#[wasm_bindgen]
pub fn run_sa(config_json: JsValue) -> Result<JsValue, JsValue> {
    let config: SaConfig = serde_wasm_bindgen::from_value(config_json)
        .map_err(|e| JsValue::from_str(&format!("invalid config: {e}")))?;

    let n = config.nodes.len();
    if n < 2 {
        return Err(JsValue::from_str("need at least 2 nodes"));
    }
    if config.initial_temp <= 0.0 {
        return Err(JsValue::from_str("initial_temp must be positive"));
    }
    if config.cooling_rate <= 0.0 || config.cooling_rate >= 1.0 {
        return Err(JsValue::from_str("cooling_rate must be in (0, 1)"));
    }
    if config.iterations == 0 {
        return Err(JsValue::from_str("iterations must be at least 1"));
    }

    let mut rng = WasmRng::new();
    let nodes = &config.nodes;

    // Start from a nearest-neighbour tour
    let mut current = nearest_neighbour_tour(nodes);
    let mut current_dist = tour_distance(&current, nodes);
    let mut best = current.clone();
    let mut best_dist = current_dist;

    let mut temp = config.initial_temp;

    for _ in 0..config.iterations {
        // 2-opt neighbour: reverse a random sub-segment
        let i = rng.next_usize(n);
        let j = rng.next_usize(n);
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };

        if lo == hi {
            temp *= config.cooling_rate;
            continue;
        }

        // Compute delta without full tour recalculation (O(1) for 2-opt)
        let before_lo = if lo == 0 { n - 1 } else { lo - 1 };
        let after_hi = (hi + 1) % n;

        let old_cost = node_dist(&nodes[current[before_lo]], &nodes[current[lo]])
            + node_dist(&nodes[current[hi]], &nodes[current[after_hi]]);
        let new_cost = node_dist(&nodes[current[before_lo]], &nodes[current[hi]])
            + node_dist(&nodes[current[lo]], &nodes[current[after_hi]]);
        let delta = new_cost - old_cost;

        let accept = if delta < 0.0 {
            true
        } else if temp > 1e-15 {
            rng.next_f64() < (-delta / temp).exp()
        } else {
            false
        };

        if accept {
            current[lo..=hi].reverse();
            current_dist += delta;

            if current_dist < best_dist {
                best_dist = current_dist;
                best = current.clone();
            }
        }

        temp *= config.cooling_rate;
    }

    let result = SaResult {
        best_distance: best_dist,
        best_tour: best,
        iterations_run: config.iterations,
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("serialization error: {e}")))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nodes() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }

    #[test]
    fn test_tour_distance_square() {
        let nodes = make_nodes();
        let tour = vec![0, 1, 2, 3];
        let d = tour_distance(&tour, &nodes);
        assert!(
            (d - 4.0).abs() < 1e-10,
            "square perimeter should be 4, got {d}"
        );
    }

    #[test]
    fn test_nearest_neighbour_tour_length() {
        let nodes = make_nodes();
        let tour = nearest_neighbour_tour(&nodes);
        assert_eq!(tour.len(), 4);
        // All nodes visited exactly once
        let mut seen = [false; 4];
        for &n in &tour {
            assert!(!seen[n], "node {n} visited twice");
            seen[n] = true;
        }
    }

    #[test]
    fn test_wasm_rng_range() {
        let mut rng = WasmRng { state: 12345 };
        for _ in 0..1000 {
            let v = rng.next_usize(10);
            assert!(v < 10);
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn test_ox_crossover_valid_permutation() {
        let mut rng = WasmRng { state: 42 };
        let p1 = vec![0, 1, 2, 3, 4];
        let p2 = vec![4, 3, 2, 1, 0];
        let child = ox_crossover(&p1, &p2, &mut rng);
        assert_eq!(child.len(), 5);
        let mut seen = [false; 5];
        for &g in &child {
            assert!(!seen[g], "duplicate gene {g}");
            seen[g] = true;
        }
    }

    #[test]
    fn test_tour_distance_empty() {
        assert_eq!(tour_distance(&[], &[]), 0.0);
    }
}
