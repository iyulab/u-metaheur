//! WASM bindings for u-metaheur.
//!
//! Exposes TSP (Travelling Salesman Problem) solvers to JavaScript via
//! `wasm-bindgen`. Only compiled when the `wasm` feature is enabled.
//!
//! Both `run_ga` and `run_sa` accept a native JS config object (not a JSON
//! string) and return a plain JS result object.
//! They use self-contained TSP implementations to avoid WASM incompatibilities
//! in the generic runners (e.g. `std::time::Instant`).
//!
//! # Usage (JavaScript)
//! ```js
//! import { run_ga, run_sa } from '@iyulab/u-metaheur';
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
//!
//! Every refusal throws an `Error` whose `message` is readable text and which
//! carries `code` -- a stable reason -- and the values behind it
//! (`parameter`, `min`, `max`, `got`). See the README's *Errors*.

use serde::{Deserialize, Serialize};
use serde_json::json;
use wasm_bindgen::prelude::*;

// ============================================================================
// Shared utilities
// ============================================================================

/// A refusal on its way to JavaScript: the text for `Error.message`, and the
/// fields -- `code` first among them -- copied onto the `Error`.
#[derive(Debug)]
struct WireError {
    message: String,
    fields: serde_json::Value,
}

impl WireError {
    fn new(code: &str, message: String, mut extra: serde_json::Value) -> Self {
        let mut fields = serde_json::Map::new();
        fields.insert("code".into(), json!(code));
        if let Some(extra) = extra.as_object_mut() {
            fields.append(extra);
        }
        WireError {
            message,
            fields: serde_json::Value::Object(fields),
        }
    }

    /// An argument that is not the shape the function takes: a JSON string
    /// instead of a value, a wrong type, a missing or unknown key.
    fn malformed_input(parameter: &str, message: String) -> Self {
        Self::new(
            "malformed_input",
            message,
            json!({ "parameter": parameter }),
        )
    }

    /// A setting outside the range the solver accepts. `max` is `None` when
    /// the range is open above; the message says whether a bound is included.
    fn out_of_range(
        parameter: &str,
        min: f64,
        max: Option<f64>,
        got: f64,
        message: String,
    ) -> Self {
        Self::new(
            "parameter_out_of_range",
            message,
            json!({ "parameter": parameter, "min": min, "max": max, "got": got }),
        )
    }

    /// Fewer nodes than a tour needs.
    fn too_few_nodes(got: usize) -> Self {
        Self::new(
            "insufficient_data",
            format!("need at least 2 nodes, got {got}"),
            json!({ "parameter": "nodes", "min": 2, "got": got }),
        )
    }
}

/// Every refusal crosses into JavaScript as an `Error` whose `message` is the
/// readable text and which carries `code` and the values behind it as further
/// properties.
fn js_err(error: WireError) -> JsValue {
    let js = js_sys::Error::new(&error.message);
    // `json_compatible` turns the map into a plain object; the default would
    // produce a JavaScript `Map`, which `Object.assign` does not read.
    if let Ok(fields) = error
        .fields
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    {
        js_sys::Object::assign(&js, &fields.into());
    }
    js.into()
}

/// Serializes a response; a failure is reported rather than unwrapped.
fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value)
        .map_err(|e| js_err(WireError::malformed_input("result", e.to_string())))
}

/// A NaN or ±Infinity found in a JS argument, and where it sits.
///
/// JSON has no non-finite numbers, so on the way to the wire schema
/// `serde_json` turns one into `null` and the caller would be told a value has
/// the wrong type. [`find_non_finite`] looks before that happens, so the
/// refusal names the real reason and the place.
struct NonFinite {
    /// The argument's name, then `.key` and `[i]` steps down to the array or
    /// field that holds the number.
    parameter: String,
    /// The number's position, when it is an array element.
    index: Option<usize>,
    value: f64,
}

impl NonFinite {
    fn message(&self) -> String {
        let at = match self.index {
            Some(i) => format!("{}[{i}]", self.parameter),
            None => self.parameter.clone(),
        };
        let got = if self.value.is_nan() {
            "NaN"
        } else if self.value > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        };
        format!("{at}: expected a finite number, got {got}")
    }

    /// `parameter` and `index` (`null` when the number is not an array element).
    fn fields(&self) -> serde_json::Value {
        serde_json::json!({ "parameter": self.parameter, "index": self.index })
    }
}

/// The first NaN or ±Infinity in `value`, searching arrays, iterables and
/// plain objects. `allow_nan` lets NaN through for an input that reads it as a
/// missing value; it then arrives as `null`.
fn find_non_finite(value: &JsValue, parameter: &str, allow_nan: bool) -> Option<NonFinite> {
    let refused = |n: f64| !n.is_finite() && !(allow_nan && n.is_nan());
    let found = |index: Option<usize>, value: f64| NonFinite {
        parameter: parameter.to_string(),
        index,
        value,
    };
    if let Some(n) = value.as_f64() {
        return refused(n).then(|| found(None, n));
    }
    if !value.is_object() {
        return None;
    }
    if let Ok(Some(items)) = js_sys::try_iter(value) {
        for (i, item) in items.enumerate() {
            // An iterator that throws is left for serde to report.
            let item = item.ok()?;
            match item.as_f64() {
                Some(n) if refused(n) => return Some(found(Some(i), n)),
                Some(_) => {}
                None => {
                    let inner = find_non_finite(&item, &format!("{parameter}[{i}]"), allow_nan);
                    if inner.is_some() {
                        return inner;
                    }
                }
            }
        }
        return None;
    }
    let object: &js_sys::Object = wasm_bindgen::JsCast::unchecked_ref(value);
    for entry in js_sys::Object::entries(object).iter() {
        let pair: js_sys::Array = wasm_bindgen::JsCast::unchecked_into(entry);
        let key = pair.get(0).as_string().unwrap_or_default();
        let inner = find_non_finite(&pair.get(1), &format!("{parameter}.{key}"), allow_nan);
        if inner.is_some() {
            return inner;
        }
    }
    None
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    let refuse = |message: String| js_err(WireError::malformed_input(param, message));
    if value.as_string().is_some() {
        return Err(refuse(format!(
            "{param}: expected a native JS object/array, got a string — \
             pass the value directly, not JSON.stringify(...)"
        )));
    }
    if let Some(found) = find_non_finite(&value, param, false) {
        return Err(js_err(WireError::new(
            "value_not_finite",
            found.message(),
            found.fields(),
        )));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value =
        serde_wasm_bindgen::from_value(value).map_err(|e| refuse(format!("{param}: {e}")))?;
    serde_json::from_value(json).map_err(|e| refuse(format!("{param}: {e}")))
}

/// The settings `run_ga` refuses.
fn check_ga(config: &GaConfig) -> Result<(), WireError> {
    if config.nodes.len() < 2 {
        return Err(WireError::too_few_nodes(config.nodes.len()));
    }
    if config.population_size < 2 {
        return Err(WireError::out_of_range(
            "population_size",
            2.0,
            None,
            config.population_size as f64,
            format!(
                "population_size must be at least 2, got {}",
                config.population_size
            ),
        ));
    }
    if config.generations == 0 {
        return Err(WireError::out_of_range(
            "generations",
            1.0,
            None,
            0.0,
            "generations must be at least 1, got 0".to_string(),
        ));
    }
    Ok(())
}

/// The settings `run_sa` refuses.
fn check_sa(config: &SaConfig) -> Result<(), WireError> {
    if config.nodes.len() < 2 {
        return Err(WireError::too_few_nodes(config.nodes.len()));
    }
    if config.initial_temp <= 0.0 {
        return Err(WireError::out_of_range(
            "initial_temp",
            0.0,
            None,
            config.initial_temp,
            format!(
                "initial_temp must be positive (> 0), got {}",
                config.initial_temp
            ),
        ));
    }
    if config.cooling_rate <= 0.0 || config.cooling_rate >= 1.0 {
        return Err(WireError::out_of_range(
            "cooling_rate",
            0.0,
            Some(1.0),
            config.cooling_rate,
            format!(
                "cooling_rate must be in (0, 1), both excluded, got {}",
                config.cooling_rate
            ),
        ));
    }
    if config.iterations == 0 {
        return Err(WireError::out_of_range(
            "iterations",
            1.0,
            None,
            0.0,
            "iterations must be at least 1, got 0".to_string(),
        ));
    }
    Ok(())
}

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
        use rand::Rng;
        let seed = rand::rng().next_u64();
        Self { state: seed }
    }

    #[cfg(test)]
    fn with_seed(seed: u64) -> Self {
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

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct GaConfig {
    nodes: Vec<[f64; 2]>,
    #[serde(default = "default_pop_size")]
    #[tsify(optional)]
    population_size: usize,
    #[serde(default = "default_generations")]
    #[tsify(optional)]
    generations: usize,
    #[serde(default = "default_mutation_rate")]
    #[tsify(optional)]
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

#[derive(Serialize, tsify::Tsify)]
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
/// `config` — native JS object with fields:
/// - `nodes`: `[[x, y], ...]` (required)
/// - `population_size`: integer (default 100)
/// - `generations`: integer (default 200)
/// - `mutation_rate`: float 0–1 (default 0.05)
///
/// # Returns
/// JS object with `best_distance`, `best_tour`, `generations_run`.
#[wasm_bindgen(unchecked_return_type = "GaResult")]
pub fn run_ga(
    #[wasm_bindgen(unchecked_param_type = "GaConfig")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: GaConfig = from_js(config, "config")?;
    check_ga(&config).map_err(js_err)?;
    let n = config.nodes.len();

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

    to_js(&result)
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

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct SaConfig {
    nodes: Vec<[f64; 2]>,
    #[serde(default = "default_temp")]
    #[tsify(optional)]
    initial_temp: f64,
    #[serde(default = "default_cooling")]
    #[tsify(optional)]
    cooling_rate: f64,
    #[serde(default = "default_iterations")]
    #[tsify(optional)]
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

#[derive(Serialize, tsify::Tsify)]
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
/// `config` — native JS object with fields:
/// - `nodes`: `[[x, y], ...]` (required)
/// - `initial_temp`: float (default 1000.0)
/// - `cooling_rate`: float in (0, 1) (default 0.995)
/// - `iterations`: integer (default 5000)
///
/// # Returns
/// JS object with `best_distance`, `best_tour`, `iterations_run`.
#[wasm_bindgen(unchecked_return_type = "SaResult")]
pub fn run_sa(
    #[wasm_bindgen(unchecked_param_type = "SaConfig")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: SaConfig = from_js(config, "config")?;
    check_sa(&config).map_err(js_err)?;

    let mut rng = WasmRng::new();
    let (best, best_dist) = anneal(
        &config.nodes,
        config.initial_temp,
        config.cooling_rate,
        config.iterations,
        &mut rng,
    );

    let result = SaResult {
        best_distance: best_dist,
        best_tour: best,
        iterations_run: config.iterations,
    };

    to_js(&result)
}

/// Simulated annealing over 2-opt moves, from a nearest-neighbour tour.
///
/// Returns the best tour seen and its length. The length is tracked
/// incrementally from each move's delta, which is exact for every move except
/// one: reversing the whole tour (`lo == 0`, `hi == n - 1`). On a cycle that
/// changes nothing, but the two edges the delta formula reads are then the
/// same edge, so the formula reported a saving of twice its length. Summed
/// over a run, the tracked length drifted below zero and the "best" tour was
/// chosen by a length it did not have; the move is skipped like `lo == hi`.
fn anneal(
    nodes: &[[f64; 2]],
    initial_temp: f64,
    cooling_rate: f64,
    iterations: usize,
    rng: &mut WasmRng,
) -> (Vec<usize>, f64) {
    let n = nodes.len();
    let mut current = nearest_neighbour_tour(nodes);
    let mut current_dist = tour_distance(&current, nodes);
    let mut best = current.clone();
    let mut best_dist = current_dist;

    let mut temp = initial_temp;

    for _ in 0..iterations {
        // 2-opt neighbour: reverse a random sub-segment
        let i = rng.next_usize(n);
        let j = rng.next_usize(n);
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };

        if lo == hi || hi - lo == n - 1 {
            temp *= cooling_rate;
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

        temp *= cooling_rate;
    }

    (best, best_dist)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    /// Every refusal names its reason as a `code` and carries the values
    /// behind it.
    #[test]
    fn refusals_carry_their_code_and_values() {
        let ga = |nodes: usize, population_size: usize, generations: usize| GaConfig {
            nodes: vec![[0.0, 0.0]; nodes],
            population_size,
            generations,
            ..serde_json::from_value(serde_json::json!({ "nodes": [] })).expect("defaults")
        };
        let err = check_ga(&ga(1, 10, 10)).expect_err("one node");
        assert_eq!(
            err.fields,
            json!({ "code": "insufficient_data", "parameter": "nodes", "min": 2, "got": 1 })
        );
        let err = check_ga(&ga(3, 1, 10)).expect_err("population of one");
        assert_eq!(err.fields["parameter"], "population_size");
        assert_eq!(err.fields["max"], serde_json::Value::Null);
        assert!(check_ga(&ga(3, 2, 1)).is_ok());

        let sa: SaConfig = serde_json::from_value(serde_json::json!({
            "nodes": [[0.0, 0.0], [1.0, 1.0]],
            "cooling_rate": 1.0,
        }))
        .expect("an SA config");
        let err = check_sa(&sa).expect_err("cooling rate of 1");
        assert_eq!(
            err.fields,
            json!({
                "code": "parameter_out_of_range",
                "parameter": "cooling_rate",
                "min": 0.0,
                "max": 1.0,
                "got": 1.0,
            })
        );
    }

    use super::*;

    /// The reported length is the length of the reported tour, whatever the
    /// seed. The whole-tour reversal made the tracked length drift below zero:
    /// four nodes, 200 iterations, and the result said -72.8.
    #[test]
    fn anneal_reports_the_length_of_the_tour_it_returns() {
        let nodes = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [1.0, -1.0], [3.0, 2.0]];
        for seed in 0..200u64 {
            let mut rng = WasmRng::with_seed(seed);
            let (tour, dist) = anneal(&nodes, 1000.0, 0.995, 500, &mut rng);
            let actual = tour_distance(&tour, &nodes);
            assert!(
                (dist - actual).abs() < 1e-9,
                "seed {seed}: reported {dist}, tour measures {actual}"
            );
            let mut sorted = tour.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..nodes.len()).collect::<Vec<_>>());
        }
    }

    /// On two nodes the only move is the whole-tour reversal, which the delta
    /// formula prices at minus twice the edge.
    #[test]
    fn anneal_on_two_nodes_keeps_the_single_tour_length() {
        let nodes = vec![[0.0, 0.0], [3.0, 4.0]];
        let mut rng = WasmRng::with_seed(7);
        let (tour, dist) = anneal(&nodes, 1000.0, 0.995, 100, &mut rng);
        assert!((dist - 10.0).abs() < 1e-12, "got {dist}");
        assert!((tour_distance(&tour, &nodes) - 10.0).abs() < 1e-12);
    }

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

// ── Wire-schema strictness tests ─────────────────────────────────────

#[cfg(test)]
mod dto_strictness_tests {
    use serde_json::json;

    fn assert_rejects_unknown<T: serde::de::DeserializeOwned>(v: serde_json::Value) {
        match serde_json::from_value::<T>(v) {
            Ok(_) => panic!("unknown key must be rejected"),
            Err(e) => assert!(e.to_string().contains("unknown field"), "{e}"),
        }
    }

    #[test]
    fn ga_config_rejects_unknown_keys() {
        assert_rejects_unknown::<super::GaConfig>(
            json!({ "nodes": [[0.0, 0.0], [1.0, 1.0]], "populationSize": 10 }),
        );
    }

    #[test]
    fn sa_config_rejects_unknown_keys() {
        assert_rejects_unknown::<super::SaConfig>(
            json!({ "nodes": [[0.0, 0.0], [1.0, 1.0]], "cooling": 0.9 }),
        );
    }
}
