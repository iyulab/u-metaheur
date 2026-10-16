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
//! (`parameter`, `min`, `max`, `exclusive_min`, `exclusive_max`, `got`). See
//! the README's *Errors*.

use std::ops::Bound::{Excluded, Unbounded};
use std::ops::RangeBounds;

use serde::{Deserialize, Serialize};
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::config_error::Bounds;

// ============================================================================
// Shared utilities
// ============================================================================

/// Every reason a function of this binding refuses with: the `code` an
/// `Error` carries, and the fields that can stand beside it. A refusal is
/// built from one of these ([`WireError::new`]), so [`codes`] -- the list a caller
/// reads -- is the list the binding refuses from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Code {
    MalformedInput,
    ValueNotFinite,
    ParameterOutOfRange,
    InsufficientData,
    /// The package failed, not the input.
    Internal,
}

impl Code {
    const ALL: [Code; 5] = [
        Code::MalformedInput,
        Code::ValueNotFinite,
        Code::ParameterOutOfRange,
        Code::InsufficientData,
        Code::Internal,
    ];

    fn name(self) -> &'static str {
        match self {
            Code::MalformedInput => "malformed_input",
            Code::ValueNotFinite => "value_not_finite",
            Code::ParameterOutOfRange => "parameter_out_of_range",
            Code::InsufficientData => "insufficient_data",
            Code::Internal => "internal",
        }
    }

    /// The fields a refusal with this code can carry besides `code`.
    fn fields(self) -> &'static [&'static str] {
        match self {
            Code::MalformedInput => &["parameter", "index"],
            Code::ValueNotFinite => &["parameter", "index"],
            Code::ParameterOutOfRange => &[
                "parameter",
                "min",
                "max",
                "exclusive_min",
                "exclusive_max",
                "got",
            ],
            Code::InsufficientData => &["parameter", "min", "got"],
            Code::Internal => &[],
        }
    }
}

/// A refusal reason: the `code` an `Error` carries and the fields that can
/// stand beside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct RefusalCode {
    code: String,
    fields: Vec<String>,
}

/// A violation a checker reports. This package has no checker; the type is
/// here so every engine's [`Codes`] has one shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct ViolationCode {
    code: String,
    fields: Vec<String>,
    checkers: Vec<String>,
}

/// What this package can answer with besides a result: every refusal
/// `code`, and no violations -- it has no checker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct Codes {
    refusals: Vec<RefusalCode>,
    violations: Vec<ViolationCode>,
}

fn code_catalog() -> Codes {
    Codes {
        refusals: Code::ALL
            .into_iter()
            .map(|c| RefusalCode {
                code: c.name().to_string(),
                fields: c.fields().iter().map(|f| f.to_string()).collect(),
            })
            .collect(),
        violations: vec![],
    }
}

/// Every refusal `code` the functions of this package throw, with the fields
/// that can come with it: `{ refusals: [{ code, fields }], violations: [] }`.
///
/// It is built from the table the refusals come from, so a caller that
/// translates codes can check its list against the package's.
#[wasm_bindgen(unchecked_return_type = "Codes")]
pub fn codes() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&code_catalog())
        .map_err(|e| js_err(WireError::new(Code::Internal, e.to_string(), json!({}))))
}

/// A refusal on its way to JavaScript: the text for `Error.message`, and the
/// fields -- `code` first among them -- copied onto the `Error`.
#[derive(Debug)]
struct WireError {
    message: String,
    fields: serde_json::Value,
}

impl WireError {
    fn new(code: Code, message: String, mut extra: serde_json::Value) -> Self {
        if let Some(given) = extra.as_object() {
            debug_assert!(
                given.keys().all(|k| code.fields().contains(&k.as_str())),
                "{} refusal with undeclared fields: {:?}",
                code.name(),
                given.keys().collect::<Vec<_>>()
            );
        }
        let mut fields = serde_json::Map::new();
        fields.insert("code".into(), json!(code.name()));
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
            Code::MalformedInput,
            message,
            json!({ "parameter": parameter }),
        )
    }

    /// A setting outside the range the solver accepts, each end open or closed
    /// as it is. A closed end goes out as `min` / `max`, an open end as
    /// `exclusive_min` / `exclusive_max`; `min` and `max` are always present
    /// (`null` on a side that is open or unbounded), the `exclusive_*` fields
    /// only when that end is open.
    fn out_of_range(
        parameter: &str,
        accepted: impl RangeBounds<f64>,
        got: f64,
        message: String,
    ) -> Self {
        let bounds = Bounds::of(accepted);
        let mut fields = json!({
            "parameter": parameter,
            "min": bounds.min,
            "max": bounds.max,
            "got": got,
        });
        for (key, open) in [
            ("exclusive_min", bounds.exclusive_min),
            ("exclusive_max", bounds.exclusive_max),
        ] {
            if let Some(value) = open {
                fields[key] = json!(value);
            }
        }
        Self::new(Code::ParameterOutOfRange, message, fields)
    }

    /// Fewer nodes than a tour needs.
    fn too_few_nodes(got: usize) -> Self {
        Self::new(
            Code::InsufficientData,
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
        .map_err(|e| js_err(WireError::new(Code::Internal, e.to_string(), json!({}))))
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
            Code::ValueNotFinite,
            found.message(),
            found.fields(),
        )));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value =
        serde_wasm_bindgen::from_value(value).map_err(|e| refuse(format!("{param}: {e}")))?;
    from_json(json, param).map_err(js_err)
}

/// The half of [`from_js`] that enforces the wire schema, apart from the
/// `JsValue` (which cannot be built off `wasm32`) so tests walk the same path a
/// JS caller does. A refusal names where it stopped: see [`failure_site`].
fn from_json<T: serde::de::DeserializeOwned>(
    json: serde_json::Value,
    param: &str,
) -> Result<T, WireError> {
    serde_path_to_error::deserialize(json).map_err(|e| {
        let (parameter, index) = failure_site(param, e.path());
        let mut err =
            WireError::malformed_input(&parameter, format!("{param}: {}: {}", e.path(), e.inner()));
        if let Some(i) = index {
            err.fields["index"] = serde_json::json!(i);
        }
        err
    })
}

/// Where in the argument `param` a request stopped deserializing, as the
/// refusal reports it: the field (`design[1]`, `points[0].y`) and, when the
/// failure sits in an array, its position there -- the same `parameter` /
/// `index` a number array read element by element reports. Without it a
/// `null` three levels down was "invalid type: null, expected f64" with no
/// way to say which row (the gap `read_numbers` closed for bare arrays).
fn failure_site(param: &str, path: &serde_path_to_error::Path) -> (String, Option<usize>) {
    use serde_path_to_error::Segment;
    let segments: Vec<&Segment> = path.iter().collect();
    let index = segments.iter().rev().find_map(|s| match s {
        Segment::Seq { index } => Some(*index),
        _ => None,
    });
    // A trailing `[i]` is the index, not part of the name.
    let named = match segments.last() {
        Some(Segment::Seq { .. }) => &segments[..segments.len() - 1],
        _ => &segments[..],
    };
    let mut name = String::new();
    for segment in named {
        match segment {
            Segment::Seq { index } => name.push_str(&format!("[{index}]")),
            Segment::Map { key } | Segment::Enum { variant: key } => {
                if !name.is_empty() {
                    name.push('.');
                }
                name.push_str(key);
            }
            Segment::Unknown => name.push_str(".?"),
        }
    }
    // A top-level array argument (`data[1][2]`) or a failure at the root
    // (a missing field) is named by the argument itself.
    if name.is_empty() || name.starts_with('[') {
        name.insert_str(0, param);
    }
    (name, index)
}

// ============================================================================
// Whole numbers
// ============================================================================

/// A whole number exactly as the caller wrote it: any integer, or a number
/// with no fractional part.
///
/// Counts are read through this rather than straight into `usize`. As a
/// `usize`, `-1` failed as a type mismatch ("invalid value: integer `-1`,
/// expected usize", `malformed_input`) while `1` was refused by the range
/// check -- two answers to one mistake. Read signed and wide, every whole
/// number reaches the range check, which names the accepted range ([`count`]).
/// A fractional part is refused here: whether a count is whole is a matter of
/// its type, so `2.5` is `malformed_input`.
struct Whole(i128);

impl<'de> Deserialize<'de> for Whole {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct WholeVisitor;

        impl serde::de::Visitor<'_> for WholeVisitor {
            type Value = Whole;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a whole number")
            }

            fn visit_i64<E>(self, v: i64) -> Result<Whole, E> {
                Ok(Whole(v.into()))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Whole, E> {
                Ok(Whole(v.into()))
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Whole, E> {
                if v.fract() == 0.0 {
                    // `as` saturates only past ±1.7e38, far outside every
                    // range a setting accepts.
                    Ok(Whole(v as i128))
                } else {
                    Err(E::custom(format_args!("must be a whole number, got {v}")))
                }
            }
        }

        deserializer.deserialize_any(WholeVisitor)
    }
}

/// Reads a whole-number field (`#[serde(deserialize_with = "whole")]`); see
/// [`Whole`].
fn whole<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<i128, D::Error> {
    Whole::deserialize(deserializer).map(|w| w.0)
}

/// A whole-number setting as the `usize` the solver counts in.
///
/// Below `min` -- a negative count as much as a too-small one -- it is
/// refused as `parameter_out_of_range` with `min` and no upper bound; above
/// the largest `usize` the target holds, with that as `max`.
fn count(parameter: &str, value: i128, min: i128) -> Result<usize, WireError> {
    if value < min {
        return Err(WireError::out_of_range(
            parameter,
            min as f64..,
            value as f64,
            format!("{parameter} must be at least {min}, got {value}"),
        ));
    }
    usize::try_from(value).map_err(|_| {
        WireError::out_of_range(
            parameter,
            min as f64..=usize::MAX as f64,
            value as f64,
            format!(
                "{parameter} must be from {min} to {}, got {value}",
                usize::MAX
            ),
        )
    })
}

/// The counts `run_ga` runs with, once [`check_ga`] has accepted them.
#[derive(Debug, PartialEq)]
struct GaCounts {
    population_size: usize,
    generations: usize,
}

/// The settings `run_ga` refuses.
fn check_ga(config: &GaConfig) -> Result<GaCounts, WireError> {
    if config.nodes.len() < 2 {
        return Err(WireError::too_few_nodes(config.nodes.len()));
    }
    let counts = GaCounts {
        population_size: count("population_size", config.population_size, 2)?,
        generations: count("generations", config.generations, 1)?,
    };
    if !(0.0..=1.0).contains(&config.mutation_rate) {
        return Err(WireError::out_of_range(
            "mutation_rate",
            0.0..=1.0,
            config.mutation_rate,
            format!(
                "mutation_rate must be in [0, 1], got {}",
                config.mutation_rate
            ),
        ));
    }
    Ok(counts)
}

/// The settings `run_sa` refuses; the iteration count it runs with.
fn check_sa(config: &SaConfig) -> Result<usize, WireError> {
    if config.nodes.len() < 2 {
        return Err(WireError::too_few_nodes(config.nodes.len()));
    }
    if config.initial_temp <= 0.0 {
        return Err(WireError::out_of_range(
            "initial_temp",
            (Excluded(0.0), Unbounded),
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
            (Excluded(0.0), Excluded(1.0)),
            config.cooling_rate,
            format!(
                "cooling_rate must be in (0, 1), both excluded, got {}",
                config.cooling_rate
            ),
        ));
    }
    count("iterations", config.iterations, 1)
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

/// A tour over the request's nodes, carrying its closed length as fitness
/// (lower is better).
#[derive(Clone)]
struct Tour {
    order: Vec<usize>,
    length: f64,
}

impl crate::ga::Individual for Tour {
    type Fitness = f64;

    fn fitness(&self) -> f64 {
        self.length
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.length = fitness;
    }
}

/// The travelling-salesman problem over `nodes`, stated for the crate's GA
/// and SA runners: `run_ga` and `run_sa` are those runners, not loops of their
/// own, so they take the runners' settings (`seed`, `time_limit_ms`) and report
/// why they stopped.
struct Tsp {
    nodes: Vec<[f64; 2]>,
}

impl crate::ga::GaProblem for Tsp {
    type Individual = Tour;

    fn create_individual<R: rand::Rng>(&self, rng: &mut R) -> Tour {
        use rand::seq::SliceRandom;
        let mut order: Vec<usize> = (0..self.nodes.len()).collect();
        order.shuffle(rng);
        Tour {
            order,
            length: f64::INFINITY,
        }
    }

    /// The nearest-neighbour tour seeds the population.
    fn initial_individuals(&self) -> Vec<Tour> {
        vec![Tour {
            order: nearest_neighbour_tour(&self.nodes),
            length: f64::INFINITY,
        }]
    }

    fn evaluate(&self, tour: &Tour) -> f64 {
        tour_distance(&tour.order, &self.nodes)
    }

    fn crossover<R: rand::Rng>(&self, a: &Tour, b: &Tour, rng: &mut R) -> Vec<Tour> {
        let (c, d) = crate::ga::operators::order_crossover(&a.order, &b.order, rng);
        [c, d]
            .into_iter()
            .map(|order| Tour {
                order,
                length: f64::INFINITY,
            })
            .collect()
    }

    fn mutate<R: rand::Rng>(&self, tour: &mut Tour, rng: &mut R) {
        crate::ga::operators::swap_mutation(&mut tour.order, rng);
    }
}

impl crate::sa::SaProblem for Tsp {
    type Solution = Vec<usize>;

    /// Annealing starts from the nearest-neighbour tour.
    fn initial_solution<R: rand::Rng>(&self, _rng: &mut R) -> Vec<usize> {
        nearest_neighbour_tour(&self.nodes)
    }

    fn cost(&self, tour: &Vec<usize>) -> f64 {
        tour_distance(tour, &self.nodes)
    }

    /// A 2-opt move: reverse a segment that is neither one node nor the whole
    /// tour (both leave the cycle as it was).
    fn neighbor<R: rand::Rng>(&self, tour: &Vec<usize>, rng: &mut R) -> Vec<usize> {
        let n = tour.len();
        let mut next = tour.clone();
        if n < 4 {
            // Every tour of three or fewer nodes is the same cycle.
            return next;
        }
        loop {
            use rand::RngExt;
            let (i, j) = (rng.random_range(0..n), rng.random_range(0..n));
            let (lo, hi) = (i.min(j), i.max(j));
            if lo != hi && hi - lo != n - 1 {
                next[lo..=hi].reverse();
                return next;
            }
        }
    }
}

/// An optional whole-number setting (`seed`, `time_limit_ms`); see [`Whole`].
fn optional_whole<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<i128>, D::Error> {
    Option::<Whole>::deserialize(deserializer).map(|w| w.map(|w| w.0))
}

/// A seed: a whole number from 0 to 2^64 - 1.
fn seed(value: Option<i128>) -> Result<Option<u64>, WireError> {
    value
        .map(|v| {
            u64::try_from(v).map_err(|_| {
                WireError::out_of_range(
                    "seed",
                    0.0..=u64::MAX as f64,
                    v as f64,
                    format!("seed must be from 0 to {}, got {v}", u64::MAX),
                )
            })
        })
        .transpose()
}

/// A wall-clock budget in milliseconds, at least 1.
fn time_limit(value: Option<i128>) -> Result<Option<u64>, WireError> {
    value
        .map(|v| count("time_limit_ms", v, 1).map(|ms| ms as u64))
        .transpose()
}

/// The runner refusing settings this binding already checked is the
/// package's fault, not the caller's.
fn runner_refused(e: crate::ConfigError) -> JsValue {
    js_err(WireError::new(
        Code::Internal,
        format!("internal: the runner refused checked settings: {e}"),
        json!({}),
    ))
}

// ============================================================================
// GA — Genetic Algorithm for TSP
// ============================================================================

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct GaConfig {
    nodes: Vec<[f64; 2]>,
    /// At least 2 (`run_ga` checks it; see [`Whole`] for why it is signed).
    #[serde(default = "default_pop_size", deserialize_with = "whole")]
    #[tsify(optional, type = "number")]
    population_size: i128,
    /// At least 1.
    #[serde(default = "default_generations", deserialize_with = "whole")]
    #[tsify(optional, type = "number")]
    generations: i128,
    #[serde(default = "default_mutation_rate")]
    #[tsify(optional)]
    mutation_rate: f64,
    /// RNG seed, a whole number from 0 to 2^64 - 1: the same request and seed
    /// give the same tour when the run ends on its own budget.
    #[serde(default, deserialize_with = "optional_whole")]
    #[tsify(optional, type = "number | null")]
    seed: Option<i128>,
    /// Wall-clock budget in milliseconds, at least 1: the run returns the best
    /// tour found by then (`stop_reason: "time_limit"`).
    #[serde(default, deserialize_with = "optional_whole")]
    #[tsify(optional, type = "number | null")]
    time_limit_ms: Option<i128>,
}

fn default_pop_size() -> i128 {
    100
}
fn default_generations() -> i128 {
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
    /// Why the run stopped: `"iteration_limit"` (it ran its `generations`) or
    /// `"time_limit"`.
    #[tsify(type = "\"iteration_limit\" | \"time_limit\"")]
    stop_reason: String,
}

/// Runs the crate's genetic algorithm ([`crate::ga::GaRunner`]) on the
/// travelling-salesman tour over `nodes`: order crossover, swap mutation,
/// tournament selection, a nearest-neighbour tour in the first population.
///
/// # Arguments
/// `config` — native JS object with fields:
/// - `nodes`: `[[x, y], ...]` (required, at least 2)
/// - `population_size`: integer (default 100, at least 2)
/// - `generations`: integer (default 200, at least 1) — run as given
/// - `mutation_rate`: float 0–1 (default 0.05)
/// - `seed`: whole number 0 to 2^64 − 1 (optional) — reproducible runs
/// - `time_limit_ms`: whole number, at least 1 (optional)
///
/// # Returns
/// JS object with `best_distance`, `best_tour`, `generations_run`,
/// `stop_reason`.
#[wasm_bindgen(unchecked_return_type = "GaResult")]
pub fn run_ga(
    #[wasm_bindgen(unchecked_param_type = "GaConfig")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: GaConfig = from_js(config, "config")?;
    let GaCounts {
        population_size,
        generations,
    } = check_ga(&config).map_err(js_err)?;
    let seed = seed(config.seed).map_err(js_err)?;
    let time_limit = time_limit(config.time_limit_ms).map_err(js_err)?;

    // Every child comes from order crossover, the best tenth (at least one)
    // is carried over, and the run goes the generations it is given.
    let elite_ratio = (0.1_f64).max(1.0 / population_size as f64).min(0.5);
    let mut runner_config = crate::ga::GaConfig::default()
        .with_population_size(population_size)
        .with_max_generations(generations)
        .with_selection(crate::ga::Selection::Tournament(3))
        .with_elite_ratio(elite_ratio)
        .with_crossover_rate(1.0)
        .with_mutation_rate(config.mutation_rate)
        .with_stagnation_limit(0)
        .with_parallel(false);
    if let Some(seed) = seed {
        runner_config = runner_config.with_seed(seed);
    }
    if let Some(ms) = time_limit {
        runner_config = runner_config.with_time_limit_ms(ms);
    }
    let problem = Tsp {
        nodes: config.nodes,
    };
    let result = crate::ga::GaRunner::run(&problem, &runner_config).map_err(runner_refused)?;

    to_js(&GaResult {
        best_distance: result.best_fitness,
        best_tour: result.best.order,
        generations_run: result.generations,
        stop_reason: result.stop_reason.name().to_string(),
    })
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
    /// At least 1 (`run_sa` checks it; see [`Whole`] for why it is signed).
    #[serde(default = "default_iterations", deserialize_with = "whole")]
    #[tsify(optional, type = "number")]
    iterations: i128,
    /// RNG seed, a whole number from 0 to 2^64 - 1 (as `run_ga`'s).
    #[serde(default, deserialize_with = "optional_whole")]
    #[tsify(optional, type = "number | null")]
    seed: Option<i128>,
    /// Wall-clock budget in milliseconds, at least 1 (as `run_ga`'s).
    #[serde(default, deserialize_with = "optional_whole")]
    #[tsify(optional, type = "number | null")]
    time_limit_ms: Option<i128>,
}

fn default_temp() -> f64 {
    1000.0
}
fn default_cooling() -> f64 {
    0.995
}
fn default_iterations() -> i128 {
    5000
}

#[derive(Serialize, tsify::Tsify)]
struct SaResult {
    best_distance: f64,
    best_tour: Vec<usize>,
    iterations_run: usize,
    /// Why the run stopped: `"iteration_limit"` (it ran its `iterations`),
    /// `"completed"` (the temperature cooled to nothing first) or
    /// `"time_limit"`.
    #[tsify(type = "\"iteration_limit\" | \"completed\" | \"time_limit\"")]
    stop_reason: String,
}

/// Runs the crate's simulated annealing ([`crate::sa::SaRunner`]) on the
/// travelling-salesman tour over `nodes`, from the nearest-neighbour tour,
/// with 2-opt moves and geometric cooling (`T *= cooling_rate` per move).
///
/// # Arguments
/// `config` — native JS object with fields:
/// - `nodes`: `[[x, y], ...]` (required, at least 2)
/// - `initial_temp`: float (default 1000.0, > 0)
/// - `cooling_rate`: float in (0, 1) (default 0.995)
/// - `iterations`: integer (default 5000, at least 1)
/// - `seed`, `time_limit_ms`: as `run_ga`'s
///
/// # Returns
/// JS object with `best_distance`, `best_tour`, `iterations_run`,
/// `stop_reason`.
#[wasm_bindgen(unchecked_return_type = "SaResult")]
pub fn run_sa(
    #[wasm_bindgen(unchecked_param_type = "SaConfig")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: SaConfig = from_js(config, "config")?;
    let iterations = check_sa(&config).map_err(js_err)?;
    let seed = seed(config.seed).map_err(js_err)?;
    let time_limit = time_limit(config.time_limit_ms).map_err(js_err)?;
    to_js(&anneal(
        config.nodes,
        config.initial_temp,
        config.cooling_rate,
        iterations,
        seed,
        time_limit,
    )?)
}

/// The crate's SA runner over 2-opt moves from the nearest-neighbour tour,
/// cooling geometrically once per move.
fn anneal(
    nodes: Vec<[f64; 2]>,
    initial_temp: f64,
    cooling_rate: f64,
    iterations: usize,
    seed: Option<u64>,
    time_limit: Option<u64>,
) -> Result<SaResult, JsValue> {
    let mut runner_config = crate::sa::SaConfig::default()
        .with_initial_temperature(initial_temp)
        .with_min_temperature(f64::MIN_POSITIVE.min(initial_temp / 2.0))
        .with_cooling(crate::sa::CoolingSchedule::Geometric {
            alpha: cooling_rate,
        })
        .with_iterations_per_temperature(1)
        .with_max_iterations(iterations);
    if let Some(seed) = seed {
        runner_config = runner_config.with_seed(seed);
    }
    if let Some(ms) = time_limit {
        runner_config = runner_config.with_time_limit_ms(ms);
    }
    let result =
        crate::sa::SaRunner::run(&Tsp { nodes }, &runner_config).map_err(runner_refused)?;
    Ok(SaResult {
        best_distance: result.best_cost,
        best_tour: result.best,
        iterations_run: result.iterations,
        stop_reason: result.stop_reason.name().to_string(),
    })
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
        let ga = |nodes: usize, population_size: i128, generations: i128| GaConfig {
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
        let err = check_ga(&GaConfig {
            mutation_rate: 5.0,
            ..ga(3, 10, 10)
        })
        .expect_err("mutation rate above 1");
        assert_eq!(
            err.fields,
            json!({ "code": "parameter_out_of_range", "parameter": "mutation_rate",
                    "min": 0.0, "max": 1.0, "got": 5.0 })
        );

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
                "min": null,
                "max": null,
                "exclusive_min": 0.0,
                "exclusive_max": 1.0,
                "got": 1.0,
            })
        );
        let err = check_sa(&SaConfig {
            initial_temp: 0.0,
            ..sa
        })
        .expect_err("a temperature of zero");
        assert_eq!(
            err.fields,
            json!({
                "code": "parameter_out_of_range",
                "parameter": "initial_temp",
                "min": null,
                "max": null,
                "exclusive_min": 0.0,
                "got": 0.0,
            })
        );
    }

    use super::*;

    /// The reported length is the length of the reported tour, and a seed
    /// reproduces the run.
    #[test]
    fn anneal_reports_its_tour_and_a_seed_reproduces_it() {
        let nodes = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [1.0, -1.0], [3.0, 2.0]];
        for seed in 0..50 {
            let r = anneal(nodes.clone(), 1000.0, 0.995, 500, Some(seed), None)
                .unwrap_or_else(|_| panic!("seed {seed}"));
            let actual = tour_distance(&r.best_tour, &nodes);
            assert!((r.best_distance - actual).abs() < 1e-9, "seed {seed}");
            assert_eq!(r.iterations_run, 500);
            assert_eq!(r.stop_reason, "iteration_limit");
            let again = anneal(nodes.clone(), 1000.0, 0.995, 500, Some(seed), None)
                .unwrap_or_else(|_| panic!("seed {seed}"));
            assert_eq!(again.best_tour, r.best_tour);
        }
    }

    #[test]
    fn anneal_on_two_nodes_keeps_the_single_tour_length() {
        let nodes = vec![[0.0, 0.0], [5.0, 0.0]];
        let r = anneal(nodes.clone(), 1000.0, 0.995, 100, Some(7), None).expect("runs");
        assert!((r.best_distance - 10.0).abs() < 1e-12);
        assert!((tour_distance(&r.best_tour, &nodes) - 10.0).abs() < 1e-12);
    }

    /// The GA is the crate's runner: a seed reproduces it and it says why it
    /// stopped.
    #[test]
    fn the_ga_is_reproducible_from_a_seed() {
        use crate::ga::{GaConfig as Runner, GaRunner};
        let problem = Tsp {
            nodes: make_nodes(),
        };
        let config = Runner::default()
            .with_population_size(20)
            .with_max_generations(30)
            .with_stagnation_limit(0)
            .with_parallel(false)
            .with_seed(11);
        let a = GaRunner::run(&problem, &config).expect("runs");
        let b = GaRunner::run(&problem, &config).expect("runs");
        assert_eq!(a.best.order, b.best.order);
        assert_eq!(a.generations, 30);
        assert_eq!(a.stop_reason.name(), "iteration_limit");
        assert!((a.best_fitness - tour_distance(&a.best.order, &problem.nodes)).abs() < 1e-9);
    }

    #[test]
    fn a_seed_and_a_time_limit_are_checked() {
        assert_eq!(
            seed(Some(-1)).expect_err("negative").fields["parameter"],
            "seed"
        );
        assert_eq!(seed(Some(7)).expect("in range"), Some(7));
        let e = time_limit(Some(0)).expect_err("zero");
        assert_eq!(e.fields["parameter"], "time_limit_ms");
        assert_eq!(e.fields["min"], 1.0);
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

#[cfg(test)]
mod path_tests {
    //! A value of the wrong type deep inside an argument is refused where it
    //! sits: `parameter` names the field, `index` its array position.

    #[test]
    fn a_bad_coordinate_is_refused_at_its_node() {
        let err = super::from_json::<super::GaConfig>(
            serde_json::json!({ "nodes": [[0.0, 0.0], [1.0, null]] }),
            "config",
        )
        .err()
        .expect("null is not a number");
        assert_eq!(err.fields["code"], "malformed_input");
        assert_eq!(err.fields["parameter"], "nodes[1]");
        assert_eq!(err.fields["index"], 1);
    }
}

#[cfg(test)]
mod whole_number_tests {
    //! Counts are whole numbers read signed: a negative one is refused by the
    //! same range check as a too-small one, a fractional one as malformed.
    //!
    //! `from_json` is the half of `from_js` after serde-wasm-bindgen has read
    //! the JS value into JSON. That reader turns a JS number that is a safe
    //! integer (`3`, `-1`, `3.0` -- the same number in JS) into a JSON integer
    //! and any other number (`2.5`) into a JSON float, so these JSON values are
    //! what a JS caller's arguments arrive as.

    use super::*;

    fn ga(settings: serde_json::Value) -> Result<GaCounts, WireError> {
        let mut config = json!({ "nodes": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]] });
        config
            .as_object_mut()
            .expect("an object")
            .extend(settings.as_object().expect("an object").clone());
        check_ga(&from_json::<GaConfig>(config, "config")?)
    }

    fn sa(settings: serde_json::Value) -> Result<usize, WireError> {
        let mut config = json!({ "nodes": [[0.0, 0.0], [1.0, 0.0]] });
        config
            .as_object_mut()
            .expect("an object")
            .extend(settings.as_object().expect("an object").clone());
        check_sa(&from_json::<SaConfig>(config, "config")?)
    }

    #[test]
    fn a_negative_count_is_refused_like_a_too_small_one() {
        for got in [-1.0, 1.0] {
            let err = ga(json!({ "population_size": got as i64 })).expect_err("below 2");
            assert_eq!(
                err.fields,
                json!({ "code": "parameter_out_of_range", "parameter": "population_size",
                        "min": 2.0, "max": null, "got": got })
            );
        }
        for got in [-1.0, 0.0] {
            let err = ga(json!({ "generations": got as i64 })).expect_err("below 1");
            assert_eq!(
                err.fields,
                json!({ "code": "parameter_out_of_range", "parameter": "generations",
                        "min": 1.0, "max": null, "got": got })
            );
            let err = sa(json!({ "iterations": got as i64 })).expect_err("below 1");
            assert_eq!(
                err.fields,
                json!({ "code": "parameter_out_of_range", "parameter": "iterations",
                        "min": 1.0, "max": null, "got": got })
            );
        }
        let err = ga(json!({ "population_size": -1 })).expect_err("negative");
        assert_eq!(err.message, "population_size must be at least 2, got -1");
    }

    #[test]
    fn a_fractional_count_is_malformed_at_its_field() {
        let err = ga(json!({ "population_size": 2.5 })).expect_err("not whole");
        assert_eq!(
            err.fields,
            json!({ "code": "malformed_input", "parameter": "population_size" })
        );
        assert!(
            err.message.contains("must be a whole number, got 2.5"),
            "{}",
            err.message
        );
        let err = sa(json!({ "iterations": 0.5 })).expect_err("not whole");
        assert_eq!(err.fields["parameter"], "iterations");
        assert_eq!(err.fields["code"], "malformed_input");
    }

    #[test]
    fn a_whole_count_reads_as_itself() {
        assert_eq!(
            ga(json!({ "population_size": 3, "generations": 1 })).expect("accepted"),
            GaCounts {
                population_size: 3,
                generations: 1
            }
        );
        // A whole number written with a fractional part of zero is the same
        // number -- in JS `3.0` is `3`.
        assert_eq!(
            ga(json!({ "population_size": 3.0 }))
                .expect("accepted")
                .population_size,
            3
        );
        assert_eq!(sa(json!({ "iterations": 7 })).expect("accepted"), 7);
        // Omitted: the defaults.
        assert_eq!(
            ga(json!({})).expect("defaults"),
            GaCounts {
                population_size: 100,
                generations: 200
            }
        );
        assert_eq!(sa(json!({})).expect("defaults"), 5000);
    }

    #[test]
    fn a_count_that_is_not_a_number_stays_malformed() {
        for bad in [json!(null), json!("10"), json!(true)] {
            let err = ga(json!({ "population_size": bad })).expect_err("not a number");
            assert_eq!(err.fields["code"], "malformed_input", "{bad}");
            assert_eq!(err.fields["parameter"], "population_size", "{bad}");
        }
    }
}

#[cfg(test)]
mod code_catalog_tests {
    use super::*;

    /// The README's code tables list exactly the published vocabulary, so the
    /// tables and `codes()` cannot drift apart.
    #[test]
    fn the_readme_documents_every_code_and_no_other() {
        let readme = include_str!("../README.md");
        let mut documented = std::collections::BTreeSet::new();
        let mut in_table = false;
        for line in readme.lines() {
            if line.starts_with("| `code` |") {
                in_table = true;
                continue;
            }
            if !line.starts_with('|') {
                in_table = false;
                continue;
            }
            if in_table && !line.starts_with("|---") {
                let first = line.split('|').nth(1).expect("a cell");
                for code in first.split('·') {
                    documented.insert(code.trim().trim_matches('`').to_string());
                }
            }
        }
        let catalog = code_catalog();
        let published: std::collections::BTreeSet<String> = catalog
            .refusals
            .iter()
            .map(|r| r.code.clone())
            .chain(catalog.violations.iter().map(|v| v.code.clone()))
            .collect();
        assert_eq!(documented, published);
    }
}
