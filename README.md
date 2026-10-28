# u-metaheur

**Domain-agnostic metaheuristic optimization framework**

[![Crates.io](https://img.shields.io/crates/v/u-metaheur.svg)](https://crates.io/crates/u-metaheur)
[![docs.rs](https://docs.rs/u-metaheur/badge.svg)](https://docs.rs/u-metaheur)
[![CI](https://github.com/iyulab/u-metaheur/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/u-metaheur/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

u-metaheur provides generic implementations of common metaheuristic algorithms. It contains no domain-specific concepts — scheduling, nesting, routing, etc. are defined by the user through trait implementations.

## Algorithms

| Module | Algorithm | Description |
|--------|-----------|-------------|
| `ga` | Genetic Algorithm | Population-based evolutionary optimization with pluggable selection, crossover, and mutation operators |
| `brkga` | BRKGA | Biased Random-Key GA — user implements only a decoder; all evolutionary mechanics are handled generically |
| `sa` | Simulated Annealing | Single-solution trajectory optimization with pluggable cooling schedules |
| `alns` | ALNS | Adaptive Large Neighborhood Search — destroy/repair operators with adaptive weight selection |
| `cp` | Constraint Programming | Domain-agnostic modeling layer for constrained optimization with interval, integer, and boolean variables; an exact branch-and-bound solver for disjunctive scheduling models (job shop with release dates, deadlines and time lags) |
| `dispatching` | Dispatching | Generic priority rule composition engine for multi-rule item ranking |

## Key Traits

Each algorithm asks for one small trait from your domain:

| Algorithm | Implement | You provide |
|---|---|---|
| GA | `ga::GaProblem` (+ `ga::Individual`) | create, evaluate, crossover and mutate individuals |
| BRKGA | `brkga::BrkgaDecoder` | `decode(&[f64]) -> f64` — random keys to a cost |
| SA | `sa::SaProblem` | an initial solution, its `cost`, a `neighbor` |
| ALNS | `alns::AlnsProblem` + `DestroyOperator` / `RepairOperator` | a solution, its cost, destroy and repair moves |

Every cost is minimised.

```rust
use rand::{Rng, RngExt};
use u_metaheur::brkga::{BrkgaConfig, BrkgaDecoder, BrkgaRunner};
use u_metaheur::sa::{SaConfig, SaProblem, SaRunner};

// BRKGA: the decoder turns random keys into a cost -- here, how far the keys
// are from ascending order.
struct Sorted;
impl BrkgaDecoder for Sorted {
    fn decode(&self, keys: &[f64]) -> f64 {
        keys.windows(2).filter(|w| w[0] > w[1]).count() as f64
    }
}
let config = BrkgaConfig::new(6).with_stagnation_limit(200).with_seed(7);
let result = BrkgaRunner::run(&Sorted, &config).unwrap();
assert_eq!(result.best_cost, 0.0);

// SA: minimise (x - 3)^2 by small random steps.
struct Parabola;
impl SaProblem for Parabola {
    type Solution = f64;
    fn initial_solution<R: Rng>(&self, _rng: &mut R) -> f64 {
        0.0
    }
    fn cost(&self, x: &f64) -> f64 {
        (x - 3.0).powi(2)
    }
    fn neighbor<R: Rng>(&self, x: &f64, rng: &mut R) -> f64 {
        x + rng.random_range(-0.5..0.5)
    }
}
let result = SaRunner::run(&Parabola, &SaConfig::default().with_seed(7)).expect("valid config");
assert!((result.best - 3.0).abs() < 0.1);
```

## Features

- **`serde`** — Enable serde serialization for algorithm parameters

## Quick Start

```toml
[dependencies]
u-metaheur = "0.14"

# with serde support
u-metaheur = { version = "0.14", features = ["serde"] }
```

## Build & Test

```bash
cargo build
cargo test
cargo bench  # criterion benchmarks
```

## Dependencies

- [u-numflow](https://github.com/iyulab/u-numflow) — Mathematical primitives (statistics, RNG)
- `rand` 0.9 — Random number generation
- `rayon` 1.10 — Parallel computation
- `serde` 1.0 — Serialization (optional)

## License

MIT License — see [LICENSE](LICENSE).

## npm (WebAssembly)

```bash
npm install @iyulab/u-metaheur
```

The package resolves per environment via a conditional `exports` map:

| Environment | Entry |
|---|---|
| Bundlers (webpack, Vite, …) | ESM + WebAssembly ESM-integration (`default` condition) |
| Node.js — `require()`, ESM `import`, CJS TS runners (`tsx`, `ts-node`) | CJS glue loading the wasm from the filesystem (`node` condition) — no loader hooks or flags |

A browser **without** a bundler is not supported: the package loads its `.wasm`
file with an ES module import, which browsers refuse (`application/wasm` is not a
module script type), so `<script type="module">` from a CDN fails, and CDN
re-bundling services fail on the same import. Use a bundler or Node.

### Quick Start

Both functions run this crate's own runners — `GaRunner` and `SaRunner` — on a
travelling-salesman tour over `nodes` (`[x, y]` pairs), and return the best tour
found, its closed length, and why the run stopped:

```js
import { run_ga, run_sa } from '@iyulab/u-metaheur';

// The corners of a unit square: the shortest closed tour is its perimeter, 4.
const nodes = [[0, 0], [1, 0], [1, 1], [0, 1]];

const ga = run_ga({ nodes, population_size: 30, generations: 50, seed: 7 });
const sa = run_sa({ nodes, iterations: 2000, seed: 7 });
console.log(ga.best_distance, ga.best_tour, ga.stop_reason); // 4 [ ... ] iteration_limit
console.log(sa.best_distance, sa.iterations_run);

// The same seed gives the same tour.
console.log(run_ga({ nodes, population_size: 30, generations: 50, seed: 7 }).best_tour.join() === ga.best_tour.join()); // true
```

`run_ga` also takes `mutation_rate`; `run_sa` takes `initial_temp` and
`cooling_rate`. Without `initial_temp` the annealing starts at a twentieth of the
typical worsening of a 2-opt move (measured before the run), so the same cities
in other coordinates anneal the same way; give `initial_temp` for an absolute
temperature in the unit of the tour length. Both take `seed` (a whole number from 0 to 2^64 − 1 — a run that
ends on its own budget is reproduced exactly) and `time_limit_ms` (at least 1 —
the run returns the best tour found by then). `stop_reason` is
`"iteration_limit"` when the run spent its `generations` / `iterations`,
`"time_limit"` when the clock ran out first, and for `run_sa` `"completed"` when
the temperature cooled to nothing first. Every setting is optional except
`nodes`, and a setting the function does not have is rejected rather than
ignored.

`codes()` returns every refusal `code` the functions throw, with the fields that
can come with it — `{ refusals: [{ code, fields }], violations: [] }` — built from
the table the refusals come from; the table below explains each.

### Errors

A refusal throws an `Error` whose `message` is readable text and which carries a
`code` naming the reason, next to the values behind it:

```js
import { run_sa } from '@iyulab/u-metaheur';

try {
  run_sa({ nodes: [[0, 0], [1, 0], [1, 1]], cooling_rate: 1 });
} catch (err) {
  console.log(err.code, err.parameter, err.exclusive_min, err.exclusive_max, err.got); // parameter_out_of_range cooling_rate 0 1 1
}
```

| `code` | Fields | Meaning |
|---|---|---|
| `insufficient_data` | `parameter` (`"nodes"`), `min`, `got` | Fewer than 2 nodes |
| `parameter_out_of_range` | `parameter`, `min`, `max`, `exclusive_min`, `exclusive_max`, `got` | `population_size < 2`, `generations`, `iterations` or `time_limit_ms` below 1, a `seed` outside 0 to 2^64 − 1 (a negative count included: `population_size: -1` is `{ min: 2, max: null, got: -1 }`, the same as `1`), `initial_temp ≤ 0`, or `cooling_rate` outside (0, 1). `min` / `max` are the closed (included) bounds and are always present — `null` on a side that is open or unbounded; `exclusive_min` / `exclusive_max` are the open (excluded) bounds, present only when that end is open: `initial_temp` (> 0) is refused with `min: null, exclusive_min: 0`, `cooling_rate` with `exclusive_min: 0, exclusive_max: 1` |
| `value_not_finite` | `parameter`, `index` | A NaN or ±Infinity anywhere in an argument — `parameter` is the path to it (`config.nodes[1]`), `index` its position in that array, or `null` |
| `malformed_input` | `parameter`, `index` (inside an array) | An argument of the wrong shape or type (a missing or unknown key), a count that is not a whole number (`population_size: 2.5` — `parameter` is the field), or a JSON string |
| `internal` | — | A result the package could not hand back — a fault in the package, not in the input; please report it |

### TypeScript

Every exported function declares its parameter and return types, and the
declarations are generated from the same structs the binding reads and
serialises, so they cannot drift from what it actually accepts and returns:

```ts
export function run_ga(config: GaConfig): GaResult;
```

An absent optional value is declared `T | undefined`, which is what the binding
sends. Nothing needs an `as` cast -- and a wrong assumption about a result's
shape is a compile error rather than something that fails at run time.

The same holds on the way in: a setting the configuration does not have
does not compile. The binding still validates every input at the boundary, for
JavaScript callers and for values that reach it through a cast, and a rejected
one says what was wrong.

## Related

- [u-numflow](https://github.com/iyulab/u-numflow) — Mathematical primitives
- [u-geometry](https://github.com/iyulab/u-geometry) — Computational geometry
- [u-schedule](https://github.com/iyulab/u-schedule) — Scheduling framework
- [u-nesting](https://github.com/iyulab/U-Nesting) — 2D/3D nesting and bin packing
