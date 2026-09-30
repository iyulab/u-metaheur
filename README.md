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
| `cp` | Constraint Programming | Domain-agnostic modeling layer for constrained optimization with interval, integer, and boolean variables |
| `dispatching` | Dispatching | Generic priority rule composition engine for multi-rule item ranking |

## Key Traits

```rust
// GA — implement these for your domain
trait Chromosome: Clone + Send + Sync {
    fn fitness(&self) -> f64;
}
trait Crossover<C: Chromosome> {
    fn crossover(&self, parent1: &C, parent2: &C, rng: &mut Rng) -> C;
}
trait Mutation<C: Chromosome> {
    fn mutate(&self, chromosome: &mut C, rng: &mut Rng);
}

// BRKGA — implement only the decoder
trait BrkgaDecoder: Send + Sync {
    type Solution;
    fn decode(&self, keys: &[f64]) -> Self::Solution;
    fn fitness(&self, solution: &Self::Solution) -> f64;
}

// ALNS — implement destroy and repair operators
trait DestroyOperator<S> {
    fn destroy(&self, solution: &S, rng: &mut Rng) -> S;
}
trait RepairOperator<S> {
    fn repair(&self, solution: &S, rng: &mut Rng) -> S;
}
```

## Features

- **`serde`** — Enable serde serialization for algorithm parameters

## Quick Start

```toml
[dependencies]
u-metaheur = { git = "https://github.com/iyulab/u-metaheur" }

# with serde support
u-metaheur = { git = "https://github.com/iyulab/u-metaheur", features = ["serde"] }
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

### Quick Start

Both functions solve a travelling-salesman tour over `nodes` (`[x, y]` pairs)
and return the best tour found and its closed length:

```js
import { run_ga, run_sa } from '@iyulab/u-metaheur';

// The corners of a unit square: the shortest closed tour is its perimeter, 4.
const nodes = [[0, 0], [1, 0], [1, 1], [0, 1]];

const ga = run_ga({ nodes, population_size: 30, generations: 50 });
const sa = run_sa({ nodes, iterations: 2000 });
console.log(ga.best_distance, ga.best_tour); // 4 [ ... ]
console.log(sa.best_distance, sa.iterations_run);
```

`run_ga` also takes `mutation_rate`; `run_sa` takes `initial_temp` and
`cooling_rate`. Every setting is optional except `nodes`, and a setting the
function does not have is rejected rather than ignored.

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
