# u-metaheur

**Domain-agnostic metaheuristic optimization framework**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

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

- [u-optim](https://github.com/iyulab/u-optim) — Mathematical primitives (statistics, RNG)
- `rand` 0.9 — Random number generation
- `rayon` 1.10 — Parallel computation
- `serde` 1.0 — Serialization (optional)

## License

MIT License — see [LICENSE](LICENSE).

## Related

- [u-optim](https://github.com/iyulab/u-optim) — Mathematical primitives
- [u-geometry](https://github.com/iyulab/u-geometry) — Computational geometry
- [u-schedule](https://github.com/iyulab/u-schedule) — Scheduling framework
- [u-nesting](https://github.com/iyulab/U-Nesting) — 2D/3D nesting and bin packing
