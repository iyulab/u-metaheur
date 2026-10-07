# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.1 onward; earlier entries list release dates only (see git history).

## [0.7.0] - 2026-10-07

Depends on u-numflow 0.10.

### Changed

- **Breaking:** `GaConfig`, `BrkgaConfig`, `SaConfig` and `AlnsConfig` `validate()`,
  and `GaRunner`, `BrkgaRunner`, `SaRunner` and `AlnsRunner` `run`, return
  `ConfigError` instead of `String`: `OutOfRange { parameter, min, max, got, range }`
  for a value outside what the setting accepts, `Invalid { parameter, reason }` for a
  setting wrong only beside the others (an elite ratio that leaves no elite) or a
  missing operator. `parameter()` names the field; `Display` reads as before
  (`population_size must be at least 2, got 1`).

## [0.6.3] - 2026-10-07

### Fixed

- WebAssembly: a value of the wrong type inside an argument -- a `null` or a
  string where a number belongs (`nodes[1]`), or a missing field -- is refused as
  `malformed_input` with `parameter` naming the field and `index` its position in
  its array. It named only the argument ("invalid type: null, expected f64"),
  so a caller could not say which row was wrong.

## [0.6.2] - 2026-10-07

Depends on u-numflow 0.9. No other change.

## [0.6.1] - 2026-10-04

### Changed

- Depends on u-numflow 0.8.


## [0.6.0] - 2026-10-03

### Changed

- **Breaking:** `SaRunner::run` and `SaRunner::run_with_cancel` return
  `Result<SaResult, String>`, like the GA, BRKGA and ALNS runners. An invalid
  `SaConfig` used to panic inside the library.

### Fixed

- **Breaking:** the configuration builders keep the value they are given and
  `validate()` refuses one out of range, with the value in the message. They
  used to clamp it silently: `with_mutation_rate(5.0)` ran at 1.0,
  `with_elite_inheritance_prob(0.3)` became 0.5 and was then refused as
  "must be > 0.5", `with_destroy_degree(0.5, 0.2)` raised the maximum so the
  "min <= max" check could never fire, and a NaN passed or panicked inside
  `clamp`. Covers `GaConfig` (elite ratio, crossover and mutation rate,
  convergence threshold), `BrkgaConfig` (elite and mutant fraction, elite
  inheritance probability) and `AlnsConfig` (destroy degree).
- `validate()` refuses NaN everywhere it checks a range; comparisons such as
  `x <= 0.0` let a NaN through.
- WASM `run_ga` refuses a `mutation_rate` outside [0, 1]
  (`parameter_out_of_range`); 5 used to behave as 1.

## [0.5.1] - 2026-10-03

### Fixed

- WASM: a NaN or ±Infinity anywhere in an argument is refused with
  `value_not_finite`, with `parameter` the path to it and `index` its position
  in that array. JSON has no such numbers, so it used to reach the wire schema
  as `null` and be refused as `malformed_input` ("invalid type: null, expected
  f64") — the wrong reason, and the library's own non-finite checks behind the
  binding could not be reached.

## [0.5.0] - 2026-09-30

### Changed

- Depends on u-numflow 0.7.

- **Breaking:** `run_ga` and `run_sa` throw an `Error` carrying a stable `code`
  (`insufficient_data`, `parameter_out_of_range`, `malformed_input`) and the
  values behind it (`parameter`, `min`, `max`, `got`) instead of a bare string.
  `err.message` reads as before, now with the value that was refused, but
  `String(err)` starts with `Error: `.

- The README says a browser without a bundler is not supported (the package
  loads its `.wasm` through an ES module import, which browsers refuse), instead
  of listing only the environments that work.

## [0.4.3] - 2026-09-30

### Added

- README: a quick-start example. The npm section described how the package
  resolves but never showed a call.

### Changed

- The publishing workflow runs the README's JavaScript examples against the
  built package before it publishes, so an example that throws is caught
  before a reader copies it.

### Fixed

- The README's "Key Traits" block showed trait definitions that do not match
  the crate (`Chromosome`, a BRKGA decoder returning a solution). It is now a
  table of the real traits and a runnable BRKGA and SA example. The Quick
  Start pointed at the git repository instead of the published crate.
  The README's Rust examples are now compiled and run with the doc-tests,
  so an example that stops matching the API fails CI.

## [0.4.2] - 2026-09-29

### Fixed

- **`run_sa` reported a tour length below zero, and chose its best tour by
  it.** The annealer tracks the tour length from each 2-opt move's delta, and
  the delta formula is wrong for one move: reversing the whole tour. On a
  cycle that changes nothing, but the two edges the formula reads are then the
  same edge, so it priced the move at minus twice that edge. Every run drifted:
  four nodes and 200 iterations gave `best_distance` = -72.8 (and 20 runs in
  20 were negative), and the tour returned as best was the one with the most
  drift, not the shortest. The move is now skipped, as reversing a single
  node already was; `best_distance` is the length of `best_tour`.

### Changed

- **Every exported WASM function declares its parameter types.** Inputs were
  typed `any`; `run_ga` and `run_sa` now take `GaConfig` and `SaConfig`, with
  the defaulted settings optional, so a misspelt setting does not compile.
  The runtime path is unchanged.
- The publishing workflow now also fails if an exported function takes a
  parameter typed `any` (`check-typed-dts.sh`).

## [0.4.1] - 2026-09-20

### Added

- **Every exported WASM function declares its return type.** They were typed
  `(...) => any`, with the output's field *names* in the doc comment and the
  element types only in the README -- so a consumer's wrong assumption about a
  result's shape compiled and shipped. `as` is the only thing that can be
  written against `any`, and it is exactly the construct that silences this.

  The declarations are derived from the structs the binding already
  serialises, so there is no second copy to drift: `tsify` emits the interface
  and `unchecked_return_type` names it in the signature. The runtime path is
  unchanged -- same serializer, same bytes. An optional field is declared
  `T | undefined`, which is what the binding sends.

  A publish-path check (`scripts/check-typed-dts.sh`) fails the release if any
  exported function returns `any`, or if a declaration names a type the file
  does not declare. It runs before publishing rather than beside it in CI,
  because the two run on the same push.

  Inputs remain `any`; they are validated at the boundary.

### Changed

- `u-numflow` pin moves to 0.6 (tail-precise normal functions). No change in
  this crate's own code or output; not a release on its own.

## [0.4.0] - 2026-09-07

### Changed (breaking)

- **`rand` is now 0.10** (previously 0.9). `rand::Rng` appears in this crate's
  public signatures — the genetic operators (`order_crossover`, `pmx_crossover`,
  `swap_mutation`, `insert_mutation`, `invert_mutation`) and the selection
  strategies are generic over `R: Rng` — so the two versions are not
  interchangeable at the boundary. Callers must move to `rand` 0.10 as well.
  The generated sequences for a given seed are unchanged, so seeded runs
  reproduce the previous release's results.
- **The minimum supported Rust version is now declared as 1.87** and is verified
  by building on that exact toolchain; 1.86 and below fail. The requirement comes
  from this crate's own use of `unsigned_is_multiple_of`, stabilised in 1.87 —
  not from its dependencies, which would suggest a lower figure. The crate
  previously declared no `rust-version` at all.
- **`u-numflow` is now required at 0.4** (previously 0.3), following that crate's
  own `rand` 0.10 break.

### Changed

- **`getrandom` is now 0.4** on WebAssembly targets, reaching the browser entropy
  source through its `wasm_js` crate feature alone. The
  `RUSTFLAGS --cfg getrandom_backend="wasm_js"` that 0.3 required is no longer
  needed.

## [0.3.2] - 2026-07-05

### Fixed

- npm: expose the `./package.json` subpath in the `exports` map so tools
  that `require('<pkg>/package.json')` (license scanners, version
  reporters) keep working alongside the conditional exports introduced in
  the previous release (`ERR_PACKAGE_PATH_NOT_EXPORTED`).

## [0.3.1] - 2026-07-05

### Fixed

- **npm packaging — Node-compatible entry.** The npm package previously
  shipped only the wasm-bindgen *bundler*-target output, whose static
  `.wasm` import fails on Node's CJS path (`tsx`/`ts-node` in non-ESM
  packages) with an opaque `SyntaxError: Invalid or unexpected token`.
  The package now additionally ships the *nodejs*-target CJS glue under
  `node/` and routes Node consumers to it via a conditional `exports`
  map (`node` → CJS with filesystem wasm loading, `default` → bundler
  ESM). `require()`, native ESM `import`, and CJS TS runners all work
  without loader hooks. A pre-publish smoke test (CJS `require` + ESM
  `import`) now guards this path in CI. Rust API unchanged.

### Changed

- `u-numflow` dependency `^0.2` → `^0.3` (compatible; 0.3.0 publishes the
  previously-unreleased `wasm` feature and input-validation hardening —
  no API used by this crate changed).


## [0.3.0] - 2026-06-12

### Changed — BREAKING (WASM)

- WASM config objects (`run_ga`, `run_sa`) now **reject unknown keys** with an
  explicit `unknown field` error instead of silently ignoring them
  (`serde(deny_unknown_fields)`). Typos and unsupported options previously
  failed silently; remove any extra keys from config objects when upgrading.

## [0.2.1] - 2026-06-10

### Changed

- WASM: dropped legacy `*_json` parameter-name suffixes — exported functions
  take native JS objects/arrays, and JSON-string arguments are now rejected
  early with a descriptive error.

## Earlier releases

- 0.2.0 — 2026-03-08
- 0.1.0 — 2026-02-09
