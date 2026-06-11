# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.1 onward; earlier entries list release dates only (see git history).

## [Unreleased]

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
