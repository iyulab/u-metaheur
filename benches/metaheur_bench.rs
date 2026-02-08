//! Criterion benchmarks for u-metaheur optimization algorithms.
//!
//! Uses synthetic problems (Sphere function, OneMax) to measure
//! pure algorithm overhead independent of any domain.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use u_metaheur::brkga::{BrkgaConfig, BrkgaDecoder, BrkgaRunner};
use u_metaheur::ga::{GaConfig, GaProblem, GaRunner, Individual};
use u_metaheur::sa::{SaConfig, SaProblem, SaRunner};

// ===========================================================================
// Sphere function: minimize sum(x_i^2)
// ===========================================================================

#[derive(Clone)]
struct SphereIndividual {
    genes: Vec<f64>,
    fitness: f64,
}

impl Individual for SphereIndividual {
    type Fitness = f64;
    fn fitness(&self) -> f64 {
        self.fitness
    }
    fn set_fitness(&mut self, f: f64) {
        self.fitness = f;
    }
}

struct SphereProblem {
    dim: usize,
}

impl GaProblem for SphereProblem {
    type Individual = SphereIndividual;

    fn create_individual<R: Rng>(&self, rng: &mut R) -> SphereIndividual {
        let genes: Vec<f64> = (0..self.dim).map(|_| rng.random_range(-5.0..5.0)).collect();
        SphereIndividual {
            genes,
            fitness: f64::INFINITY,
        }
    }

    fn evaluate(&self, ind: &SphereIndividual) -> f64 {
        ind.genes.iter().map(|x| x * x).sum()
    }

    fn crossover<R: Rng>(
        &self,
        p1: &SphereIndividual,
        p2: &SphereIndividual,
        rng: &mut R,
    ) -> Vec<SphereIndividual> {
        let point = rng.random_range(0..self.dim);
        let mut child = p1.clone();
        child.genes[point..].copy_from_slice(&p2.genes[point..]);
        vec![child]
    }

    fn mutate<R: Rng>(&self, ind: &mut SphereIndividual, rng: &mut R) {
        let i = rng.random_range(0..self.dim);
        ind.genes[i] += rng.random_range(-0.5..0.5);
    }
}

// ===========================================================================
// Sphere for SA
// ===========================================================================

struct SphereSa {
    dim: usize,
}

impl SaProblem for SphereSa {
    type Solution = Vec<f64>;

    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        (0..self.dim).map(|_| rng.random_range(-5.0..5.0)).collect()
    }

    fn cost(&self, sol: &Vec<f64>) -> f64 {
        sol.iter().map(|x| x * x).sum()
    }

    fn neighbor<R: Rng>(&self, sol: &Vec<f64>, rng: &mut R) -> Vec<f64> {
        let mut new = sol.clone();
        let i = rng.random_range(0..self.dim);
        new[i] += rng.random_range(-0.5..0.5);
        new
    }
}

// ===========================================================================
// OneMax for BRKGA: maximize number of 1-bits (minimize -count)
// ===========================================================================

struct OneMaxDecoder {
    _n: usize,
}

impl BrkgaDecoder for OneMaxDecoder {
    fn decode(&self, keys: &[f64]) -> f64 {
        let ones: usize = keys.iter().filter(|&&k| k > 0.5).count();
        -(ones as f64) // minimize negative = maximize count
    }
}

// ===========================================================================
// Benchmarks
// ===========================================================================

fn bench_ga_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga_sphere");
    group.sample_size(10);

    for (dim, pop, gen) in [(10usize, 50usize, 50usize), (50, 100, 30), (100, 100, 20)] {
        let problem = SphereProblem { dim };
        let config = GaConfig {
            population_size: pop,
            max_generations: gen,
            seed: Some(42),
            ..GaConfig::default()
        };
        group.bench_with_input(
            BenchmarkId::new(format!("d{}_p{}_g{}", dim, pop, gen), dim),
            &(problem, config),
            |b, (p, c)| {
                b.iter(|| {
                    let result = GaRunner::run(black_box(p), black_box(c));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_sa_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("sa_sphere");
    group.sample_size(10);

    for &dim in &[10, 50, 100] {
        let problem = SphereSa { dim };
        let config = SaConfig::default()
            .with_initial_temperature(100.0)
            .with_min_temperature(0.01)
            .with_max_iterations(1000)
            .with_seed(42);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &(problem, config), |b, (p, c)| {
            b.iter(|| {
                let result = SaRunner::run(black_box(p), black_box(c));
                black_box(result)
            })
        });
    }
    group.finish();
}

fn bench_brkga_onemax(c: &mut Criterion) {
    let mut group = c.benchmark_group("brkga_onemax");
    group.sample_size(10);

    for &n in &[20, 50, 100] {
        let decoder = OneMaxDecoder { _n: n };
        let config = BrkgaConfig::new(n)
            .with_population_size(100)
            .with_max_generations(50)
            .with_seed(42);
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(decoder, config),
            |b, (d, c)| {
                b.iter(|| {
                    let result = BrkgaRunner::run(black_box(d), black_box(c));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_ga_sphere, bench_sa_sphere, bench_brkga_onemax);
criterion_main!(benches);
