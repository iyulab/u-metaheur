//! Core trait definitions for the GA framework.
//!
//! The two central traits — [`Individual`] and [`GaProblem`] — define the
//! contract between the generic GA engine and domain-specific problem
//! implementations.

use rand::Rng;

/// Marker trait for fitness values.
///
/// Fitness must support comparison and be cheaply copyable.
/// Lower fitness is considered better (minimization).
///
/// Built-in implementations exist for `f64` and `f32`.
/// For maximization problems, negate the fitness or use a wrapper type.
pub trait Fitness: PartialOrd + Copy + Send + Sync + std::fmt::Debug + 'static {
    /// Returns a value representing the worst possible fitness.
    ///
    /// Used for initial/uninitialized individuals.
    fn worst() -> Self;

    /// Converts the fitness to `f64` for logging and statistics.
    fn to_f64(self) -> f64;
}

impl Fitness for f64 {
    fn worst() -> Self {
        f64::INFINITY
    }

    fn to_f64(self) -> f64 {
        self
    }
}

impl Fitness for f32 {
    fn worst() -> Self {
        f32::INFINITY
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// A candidate solution in the GA population.
///
/// Individuals carry their own fitness value. The GA framework calls
/// [`GaProblem::evaluate`] to compute fitness, then stores it via
/// [`set_fitness`](Individual::set_fitness).
///
/// # Implementing
///
/// ```ignore
/// #[derive(Clone)]
/// struct MySolution {
///     genes: Vec<f64>,
///     fitness: f64,
/// }
///
/// impl Individual for MySolution {
///     type Fitness = f64;
///     fn fitness(&self) -> f64 { self.fitness }
///     fn set_fitness(&mut self, f: f64) { self.fitness = f; }
/// }
/// ```
pub trait Individual: Clone + Send + Sync {
    /// The fitness type. Must implement [`Fitness`].
    type Fitness: Fitness;

    /// Returns the current fitness of this individual.
    fn fitness(&self) -> Self::Fitness;

    /// Sets the fitness of this individual.
    ///
    /// Called by the GA framework after evaluation.
    fn set_fitness(&mut self, fitness: Self::Fitness);
}

/// Defines a GA optimization problem.
///
/// This is the main trait that users implement to plug their domain-specific
/// logic into the generic GA framework. It covers:
///
/// 1. **Initialization**: How to create random individuals
/// 2. **Evaluation**: How to compute fitness
/// 3. **Crossover**: How to recombine two parents
/// 4. **Mutation**: How to perturb an individual
///
/// # Type Parameters
///
/// The associated type `Individual` links this problem to its solution
/// representation.
///
/// # Thread Safety
///
/// `GaProblem` must be `Send + Sync` because the GA runner may evaluate
/// individuals in parallel using rayon.
pub trait GaProblem: Send + Sync {
    /// The individual (solution) type for this problem.
    type Individual: Individual;

    /// Creates a random individual.
    ///
    /// Called during population initialization. The implementation should
    /// produce a valid (but not necessarily good) solution.
    fn create_individual<R: Rng>(&self, rng: &mut R) -> Self::Individual;

    /// Evaluates an individual and returns its fitness.
    ///
    /// This is typically the most expensive operation. The GA framework
    /// may call this in parallel across the population.
    ///
    /// Lower fitness values are considered better (minimization).
    fn evaluate(&self, individual: &Self::Individual) -> <Self::Individual as Individual>::Fitness;

    /// Produces one or two offspring by recombining two parents.
    ///
    /// Returns a `Vec` of 1 or 2 children. The framework handles sizing.
    ///
    /// The default implementation clones parent1 (no crossover).
    fn crossover<R: Rng>(
        &self,
        parent1: &Self::Individual,
        _parent2: &Self::Individual,
        _rng: &mut R,
    ) -> Vec<Self::Individual> {
        vec![parent1.clone()]
    }

    /// Mutates an individual in place.
    ///
    /// The default implementation is a no-op.
    fn mutate<R: Rng>(&self, _individual: &mut Self::Individual, _rng: &mut R) {}

    /// Called at the end of each generation with the current best fitness.
    ///
    /// Useful for logging, adaptive parameter control, or external
    /// communication. The default implementation is a no-op.
    fn on_generation(
        &self,
        _generation: usize,
        _best_fitness: <Self::Individual as Individual>::Fitness,
    ) {
    }
}
