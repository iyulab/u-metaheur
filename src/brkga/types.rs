//! Core trait for BRKGA.

use rand::Rng;

/// Decoder trait for BRKGA.
///
/// This is the **only** trait a user must implement to use BRKGA.
/// It maps a random-key chromosome (a slice of `f64` in `[0, 1)`)
/// to a cost value. Lower cost is better (minimization).
///
/// # Examples
///
/// ```ignore
/// struct KnapsackDecoder { weights: Vec<f64>, values: Vec<f64>, capacity: f64 }
///
/// impl BrkgaDecoder for KnapsackDecoder {
///     fn decode(&self, keys: &[f64]) -> f64 {
///         // keys[i] > 0.5 means include item i
///         let (total_w, total_v) = keys.iter().enumerate()
///             .filter(|(_, &k)| k > 0.5)
///             .fold((0.0, 0.0), |(w, v), (i, _)| (w + self.weights[i], v + self.values[i]));
///         if total_w > self.capacity { f64::INFINITY } else { -total_v }
///     }
/// }
/// ```
///
/// # References
///
/// Bean (1994), Goncalves & Resende (2011)
pub trait BrkgaDecoder: Send + Sync {
    /// Decodes a random-key chromosome and returns its cost.
    ///
    /// # Arguments
    /// * `keys` - A slice of `f64` values in `[0.0, 1.0)`.
    ///   Length equals [`super::BrkgaConfig::chromosome_length`].
    ///
    /// Lower cost is better (minimization).
    fn decode(&self, keys: &[f64]) -> f64;

    /// Creates a custom initial chromosome.
    ///
    /// Override this to seed the population with domain-specific
    /// heuristic solutions. The default returns `None` (use random keys).
    fn seed_chromosome<R: Rng>(&self, _rng: &mut R) -> Option<Vec<f64>> {
        None
    }
}
