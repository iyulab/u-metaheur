//! Core trait for priority rules.

/// A scoring rule that assigns a priority value to an item.
///
/// Rules return `f64` scores where **lower is higher priority**
/// (consistent with the minimization convention used throughout u-metaheur).
///
/// # Type Parameters
///
/// * `T` - The item type being scored
/// * `C` - The context type providing state information
///
/// # Examples
///
/// ```ignore
/// // Scheduling: Shortest Processing Time
/// struct Spt;
///
/// impl PriorityRule<Task, SchedulingContext> for Spt {
///     fn name(&self) -> &str { "SPT" }
///     fn score(&self, task: &Task, _ctx: &SchedulingContext) -> f64 {
///         task.processing_time as f64
///     }
/// }
///
/// // Nesting: Largest Area First
/// struct LargestArea;
///
/// impl PriorityRule<Piece, NestingContext> for LargestArea {
///     fn name(&self) -> &str { "LargestArea" }
///     fn score(&self, piece: &Piece, _ctx: &NestingContext) -> f64 {
///         -piece.area() // negate: larger area = lower score = higher priority
///     }
/// }
/// ```
pub trait PriorityRule<T, C>: Send + Sync {
    /// Returns the name of this rule.
    fn name(&self) -> &str;

    /// Computes a priority score for the given item.
    ///
    /// Lower scores indicate higher priority.
    fn score(&self, item: &T, context: &C) -> f64;
}
