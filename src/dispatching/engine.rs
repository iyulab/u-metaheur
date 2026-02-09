//! Rule composition engine.

use super::types::PriorityRule;

/// How multiple rules are combined to produce a final ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationMode {
    /// Rules are applied in order. A later rule is only consulted
    /// when the previous rule produces a tie (within epsilon).
    Sequential,

    /// All rules contribute simultaneously via weighted sum.
    Weighted,
}

/// Strategy for breaking ties when all rules produce equal scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieBreaker {
    /// Keep the original order (stable sort).
    PreserveOrder,

    /// Break ties by item index (deterministic).
    ByIndex,
}

/// A rule paired with its weight (used in Weighted mode).
struct WeightedRule<T, C> {
    rule: Box<dyn PriorityRule<T, C>>,
    weight: f64,
}

/// Engine for composing and applying multiple priority rules.
///
/// # Examples
///
/// ```ignore
/// let engine = RuleEngine::new()
///     .with_rule(Spt)
///     .with_tie_breaker(TieBreaker::ByIndex);
///
/// let sorted = engine.sort(&tasks, &context);
/// ```
///
/// # Weighted Mode
///
/// ```ignore
/// let engine = RuleEngine::new()
///     .with_mode(EvaluationMode::Weighted)
///     .with_weighted_rule(Spt, 0.6)
///     .with_weighted_rule(Edd, 0.4);
///
/// let sorted = engine.sort(&tasks, &context);
/// ```
pub struct RuleEngine<T, C> {
    rules: Vec<WeightedRule<T, C>>,
    mode: EvaluationMode,
    tie_breaker: TieBreaker,
    epsilon: f64,
}

impl<T, C> RuleEngine<T, C> {
    /// Creates a new engine with sequential evaluation mode.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            mode: EvaluationMode::Sequential,
            tie_breaker: TieBreaker::PreserveOrder,
            epsilon: 1e-9,
        }
    }

    /// Sets the evaluation mode.
    pub fn with_mode(mut self, mode: EvaluationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the tie-breaking strategy.
    pub fn with_tie_breaker(mut self, tb: TieBreaker) -> Self {
        self.tie_breaker = tb;
        self
    }

    /// Sets the epsilon for floating-point comparison.
    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Adds a rule with weight 1.0.
    pub fn with_rule<R: PriorityRule<T, C> + 'static>(mut self, rule: R) -> Self {
        self.rules.push(WeightedRule {
            rule: Box::new(rule),
            weight: 1.0,
        });
        self
    }

    /// Adds a rule with a custom weight (for Weighted mode).
    pub fn with_weighted_rule<R: PriorityRule<T, C> + 'static>(
        mut self,
        rule: R,
        weight: f64,
    ) -> Self {
        self.rules.push(WeightedRule {
            rule: Box::new(rule),
            weight,
        });
        self
    }

    /// Returns the number of rules in this engine.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Returns the names of all rules in order.
    pub fn rule_names(&self) -> Vec<&str> {
        self.rules.iter().map(|wr| wr.rule.name()).collect()
    }

    /// Computes the composite score for a single item.
    ///
    /// In Sequential mode, returns the vector of individual rule scores.
    /// In Weighted mode, returns a single-element vector with the weighted sum.
    fn compute_scores(&self, item: &T, context: &C) -> Vec<f64> {
        match self.mode {
            EvaluationMode::Sequential => self
                .rules
                .iter()
                .map(|wr| wr.rule.score(item, context))
                .collect(),
            EvaluationMode::Weighted => {
                let sum: f64 = self
                    .rules
                    .iter()
                    .map(|wr| wr.rule.score(item, context) * wr.weight)
                    .sum();
                vec![sum]
            }
        }
    }

    /// Sorts items by priority (lowest score first = highest priority).
    ///
    /// Returns indices into the original slice, sorted by priority.
    pub fn sort_indices(&self, items: &[T], context: &C) -> Vec<usize> {
        if self.rules.is_empty() {
            return (0..items.len()).collect();
        }

        let scores: Vec<Vec<f64>> = items
            .iter()
            .map(|item| self.compute_scores(item, context))
            .collect();

        let mut indices: Vec<usize> = (0..items.len()).collect();

        indices.sort_by(|&a, &b| {
            let sa = &scores[a];
            let sb = &scores[b];

            for (va, vb) in sa.iter().zip(sb.iter()) {
                let diff = va - vb;
                if diff.abs() > self.epsilon {
                    return va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal);
                }
            }

            // All rules tied — apply tie-breaker
            match self.tie_breaker {
                TieBreaker::PreserveOrder => std::cmp::Ordering::Equal,
                TieBreaker::ByIndex => a.cmp(&b),
            }
        });

        indices
    }

    /// Sorts items by priority and returns references in sorted order.
    pub fn sort<'a>(&self, items: &'a [T], context: &C) -> Vec<&'a T> {
        self.sort_indices(items, context)
            .into_iter()
            .map(|i| &items[i])
            .collect()
    }

    /// Returns the index of the highest-priority item (lowest score).
    ///
    /// Returns `None` if the slice is empty.
    pub fn select_best(&self, items: &[T], context: &C) -> Option<usize> {
        self.sort_indices(items, context).first().copied()
    }

    /// Scores a single item using the composite scoring.
    ///
    /// In Weighted mode, returns the weighted sum.
    /// In Sequential mode, returns the first rule's score.
    pub fn score(&self, item: &T, context: &C) -> f64 {
        let scores = self.compute_scores(item, context);
        scores.first().copied().unwrap_or(0.0)
    }
}

impl<T, C> Default for RuleEngine<T, C> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test domain: items with value and weight
    #[derive(Debug, Clone)]
    struct Item {
        value: f64,
        weight: f64,
    }

    struct EmptyContext;

    // Rule: sort by value (ascending = lower value first)
    struct ByValue;
    impl PriorityRule<Item, EmptyContext> for ByValue {
        fn name(&self) -> &str {
            "ByValue"
        }
        fn score(&self, item: &Item, _ctx: &EmptyContext) -> f64 {
            item.value
        }
    }

    // Rule: sort by weight (ascending = lighter first)
    struct ByWeight;
    impl PriorityRule<Item, EmptyContext> for ByWeight {
        fn name(&self) -> &str {
            "ByWeight"
        }
        fn score(&self, item: &Item, _ctx: &EmptyContext) -> f64 {
            item.weight
        }
    }

    fn test_items() -> Vec<Item> {
        vec![
            Item {
                value: 3.0,
                weight: 1.0,
            },
            Item {
                value: 1.0,
                weight: 2.0,
            },
            Item {
                value: 2.0,
                weight: 1.0,
            },
            Item {
                value: 1.0,
                weight: 3.0,
            },
        ]
    }

    #[test]
    fn test_single_rule_sort() {
        let engine = RuleEngine::new().with_rule(ByValue);
        let items = test_items();
        let sorted = engine.sort(&items, &EmptyContext);

        assert!((sorted[0].value - 1.0).abs() < 1e-10);
        assert!((sorted[1].value - 1.0).abs() < 1e-10);
        assert!((sorted[2].value - 2.0).abs() < 1e-10);
        assert!((sorted[3].value - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sequential_tie_breaking() {
        // Items 1 and 3 both have value=1.0
        // ByWeight should break the tie: item 1 (weight=2) vs item 3 (weight=3)
        let engine = RuleEngine::new().with_rule(ByValue).with_rule(ByWeight);
        let items = test_items();
        let sorted = engine.sort(&items, &EmptyContext);

        // First two should be the value=1.0 items, ordered by weight
        assert!((sorted[0].value - 1.0).abs() < 1e-10);
        assert!((sorted[0].weight - 2.0).abs() < 1e-10); // lighter first
        assert!((sorted[1].value - 1.0).abs() < 1e-10);
        assert!((sorted[1].weight - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mode() {
        let engine = RuleEngine::new()
            .with_mode(EvaluationMode::Weighted)
            .with_weighted_rule(ByValue, 0.5)
            .with_weighted_rule(ByWeight, 0.5);
        let items = test_items();
        let indices = engine.sort_indices(&items, &EmptyContext);

        // Weighted score = 0.5*value + 0.5*weight
        // item 0: 0.5*3 + 0.5*1 = 2.0
        // item 1: 0.5*1 + 0.5*2 = 1.5
        // item 2: 0.5*2 + 0.5*1 = 1.5
        // item 3: 0.5*1 + 0.5*3 = 2.0
        // Sorted: [1 or 2, 1 or 2, 0 or 3, 0 or 3]
        let first_score = 0.5 * items[indices[0]].value + 0.5 * items[indices[0]].weight;
        let last_score = 0.5 * items[indices[3]].value + 0.5 * items[indices[3]].weight;
        assert!(first_score <= last_score + 1e-10);
    }

    #[test]
    fn test_select_best() {
        let engine = RuleEngine::new().with_rule(ByValue);
        let items = test_items();
        let best = engine.select_best(&items, &EmptyContext);

        assert!(best.is_some());
        let idx = best.unwrap();
        assert!((items[idx].value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_select_best_empty() {
        let engine = RuleEngine::<Item, EmptyContext>::new().with_rule(ByValue);
        let items: Vec<Item> = vec![];
        assert!(engine.select_best(&items, &EmptyContext).is_none());
    }

    #[test]
    fn test_no_rules() {
        let engine = RuleEngine::<Item, EmptyContext>::new();
        let items = test_items();
        let indices = engine.sort_indices(&items, &EmptyContext);

        // Without rules, preserve original order
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_rule_names() {
        let engine = RuleEngine::<Item, EmptyContext>::new()
            .with_rule(ByValue)
            .with_rule(ByWeight);

        assert_eq!(engine.rule_names(), vec!["ByValue", "ByWeight"]);
        assert_eq!(engine.rule_count(), 2);
    }

    #[test]
    fn test_epsilon_comparison() {
        // Two items with nearly identical values (within epsilon)
        let items = vec![
            Item {
                value: 1.0,
                weight: 5.0,
            },
            Item {
                value: 1.0 + 1e-12,
                weight: 1.0,
            },
        ];

        // With ByValue alone, they should tie; ByWeight breaks tie
        let engine = RuleEngine::new().with_rule(ByValue).with_rule(ByWeight);
        let sorted = engine.sort(&items, &EmptyContext);

        // Item with weight=1 should come first (within epsilon on value,
        // so weight breaks the tie)
        assert!((sorted[0].weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_by_index_tie_breaker() {
        let items = vec![
            Item {
                value: 1.0,
                weight: 1.0,
            },
            Item {
                value: 1.0,
                weight: 1.0,
            },
            Item {
                value: 1.0,
                weight: 1.0,
            },
        ];

        let engine = RuleEngine::new()
            .with_rule(ByValue)
            .with_tie_breaker(TieBreaker::ByIndex);

        let indices = engine.sort_indices(&items, &EmptyContext);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_score_single_item() {
        let engine = RuleEngine::new()
            .with_mode(EvaluationMode::Weighted)
            .with_weighted_rule(ByValue, 2.0)
            .with_weighted_rule(ByWeight, 1.0);

        let item = Item {
            value: 3.0,
            weight: 5.0,
        };
        let s = engine.score(&item, &EmptyContext);

        // 2.0 * 3.0 + 1.0 * 5.0 = 11.0
        assert!((s - 11.0).abs() < 1e-10);
    }

    // ---- Context-dependent rule ----

    struct Threshold {
        min_value: f64,
    }

    struct PenalizeBelow;
    impl PriorityRule<Item, Threshold> for PenalizeBelow {
        fn name(&self) -> &str {
            "PenalizeBelow"
        }
        fn score(&self, item: &Item, ctx: &Threshold) -> f64 {
            if item.value < ctx.min_value {
                1000.0 // heavy penalty
            } else {
                item.value
            }
        }
    }

    #[test]
    fn test_context_dependent_rule() {
        let items = vec![
            Item {
                value: 5.0,
                weight: 1.0,
            },
            Item {
                value: 0.5,
                weight: 1.0,
            }, // below threshold
            Item {
                value: 3.0,
                weight: 1.0,
            },
        ];

        let ctx = Threshold { min_value: 1.0 };
        let engine = RuleEngine::new().with_rule(PenalizeBelow);
        let sorted = engine.sort(&items, &ctx);

        // Item with value=0.5 is penalized to 1000, should be last
        assert!((sorted[0].value - 3.0).abs() < 1e-10);
        assert!((sorted[1].value - 5.0).abs() < 1e-10);
        assert!((sorted[2].value - 0.5).abs() < 1e-10);
    }
}
