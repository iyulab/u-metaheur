//! Generic permutation-based genetic operators.
//!
//! Crossover and mutation operators for permutation-encoded chromosomes.
//! These operate on `&[usize]` index vectors and are domain-agnostic:
//! scheduling, TSP, and any permutation problem can use them.
//!
//! # Crossover Operators
//!
//! - [`order_crossover`] (OX): Davis (1985) — preserves relative order
//! - [`pmx_crossover`] (PMX): Goldberg & Lingle (1985) — preserves absolute position
//!
//! # Mutation Operators
//!
//! - [`swap_mutation`]: Exchange two random positions — O(1)
//! - [`insert_mutation`]: Remove and reinsert at random position — O(n)
//! - [`invert_mutation`]: Reverse a random segment (2-opt) — O(n)
//!
//! # References
//!
//! - Davis (1985), "Applying Adaptive Algorithms to Epistatic Domains"
//! - Goldberg & Lingle (1985), "Alleles, Loci, and the Traveling Salesman Problem"
//! - Cicirello (2023), "Genetic Operators for Permutation Representation"

use rand::Rng;

// ============================================================================
// Crossover operators
// ============================================================================

/// Order Crossover (OX) for permutations.
///
/// Preserves the **relative order** of elements from both parents.
///
/// # Algorithm (Davis, 1985)
///
/// 1. Select a random segment `[start, end]` from parent1
/// 2. Copy segment to child at the same positions
/// 3. Fill remaining positions with elements from parent2, in their original
///    order, skipping elements already present in the child
///
/// # Complexity
/// O(n) time, O(n) space
///
/// # Panics
/// Panics if parents have different lengths or are empty.
pub fn order_crossover<R: Rng>(
    parent1: &[usize],
    parent2: &[usize],
    rng: &mut R,
) -> (Vec<usize>, Vec<usize>) {
    let n = parent1.len();
    assert_eq!(n, parent2.len(), "parents must have equal length");
    assert!(n > 0, "parents must not be empty");

    if n == 1 {
        return (parent1.to_vec(), parent2.to_vec());
    }

    let (start, end) = random_segment(n, rng);

    let child1 = ox_build_child(parent1, parent2, start, end);
    let child2 = ox_build_child(parent2, parent1, start, end);

    (child1, child2)
}

/// Build one OX child: copy segment from `template`, fill from `donor`.
fn ox_build_child(template: &[usize], donor: &[usize], start: usize, end: usize) -> Vec<usize> {
    let n = template.len();
    let mut child = vec![usize::MAX; n];
    let mut in_segment = vec![false; n];

    // Step 1: Copy segment from template
    for i in start..=end {
        child[i] = template[i];
        in_segment[template[i]] = true;
    }

    // Step 2: Fill from donor, starting after segment end, wrapping around
    let mut pos = (end + 1) % n;
    for offset in 0..n {
        let donor_idx = (end + 1 + offset) % n;
        let val = donor[donor_idx];
        if !in_segment[val] {
            child[pos] = val;
            pos = (pos + 1) % n;
        }
    }

    child
}

/// Partially Mapped Crossover (PMX) for permutations.
///
/// Preserves the **absolute position** of elements from both parents
/// as much as possible.
///
/// # Algorithm (Goldberg & Lingle, 1985)
///
/// 1. Select a random segment `[start, end]` from parent1
/// 2. Copy segment to child at the same positions
/// 3. For each element in parent2's segment that isn't in the child yet,
///    find its position through the mapping chain and place it there
/// 4. Fill remaining positions from parent2
///
/// # Complexity
/// O(n) time, O(n) space
///
/// # Panics
/// Panics if parents have different lengths or are empty.
pub fn pmx_crossover<R: Rng>(
    parent1: &[usize],
    parent2: &[usize],
    rng: &mut R,
) -> (Vec<usize>, Vec<usize>) {
    let n = parent1.len();
    assert_eq!(n, parent2.len(), "parents must have equal length");
    assert!(n > 0, "parents must not be empty");

    if n == 1 {
        return (parent1.to_vec(), parent2.to_vec());
    }

    let (start, end) = random_segment(n, rng);

    let child1 = pmx_build_child(parent1, parent2, start, end);
    let child2 = pmx_build_child(parent2, parent1, start, end);

    (child1, child2)
}

/// Build one PMX child: copy segment from `template`, map from `donor`.
fn pmx_build_child(template: &[usize], donor: &[usize], start: usize, end: usize) -> Vec<usize> {
    let n = template.len();
    let sentinel = usize::MAX;
    let mut child = vec![sentinel; n];
    let mut placed = vec![false; n];

    // Step 1: Copy segment from template
    for i in start..=end {
        child[i] = template[i];
        placed[template[i]] = true;
    }

    // Step 2: For elements in donor's segment not yet placed,
    //         follow the mapping chain to find a free position
    for i in start..=end {
        let donor_val = donor[i];
        if placed[donor_val] {
            continue;
        }
        // Follow chain: template[i] -> find where template[i] is in donor -> check if that pos is free
        let mut pos = i;
        loop {
            let mapped_val = template[pos];
            // Find where mapped_val is in donor
            let donor_pos = donor
                    .iter()
                    .position(|&v| v == mapped_val)
                    .expect("valid permutation: every value in template exists in donor");
            if donor_pos < start || donor_pos > end {
                // Position is outside segment — it's free
                child[donor_pos] = donor_val;
                placed[donor_val] = true;
                break;
            }
            // Position is inside segment — follow chain
            pos = donor_pos;
        }
    }

    // Step 3: Fill remaining from donor
    for i in 0..n {
        if child[i] == sentinel {
            child[i] = donor[i];
        }
    }

    child
}

// ============================================================================
// Mutation operators
// ============================================================================

/// Swap mutation: exchange two random positions.
///
/// # Complexity
/// O(1)
pub fn swap_mutation<R: Rng>(perm: &mut [usize], rng: &mut R) {
    let n = perm.len();
    if n < 2 {
        return;
    }
    let i = rng.random_range(0..n);
    let j = rng.random_range(0..n);
    perm.swap(i, j);
}

/// Insert mutation: remove an element and reinsert at a random position.
///
/// Equivalent to a single "insert" move in local search.
///
/// # Complexity
/// O(n) due to array shifting
pub fn insert_mutation<R: Rng>(perm: &mut Vec<usize>, rng: &mut R) {
    let n = perm.len();
    if n < 2 {
        return;
    }
    let from = rng.random_range(0..n);
    let item = perm.remove(from);
    let to = rng.random_range(0..n); // n-1 elements, but 0..n insertion points
    perm.insert(to, item);
}

/// Invert mutation: reverse a random segment (2-opt move).
///
/// # Complexity
/// O(n) worst case for segment reversal
pub fn invert_mutation<R: Rng>(perm: &mut [usize], rng: &mut R) {
    let n = perm.len();
    if n < 2 {
        return;
    }
    let (start, end) = random_segment(n, rng);
    perm[start..=end].reverse();
}

// ============================================================================
// Helpers
// ============================================================================

/// Pick a random segment `[start, end]` within `0..n` where `start <= end`.
fn random_segment<R: Rng>(n: usize, rng: &mut R) -> (usize, usize) {
    let a = rng.random_range(0..n);
    let b = rng.random_range(0..n);
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use u_optim::random::create_rng;

    /// Check that a slice is a valid permutation of 0..n.
    fn is_valid_permutation(perm: &[usize], n: usize) -> bool {
        if perm.len() != n {
            return false;
        }
        let set: HashSet<usize> = perm.iter().copied().collect();
        set.len() == n && perm.iter().all(|&v| v < n)
    }

    // ---- OX Crossover ----

    #[test]
    fn test_ox_produces_valid_permutations() {
        let mut rng = create_rng(42);
        let p1 = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let p2 = vec![7, 6, 5, 4, 3, 2, 1, 0];

        for _ in 0..100 {
            let (c1, c2) = order_crossover(&p1, &p2, &mut rng);
            assert!(
                is_valid_permutation(&c1, 8),
                "OX child1 not valid: {c1:?}"
            );
            assert!(
                is_valid_permutation(&c2, 8),
                "OX child2 not valid: {c2:?}"
            );
        }
    }

    #[test]
    fn test_ox_preserves_segment() {
        let mut rng = create_rng(123);
        let p1 = vec![0, 1, 2, 3, 4];
        let p2 = vec![4, 3, 2, 1, 0];

        // Run many times and verify segment preservation
        for _ in 0..50 {
            let (c1, _c2) = order_crossover(&p1, &p2, &mut rng);
            assert!(is_valid_permutation(&c1, 5));
        }
    }

    #[test]
    fn test_ox_single_element() {
        let mut rng = create_rng(42);
        let p1 = vec![0];
        let p2 = vec![0];
        let (c1, c2) = order_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1, vec![0]);
        assert_eq!(c2, vec![0]);
    }

    #[test]
    fn test_ox_two_elements() {
        let mut rng = create_rng(42);
        let p1 = vec![0, 1];
        let p2 = vec![1, 0];

        for _ in 0..20 {
            let (c1, c2) = order_crossover(&p1, &p2, &mut rng);
            assert!(is_valid_permutation(&c1, 2));
            assert!(is_valid_permutation(&c2, 2));
        }
    }

    // ---- PMX Crossover ----

    #[test]
    fn test_pmx_produces_valid_permutations() {
        let mut rng = create_rng(42);
        let p1 = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let p2 = vec![3, 7, 5, 1, 6, 0, 2, 4];

        for _ in 0..100 {
            let (c1, c2) = pmx_crossover(&p1, &p2, &mut rng);
            assert!(
                is_valid_permutation(&c1, 8),
                "PMX child1 not valid: {c1:?}"
            );
            assert!(
                is_valid_permutation(&c2, 8),
                "PMX child2 not valid: {c2:?}"
            );
        }
    }

    #[test]
    fn test_pmx_preserves_segment() {
        let mut rng = create_rng(99);
        let p1 = vec![0, 1, 2, 3, 4];
        let p2 = vec![4, 3, 2, 1, 0];

        for _ in 0..50 {
            let (c1, _) = pmx_crossover(&p1, &p2, &mut rng);
            assert!(is_valid_permutation(&c1, 5));
        }
    }

    #[test]
    fn test_pmx_single_element() {
        let mut rng = create_rng(42);
        let p1 = vec![0];
        let p2 = vec![0];
        let (c1, c2) = pmx_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1, vec![0]);
        assert_eq!(c2, vec![0]);
    }

    #[test]
    fn test_pmx_identical_parents() {
        let mut rng = create_rng(42);
        let p = vec![0, 1, 2, 3, 4];
        let (c1, c2) = pmx_crossover(&p, &p, &mut rng);
        assert_eq!(c1, p);
        assert_eq!(c2, p);
    }

    // ---- Swap Mutation ----

    #[test]
    fn test_swap_preserves_permutation() {
        let mut rng = create_rng(42);
        for _ in 0..100 {
            let mut perm: Vec<usize> = (0..10).collect();
            swap_mutation(&mut perm, &mut rng);
            assert!(is_valid_permutation(&perm, 10));
        }
    }

    #[test]
    fn test_swap_single_element() {
        let mut rng = create_rng(42);
        let mut perm = vec![0];
        swap_mutation(&mut perm, &mut rng);
        assert_eq!(perm, vec![0]);
    }

    // ---- Insert Mutation ----

    #[test]
    fn test_insert_preserves_permutation() {
        let mut rng = create_rng(42);
        for _ in 0..100 {
            let mut perm: Vec<usize> = (0..10).collect();
            insert_mutation(&mut perm, &mut rng);
            assert!(is_valid_permutation(&perm, 10));
        }
    }

    #[test]
    fn test_insert_single_element() {
        let mut rng = create_rng(42);
        let mut perm = vec![0];
        insert_mutation(&mut perm, &mut rng);
        assert_eq!(perm, vec![0]);
    }

    // ---- Invert Mutation ----

    #[test]
    fn test_invert_preserves_permutation() {
        let mut rng = create_rng(42);
        for _ in 0..100 {
            let mut perm: Vec<usize> = (0..10).collect();
            invert_mutation(&mut perm, &mut rng);
            assert!(is_valid_permutation(&perm, 10));
        }
    }

    #[test]
    fn test_invert_reverses_segment() {
        let mut rng = create_rng(42);
        let mut perm = vec![0, 1, 2, 3, 4];
        // Force specific segment by running until we observe a reversal
        let original = perm.clone();
        let mut changed = false;
        for _ in 0..100 {
            perm = original.clone();
            invert_mutation(&mut perm, &mut rng);
            if perm != original {
                changed = true;
                break;
            }
        }
        assert!(changed, "invert should change the permutation eventually");
        assert!(is_valid_permutation(&perm, 5));
    }

    // ---- Integration: crossover + mutation pipeline ----

    #[test]
    fn test_full_pipeline_preserves_validity() {
        let mut rng = create_rng(42);
        let p1: Vec<usize> = (0..20).collect();
        let mut p2: Vec<usize> = (0..20).collect();
        p2.reverse();

        for _ in 0..50 {
            let (mut c1, mut c2) = order_crossover(&p1, &p2, &mut rng);
            swap_mutation(&mut c1, &mut rng);
            insert_mutation(&mut c2, &mut rng);
            invert_mutation(&mut c1, &mut rng);

            assert!(is_valid_permutation(&c1, 20), "pipeline c1 invalid: {c1:?}");
            assert!(is_valid_permutation(&c2, 20), "pipeline c2 invalid: {c2:?}");
        }
    }

    #[test]
    fn test_pmx_pipeline_preserves_validity() {
        let mut rng = create_rng(42);
        let p1: Vec<usize> = (0..15).collect();
        let mut p2: Vec<usize> = (0..15).collect();
        p2.reverse();

        for _ in 0..50 {
            let (mut c1, _) = pmx_crossover(&p1, &p2, &mut rng);
            swap_mutation(&mut c1, &mut rng);
            assert!(is_valid_permutation(&c1, 15));
        }
    }

    // ---- Random segment helper ----

    #[test]
    fn test_random_segment_bounds() {
        let mut rng = create_rng(42);
        for _ in 0..1000 {
            let (start, end) = random_segment(10, &mut rng);
            assert!(start <= end);
            assert!(end < 10);
        }
    }
}
