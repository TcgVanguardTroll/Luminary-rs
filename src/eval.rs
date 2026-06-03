//! Offline ranking-quality metrics for evaluating the recommender against the
//! user's own liked set (leave-one-out), borrowed from standard IR / recommender
//! practice (precision@k, recall@k, MAP, NDCG@k — see recommenders-team patterns).
//!
//! Pure functions over a *ranked relevance list*: `rels[i] == true` means the
//! item at rank `i` (0-based, best-first) is relevant. No I/O — the CLI harness
//! builds the ranked list (by running the blend over the body index) and feeds it
//! here. This is the objective the learned-weights work (#18) will optimise, so
//! it is kept small, dependency-free, and exhaustively unit-tested.

/// Precision@k — fraction of the top-`k` that are relevant. `k` is clamped to the
/// list length.
pub fn precision_at_k(rels: &[bool], k: usize) -> f64 {
    let k = k.min(rels.len());
    if k == 0 {
        return 0.0;
    }
    let hits = rels[..k].iter().filter(|r| **r).count();
    hits as f64 / k as f64
}

/// Recall@k — fraction of all `total_relevant` relevant items captured in the
/// top-`k`. Returns 0 when there are no relevant items.
pub fn recall_at_k(rels: &[bool], k: usize, total_relevant: usize) -> f64 {
    if total_relevant == 0 {
        return 0.0;
    }
    let k = k.min(rels.len());
    let hits = rels[..k].iter().filter(|r| **r).count();
    hits as f64 / total_relevant as f64
}

/// Average Precision — the mean of precision@i taken at each rank where a relevant
/// item appears, normalised by `total_relevant`. The per-query basis of MAP
/// (Mean Average Precision = mean of this over all queries). Returns 0 when there
/// are no relevant items.
pub fn average_precision(rels: &[bool], total_relevant: usize) -> f64 {
    if total_relevant == 0 {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut sum = 0.0;
    for (i, &r) in rels.iter().enumerate() {
        if r {
            hits += 1;
            sum += hits as f64 / (i + 1) as f64; // precision@(i+1) at this hit
        }
    }
    sum / total_relevant as f64
}

/// Discounted Cumulative Gain @k with binary gains: sum of `1 / log2(rank+2)` over
/// relevant items in the top-`k` (rank 0-based).
fn dcg_at_k(rels: &[bool], k: usize) -> f64 {
    let k = k.min(rels.len());
    rels[..k]
        .iter()
        .enumerate()
        .filter(|(_, r)| **r)
        .map(|(i, _)| 1.0 / ((i as f64) + 2.0).log2())
        .sum()
}

/// Normalised DCG@k (binary gains): `DCG@k / IDCG@k`, where the ideal ranking
/// puts all relevant items first. Range 0..=1; returns 0 when nothing is relevant.
pub fn ndcg_at_k(rels: &[bool], k: usize) -> f64 {
    let total_relevant = rels.iter().filter(|r| **r).count();
    if total_relevant == 0 {
        return 0.0;
    }
    // Ideal: as many leading relevant items as fit in k.
    let ideal_hits = total_relevant.min(k.min(rels.len()));
    let idcg: f64 = (0..ideal_hits)
        .map(|i| 1.0 / ((i as f64) + 2.0).log2())
        .sum();
    if idcg == 0.0 {
        return 0.0;
    }
    let v = dcg_at_k(rels, k) / idcg;
    // Explicit compare (not `.max(0.0)`, whose signed-zero behaviour is
    // implementation-defined) so an empty top-k DCG yields +0.0, never "-0.00".
    if v <= 0.0 {
        0.0
    } else {
        v
    }
}

/// The aggregate scores reported by the harness, averaged over all queries.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct EvalScores {
    pub queries: usize,
    pub precision_at_5: f64,
    pub precision_at_10: f64,
    pub recall_at_10: f64,
    pub map: f64,
    pub ndcg_at_10: f64,
}

/// Averages per-query metrics into one [`EvalScores`]. Each tuple is
/// `(ranked_relevance_list, total_relevant_for_that_query)`.
pub fn aggregate(per_query: &[(Vec<bool>, usize)]) -> EvalScores {
    let n = per_query.len();
    if n == 0 {
        return EvalScores::default();
    }
    let mut s = EvalScores {
        queries: n,
        ..Default::default()
    };
    for (rels, total) in per_query {
        s.precision_at_5 += precision_at_k(rels, 5);
        s.precision_at_10 += precision_at_k(rels, 10);
        s.recall_at_10 += recall_at_k(rels, 10, *total);
        s.map += average_precision(rels, *total);
        s.ndcg_at_10 += ndcg_at_k(rels, 10);
    }
    let d = n as f64;
    s.precision_at_5 /= d;
    s.precision_at_10 /= d;
    s.recall_at_10 /= d;
    s.map /= d;
    s.ndcg_at_10 /= d;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-4, "expected {b}, got {a}");
    }

    #[test]
    fn precision_at_k_basics() {
        approx(precision_at_k(&[true, false, true, false], 2), 0.5);
        approx(precision_at_k(&[true, true, false], 2), 1.0);
        // k beyond the list clamps to the list length.
        approx(precision_at_k(&[true, false], 10), 0.5);
        approx(precision_at_k(&[], 5), 0.0);
    }

    #[test]
    fn recall_at_k_basics() {
        approx(recall_at_k(&[true, false, true, false], 2, 2), 0.5);
        approx(recall_at_k(&[true, false, true], 3, 2), 1.0);
        approx(recall_at_k(&[false, false], 2, 0), 0.0); // no relevant
    }

    #[test]
    fn average_precision_matches_hand_calc() {
        // hits at rank 1 (P=1/1) and rank 3 (P=2/3); AP = (1 + 0.6667) / 2.
        approx(average_precision(&[true, false, true], 2), 0.8333);
        // a single relevant at the top is a perfect AP.
        approx(average_precision(&[true, false, false], 1), 1.0);
        approx(average_precision(&[false, true], 0), 0.0);
    }

    #[test]
    fn ndcg_matches_hand_calc() {
        // DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        // IDCG (2 relevant) = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        approx(ndcg_at_k(&[true, false, true], 3), 1.5 / 1.6309);
        // perfect ranking -> 1.0
        approx(ndcg_at_k(&[true, true, false], 10), 1.0);
        approx(ndcg_at_k(&[false, false], 5), 0.0);
    }

    #[test]
    fn ndcg_rewards_putting_relevant_first() {
        let good = ndcg_at_k(&[true, false, false, true], 10);
        let bad = ndcg_at_k(&[false, true, true, false], 10);
        assert!(good > bad);
    }

    #[test]
    fn aggregate_averages_across_queries() {
        let per_query = vec![(vec![true, false, true], 2), (vec![false, true, false], 1)];
        let s = aggregate(&per_query);
        assert_eq!(s.queries, 2);
        // P@5 for q1 = 2/3, q2 = 1/3 -> mean 0.5
        approx(s.precision_at_5, 0.5);
        approx(aggregate(&[]).map, 0.0);
    }
}
