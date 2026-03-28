//! Self-contained k-means clustering (Lloyd's algorithm with kmeans++ init).
//!
//! Used by [`PalettizePass`](super::palettize::PalettizePass) to compress
//! weight tensors into lookup-table palettes.

/// Run k-means clustering on 1-D `data` producing `k` centroids.
///
/// Returns `(centroids, assignments)` where `centroids` has length `k` and
/// `assignments[i]` is the index of the centroid closest to `data[i]`.
///
/// Uses kmeans++ initialization and converges when assignments stabilise or
/// `max_iter` iterations have been performed.
pub fn kmeans(data: &[f32], k: usize, max_iter: usize) -> (Vec<f32>, Vec<usize>) {
    assert!(k > 0, "k must be at least 1");

    if data.is_empty() {
        return (vec![0.0; k], vec![]);
    }

    // If k >= n_values, each value gets its own centroid.
    let n = data.len();
    if k >= n {
        let mut centroids: Vec<f32> = data.to_vec();
        centroids.resize(k, *data.last().unwrap());
        let assignments: Vec<usize> = (0..n).collect();
        return (centroids, assignments);
    }

    // Deduplicate — if there are fewer unique values than k, cap k.
    let unique = count_unique(data);
    let effective_k = k.min(unique);

    let mut centroids = kmeans_pp_init(data, effective_k);
    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let new_assignments = assign(data, &centroids);
        let converged = new_assignments == assignments;
        assignments = new_assignments;
        if converged {
            break;
        }
        update_centroids(data, &mut centroids, &mut assignments, effective_k);
    }

    // Pad centroids to requested k if effective_k < k.
    if effective_k < k {
        let pad_val = *centroids.last().unwrap_or(&0.0);
        centroids.resize(k, pad_val);
    }

    (centroids, assignments)
}

/// kmeans++ initialization: greedily pick centroids that are spread out.
fn kmeans_pp_init(data: &[f32], k: usize) -> Vec<f32> {
    use std::collections::HashSet;

    let n = data.len();
    let mut centroids = Vec::with_capacity(k);
    let mut chosen_indices = HashSet::with_capacity(k);

    // First centroid: pick the median-index element for determinism.
    let first_idx = n / 2;
    centroids.push(data[first_idx]);
    chosen_indices.insert(first_idx);

    let mut dist_sq = vec![f32::MAX; n];

    for _ in 1..k {
        // Update distances to nearest chosen centroid.
        let last_c = *centroids.last().unwrap();
        for (i, d) in dist_sq.iter_mut().enumerate() {
            let dd = (data[i] - last_c) * (data[i] - last_c);
            if dd < *d {
                *d = dd;
            }
        }

        // Pick the point with the largest minimum distance (deterministic
        // D² selection without randomness — max-distance variant).
        let best_idx = dist_sq
            .iter()
            .enumerate()
            .filter(|(i, _)| !chosen_indices.contains(i))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        centroids.push(data[best_idx]);
        chosen_indices.insert(best_idx);
    }

    centroids
}

/// Assign each data point to the nearest centroid.
fn assign(data: &[f32], centroids: &[f32]) -> Vec<usize> {
    data.iter()
        .map(|&x| {
            centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (x - **a).abs();
                    let db = (x - **b).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

/// Recompute centroids as the mean of assigned points.
/// Handles empty clusters by reassigning them to the data point farthest
/// from its own assigned centroid, updating assignments to avoid duplicates.
fn update_centroids(data: &[f32], centroids: &mut [f32], assignments: &mut [usize], k: usize) {
    let mut sums = vec![0.0f64; k];
    let mut counts = vec![0usize; k];

    for (i, &a) in assignments.iter().enumerate() {
        sums[a] += data[i] as f64;
        counts[a] += 1;
    }

    for c in 0..k {
        if counts[c] > 0 {
            centroids[c] = (sums[c] / counts[c] as f64) as f32;
        } else {
            // Find the point with the maximum distance to its own centroid.
            let farthest_idx = data
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| {
                    let da = (**a - centroids[assignments[*i]]).abs();
                    let db = (**b - centroids[assignments[*j]]).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap();
            centroids[c] = data[farthest_idx];
            assignments[farthest_idx] = c;
        }
    }
}

/// Count distinct f32 values (using bitwise equality).
fn count_unique(data: &[f32]) -> usize {
    use std::collections::HashSet;
    let set: HashSet<u32> = data.iter().map(|v| v.to_bits()).collect();
    set.len()
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_convergence() {
        // Two clear clusters around 0.0 and 10.0
        let data: Vec<f32> = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2];
        let (centroids, assignments) = kmeans(&data, 2, 100);

        assert_eq!(centroids.len(), 2);
        // Each point in the first cluster should share an assignment, likewise
        // for the second cluster.
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        assert_ne!(assignments[0], assignments[3]);
    }

    #[test]
    fn all_same_values() {
        let data = vec![5.0; 20];
        let (centroids, assignments) = kmeans(&data, 4, 50);

        // With only 1 unique value, effective_k = 1; remaining centroids are
        // padded to the requested k.
        assert_eq!(centroids.len(), 4);
        // All assignments must point to the single real centroid (index 0).
        assert!(assignments.iter().all(|&a| a == 0));
        assert!((centroids[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn k_greater_than_n_values() {
        let data = vec![1.0, 2.0, 3.0];
        let (centroids, assignments) = kmeans(&data, 8, 50);

        assert_eq!(centroids.len(), 8);
        assert_eq!(assignments.len(), 3);
        // Each point should map to its own centroid.
        assert_ne!(assignments[0], assignments[1]);
        assert_ne!(assignments[1], assignments[2]);
    }

    #[test]
    fn kmeans_pp_produces_distinct_centroids() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let centroids = kmeans_pp_init(&data, 4);
        assert_eq!(centroids.len(), 4);

        // All centroids should be distinct.
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                assert_ne!(
                    centroids[i].to_bits(),
                    centroids[j].to_bits(),
                    "centroids {i} and {j} should be distinct"
                );
            }
        }
    }

    #[test]
    fn empty_data() {
        let data: Vec<f32> = vec![];
        let (centroids, assignments) = kmeans(&data, 4, 50);
        assert_eq!(centroids.len(), 4);
        assert!(assignments.is_empty());
    }

    #[test]
    fn single_element() {
        let data = vec![42.0];
        let (centroids, assignments) = kmeans(&data, 2, 50);
        assert_eq!(centroids.len(), 2);
        assert_eq!(assignments.len(), 1);
    }

    #[test]
    fn four_clusters() {
        let mut data = Vec::new();
        for &center in &[0.0, 100.0, 200.0, 300.0] {
            for i in 0..10 {
                data.push(center + i as f32 * 0.1);
            }
        }
        let (centroids, assignments) = kmeans(&data, 4, 100);
        assert_eq!(centroids.len(), 4);

        // Points in the same original cluster should share an assignment.
        for chunk in assignments.chunks(10) {
            let first = chunk[0];
            assert!(chunk.iter().all(|&a| a == first));
        }
    }
}
