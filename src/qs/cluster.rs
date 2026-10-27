//! Automatic domain clustering for non-stationary TIs.
//!
//! For each cell in the TI we build a feature vector of values at the
//! template's lag offsets (boundary lags are masked out). K-means then
//! assigns each TI cell to one of `K` clusters. At simulation time, each
//! target cell's data event is classified into the same cluster space, and
//! the top-K mismatch search is restricted to TI cells with the matching
//! cluster label. This is a fully automatic stand-in for a hand-built
//! auxiliary variable / domain map.
//!
//! Continuous features: raw values at populated lags, masked L2 distance.
//! Categorical features: one-hot encoded values per lag, masked L2 distance
//! (equivalent to a weighted Hamming distance).

use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;

use super::grid::Grid3;
use super::template::QsTemplate;

/// Result of clustering. `labels` is a per-cell cluster id co-located with
/// the TI; `centroids` are the K patterns in feature-space (length = number
/// of lags for continuous, K_classes * n_lags for categorical).
#[derive(Clone, Debug)]
pub struct ClusterMap {
    pub labels: Grid3<u32>,
    pub centroids: Vec<Vec<f64>>,
    pub feature_dim: usize,
    pub is_categorical: bool,
    pub n_classes: u32, // 0 if continuous
}

impl ClusterMap {
    pub fn n_clusters(&self) -> usize {
        self.centroids.len()
    }
}

/// Cluster a continuous TI by local pattern signature. Each TI cell's feature
/// is the vector of values at the template's lag offsets (boundary lags get
/// value=0, mask=0). Returns labels co-located with the TI.
pub fn cluster_ti_continuous(
    ti: &Grid3<f64>,
    template: &QsTemplate,
    n_clusters: usize,
    max_iter: usize,
    seed: u64,
) -> ClusterMap {
    let n_lags = template.lags.len();
    let n_cells = ti.data.len();

    let mut features = vec![0.0f64; n_cells * n_lags];
    let mut masks = vec![0.0f64; n_cells * n_lags];
    for z in 0..ti.nz {
        for y in 0..ti.ny {
            for x in 0..ti.nx {
                let cell_idx = (z * ti.ny + y) * ti.nx + x;
                let base = cell_idx * n_lags;
                for (i, lag) in template.lags.iter().enumerate() {
                    let xx = x as i64 + lag.dx as i64;
                    let yy = y as i64 + lag.dy as i64;
                    let zz = z as i64 + lag.dz as i64;
                    if ti.in_bounds(xx, yy, zz) {
                        features[base + i] = *ti.get(xx as usize, yy as usize, zz as usize);
                        masks[base + i] = 1.0;
                    }
                }
            }
        }
    }

    let (centroids, labels_flat) = kmeans(
        &features, &masks, n_cells, n_lags, n_clusters, max_iter, seed,
    );
    let labels = Grid3::from_data(ti.nx, ti.ny, ti.nz, labels_flat);

    ClusterMap {
        labels,
        centroids,
        feature_dim: n_lags,
        is_categorical: false,
        n_classes: 0,
    }
}

/// Cluster a categorical TI by local pattern. Each TI cell's feature is the
/// one-hot encoded vector of categories at template lags (length =
/// `n_classes * n_lags`).
pub fn cluster_ti_categorical(
    ti: &Grid3<u32>,
    n_classes: u32,
    template: &QsTemplate,
    n_clusters: usize,
    max_iter: usize,
    seed: u64,
) -> ClusterMap {
    let n_lags = template.lags.len();
    let feature_dim = n_classes as usize * n_lags;
    let n_cells = ti.data.len();

    let mut features = vec![0.0f64; n_cells * feature_dim];
    let mut masks = vec![0.0f64; n_cells * feature_dim];

    for z in 0..ti.nz {
        for y in 0..ti.ny {
            for x in 0..ti.nx {
                let cell_idx = (z * ti.ny + y) * ti.nx + x;
                let base = cell_idx * feature_dim;
                for (i, lag) in template.lags.iter().enumerate() {
                    let xx = x as i64 + lag.dx as i64;
                    let yy = y as i64 + lag.dy as i64;
                    let zz = z as i64 + lag.dz as i64;
                    if !ti.in_bounds(xx, yy, zz) {
                        continue;
                    }
                    let v = *ti.get(xx as usize, yy as usize, zz as usize);
                    let lag_base = base + i * n_classes as usize;
                    for k in 0..n_classes as usize {
                        masks[lag_base + k] = 1.0;
                        features[lag_base + k] = if v as usize == k { 1.0 } else { 0.0 };
                    }
                }
            }
        }
    }

    let (centroids, labels_flat) = kmeans(
        &features,
        &masks,
        n_cells,
        feature_dim,
        n_clusters,
        max_iter,
        seed,
    );
    let labels = Grid3::from_data(ti.nx, ti.ny, ti.nz, labels_flat);

    ClusterMap {
        labels,
        centroids,
        feature_dim,
        is_categorical: true,
        n_classes,
    }
}

/// K-means on a flat feature matrix. `features` and `masks` are stored as
/// row-major `(n_samples, feature_dim)`. Returns (centroids, labels).
fn kmeans(
    features: &[f64],
    masks: &[f64],
    n_samples: usize,
    feature_dim: usize,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<u32>) {
    assert!(k >= 1);
    let mut rng = StdRng::seed_from_u64(seed);

    // K-means++ initialization.
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = rng.random_range(0..n_samples);
    centroids.push(features[first * feature_dim..(first + 1) * feature_dim].to_vec());
    let mut dists = vec![f64::INFINITY; n_samples];
    for _ in 1..k {
        // Update minimum distance to existing centroids.
        for i in 0..n_samples {
            let fbase = i * feature_dim;
            let mbase = i * feature_dim;
            let last_centroid = &centroids[centroids.len() - 1];
            let mut sumsq = 0.0;
            for j in 0..feature_dim {
                let m = masks[mbase + j];
                if m > 0.0 {
                    let diff = features[fbase + j] - last_centroid[j];
                    sumsq += m * diff * diff;
                }
            }
            if sumsq < dists[i] {
                dists[i] = sumsq;
            }
        }
        // Weighted pick by distance squared.
        let total: f64 = dists.iter().filter(|d| d.is_finite()).copied().sum();
        if total <= 0.0 {
            // Degenerate: pick uniformly.
            centroids.push(
                features[rng.random_range(0..n_samples) * feature_dim..][..feature_dim].to_vec(),
            );
        } else {
            let pick = rng.random::<f64>() * total;
            let mut cum = 0.0;
            let mut chosen = 0;
            for (i, &d) in dists.iter().enumerate() {
                if !d.is_finite() {
                    continue;
                }
                cum += d;
                if cum >= pick {
                    chosen = i;
                    break;
                }
            }
            centroids.push(features[chosen * feature_dim..(chosen + 1) * feature_dim].to_vec());
        }
    }

    let mut labels = vec![0u32; n_samples];
    for _iter in 0..max_iter {
        let mut changed = 0usize;
        for i in 0..n_samples {
            let fbase = i * feature_dim;
            let mbase = i * feature_dim;
            let mut best_k = 0;
            let mut best_d = f64::INFINITY;
            for (kk, centroid) in centroids.iter().enumerate() {
                let mut sumsq = 0.0;
                for j in 0..feature_dim {
                    let m = masks[mbase + j];
                    if m > 0.0 {
                        let diff = features[fbase + j] - centroid[j];
                        sumsq += m * diff * diff;
                    }
                }
                if sumsq < best_d {
                    best_d = sumsq;
                    best_k = kk;
                }
            }
            if labels[i] != best_k as u32 {
                labels[i] = best_k as u32;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }

        // Update centroids: per-dimension mean over assigned cells, weighted
        // by mask so missing-feature dimensions don't bias the centroid.
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0; feature_dim]; k];
        let mut counts: Vec<Vec<f64>> = vec![vec![0.0; feature_dim]; k];
        for i in 0..n_samples {
            let cluster = labels[i] as usize;
            let fbase = i * feature_dim;
            let mbase = i * feature_dim;
            for j in 0..feature_dim {
                let m = masks[mbase + j];
                if m > 0.0 {
                    sums[cluster][j] += m * features[fbase + j];
                    counts[cluster][j] += m;
                }
            }
        }
        for kk in 0..k {
            for j in 0..feature_dim {
                if counts[kk][j] > 0.0 {
                    centroids[kk][j] = sums[kk][j] / counts[kk][j];
                }
            }
        }
    }

    (centroids, labels)
}

/// Classify a continuous-target event (parallel to `template.lags`) into the
/// nearest cluster. `obs[i]` is the observed value at lag `i`, `mask[i]` is
/// 1.0 if populated. Returns the cluster index 0..K-1.
pub fn classify_event_continuous(cluster: &ClusterMap, obs: &[f64], mask: &[f64]) -> u32 {
    let dim = cluster.feature_dim;
    assert_eq!(obs.len(), dim);
    assert_eq!(mask.len(), dim);
    let mut best = 0u32;
    let mut best_d = f64::INFINITY;
    for (k, centroid) in cluster.centroids.iter().enumerate() {
        let mut sumsq = 0.0;
        let mut wsum = 0.0;
        for j in 0..dim {
            let m = mask[j];
            if m > 0.0 {
                let diff = obs[j] - centroid[j];
                sumsq += m * diff * diff;
                wsum += m;
            }
        }
        // If no overlap, treat as infinitely far - the cluster carries no
        // information for this event.
        if wsum > 0.0 && sumsq < best_d {
            best_d = sumsq;
            best = k as u32;
        }
    }
    best
}

/// Classify a categorical-target event by one-hot encoding the observation
/// vector and comparing against centroids in the one-hot space.
pub fn classify_event_categorical(cluster: &ClusterMap, obs: &[u32], mask: &[f64]) -> u32 {
    let n_lags = obs.len();
    let n_classes = cluster.n_classes as usize;
    assert_eq!(cluster.feature_dim, n_lags * n_classes);
    let mut best = 0u32;
    let mut best_d = f64::INFINITY;
    for (k, centroid) in cluster.centroids.iter().enumerate() {
        let mut sumsq = 0.0;
        let mut wsum = 0.0;
        for (i, &v) in obs.iter().enumerate() {
            if mask[i] <= 0.0 {
                continue;
            }
            let base = i * n_classes;
            for kk in 0..n_classes {
                let target_val = if v as usize == kk { 1.0 } else { 0.0 };
                let diff = target_val - centroid[base + kk];
                sumsq += mask[i] * diff * diff;
                wsum += mask[i];
            }
        }
        if wsum > 0.0 && sumsq < best_d {
            best_d = sumsq;
            best = k as u32;
        }
    }
    best
}
