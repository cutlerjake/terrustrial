//! Quick Sampling driver.
//!
//! The simulation grid mirrors the TI's regular cell topology: every
//! simulated value lives at integer `(x, y, z)`. Hard data is supplied as a
//! list of `(x, y, z, value)` triples that are placed into the simulation
//! grid before the walk begins and treated as fixed.
//!
//! At each unsimulated cell along a random or column-sequential path:
//! 1. Build the data event from the simulation grid (hard data + previously
//!    simulated). Cells without a populated neighbour at a given template lag
//!    are masked out.
//! 2. Compute the mismatch field over the *entire* TI in one FFT pass.
//! 3. Pick uniformly at random from the `top_k` TI cells with the smallest
//!    mismatch.
//! 4. Copy that TI cell's value into the simulation grid.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::RngExt;
use rand::SeedableRng;
use rayon::prelude::*;

use super::cluster::{
    classify_event_categorical, classify_event_continuous, cluster_ti_categorical,
    cluster_ti_continuous, ClusterMap,
};
use super::grid::Grid3;
use super::mismatch::{mismatch_categorical, mismatch_continuous, FftCtx, TiFreq};
use super::template::QsTemplate;

/// Send + Sync raw-pointer wrapper for shared mutable access to a `Grid3<T>`'s
/// underlying `Vec<T>`. Parallel column simulation writes to disjoint cells
/// (one cell per column step) so the writes themselves never race. Reads of
/// other columns' cells may race with their writes — the resulting "see a
/// stale value" is the standard QS approximation under parallel paths and is
/// well-tolerated in practice; it's how DEESSE/G2S also parallelise.
struct SharedOut<T> {
    ptr: *mut T,
    len: usize,
}
unsafe impl<T: Send> Send for SharedOut<T> {}
unsafe impl<T: Sync> Sync for SharedOut<T> {}

impl<T: Copy> SharedOut<T> {
    fn new(grid: &mut Grid3<T>) -> Self {
        Self {
            ptr: grid.data.as_mut_ptr(),
            len: grid.data.len(),
        }
    }
    #[inline]
    unsafe fn read(&self, idx: usize) -> T {
        debug_assert!(idx < self.len);
        *self.ptr.add(idx)
    }
    #[inline]
    unsafe fn write(&self, idx: usize, value: T) {
        debug_assert!(idx < self.len);
        *self.ptr.add(idx) = value;
    }
}

/// Shared QS parameters.
#[derive(Clone, Debug)]
pub struct QsParams {
    pub template: QsTemplate,
    /// Pick the simulated value uniformly from the `top_k` TI cells with the
    /// smallest mismatch. `top_k = 1` is strict argmin.
    pub top_k: usize,
    /// Hard cap on cells considered when selecting top-K. Pure performance
    /// knob; with `f = 1.0` we evaluate every TI cell; smaller values trade
    /// quality for speed by subsampling.
    pub ti_sample_fraction: f64,
    /// Random seed.
    pub seed: u64,
    /// If true, columns are simulated bottom-up sequentially (`(x, y)` random
    /// across columns, `z` sequential within each). Otherwise: fully random
    /// per-cell path.
    pub column_sequential: bool,
    /// Number of multi-grid levels. `1` runs a single full-resolution pass
    /// (classical QS). Values `>= 2` enable multi-grid: the simulation is
    /// run at decreasing strides `2^(L-1), …, 2, 1`, with each level using
    /// the previous one's output as additional conditioning. This captures
    /// long-range structure that a single-grid pass cannot, and is the
    /// standard MPS approach (Goovaerts; Strebelle; G2S). At each level the
    /// template is scaled isotropically by the stride. Default `1` to
    /// reproduce the original behaviour; `3` is a good general default.
    pub n_grid_levels: usize,
    /// Number of auto-derived domain clusters. `0` or `1` disables
    /// non-stationarity handling. With `>= 2`, the TI is k-means-clustered
    /// by local pattern signature at setup time; each target's data event
    /// is then classified into the same cluster space, and the top-K
    /// candidate search is restricted to TI cells whose cluster label
    /// matches the target's. This is a fully automatic stand-in for an
    /// auxiliary domain map and improves results on TIs whose patterns
    /// vary spatially (different facies, channel orientations, etc.).
    pub n_clusters: usize,
    /// K-means max iterations for the cluster fit. Default 20 is plenty.
    pub cluster_max_iter: usize,
}

impl QsParams {
    pub fn new(template: QsTemplate) -> Self {
        Self {
            template,
            top_k: 5,
            ti_sample_fraction: 1.0,
            seed: 0xC0FFEE,
            column_sequential: true,
            n_grid_levels: 1,
            n_clusters: 0,
            cluster_max_iter: 20,
        }
    }
}

/// A hard-data sample at integer cell coordinates.
#[derive(Clone, Copy, Debug)]
pub struct HardSample<T> {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub value: T,
}

/// Continuous Quick Sampling. `hard_data` cells are fixed; every other cell
/// in the output grid is simulated. Returns a grid of the same shape as
/// `sim_shape`.
///
/// In `column_sequential` mode, columns are processed *in parallel* across
/// rayon workers, with z walked sequentially within each column. This is the
/// production-grade configuration: it gives near-linear speedup with cores
/// because each column carries its own FFT context and writes to disjoint
/// cells. Reads of neighbouring columns' values may be racy (the standard
/// QS approximation under parallel paths), which is well-tolerated in
/// practice. In random-path mode (`column_sequential = false`) we fall back
/// to the sequential driver.
pub fn simulate_continuous(
    ti: &Grid3<f64>,
    sim_shape: (usize, usize, usize),
    hard_data: &[HardSample<f64>],
    params: &QsParams,
) -> Grid3<f64> {
    let (nx, ny, nz) = sim_shape;
    let mut out = Grid3::filled(nx, ny, nz, f64::NAN);
    place_hard_data(&mut out, hard_data);

    let mut rng = StdRng::seed_from_u64(params.seed);
    let n_levels = params.n_grid_levels.max(1);

    // Optional auto-clustering of the TI by local pattern signature.
    // Built once with the *full-resolution* template; the same cluster
    // assignment is reused across multi-grid levels.
    let cluster_map: Option<ClusterMap> = if params.n_clusters >= 2 {
        Some(cluster_ti_continuous(
            ti,
            &params.template,
            params.n_clusters,
            params.cluster_max_iter,
            params.seed.wrapping_add(0xC1057E2),
        ))
    } else {
        None
    };

    for level in 0..n_levels {
        let stride: usize = 1 << (n_levels - 1 - level);
        let scaled_template = params
            .template
            .scaled(stride as i32, stride as i32, stride as i32);
        if !template_fits_ti(&scaled_template, ti.dims()) {
            continue;
        }
        let level_params = QsParams {
            template: scaled_template,
            seed: params
                .seed
                .wrapping_add(level as u64 * 0x9E37_79B9_7F4A_7C15),
            ..params.clone()
        };

        let mut seed_ctx = FftCtx::new(ti.dims(), level_params.template.extents());
        let ti_freq = TiFreq::for_continuous(ti, &mut seed_ctx);

        if level_params.column_sequential {
            let columns = build_columns_continuous_strided(nx, ny, nz, &out, &mut rng, stride);
            if !columns.is_empty() {
                run_parallel_columns_continuous(
                    ti,
                    &ti_freq,
                    &columns,
                    &mut out,
                    &level_params,
                    cluster_map.as_ref(),
                );
            }
        } else {
            let path = build_path_strided(nx, ny, nz, &out, &mut rng, stride);
            run_sequential_path_continuous(
                ti,
                &ti_freq,
                &path,
                &mut out,
                &level_params,
                &mut rng,
                cluster_map.as_ref(),
            );
        }
    }

    out
}

/// A template fits a TI if at least half the TI extent along each axis
/// exceeds the template's extent. Levels whose scaled template fails this
/// are skipped — they would produce events with too many out-of-bounds
/// (masked) lags and the mismatch field would be statistically meaningless.
fn template_fits_ti(template: &QsTemplate, ti_dims: (usize, usize, usize)) -> bool {
    let (ex, ey, ez) = template.extents();
    let (nx, ny, nz) = ti_dims;
    (ex as usize) * 2 < nx && (ey as usize) * 2 < ny && (ez as usize) * 2 < nz
}

fn build_columns_continuous(
    nx: usize,
    ny: usize,
    nz: usize,
    out: &Grid3<f64>,
    rng: &mut StdRng,
) -> Vec<Vec<(usize, usize, usize)>> {
    build_columns_continuous_strided(nx, ny, nz, out, rng, 1)
}

/// Build column-grouped cells where every coordinate satisfies
/// `c % stride == 0`. Used by the multi-grid driver.
fn build_columns_continuous_strided(
    nx: usize,
    ny: usize,
    nz: usize,
    out: &Grid3<f64>,
    rng: &mut StdRng,
    stride: usize,
) -> Vec<Vec<(usize, usize, usize)>> {
    let mut cols: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    let mut y = 0;
    while y < ny {
        let mut x = 0;
        while x < nx {
            let mut col: Vec<(usize, usize, usize)> = Vec::new();
            let mut z = 0;
            while z < nz {
                if out.get(x, y, z).is_nan() {
                    col.push((x, y, z));
                }
                z += stride;
            }
            if !col.is_empty() {
                cols.push(col);
            }
            x += stride;
        }
        y += stride;
    }
    cols.shuffle(rng);
    cols
}

/// Sequential-path variant for multi-grid (random per-cell, stride-filtered).
fn build_path_strided(
    nx: usize,
    ny: usize,
    nz: usize,
    out: &Grid3<f64>,
    rng: &mut StdRng,
    stride: usize,
) -> Vec<(usize, usize, usize)> {
    let mut cells = Vec::new();
    let mut z = 0;
    while z < nz {
        let mut y = 0;
        while y < ny {
            let mut x = 0;
            while x < nx {
                if out.get(x, y, z).is_nan() {
                    cells.push((x, y, z));
                }
                x += stride;
            }
            y += stride;
        }
        z += stride;
    }
    cells.shuffle(rng);
    cells
}

fn run_parallel_columns_continuous(
    ti: &Grid3<f64>,
    ti_freq: &TiFreq,
    columns: &[Vec<(usize, usize, usize)>],
    out: &mut Grid3<f64>,
    params: &QsParams,
    cluster_map: Option<&ClusterMap>,
) {
    let n_lags = params.template.lags.len();
    let nx = out.nx;
    let ny = out.ny;
    let nz = out.nz;
    let shared = SharedOut::new(out);
    let ti_mean_fallback = ti_global_mean(ti);

    columns.par_iter().enumerate().for_each_init(
        || {
            let ctx = FftCtx::new(ti.dims(), params.template.extents());
            let mismatch_buf = Grid3::<f64>::filled(ti.nx, ti.ny, ti.nz, 0.0);
            let obs = vec![0.0f64; n_lags];
            let mask = vec![0.0f64; n_lags];
            let scratch: Vec<num_complex::Complex<f32>> = Vec::new();
            (ctx, mismatch_buf, obs, mask, scratch)
        },
        |(ctx, mismatch_buf, obs, mask, scratch), (col_idx, column)| {
            // Each column gets its own deterministic RNG so realizations
            // are reproducible regardless of rayon's scheduling.
            let mut rng = StdRng::seed_from_u64(params.seed.wrapping_add(col_idx as u64));
            for &(x, y, z) in column.iter() {
                let idx = (z * ny + y) * nx + x;
                // safety: SharedOut::read is sound because we never
                // alias mut access; reads may see stale neighbour values.
                if !unsafe { shared.read(idx) }.is_nan() {
                    continue;
                }
                build_event_continuous_shared(
                    &shared,
                    x,
                    y,
                    z,
                    nx,
                    ny,
                    nz,
                    &params.template,
                    obs,
                    mask,
                );
                let n_active: f64 = mask.iter().sum();
                if n_active < 1.0 {
                    unsafe { shared.write(idx, ti_mean_fallback) };
                    continue;
                }
                mismatch_continuous(
                    ctx,
                    ti_freq,
                    &params.template,
                    obs,
                    mask,
                    mismatch_buf,
                    scratch,
                );
                if let Some(cmap) = cluster_map {
                    // The target's full-resolution event is what's stored
                    // in `obs`/`mask` here; classify against the same
                    // feature space we used to cluster the TI.
                    apply_cluster_filter_continuous(mismatch_buf, cmap, obs, mask);
                }
                let value = pick_top_k_continuous(ti, mismatch_buf, params, &mut rng);
                unsafe { shared.write(idx, value) };
            }
        },
    );
}

/// Penalise mismatch cells whose TI cluster label doesn't match the target's
/// inferred cluster. Sets `+∞` on non-matching cells; the existing top-K
/// selection then naturally skips them.
fn apply_cluster_filter_continuous(
    mismatch: &mut Grid3<f64>,
    cluster: &ClusterMap,
    obs: &[f64],
    mask: &[f64],
) {
    if obs.len() != cluster.feature_dim {
        return;
    }
    let target = classify_event_continuous(cluster, obs, mask);
    for (m, &lbl) in mismatch.data.iter_mut().zip(cluster.labels.data.iter()) {
        if lbl != target {
            *m = f64::INFINITY;
        }
    }
}

fn run_sequential_path_continuous(
    ti: &Grid3<f64>,
    ti_freq: &TiFreq,
    path: &[(usize, usize, usize)],
    out: &mut Grid3<f64>,
    params: &QsParams,
    rng: &mut StdRng,
    cluster_map: Option<&ClusterMap>,
) {
    let mut ctx = FftCtx::new(ti.dims(), params.template.extents());
    let mut mismatch_buf = Grid3::filled(ti.nx, ti.ny, ti.nz, 0.0);
    let mut scratch: Vec<num_complex::Complex<f32>> = Vec::new();
    let mut obs = vec![0.0; params.template.lags.len()];
    let mut mask = vec![0.0; params.template.lags.len()];

    for &(x, y, z) in path.iter() {
        let idx = out.idx(x, y, z);
        if !out.data[idx].is_nan() {
            continue;
        }
        build_event_continuous(out, x, y, z, &params.template, &mut obs, &mut mask);
        let n_active: f64 = mask.iter().sum();
        if n_active < 1.0 {
            out.data[idx] = ti_global_mean(ti);
            continue;
        }
        mismatch_continuous(
            &mut ctx,
            ti_freq,
            &params.template,
            &obs,
            &mask,
            &mut mismatch_buf,
            &mut scratch,
        );
        if let Some(cmap) = cluster_map {
            apply_cluster_filter_continuous(&mut mismatch_buf, cmap, &obs, &mask);
        }
        let value = pick_top_k_continuous(ti, &mismatch_buf, params, rng);
        out.data[idx] = value;
    }
}

fn build_event_continuous_shared(
    shared: &SharedOut<f64>,
    x: usize,
    y: usize,
    z: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    template: &QsTemplate,
    obs: &mut [f64],
    mask: &mut [f64],
) {
    for (i, lag) in template.lags.iter().enumerate() {
        let xx = x as i64 + lag.dx as i64;
        let yy = y as i64 + lag.dy as i64;
        let zz = z as i64 + lag.dz as i64;
        if xx < 0 || yy < 0 || zz < 0 || xx as usize >= nx || yy as usize >= ny || zz as usize >= nz
        {
            obs[i] = 0.0;
            mask[i] = 0.0;
            continue;
        }
        let idx = (zz as usize * ny + yy as usize) * nx + xx as usize;
        let v = unsafe { shared.read(idx) };
        if v.is_nan() {
            obs[i] = 0.0;
            mask[i] = 0.0;
        } else {
            obs[i] = v;
            mask[i] = 1.0;
        }
    }
}

/// Categorical Quick Sampling. Missing simulated cells are returned as
/// `u32::MAX`. Parallelised across (x, y) columns under `column_sequential`,
/// same shape as the continuous driver.
pub fn simulate_categorical(
    ti: &Grid3<u32>,
    n_classes: u32,
    sim_shape: (usize, usize, usize),
    hard_data: &[HardSample<u32>],
    params: &QsParams,
) -> Grid3<u32> {
    let (nx, ny, nz) = sim_shape;
    let mut out = Grid3::filled(nx, ny, nz, u32::MAX);
    place_hard_data(&mut out, hard_data);

    let mut rng = StdRng::seed_from_u64(params.seed);
    let n_levels = params.n_grid_levels.max(1);

    let cluster_map: Option<ClusterMap> = if params.n_clusters >= 2 {
        Some(cluster_ti_categorical(
            ti,
            n_classes,
            &params.template,
            params.n_clusters,
            params.cluster_max_iter,
            params.seed.wrapping_add(0xC1057E2),
        ))
    } else {
        None
    };

    for level in 0..n_levels {
        let stride: usize = 1 << (n_levels - 1 - level);
        let scaled_template = params
            .template
            .scaled(stride as i32, stride as i32, stride as i32);
        if !template_fits_ti(&scaled_template, ti.dims()) {
            continue;
        }
        let level_params = QsParams {
            template: scaled_template,
            seed: params
                .seed
                .wrapping_add(level as u64 * 0x9E37_79B9_7F4A_7C15),
            ..params.clone()
        };
        let mut seed_ctx = FftCtx::new(ti.dims(), level_params.template.extents());
        let ti_freq = TiFreq::for_categorical(ti, n_classes, &mut seed_ctx);

        if level_params.column_sequential {
            let columns = build_columns_categorical_strided(nx, ny, nz, &out, &mut rng, stride);
            if !columns.is_empty() {
                run_parallel_columns_categorical(
                    ti,
                    &ti_freq,
                    &columns,
                    &mut out,
                    &level_params,
                    cluster_map.as_ref(),
                );
            }
        } else {
            let path = build_path_categorical_strided(nx, ny, nz, &out, &mut rng, stride);
            run_sequential_path_categorical(
                ti,
                &ti_freq,
                &path,
                &mut out,
                &level_params,
                &mut rng,
                cluster_map.as_ref(),
            );
        }
    }

    out
}

fn build_columns_categorical(
    nx: usize,
    ny: usize,
    nz: usize,
    out: &Grid3<u32>,
    rng: &mut StdRng,
) -> Vec<Vec<(usize, usize, usize)>> {
    build_columns_categorical_strided(nx, ny, nz, out, rng, 1)
}

fn build_columns_categorical_strided(
    nx: usize,
    ny: usize,
    nz: usize,
    out: &Grid3<u32>,
    rng: &mut StdRng,
    stride: usize,
) -> Vec<Vec<(usize, usize, usize)>> {
    let mut cols: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    let mut y = 0;
    while y < ny {
        let mut x = 0;
        while x < nx {
            let mut col: Vec<(usize, usize, usize)> = Vec::new();
            let mut z = 0;
            while z < nz {
                if *out.get(x, y, z) == u32::MAX {
                    col.push((x, y, z));
                }
                z += stride;
            }
            if !col.is_empty() {
                cols.push(col);
            }
            x += stride;
        }
        y += stride;
    }
    cols.shuffle(rng);
    cols
}

fn build_path_categorical_strided(
    nx: usize,
    ny: usize,
    nz: usize,
    out: &Grid3<u32>,
    rng: &mut StdRng,
    stride: usize,
) -> Vec<(usize, usize, usize)> {
    let mut cells = Vec::new();
    let mut z = 0;
    while z < nz {
        let mut y = 0;
        while y < ny {
            let mut x = 0;
            while x < nx {
                if *out.get(x, y, z) == u32::MAX {
                    cells.push((x, y, z));
                }
                x += stride;
            }
            y += stride;
        }
        z += stride;
    }
    cells.shuffle(rng);
    cells
}

fn run_parallel_columns_categorical(
    ti: &Grid3<u32>,
    ti_freq: &TiFreq,
    columns: &[Vec<(usize, usize, usize)>],
    out: &mut Grid3<u32>,
    params: &QsParams,
    cluster_map: Option<&ClusterMap>,
) {
    let n_lags = params.template.lags.len();
    let nx = out.nx;
    let ny = out.ny;
    let nz = out.nz;
    let shared = SharedOut::new(out);

    columns.par_iter().enumerate().for_each_init(
        || {
            let ctx = FftCtx::new(ti.dims(), params.template.extents());
            let mismatch_buf = Grid3::<f64>::filled(ti.nx, ti.ny, ti.nz, 0.0);
            let obs = vec![0u32; n_lags];
            let mask = vec![0.0f64; n_lags];
            (ctx, mismatch_buf, obs, mask)
        },
        |(ctx, mismatch_buf, obs, mask), (col_idx, column)| {
            let mut rng = StdRng::seed_from_u64(params.seed.wrapping_add(col_idx as u64));
            for &(x, y, z) in column.iter() {
                let idx = (z * ny + y) * nx + x;
                if unsafe { shared.read(idx) } != u32::MAX {
                    continue;
                }
                build_event_categorical_shared(
                    &shared,
                    x,
                    y,
                    z,
                    nx,
                    ny,
                    nz,
                    &params.template,
                    obs,
                    mask,
                );
                let n_active: f64 = mask.iter().sum();
                if n_active < 1.0 {
                    let pick = rng.random_range(0..ti.data.len());
                    unsafe { shared.write(idx, ti.data[pick]) };
                    continue;
                }
                mismatch_categorical(ctx, ti_freq, &params.template, obs, mask, mismatch_buf);
                if let Some(cmap) = cluster_map {
                    apply_cluster_filter_categorical(mismatch_buf, cmap, obs, mask);
                }
                let value = pick_top_k_categorical(ti, mismatch_buf, params, &mut rng);
                unsafe { shared.write(idx, value) };
            }
        },
    );
}

fn apply_cluster_filter_categorical(
    mismatch: &mut Grid3<f64>,
    cluster: &ClusterMap,
    obs: &[u32],
    mask: &[f64],
) {
    if obs.len() * cluster.n_classes as usize != cluster.feature_dim {
        return;
    }
    let target = classify_event_categorical(cluster, obs, mask);
    for (m, &lbl) in mismatch.data.iter_mut().zip(cluster.labels.data.iter()) {
        if lbl != target {
            *m = f64::INFINITY;
        }
    }
}

fn run_sequential_path_categorical(
    ti: &Grid3<u32>,
    ti_freq: &TiFreq,
    path: &[(usize, usize, usize)],
    out: &mut Grid3<u32>,
    params: &QsParams,
    rng: &mut StdRng,
    cluster_map: Option<&ClusterMap>,
) {
    let mut ctx = FftCtx::new(ti.dims(), params.template.extents());
    let mut mismatch_buf = Grid3::filled(ti.nx, ti.ny, ti.nz, 0.0);
    let mut obs = vec![0u32; params.template.lags.len()];
    let mut mask = vec![0.0; params.template.lags.len()];

    for &(x, y, z) in path.iter() {
        let idx = out.idx(x, y, z);
        if out.data[idx] != u32::MAX {
            continue;
        }
        build_event_categorical(out, x, y, z, &params.template, &mut obs, &mut mask);
        let n_active: f64 = mask.iter().sum();
        if n_active < 1.0 {
            let pick = rng.random_range(0..ti.data.len());
            out.data[idx] = ti.data[pick];
            continue;
        }
        mismatch_categorical(
            &mut ctx,
            ti_freq,
            &params.template,
            &obs,
            &mask,
            &mut mismatch_buf,
        );
        if let Some(cmap) = cluster_map {
            apply_cluster_filter_categorical(&mut mismatch_buf, cmap, &obs, &mask);
        }
        let value = pick_top_k_categorical(ti, &mismatch_buf, params, rng);
        out.data[idx] = value;
    }
}

fn build_event_categorical_shared(
    shared: &SharedOut<u32>,
    x: usize,
    y: usize,
    z: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    template: &QsTemplate,
    obs: &mut [u32],
    mask: &mut [f64],
) {
    for (i, lag) in template.lags.iter().enumerate() {
        let xx = x as i64 + lag.dx as i64;
        let yy = y as i64 + lag.dy as i64;
        let zz = z as i64 + lag.dz as i64;
        if xx < 0 || yy < 0 || zz < 0 || xx as usize >= nx || yy as usize >= ny || zz as usize >= nz
        {
            obs[i] = 0;
            mask[i] = 0.0;
            continue;
        }
        let idx = (zz as usize * ny + yy as usize) * nx + xx as usize;
        let v = unsafe { shared.read(idx) };
        if v == u32::MAX {
            obs[i] = 0;
            mask[i] = 0.0;
        } else {
            obs[i] = v;
            mask[i] = 1.0;
        }
    }
}

/// Multivariate variant: `variables` are co-located on the same simulation
/// grid. At each target cell the mismatch field is the *weighted sum* of
/// per-variable mismatches; the chosen TI cell produces the simulated value
/// for every variable jointly.
///
/// All TIs must have the same shape. Continuous mismatches are scaled by TI
/// variance so cross-attribute weights are comparable. Parallelised across
/// (x, y) columns under `column_sequential`, identical to the univariate
/// drivers.
pub fn simulate_multivariate(
    variables: Vec<MultivariateVariable>,
    sim_shape: (usize, usize, usize),
    params: &QsParams,
) -> Vec<MultivariateOutput> {
    let (nx, ny, nz) = sim_shape;
    assert!(
        !variables.is_empty(),
        "multivariate QS needs at least one variable"
    );
    let ti_dims = variables[0].ti_dims();
    for v in variables.iter().skip(1) {
        assert_eq!(
            v.ti_dims(),
            ti_dims,
            "all multivariate TIs must share the same shape"
        );
    }

    let mut outputs: Vec<MultivariateOutput> = variables
        .iter()
        .map(|v| v.alloc_output(nx, ny, nz))
        .collect();

    let mut rng = StdRng::seed_from_u64(params.seed);
    let n_levels = params.n_grid_levels.max(1);

    let cont_scales: Vec<f64> = variables
        .iter()
        .map(|v| match v {
            MultivariateVariable::Continuous { ti, .. } => continuous_scale(ti),
            MultivariateVariable::Categorical { .. } => 1.0,
        })
        .collect();

    for level in 0..n_levels {
        let stride: usize = 1 << (n_levels - 1 - level);
        let scaled_template = params
            .template
            .scaled(stride as i32, stride as i32, stride as i32);
        if !template_fits_ti(&scaled_template, ti_dims) {
            continue;
        }
        let level_params = QsParams {
            template: scaled_template,
            seed: params
                .seed
                .wrapping_add(level as u64 * 0x9E37_79B9_7F4A_7C15),
            ..params.clone()
        };

        // Per-level FFT contexts and TI frequency caches.
        let mut seed_ctxs: Vec<FftCtx> = variables
            .iter()
            .map(|_| FftCtx::new(ti_dims, level_params.template.extents()))
            .collect();
        let ti_freqs: Vec<TiFreq> = variables
            .iter()
            .enumerate()
            .map(|(i, v)| match v {
                MultivariateVariable::Continuous { ti, .. } => {
                    TiFreq::for_continuous(ti, &mut seed_ctxs[i])
                }
                MultivariateVariable::Categorical { ti, n_classes, .. } => {
                    TiFreq::for_categorical(ti, *n_classes, &mut seed_ctxs[i])
                }
            })
            .collect();

        let columns = build_columns_multivariate_strided(
            nx,
            ny,
            nz,
            &outputs,
            &mut rng,
            level_params.column_sequential,
            stride,
        );
        if !columns.is_empty() {
            run_parallel_columns_multivariate(
                &variables,
                &ti_freqs,
                &cont_scales,
                &columns,
                &mut outputs,
                &level_params,
                ti_dims,
            );
        }
    }

    outputs
}

fn build_columns_multivariate(
    nx: usize,
    ny: usize,
    nz: usize,
    outputs: &[MultivariateOutput],
    rng: &mut StdRng,
    column_sequential: bool,
) -> Vec<Vec<(usize, usize, usize)>> {
    build_columns_multivariate_strided(nx, ny, nz, outputs, rng, column_sequential, 1)
}

fn build_columns_multivariate_strided(
    nx: usize,
    ny: usize,
    nz: usize,
    outputs: &[MultivariateOutput],
    rng: &mut StdRng,
    column_sequential: bool,
    stride: usize,
) -> Vec<Vec<(usize, usize, usize)>> {
    let cell_is_unsimulated = |x: usize, y: usize, z: usize| -> bool {
        outputs.iter().any(|o| match o {
            MultivariateOutput::Continuous(g) => g.get(x, y, z).is_nan(),
            MultivariateOutput::Categorical(g) => *g.get(x, y, z) == u32::MAX,
        })
    };

    if column_sequential {
        let mut cols: Vec<Vec<(usize, usize, usize)>> = Vec::new();
        let mut y = 0;
        while y < ny {
            let mut x = 0;
            while x < nx {
                let mut col: Vec<(usize, usize, usize)> = Vec::new();
                let mut z = 0;
                while z < nz {
                    if cell_is_unsimulated(x, y, z) {
                        col.push((x, y, z));
                    }
                    z += stride;
                }
                if !col.is_empty() {
                    cols.push(col);
                }
                x += stride;
            }
            y += stride;
        }
        cols.shuffle(rng);
        cols
    } else {
        let mut cells: Vec<(usize, usize, usize)> = Vec::new();
        let mut z = 0;
        while z < nz {
            let mut y = 0;
            while y < ny {
                let mut x = 0;
                while x < nx {
                    if cell_is_unsimulated(x, y, z) {
                        cells.push((x, y, z));
                    }
                    x += stride;
                }
                y += stride;
            }
            z += stride;
        }
        cells.shuffle(rng);
        vec![cells]
    }
}

fn run_parallel_columns_multivariate(
    variables: &[MultivariateVariable],
    ti_freqs: &[TiFreq],
    cont_scales: &[f64],
    columns: &[Vec<(usize, usize, usize)>],
    outputs: &mut [MultivariateOutput],
    params: &QsParams,
    ti_dims: (usize, usize, usize),
) {
    let n_lags = params.template.lags.len();
    let n_vars = variables.len();

    // Take ownership of each output's underlying Vec<T> via two SharedOuts.
    // Because outputs are Continuous OR Categorical, we hand out one shared
    // pointer per variable index, dispatched by enum at each call site.
    enum SharedVarOut {
        Continuous(SharedOut<f64>),
        Categorical(SharedOut<u32>),
    }
    let shared: Vec<SharedVarOut> = outputs
        .iter_mut()
        .map(|o| match o {
            MultivariateOutput::Continuous(g) => SharedVarOut::Continuous(SharedOut::new(g)),
            MultivariateOutput::Categorical(g) => SharedVarOut::Categorical(SharedOut::new(g)),
        })
        .collect();
    let nx0 = match &shared[0] {
        SharedVarOut::Continuous(s) => s.len,
        SharedVarOut::Categorical(s) => s.len,
    };
    let _ = nx0;

    let (out_nx, out_ny, out_nz) = match outputs[0] {
        MultivariateOutput::Continuous(ref g) => (g.nx, g.ny, g.nz),
        MultivariateOutput::Categorical(ref g) => (g.nx, g.ny, g.nz),
    };

    columns.par_iter().enumerate().for_each_init(
        || {
            let ctxs: Vec<FftCtx> = (0..n_vars)
                .map(|_| FftCtx::new(ti_dims, params.template.extents()))
                .collect();
            let mismatch_bufs: Vec<Grid3<f64>> = (0..n_vars)
                .map(|_| Grid3::filled(ti_dims.0, ti_dims.1, ti_dims.2, 0.0))
                .collect();
            let combined: Grid3<f64> = Grid3::filled(ti_dims.0, ti_dims.1, ti_dims.2, 0.0);
            let obs_cont = vec![0.0f64; n_lags];
            let obs_cat = vec![0u32; n_lags];
            let mask = vec![0.0f64; n_lags];
            let scratch: Vec<num_complex::Complex<f32>> = Vec::new();
            (
                ctxs,
                mismatch_bufs,
                combined,
                obs_cont,
                obs_cat,
                mask,
                scratch,
            )
        },
        |(ctxs, mismatch_bufs, combined, obs_cont, obs_cat, mask, scratch), (col_idx, column)| {
            let mut rng = StdRng::seed_from_u64(params.seed.wrapping_add(col_idx as u64));
            for &(x, y, z) in column.iter() {
                let idx = (z * out_ny + y) * out_nx + x;
                // Skip cells already done on every variable.
                let already = shared.iter().all(|s| match s {
                    SharedVarOut::Continuous(s) => {
                        let v = unsafe { s.read(idx) };
                        !v.is_nan()
                    }
                    SharedVarOut::Categorical(s) => {
                        let v = unsafe { s.read(idx) };
                        v != u32::MAX
                    }
                });
                if already {
                    continue;
                }

                for v in combined.data.iter_mut() {
                    *v = 0.0;
                }
                let mut weight_sum = 0.0f64;

                for vi in 0..n_vars {
                    let w_base = variables[vi].weight();
                    let n_active = match &shared[vi] {
                        SharedVarOut::Continuous(s) => {
                            build_event_continuous_shared(
                                s,
                                x,
                                y,
                                z,
                                out_nx,
                                out_ny,
                                out_nz,
                                &params.template,
                                obs_cont,
                                mask,
                            );
                            let n: f64 = mask.iter().sum();
                            if n < 1.0 {
                                0.0
                            } else {
                                mismatch_continuous(
                                    &mut ctxs[vi],
                                    &ti_freqs[vi],
                                    &params.template,
                                    obs_cont,
                                    mask,
                                    &mut mismatch_bufs[vi],
                                    scratch,
                                );
                                n
                            }
                        }
                        SharedVarOut::Categorical(s) => {
                            build_event_categorical_shared(
                                s,
                                x,
                                y,
                                z,
                                out_nx,
                                out_ny,
                                out_nz,
                                &params.template,
                                obs_cat,
                                mask,
                            );
                            let n: f64 = mask.iter().sum();
                            if n < 1.0 {
                                0.0
                            } else {
                                mismatch_categorical(
                                    &mut ctxs[vi],
                                    &ti_freqs[vi],
                                    &params.template,
                                    obs_cat,
                                    mask,
                                    &mut mismatch_bufs[vi],
                                );
                                n
                            }
                        }
                    };
                    if n_active < 1.0 {
                        continue;
                    }
                    let w = w_base / cont_scales[vi].max(1e-12);
                    for (acc, m) in combined.data.iter_mut().zip(mismatch_bufs[vi].data.iter()) {
                        *acc += w * *m;
                    }
                    weight_sum += w;
                }

                let pick = if weight_sum <= 0.0 {
                    rng.random_range(0..combined.data.len())
                } else {
                    for v in combined.data.iter_mut() {
                        *v /= weight_sum;
                    }
                    pick_top_k_index(combined, params, &mut rng)
                };

                // Commit the picked TI value to every variable.
                for (vi, s) in shared.iter().enumerate() {
                    match (s, &variables[vi]) {
                        (
                            SharedVarOut::Continuous(s),
                            MultivariateVariable::Continuous { ti, .. },
                        ) => unsafe {
                            s.write(idx, ti.data[pick]);
                        },
                        (
                            SharedVarOut::Categorical(s),
                            MultivariateVariable::Categorical { ti, .. },
                        ) => unsafe {
                            s.write(idx, ti.data[pick]);
                        },
                        _ => {}
                    }
                }
            }
        },
    );
}

/// A single co-located variable for multivariate QS.
pub enum MultivariateVariable {
    Continuous {
        ti: Grid3<f64>,
        hard_data: Vec<HardSample<f64>>,
        weight: f64,
    },
    Categorical {
        ti: Grid3<u32>,
        n_classes: u32,
        hard_data: Vec<HardSample<u32>>,
        weight: f64,
    },
}

impl MultivariateVariable {
    pub fn ti_dims(&self) -> (usize, usize, usize) {
        match self {
            MultivariateVariable::Continuous { ti, .. } => ti.dims(),
            MultivariateVariable::Categorical { ti, .. } => ti.dims(),
        }
    }

    pub fn weight(&self) -> f64 {
        match self {
            MultivariateVariable::Continuous { weight, .. } => *weight,
            MultivariateVariable::Categorical { weight, .. } => *weight,
        }
    }

    pub fn alloc_output(&self, nx: usize, ny: usize, nz: usize) -> MultivariateOutput {
        match self {
            MultivariateVariable::Continuous { hard_data, .. } => {
                let mut g = Grid3::filled(nx, ny, nz, f64::NAN);
                for hd in hard_data.iter() {
                    if g.in_bounds(hd.x as i64, hd.y as i64, hd.z as i64) {
                        *g.get_mut(hd.x as usize, hd.y as usize, hd.z as usize) = hd.value;
                    }
                }
                MultivariateOutput::Continuous(g)
            }
            MultivariateVariable::Categorical { hard_data, .. } => {
                let mut g = Grid3::filled(nx, ny, nz, u32::MAX);
                for hd in hard_data.iter() {
                    if g.in_bounds(hd.x as i64, hd.y as i64, hd.z as i64) {
                        *g.get_mut(hd.x as usize, hd.y as usize, hd.z as usize) = hd.value;
                    }
                }
                MultivariateOutput::Categorical(g)
            }
        }
    }
}

pub enum MultivariateOutput {
    Continuous(Grid3<f64>),
    Categorical(Grid3<u32>),
}

// --- helpers ---

fn place_hard_data<T: Copy>(out: &mut Grid3<T>, hard: &[HardSample<T>]) {
    for hd in hard.iter() {
        if out.in_bounds(hd.x as i64, hd.y as i64, hd.z as i64) {
            *out.get_mut(hd.x as usize, hd.y as usize, hd.z as usize) = hd.value;
        }
    }
}

fn build_path(
    nx: usize,
    ny: usize,
    nz: usize,
    out: &Grid3<f64>,
    params: &QsParams,
    rng: &mut StdRng,
) -> Vec<(usize, usize, usize)> {
    let mut cells: Vec<(usize, usize, usize)> = Vec::with_capacity(nx * ny * nz);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if out.get(x, y, z).is_nan() {
                    cells.push((x, y, z));
                }
            }
        }
    }
    if params.column_sequential {
        // Random over (x, y) columns, sequential within each column.
        let mut by_col: std::collections::HashMap<(usize, usize), Vec<(usize, usize, usize)>> =
            std::collections::HashMap::new();
        for c in cells.into_iter() {
            by_col.entry((c.0, c.1)).or_default().push(c);
        }
        for v in by_col.values_mut() {
            v.sort_by_key(|c| c.2);
        }
        let mut cols: Vec<_> = by_col.into_values().collect();
        cols.shuffle(rng);
        cols.into_iter().flatten().collect()
    } else {
        cells.shuffle(rng);
        cells
    }
}

fn build_path_categorical(
    nx: usize,
    ny: usize,
    nz: usize,
    out: &Grid3<u32>,
    params: &QsParams,
    rng: &mut StdRng,
) -> Vec<(usize, usize, usize)> {
    let mut cells: Vec<(usize, usize, usize)> = Vec::with_capacity(nx * ny * nz);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if *out.get(x, y, z) == u32::MAX {
                    cells.push((x, y, z));
                }
            }
        }
    }
    if params.column_sequential {
        let mut by_col: std::collections::HashMap<(usize, usize), Vec<(usize, usize, usize)>> =
            std::collections::HashMap::new();
        for c in cells.into_iter() {
            by_col.entry((c.0, c.1)).or_default().push(c);
        }
        for v in by_col.values_mut() {
            v.sort_by_key(|c| c.2);
        }
        let mut cols: Vec<_> = by_col.into_values().collect();
        cols.shuffle(rng);
        cols.into_iter().flatten().collect()
    } else {
        cells.shuffle(rng);
        cells
    }
}

fn build_event_continuous(
    out: &Grid3<f64>,
    x: usize,
    y: usize,
    z: usize,
    template: &QsTemplate,
    obs: &mut [f64],
    mask: &mut [f64],
) {
    for (i, lag) in template.lags.iter().enumerate() {
        let xx = x as i64 + lag.dx as i64;
        let yy = y as i64 + lag.dy as i64;
        let zz = z as i64 + lag.dz as i64;
        if !out.in_bounds(xx, yy, zz) {
            obs[i] = 0.0;
            mask[i] = 0.0;
            continue;
        }
        let v = *out.get(xx as usize, yy as usize, zz as usize);
        if v.is_nan() {
            obs[i] = 0.0;
            mask[i] = 0.0;
        } else {
            obs[i] = v;
            mask[i] = 1.0;
        }
    }
}

fn build_event_categorical(
    out: &Grid3<u32>,
    x: usize,
    y: usize,
    z: usize,
    template: &QsTemplate,
    obs: &mut [u32],
    mask: &mut [f64],
) {
    for (i, lag) in template.lags.iter().enumerate() {
        let xx = x as i64 + lag.dx as i64;
        let yy = y as i64 + lag.dy as i64;
        let zz = z as i64 + lag.dz as i64;
        if !out.in_bounds(xx, yy, zz) {
            obs[i] = 0;
            mask[i] = 0.0;
            continue;
        }
        let v = *out.get(xx as usize, yy as usize, zz as usize);
        if v == u32::MAX {
            obs[i] = 0;
            mask[i] = 0.0;
        } else {
            obs[i] = v;
            mask[i] = 1.0;
        }
    }
}

fn ti_global_mean(ti: &Grid3<f64>) -> f64 {
    if ti.data.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut n = 0;
    for v in ti.data.iter() {
        if v.is_finite() {
            sum += *v;
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

fn continuous_scale(ti: &Grid3<f64>) -> f64 {
    // Use the TI variance to normalize the mismatch for multivariate combo.
    if ti.data.is_empty() {
        return 1.0;
    }
    let mut sum = 0.0;
    let mut n = 0;
    for v in ti.data.iter() {
        if v.is_finite() {
            sum += *v;
            n += 1;
        }
    }
    if n == 0 {
        return 1.0;
    }
    let mean = sum / n as f64;
    let mut var = 0.0;
    for v in ti.data.iter() {
        if v.is_finite() {
            let d = *v - mean;
            var += d * d;
        }
    }
    (var / n as f64).max(1e-12)
}

fn pick_top_k_continuous(
    ti: &Grid3<f64>,
    mismatch: &Grid3<f64>,
    params: &QsParams,
    rng: &mut StdRng,
) -> f64 {
    let idx = pick_top_k_index(mismatch, params, rng);
    ti.data[idx]
}

fn pick_top_k_categorical(
    ti: &Grid3<u32>,
    mismatch: &Grid3<f64>,
    params: &QsParams,
    rng: &mut StdRng,
) -> u32 {
    let idx = pick_top_k_index(mismatch, params, rng);
    ti.data[idx]
}

fn pick_top_k_index(mismatch: &Grid3<f64>, params: &QsParams, rng: &mut StdRng) -> usize {
    let n = mismatch.data.len();
    let k = params.top_k.max(1).min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    let frac = params.ti_sample_fraction.clamp(1e-6, 1.0);
    if frac < 1.0 {
        let want = ((n as f64) * frac).round() as usize;
        let want = want.max(k);
        indices.shuffle(rng);
        indices.truncate(want);
    }
    // Partial select: find the k-th smallest mismatch among the candidates.
    indices.select_nth_unstable_by(k - 1, |a, b| {
        mismatch.data[*a]
            .partial_cmp(&mismatch.data[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices[rng.random_range(0..k)]
}
