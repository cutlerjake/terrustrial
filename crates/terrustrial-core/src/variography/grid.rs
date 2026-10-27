//! Exact requested-lag variograms on bounded regular grids, including files.
//! O(cells * lags) work and O(tile_cells + lags) workspace. No pairs are sampled,
//! no periodic wrapping is used, and no global input array is required.
#[cfg(feature = "fft")]
pub mod counts;
#[cfg(feature = "fft")]
pub mod spectral;
use super::{
    accumulation::{half_increment_product, ScaledMean},
    Bin,
};
use crate::{
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    grid::Source,
    Error, Result,
};
use std::collections::HashSet;

pub struct Options<'a> {
    pub variables: [usize; 2],
    /// Nonzero offsets in grid cells. Each is an independent query; when pooling
    /// unordered lag bins, include only one of h and -h to avoid duplicate pairs.
    pub offsets: &'a [[i64; 3]],
    pub spacing: [f64; 3],
    pub tile_cells: usize,
    pub budget: &'a MemoryBudget,
    pub cancellation: &'a Cancellation,
}
#[derive(Debug)]
pub struct Lag {
    pub offset: [i64; 3],
    pub statistics: Bin,
}
#[derive(Debug)]
pub struct Variogram {
    pub lags: Vec<Lag>,
    pub source_values_read: u128,
    _reservation: Reservation,
}

/// Both variables must exist at both endpoints of a cross-variogram pair.
/// Negative cross-semivariances are legitimate. Pair counts are exact u64.
/// The fixed x-fast reduction order is invariant to tile size. Source reads and
/// numerical failures are propagated, leaving no partially returned variogram.
pub fn experimental(
    source: &mut (impl Source + ?Sized),
    options: Options<'_>,
) -> Result<Variogram> {
    options.cancellation.check()?;
    if options.offsets.is_empty()
        || options.tile_cells == 0
        || options.variables.iter().any(|v| *v >= source.variables())
        || options.spacing.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(invalid(
            "grid variogram variables, offsets, spacing or tile size",
        ));
    }
    let capacity = options
        .tile_cells
        .min(usize::try_from(source.layout().cells()).unwrap_or(usize::MAX));
    let columns = if options.variables[0] == options.variables[1] {
        2
    } else {
        4
    };
    let bytes = checked_len(&[capacity, columns, std::mem::size_of::<Option<f64>>()])?
        .checked_add(checked_len(&[options.offsets.len(), 64])?)
        .ok_or(Error::SizeOverflow)?;
    let _workspace = options
        .budget
        .reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
    let reservation = options.budget.reserve(
        u64::try_from(checked_len(&[
            options.offsets.len(),
            std::mem::size_of::<Lag>(),
        ])?)
        .map_err(|_| Error::SizeOverflow)?,
    )?;
    let mut seen = HashSet::new();
    for &offset in options.offsets {
        if offset == [0; 3] || !seen.insert(offset) {
            return Err(invalid("zero or duplicate grid lag"));
        }
        distance(offset, options.spacing)?;
    }
    drop(seen);
    let mut buffers = vec![vec![None; capacity]; columns];
    let mut lags = Vec::with_capacity(options.offsets.len());
    let mut source_values_read = 0_u128;
    for &offset in options.offsets {
        lags.push(Lag {
            offset,
            statistics: direct_lag(
                source,
                options.variables,
                offset,
                options.spacing,
                &mut buffers,
                &mut source_values_read,
                options.cancellation,
            )?,
        });
    }
    Ok(Variogram {
        lags,
        source_values_read,
        _reservation: reservation,
    })
}
// Shared with spectral refinement. Scratch is retained by the caller, so a
// fallback has no per-lag allocation and exactly the streaming reduction order.
fn direct_lag(
    source: &mut (impl Source + ?Sized),
    variables: [usize; 2],
    offset: [i64; 3],
    spacing: [f64; 3],
    buffers: &mut [Vec<Option<f64>>],
    source_values_read: &mut u128,
    cancel: &Cancellation,
) -> Result<Bin> {
    let layout = source.layout();
    let dimensions = layout.dimensions();
    let capacity = buffers[0].len();
    let columns = if variables[0] == variables[1] { 2 } else { 4 };
    cancel.check()?;
    let mut mean = ScaledMean::default();
    let mut lower = [0; 3];
    let mut upper = [0; 3];
    let possible = (0..3).all(|a| offset[a].unsigned_abs() < dimensions[a]);
    if possible {
        for a in 0..3 {
            lower[a] = if offset[a] < 0 {
                offset[a].unsigned_abs()
            } else {
                0
            };
            upper[a] = dimensions[a] - if offset[a] > 0 { offset[a] as u64 } else { 0 };
        }
        // Merge adjacent valid rows/planes before issuing bounded reads.
        // This changes I/O granularity without changing pair/reduction order.
        let full_x = lower[0] == 0 && upper[0] == dimensions[0];
        let full_y = lower[1] == 0 && upper[1] == dimensions[1];
        let step_y = if full_x { upper[1] - lower[1] } else { 1 };
        let step_z = if full_x && full_y {
            upper[2] - lower[2]
        } else {
            1
        };
        let length = (upper[0] - lower[0])
            .checked_mul(step_y)
            .and_then(|v| v.checked_mul(step_z))
            .ok_or(Error::SizeOverflow)?;
        let mut z = lower[2];
        while z < upper[2] {
            let mut y = lower[1];
            while y < upper[1] {
                let a = [lower[0], y, z];
                let b = std::array::from_fn(|i| (a[i] as i128 + offset[i] as i128) as u64);
                let first = [layout.index(a)?, layout.index(b)?];
                let mut consumed = 0;
                while consumed < length {
                    cancel.check()?;
                    let n = (length - consumed).min(capacity as u64) as usize;
                    for (i, buffer) in buffers.iter_mut().take(columns).enumerate() {
                        source.read(variables[i / 2], first[i % 2] + consumed, &mut buffer[..n])?;
                        if buffer[..n].iter().flatten().any(|v| !v.is_finite()) {
                            return Err(invalid("nonfinite grid variogram value"));
                        }
                    }
                    *source_values_read = source_values_read
                        .checked_add(n as u128 * columns as u128)
                        .ok_or(Error::SizeOverflow)?;
                    for ((a, b), (c, d)) in buffers[0][..n].iter().zip(&buffers[1][..n]).zip(
                        buffers[if columns == 2 { 0 } else { 2 }][..n]
                            .iter()
                            .zip(&buffers[if columns == 2 { 1 } else { 3 }][..n]),
                    ) {
                        let (Some(a), Some(b), Some(c), Some(d)) = (*a, *b, *c, *d) else {
                            continue;
                        };
                        mean.add(half_increment_product(a, b, c, d)?)?;
                    }
                    consumed += n as u64;
                }
                y += step_y;
            }
            z += step_z;
        }
    }
    Ok(Bin {
        pairs: mean.count,
        mean_distance: if mean.count > 0 {
            Some(if offset == [0; 3] {
                0.0
            } else {
                distance(offset, spacing)?
            })
        } else {
            None
        },
        semivariance: mean.value(),
    })
}
fn distance(offset: [i64; 3], spacing: [f64; 3]) -> Result<f64> {
    let distance = crate::data::norm3(std::array::from_fn(|a| offset[a] as f64 * spacing[a]));
    if !distance.is_finite() || distance == 0.0 {
        return Err(invalid("grid lag distance is not representable"));
    }
    Ok(distance)
}
