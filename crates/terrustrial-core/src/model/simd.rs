//! Lane-parallel covariance evaluation. Reductions across structures retain
//! scalar order; no approximate transcendental functions or fused multiply-adds.
use super::*;
use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

pub(super) trait CovarianceSite {
    fn site(&self) -> &Site;
}
impl CovarianceSite for Site {
    fn site(&self) -> &Site {
        self
    }
}
impl CovarianceSite for Observation {
    fn site(&self) -> &Site {
        Observation::site(self)
    }
}

pub(super) fn covariance_line<T: CovarianceSite>(
    model: &Lmc,
    a: &Site,
    b: &[T],
    out: &mut [f64],
    observations_left: bool,
) -> Result<()> {
    if b.len() != out.len()
        || a.variable >= model.variables
        || b.iter().any(|o| o.site().variable >= model.variables)
    {
        return Err(invalid("covariance line dimensions or variables"));
    }
    fearless_simd::dispatch!(model.simd, simd => line_kernel(simd, model, a, b, out, observations_left))
}
#[simd]
fn line_kernel<S: Simd, T: CovarianceSite>(
    simd: S,
    model: &Lmc,
    a: &Site,
    b: &[T],
    out: &mut [f64],
    observations_left: bool,
) -> Result<()> {
    let lanes = S::f64s::LEN;
    for (b, out) in b.chunks(lanes).zip(out.chunks_mut(lanes)) {
        if b.len() != lanes
            || a.support.node_count() != 1
            || b.iter().any(|o| o.site().support.node_count() != 1)
        {
            for (b, out) in b.iter().zip(out) {
                *out = if observations_left {
                    model.covariance(b.site(), a)?
                } else {
                    model.covariance(a, b.site())?
                };
            }
            continue;
        }
        let point = a.support.center();
        let p = point.coordinates();
        let q: [S::f64s; 3] = std::array::from_fn(|axis| {
            S::f64s::from_fn(simd, |i| b[i].site().support.center().coordinates()[axis])
        });
        let delta: [S::f64s; 3] = std::array::from_fn(|axis| {
            let p = S::f64s::splat(simd, p[axis]);
            if observations_left {
                q[axis] - p
            } else {
                p - q[axis]
            }
        });
        let mut total = S::f64s::splat(simd, 0.0);
        for structure in &model.structures {
            let rho = correlation_lanes(simd, structure, point, b, q, delta, observations_left);
            let sill = S::f64s::from_fn(simd, |i| {
                let other = b[i].site().variable;
                let offset = if observations_left {
                    other * model.variables + a.variable
                } else {
                    a.variable * model.variables + other
                };
                structure.coregionalization.matrix[offset]
            });
            total += sill * rho;
        }
        if total.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(Error::Numerical {
                context: "LMC covariance line",
            });
        }
        total.store_slice(out);
    }
    Ok(())
}

// The structure reduction stays in the caller. Reusing this geometry-dependent
// result does not reuse a sill product or change floating-point addition order.
#[inline(always)]
fn correlation_lanes<S: Simd, T: CovarianceSite>(
    simd: S,
    structure: &Structure,
    point: Point3,
    b: &[T],
    q: [S::f64s; 3],
    delta: [S::f64s; 3],
    observations_left: bool,
) -> S::f64s {
    let p = point.coordinates();
    let lanes = S::f64s::LEN;
    if matches!(structure.correlation, Correlation::Nugget) {
        // Coincidence is geometric, not a distance that can underflow.
        S::f64s::from_fn(simd, |i| {
            if p == [q[0][i], q[1][i], q[2][i]] {
                1.0
            } else {
                0.0
            }
        })
    } else {
        let geometry = &structure.anisotropy;
        let transformed: [S::f64s; 3] = if geometry.isotropic {
            delta
        } else {
            std::array::from_fn(|axis| {
                (delta[0] * geometry.frame[axis][0]
                    + delta[1] * geometry.frame[axis][1]
                    + delta[2] * geometry.frame[axis][2])
                    / geometry.scales[axis]
            })
        };
        let squared = transformed[0] * transformed[0]
            + transformed[1] * transformed[1]
            + transformed[2] * transformed[2];
        let mut distance = squared.sqrt();
        if geometry.isotropic {
            distance /= geometry.scales[0];
        }
        for i in 0..lanes {
            // Match norm3's scaled hypot fallback, including overflow in
            // raw coordinate differences and underflow before squaring.
            if !squared[i].is_normal()
                && !(transformed[0][i] == 0.0
                    && transformed[1][i] == 0.0
                    && transformed[2][i] == 0.0)
            {
                let other = b[i].site().support.center();
                distance[i] = if observations_left {
                    geometry.normalized_distance(other, point)
                } else {
                    geometry.normalized_distance(point, other)
                };
            }
        }
        match structure.correlation {
            Correlation::Spherical => {
                let polynomial = S::f64s::splat(simd, 1.0) - distance * 1.5
                    + distance * 0.5 * distance * distance;
                distance
                    .simd_ge(1.0)
                    .select(S::f64s::splat(simd, 0.0), polynomial)
            }
            kernel => {
                // Colocated variables often occupy adjacent lanes. Reuse
                // only bit-identical distances; preserve every operation
                // in the kernel and the order of the sill reductions.
                let mut rho = S::f64s::splat(simd, 0.0);
                for i in 0..lanes {
                    rho[i] = if i > 0 && distance[i].to_bits() == distance[i - 1].to_bits() {
                        rho[i - 1]
                    } else {
                        kernel.evaluate(distance[i])
                    };
                }
                rho
            }
        }
    }
}

pub(super) fn covariance_lines(
    model: &Lmc,
    sites: &[Site],
    observations: &[Observation],
    output: &mut [&mut [f64]],
    observations_left: bool,
) -> Result<()> {
    if sites.len() != output.len()
        || output.iter().any(|row| row.len() > observations.len())
        || sites.iter().any(|site| site.variable >= model.variables)
    {
        return Err(invalid("covariance lines dimensions or variables"));
    }
    let maximum = output.iter().map(|row| row.len()).max().unwrap_or(0);
    if observations[..maximum]
        .iter()
        .any(|o| o.site().variable >= model.variables)
    {
        return Err(invalid("covariance lines variables"));
    }
    let mut first = 0;
    while first < sites.len() {
        let point = sites[first]
            .support
            .center()
            .coordinates()
            .map(f64::to_bits);
        let mut end = first + 1;
        if sites[first].support.node_count() == 1 {
            while end < sites.len()
                && end - first < 8
                && sites[end].support.node_count() == 1
                && sites[end].support.center().coordinates().map(f64::to_bits) == point
            {
                end += 1;
            }
        }
        if end == first + 1 {
            let row = &mut output[first];
            covariance_line(
                model,
                &sites[first],
                &observations[..row.len()],
                row,
                observations_left,
            )?;
        } else {
            fearless_simd::dispatch!(model.simd, simd => colocated_lines(simd, model, &sites[first..end], observations, &mut output[first..end], observations_left))?;
        }
        first = end;
    }
    Ok(())
}

#[simd]
fn colocated_lines<S: Simd>(
    simd: S,
    model: &Lmc,
    sites: &[Site],
    observations: &[Observation],
    output: &mut [&mut [f64]],
    observations_left: bool,
) -> Result<()> {
    let lanes = S::f64s::LEN;
    let common = output.iter().map(|row| row.len()).min().unwrap_or(0) / lanes * lanes;
    let point = sites[0].support.center();
    let p = point.coordinates();
    for first in (0..common).step_by(lanes) {
        let b = &observations[first..first + lanes];
        if b.iter().any(|o| o.site().support.node_count() != 1) {
            for (site, row) in sites.iter().zip(output.iter_mut()) {
                line_kernel(
                    simd,
                    model,
                    site,
                    b,
                    &mut row[first..first + lanes],
                    observations_left,
                )?;
            }
            continue;
        }
        let q: [S::f64s; 3] = std::array::from_fn(|axis| {
            S::f64s::from_fn(simd, |i| b[i].site().support.center().coordinates()[axis])
        });
        let delta: [S::f64s; 3] = std::array::from_fn(|axis| {
            let p = S::f64s::splat(simd, p[axis]);
            if observations_left {
                q[axis] - p
            } else {
                p - q[axis]
            }
        });
        let mut totals = [S::f64s::splat(simd, 0.0); 8];
        for structure in &model.structures {
            let rho = correlation_lanes(simd, structure, point, b, q, delta, observations_left);
            for (total, site) in totals.iter_mut().zip(sites) {
                let sill = S::f64s::from_fn(simd, |i| {
                    let other = b[i].site().variable;
                    let offset = if observations_left {
                        other * model.variables + site.variable
                    } else {
                        site.variable * model.variables + other
                    };
                    structure.coregionalization.matrix[offset]
                });
                *total += sill * rho;
            }
        }
        for (total, row) in totals.iter().zip(output.iter_mut()) {
            if total.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(Error::Numerical {
                    context: "LMC covariance lines",
                });
            }
            total.store_slice(&mut row[first..first + lanes]);
        }
    }
    for (site, row) in sites.iter().zip(output) {
        let len = row.len();
        line_kernel(
            simd,
            model,
            site,
            &observations[common..len],
            &mut row[common..],
            observations_left,
        )?;
    }
    Ok(())
}
