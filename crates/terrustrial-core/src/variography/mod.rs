//! Explicit pair/boundary conventions and constrained single-structure fitting.
mod accumulation;
#[cfg(feature = "sqlite")]
pub mod disk;
pub mod grid;
pub mod indexed;
pub mod sampling;

use crate::{
    data::Point3, error::invalid, execution::Cancellation, model::Correlation, Error, Result,
};

#[derive(Clone, Debug)]
pub struct Sample {
    pub point: Point3,
    pub values: Vec<Option<f64>>,
}
#[derive(Clone, Debug)]
pub struct Direction {
    axis: [f64; 3],
    cosine: f64,
    bandwidth: f64,
}
impl Direction {
    /// Unit principal direction used by this unoriented cone.
    pub fn axis(&self) -> [f64; 3] {
        self.axis
    }
    pub fn new(axis: [f64; 3], angle_radians: f64, bandwidth: f64) -> Result<Self> {
        let length = axis[0].hypot(axis[1]).hypot(axis[2]);
        if !length.is_finite()
            || length == 0.0
            || !angle_radians.is_finite()
            || !(0.0..=std::f64::consts::FRAC_PI_2).contains(&angle_radians)
            || !bandwidth.is_finite()
            || bandwidth <= 0.0
        {
            return Err(invalid("variogram direction/tolerance"));
        }
        Ok(Self {
            axis: axis.map(|x| x / length),
            cosine: if angle_radians == std::f64::consts::FRAC_PI_2 {
                0.0
            } else {
                angle_radians.cos()
            },
            bandwidth,
        })
    }
    fn accepts(&self, a: Point3, b: Point3, distance: f64) -> bool {
        if distance == 0.0 {
            return true;
        }
        let a = a.coordinates();
        let b = b.coordinates();
        let delta = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let unit = delta.map(|v| v / distance);
        let projection = (0..3).map(|i| unit[i] * self.axis[i]).sum::<f64>().abs();
        let cross = [
            unit[1] * self.axis[2] - unit[2] * self.axis[1],
            unit[2] * self.axis[0] - unit[0] * self.axis[2],
            unit[0] * self.axis[1] - unit[1] * self.axis[0],
        ];
        let transverse = crate::data::norm3(cross) * distance;
        projection + 16.0 * f64::EPSILON >= self.cosine && transverse <= self.bandwidth
    }
}
#[derive(Clone, Debug)]
pub struct Bin {
    pub pairs: u64,
    pub mean_distance: Option<f64>,
    pub semivariance: Option<f64>,
}

/// Unordered pairs once; bins `[0,e0], (e0,e1], ...`; missing endpoint values
/// are excluded. Cross-variograms require both attributes at both endpoints.
/// This O(N^2) implementation is the independent pair oracle, not the scale path.
pub fn experimental(
    samples: &[Sample],
    variable_a: usize,
    variable_b: usize,
    edges: &[f64],
    direction: Option<&Direction>,
    cancel: &Cancellation,
) -> Result<Vec<Bin>> {
    if edges.is_empty()
        || edges.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || edges.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(invalid(
            "lag edges must be finite, positive, strictly increasing",
        ));
    }
    for sample in samples {
        if sample.values.len() <= variable_a.max(variable_b)
            || sample.values.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(invalid("sample variables or nonfinite attribute"));
        }
    }
    let mut accum = vec![(0_u64, 0.0, 0.0); edges.len()];
    for (i, a) in samples.iter().enumerate() {
        cancel.check()?;
        for b in &samples[i + 1..] {
            let (Some(aa), Some(ab), Some(ba), Some(bb)) = (
                a.values[variable_a],
                a.values[variable_b],
                b.values[variable_a],
                b.values[variable_b],
            ) else {
                continue;
            };
            let distance = a.point.distance(b.point);
            if distance > *edges.last().unwrap()
                || direction.is_some_and(|d| !d.accepts(a.point, b.point, distance))
            {
                continue;
            }
            let index = edges.partition_point(|edge| distance > *edge);
            let bin = &mut accum[index];
            bin.0 = bin.0.checked_add(1).ok_or(Error::SizeOverflow)?;
            // Online means avoid storing pairs and reduce accumulation growth.
            let gamma = 0.5 * (aa - ba) * (ab - bb);
            bin.1 += (distance - bin.1) / bin.0 as f64;
            bin.2 += (gamma - bin.2) / bin.0 as f64;
            if !bin.1.is_finite() || !bin.2.is_finite() {
                return Err(Error::Numerical {
                    context: "variogram accumulation overflow",
                });
            }
        }
    }
    Ok(accum
        .into_iter()
        .map(|(pairs, distance, gamma)| Bin {
            pairs,
            mean_distance: (pairs > 0).then_some(distance),
            semivariance: (pairs > 0).then_some(gamma),
        })
        .collect())
}

#[derive(Clone, Debug)]
pub struct Fit {
    pub correlation: Correlation,
    pub scale: f64,
    pub structured_sill: f64,
    pub nugget: f64,
    pub weighted_sse: f64,
    pub at_scale_bound: bool,
}
impl Fit {
    pub fn variogram(&self, h: f64) -> f64 {
        if h == 0.0 {
            0.0
        } else {
            self.nugget + self.structured_sill * (1.0 - self.correlation.evaluate(h / self.scale))
        }
    }
}

/// Profile NNLS nugget/sill over log-scale; pair-count weighted least squares.
/// Fit each marginal only. This does not produce an admissible fitted joint LMC.
pub fn fit_single(
    bins: &[Bin],
    families: &[Correlation],
    scale_bounds: [f64; 2],
    cancel: &Cancellation,
) -> Result<Fit> {
    if families.is_empty()
        || scale_bounds.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || scale_bounds[0] >= scale_bounds[1]
    {
        return Err(invalid("fit families/scale bounds"));
    }
    let mut data = Vec::new();
    for bin in bins {
        if bin.pairs == 0 {
            continue;
        }
        let (Some(h), Some(g)) = (bin.mean_distance, bin.semivariance) else {
            return Err(invalid("nonempty bin lacks statistics"));
        };
        if !h.is_finite() || !g.is_finite() || h < 0.0 || g < 0.0 {
            return Err(invalid(
                "scalar fitting requires finite nonnegative distances and semivariances",
            ));
        }
        if h > 0.0 {
            data.push((h, g, bin.pairs as f64));
        }
    }
    if data.len() < 3 {
        return Err(invalid(
            "at least three nonempty positive-distance bins required",
        ));
    }
    let weight_scale = data.iter().map(|d| d.2).fold(0.0, f64::max);
    for row in &mut data {
        row.2 /= weight_scale;
    }
    let lo = scale_bounds[0].ln();
    let hi = scale_bounds[1].ln();
    let mut best: Option<Fit> = None;
    for &family in families {
        let evaluate = |log_scale: f64| profile(&data, family, log_scale.exp());
        let mut index = 0_usize;
        let mut local = evaluate(lo)?;
        for i in 1..=64 {
            cancel.check()?;
            let f = evaluate(lo + (hi - lo) * i as f64 / 64.0)?;
            if f.weighted_sse < local.weighted_sse {
                local = f;
                index = i;
            }
        }
        let mut a = lo + (hi - lo) * index.saturating_sub(1) as f64 / 64.0;
        let mut b = lo + (hi - lo) * (index + 1).min(64) as f64 / 64.0;
        let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
        for _ in 0..80 {
            cancel.check()?;
            let x = b - ratio * (b - a);
            let y = a + ratio * (b - a);
            let fx = evaluate(x)?;
            let fy = evaluate(y)?;
            if fx.weighted_sse < local.weighted_sse {
                local = fx.clone();
            }
            if fy.weighted_sse < local.weighted_sse {
                local = fy.clone();
            }
            if fx.weighted_sse < fy.weighted_sse {
                b = y;
            } else {
                a = x;
            }
        }
        local.at_scale_bound = (local.scale / scale_bounds[0] - 1.0).abs() < 1e-5
            || (local.scale / scale_bounds[1] - 1.0).abs() < 1e-5;
        if best
            .as_ref()
            .is_none_or(|f| local.weighted_sse < f.weighted_sse)
        {
            best = Some(local);
        }
    }
    Ok(best.unwrap())
}
fn profile(data: &[(f64, f64, f64)], family: Correlation, scale: f64) -> Result<Fit> {
    let (mut s0, mut s1, mut s2, mut t0, mut t1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(h, g, w) in data {
        let f = 1.0 - family.evaluate(h / scale);
        s0 += w;
        s1 += w * f;
        s2 += w * f * f;
        t0 += w * g;
        t1 += w * f * g;
    }
    let mut candidates = vec![(t0 / s0, 0.0), (0.0, if s2 > 0.0 { t1 / s2 } else { 0.0 })];
    let determinant = s0 * s2 - s1 * s1;
    if determinant > 1e-14 * s0 * s2 {
        let nugget = (t0 * s2 - t1 * s1) / determinant;
        let sill = (s0 * t1 - s1 * t0) / determinant;
        if nugget >= 0.0 && sill >= 0.0 {
            candidates.push((nugget, sill));
        }
    }
    let mut best = None;
    for (nugget, sill) in candidates {
        let sse = data
            .iter()
            .map(|&(h, g, w)| w * (g - nugget - sill * (1.0 - family.evaluate(h / scale))).powi(2))
            .sum::<f64>();
        if !sse.is_finite() {
            return Err(Error::Numerical {
                context: "variogram objective overflow",
            });
        }
        let fit = Fit {
            correlation: family,
            scale,
            structured_sill: sill,
            nugget,
            weighted_sse: sse,
            at_scale_bound: false,
        };
        if best.as_ref().is_none_or(|old: &Fit| sse < old.weighted_sse) {
            best = Some(fit);
        }
    }
    Ok(best.unwrap())
}
