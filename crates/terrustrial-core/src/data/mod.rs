use crate::{
    error::{checked_len, invalid},
    Result,
};

/// Fast ordinary-coordinate norm with scaled fallback for extreme magnitudes.
pub(crate) fn norm3(delta: [f64; 3]) -> f64 {
    let squared = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    if squared.is_normal() || delta == [0.0; 3] {
        squared.sqrt()
    } else {
        delta[0].hypot(delta[1]).hypot(delta[2])
    }
}

/// Finite Cartesian coordinates. The caller chooses and documents the units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3([f64; 3]);
impl Point3 {
    pub fn new(coordinates: [f64; 3]) -> Result<Self> {
        if coordinates.iter().any(|x| !x.is_finite()) {
            return Err(invalid("nonfinite coordinate"));
        }
        Ok(Self(coordinates))
    }
    pub fn coordinates(self) -> [f64; 3] {
        self.0
    }
    pub fn distance(self, other: Self) -> f64 {
        norm3([
            self.0[0] - other.0[0],
            self.0[1] - other.0[1],
            self.0[2] - other.0[2],
        ])
    }
}

/// Point support or midpoint quadrature of an axis-aligned volume average.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Support {
    center: Point3,
    half_extents: [f64; 3],
    divisions: [usize; 3],
    count: usize,
}
impl Support {
    pub fn point(point: Point3) -> Self {
        Self {
            center: point,
            half_extents: [0.0; 3],
            divisions: [1; 3],
            count: 1,
        }
    }
    pub fn block(center: Point3, half_extents: [f64; 3], divisions: [usize; 3]) -> Result<Self> {
        for axis in 0..3 {
            let h = half_extents[axis];
            if !h.is_finite()
                || h < 0.0
                || divisions[axis] == 0
                || (h == 0.0 && divisions[axis] != 1)
            {
                return Err(invalid("block extents must be finite and nonnegative; quadrature divisions positive (one on a collapsed axis)"));
            }
            if !(center.0[axis] - h).is_finite() || !(center.0[axis] + h).is_finite() {
                return Err(invalid("block bounds overflow"));
            }
        }
        let count = checked_len(&divisions)?;
        Ok(Self {
            center,
            half_extents,
            divisions,
            count,
        })
    }
    pub fn center(&self) -> Point3 {
        self.center
    }
    pub fn half_extents(&self) -> [f64; 3] {
        self.half_extents
    }
    pub fn divisions(&self) -> [usize; 3] {
        self.divisions
    }
    pub fn is_point(&self) -> bool {
        self.half_extents == [0.0; 3]
    }
    pub fn node_count(&self) -> usize {
        self.count
    }
    pub fn nodes(&self) -> impl ExactSizeIterator<Item = Point3> + '_ {
        (0..self.count).map(|mut index| {
            let mut point = self.center.0;
            for (axis, value) in point.iter_mut().enumerate() {
                let digit = index % self.divisions[axis];
                index /= self.divisions[axis];
                *value += self.half_extents[axis]
                    * (2.0 * ((digit as f64 + 0.5) / self.divisions[axis] as f64) - 1.0);
            }
            Point3(point)
        })
    }
}

/// A single variable on a spatial support, usable as an observation or a target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Site {
    pub support: Support,
    pub variable: usize,
}
impl Site {
    // Ordering consistent with geometric PartialEq, including signed-zero
    // equality. All components are finite and quadrature count is derived from
    // divisions, so no NaN or independently mutable derived fields occur here.
    pub(crate) fn geometry_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.variable
            .cmp(&other.variable)
            .then_with(|| {
                self.support
                    .center
                    .0
                    .iter()
                    .chain(&self.support.half_extents)
                    .zip(
                        other
                            .support
                            .center
                            .0
                            .iter()
                            .chain(&other.support.half_extents),
                    )
                    .map(|(a, b)| a.partial_cmp(b).expect("finite support geometry"))
                    .find(|order| !order.is_eq())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| self.support.divisions.cmp(&other.support.divisions))
    }
    pub fn point(point: Point3, variable: usize) -> Self {
        Self {
            support: Support::point(point),
            variable,
        }
    }
}

/// One observed attribute. Missing attributes are omitted, never encoded as NaN.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    id: u64,
    site: Site,
    value: f64,
    noise_variance: f64,
}
impl Observation {
    // Cache identity includes all publicly observable fields and preserves signed
    // zero. Custom covariance bulk callbacks can inspect values and IDs as well
    // as sites, so comparing only geometry or observation IDs is insufficient.
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.id == other.id
            && self.value.to_bits() == other.value.to_bits()
            && self.noise_variance.to_bits() == other.noise_variance.to_bits()
            && self.site.variable == other.site.variable
            && self.site.support.divisions == other.site.support.divisions
            && self.site.support.center.0.map(f64::to_bits)
                == other.site.support.center.0.map(f64::to_bits)
            && self.site.support.half_extents.map(f64::to_bits)
                == other.site.support.half_extents.map(f64::to_bits)
    }
    pub fn new(id: u64, site: Site, value: f64, noise_variance: f64) -> Result<Self> {
        if !value.is_finite() || !noise_variance.is_finite() || noise_variance < 0.0 {
            return Err(invalid(
                "observation value must be finite and noise variance finite and nonnegative",
            ));
        }
        Ok(Self {
            id,
            site,
            value,
            noise_variance,
        })
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn site(&self) -> &Site {
        &self.site
    }
    pub fn value(&self) -> f64 {
        self.value
    }
    pub fn noise_variance(&self) -> f64 {
        self.noise_variance
    }
}

/// A requested support and variable, with a stable caller-assigned identifier.
#[derive(Clone, Debug)]
pub struct Target {
    pub id: u64,
    pub site: Site,
}
/// Targets predicted or simulated jointly, in their supplied order.
#[derive(Clone, Debug)]
pub struct Group {
    pub id: u64,
    pub targets: Vec<Target>,
}
