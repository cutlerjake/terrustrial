//! Spatial scoring and fail-closed geological acceptance. A completed numerical
//! fit is not evidence of geological validity.
use crate::{error::invalid, Result};
pub mod trace;

/// Length-weighted errors. Unknown evidence must not be added to either class.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntervalScore {
    pub interior_length: f64,
    pub exterior_length: f64,
    pub interior_error: f64,
    pub exterior_error: f64,
}
impl IntervalScore {
    pub fn add(&mut self, inside: bool, value: f64, length: f64) -> Result<()> {
        if !value.is_finite() || !length.is_finite() || length <= 0. {
            return Err(invalid("validation value or interval weight"));
        }
        let wrong = value == 0. || (value < 0.) != inside;
        let (total, errors) = if inside {
            (&mut self.interior_length, &mut self.interior_error)
        } else {
            (&mut self.exterior_length, &mut self.exterior_error)
        };
        if !(*total + length).is_finite() {
            return Err(invalid("validation weight overflow"));
        }
        *total += length;
        if wrong {
            *errors += length;
        }
        Ok(())
    }
    /// None means there is insufficient class coverage to report balanced error.
    pub fn balanced_error(&self) -> Option<f64> {
        (self.interior_length > 0. && self.exterior_length > 0.).then(|| {
            0.5 * (self.interior_error / self.interior_length
                + self.exterior_error / self.exterior_length)
        })
    }
}

/// Contiguous spatial blocks along the widest coordinate axis. Each supplied
/// representative belongs to one whole hole, so a hole cannot leak across folds.
/// Equal projected coordinates are kept together. Input order breaks no ties.
pub fn spatial_blocks(points: &[[f64; 3]], count: usize) -> Result<Vec<usize>> {
    if count < 2 || points.len() < count || points.iter().flatten().any(|x| !x.is_finite()) {
        return Err(invalid("spatial validation blocks"));
    }
    let lo: [f64; 3] =
        std::array::from_fn(|a| points.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min));
    let hi: [f64; 3] = std::array::from_fn(|a| {
        points
            .iter()
            .map(|p| p[a])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap();
    if !(hi[axis] - lo[axis]).is_finite() {
        return Err(invalid("spatial block extent overflow"));
    }
    let mut order: Vec<_> = (0..points.len()).collect();
    order.sort_unstable_by(|&a, &b| points[a][axis].total_cmp(&points[b][axis]));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in order {
        if groups
            .last()
            .is_none_or(|g| points[g[0]][axis] != points[i][axis])
        {
            groups.push(Vec::new());
        }
        groups.last_mut().unwrap().push(i);
    }
    if groups.len() < count {
        return Err(invalid(
            "insufficient distinct locations for spatial blocks",
        ));
    }
    let mut result = vec![0; points.len()];
    let n = groups.len();
    for (rank, group) in groups.iter().enumerate() {
        for &i in group {
            result[i] = rank * count / n;
        }
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug)]
pub struct AcceptanceLimits {
    pub maximum_contact_distance_m: f64,
    pub maximum_held_out_balanced_error: f64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptanceEvidence {
    pub numerical_complete: bool,
    pub geometry_validated: bool,
    pub boundaries_validated: bool,
    pub intervals_validated: bool,
    pub independent_spatial_validation: bool,
    pub maximum_contact_distance_m: Option<f64>,
    pub held_out_balanced_error: Option<f64>,
    pub unresolved_checks: usize,
}
/// A capability created only after all mandatory evidence passes. Downstream
/// production APIs can require this rather than trusting a manifest boolean.
#[derive(Debug)]
pub struct GeologicalAcceptance {
    _private: (),
}
impl AcceptanceEvidence {
    pub fn accept(&self, limits: AcceptanceLimits) -> Result<GeologicalAcceptance> {
        if !limits.maximum_contact_distance_m.is_finite()
            || limits.maximum_contact_distance_m < 0.
            || !limits.maximum_held_out_balanced_error.is_finite()
            || !(0. ..=1.).contains(&limits.maximum_held_out_balanced_error)
        {
            return Err(invalid("geological acceptance limits"));
        }
        if !self.numerical_complete
            || !self.geometry_validated
            || !self.boundaries_validated
            || !self.intervals_validated
            || !self.independent_spatial_validation
            || self.unresolved_checks != 0
            || !self
                .maximum_contact_distance_m
                .is_some_and(|x| x.is_finite() && x >= 0. && x <= limits.maximum_contact_distance_m)
            || !self.held_out_balanced_error.is_some_and(|x| {
                x.is_finite() && (0. ..=limits.maximum_held_out_balanced_error).contains(&x)
            })
        {
            return Err(invalid("geological acceptance is incomplete or failed"));
        }
        Ok(GeologicalAcceptance { _private: () })
    }
}
