//! Shared validation and budgeted observation selection for Gaussian drivers.
use crate::{
    data::{Group, Point3},
    error::invalid,
    execution::Cancellation,
    Result,
};
mod buffers;
mod canonical;
pub(crate) use buffers::Selection;
#[cfg(test)]
pub(crate) use canonical::scalar as canonicalize_observations;
pub(crate) use canonical::CanonicalWorkspace;

pub(crate) fn group_center(group: &Group, cancel: &Cancellation) -> Result<Point3> {
    cancel.check()?;
    if group.targets.is_empty() {
        return Err(invalid("empty Gaussian group"));
    }
    for i in 0..group.targets.len() {
        for j in 0..i {
            if group.targets[i].id == group.targets[j].id
                || group.targets[i].site == group.targets[j].site
            {
                return Err(invalid("duplicate target ID or site within Gaussian group"));
            }
        }
    }
    let count = group.targets.len() as f64;
    let mut coordinates = [0.0; 3];
    for target in &group.targets {
        for (axis, x) in coordinates.iter_mut().enumerate() {
            *x += target.site.support.center().coordinates()[axis] / count;
        }
    }
    Point3::new(coordinates)
}
