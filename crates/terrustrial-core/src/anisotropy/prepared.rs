//! Immutable, budgeted local tensors for a finite conditioning/target point set.
use super::{AnchorField, Tensor, TensorField};
use crate::{
    data::Point3,
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    Error, Result,
};

/// Tensor interpolation performed once per distinct point, reusable across
/// variables, covariance matrix entries and concurrent Gaussian preparations.
/// Unknown points return an error, preventing silent interpolation-policy changes.
/// Include every quadrature node when preparing block-supported observations.
/// The cache is bound to the immutable field used to construct it; it is a
/// snapshot, not a live view. Keys normalize signed zero as Point3 equality does.
pub struct PreparedField {
    entries: Vec<(Point3, Tensor)>,
    _reservation: Reservation,
}
fn compare(a: Point3, b: Point3) -> std::cmp::Ordering {
    a.coordinates()
        .iter()
        .zip(b.coordinates())
        .map(|(x, y)| x.partial_cmp(&y).unwrap())
        .find(|o| !o.is_eq())
        .unwrap_or(std::cmp::Ordering::Equal)
}
impl PreparedField {
    pub fn new(
        field: &AnchorField,
        points: &[Point3],
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        let bytes = checked_len(&[
            points.len(),
            std::mem::size_of::<(Point3, Tensor)>() + std::mem::size_of::<Point3>(),
        ])?
        .checked_add(checked_len(&[
            field.neighbors,
            2,
            std::mem::size_of::<super::AnchorDistance>(),
        ])?)
        .ok_or(Error::SizeOverflow)?;
        let mut reservation =
            budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let mut sorted = Vec::new();
        sorted
            .try_reserve_exact(points.len())
            .map_err(|_| invalid("unable to allocate prepared tensor points"))?;
        sorted.extend_from_slice(points);
        sorted.sort_unstable_by(|a, b| compare(*a, *b));
        sorted.dedup();
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(sorted.len())
            .map_err(|_| invalid("unable to allocate prepared tensors"))?;
        let mut workspace = field.workspace()?;
        for point in sorted {
            cancel.check()?;
            entries.push((point, field.at_with_workspace(point, &mut workspace)?));
        }
        drop(workspace);
        reservation
            .resize(
                checked_len(&[entries.capacity(), std::mem::size_of::<(Point3, Tensor)>()])? as u64,
            )?;
        Ok(Self {
            entries,
            _reservation: reservation,
        })
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn accounted_bytes(&self) -> u64 {
        self._reservation.bytes()
    }
}
impl TensorField for PreparedField {
    fn at(&self, point: Point3) -> Result<Tensor> {
        self.entries
            .binary_search_by(|(p, _)| compare(*p, point))
            .map(|i| self.entries[i].1.clone())
            .map_err(|_| invalid("point absent from prepared anisotropy field"))
    }
}
