//! Shared midpoint-probability quadrature for back-transformed Gaussian moments.
use crate::{
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    Error, Result,
};
use statrs::distribution::{ContinuousCDF, Normal};

/// Immutable standard-normal abscissae. Share one instance between transforms
/// and workers. This caches the existing approximate quadrature; it does not
/// change its accuracy or the transform's tail behavior.
#[derive(Debug)]
pub struct MomentQuadrature {
    pub(super) quantiles: Vec<f64>,
    _reservation: Reservation,
}
impl MomentQuadrature {
    pub fn new(points: usize, budget: &MemoryBudget, cancel: &Cancellation) -> Result<Self> {
        cancel.check()?;
        if points == 0 {
            return Err(invalid("Gaussian moments or quadrature count"));
        }
        let bytes = checked_len(&[points, std::mem::size_of::<f64>()])?;
        let mut reservation =
            budget.reserve(u64::try_from(bytes).map_err(|_| Error::SizeOverflow)?)?;
        let mut quantiles = Vec::new();
        quantiles
            .try_reserve_exact(points)
            .map_err(|_| invalid("unable to allocate moment quadrature"))?;
        reservation
            .resize(checked_len(&[quantiles.capacity(), std::mem::size_of::<f64>()])? as u64)?;
        let normal = Normal::new(0.0, 1.0).map_err(|e| invalid(e.to_string()))?;
        for i in 0..points {
            cancel.check()?;
            quantiles.push(quantile(&normal, i, points)?);
        }
        Ok(Self {
            quantiles,
            _reservation: reservation,
        })
    }
    pub fn points(&self) -> usize {
        self.quantiles.len()
    }
    pub fn accounted_bytes(&self) -> u64 {
        self._reservation.bytes()
    }
}
pub(super) fn quantile(normal: &Normal, i: usize, points: usize) -> Result<f64> {
    let probability = (i as f64 + 0.5) / points as f64;
    if probability <= 0.0 || probability >= 1.0 {
        return Err(invalid("quadrature probability precision"));
    }
    Ok(normal.inverse_cdf(probability))
}
