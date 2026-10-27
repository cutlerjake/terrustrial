use super::*;
use crate::{data::Point3, gaussian::KnownMean};

/// Attach a caller-owned identity to custom covariance, drift or mean code.
/// Optimized bulk covariance overrides are forwarded, not replaced by defaults.
/// The caller must include every captured value, external table and semantic
/// code revision in the definition, and keep the wrapped object immutable.
pub struct Identified<T> {
    value: T,
    identity: Fingerprint,
}
impl<T> Identified<T> {
    pub fn new(value: T, namespace: &str, definition: &[u8]) -> Result<Self> {
        Ok(Self {
            value,
            identity: opaque(namespace, definition)?,
        })
    }
    pub fn get_ref(&self) -> &T {
        &self.value
    }
}
impl<T> CanonicalFingerprint for Identified<T> {
    fn fingerprint(&self) -> Result<Fingerprint> {
        Ok(self.identity)
    }
}
impl<T: Covariance> Covariance for Identified<T> {
    fn variables(&self) -> usize {
        self.value.variables()
    }
    fn point_covariance(
        &self,
        a: crate::data::Point3,
        va: usize,
        b: Point3,
        vb: usize,
    ) -> Result<f64> {
        self.value.point_covariance(a, va, b, vb)
    }
    fn covariance(&self, a: &Site, b: &Site) -> Result<f64> {
        self.value.covariance(a, b)
    }
    fn covariance_row(
        &self,
        a: &Site,
        observations: &[Observation],
        output: &mut [f64],
    ) -> Result<()> {
        self.value.covariance_row(a, observations, output)
    }
    fn covariance_column(
        &self,
        observations: &[Observation],
        b: &Site,
        output: &mut [f64],
    ) -> Result<()> {
        self.value.covariance_column(observations, b, output)
    }
}
impl<T: Drift> Drift for Identified<T> {
    fn columns(&self) -> usize {
        self.value.columns()
    }
    fn evaluate(&self, site: &Site, values: &mut [f64]) -> Result<()> {
        self.value.evaluate(site, values)
    }
}
impl<T: KnownMean> KnownMean for Identified<T> {
    fn evaluate(&self, site: &Site) -> Result<f64> {
        self.value.evaluate(site)
    }
}
