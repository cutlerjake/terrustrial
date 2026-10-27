//! Reproducibility contract owned by the Gaussian simulation driver.
use super::SimulationOptions;
use crate::{
    error::invalid,
    execution::Cancellation,
    gaussian::{Drift, NumericalTolerance},
    identity::{reader, CanonicalFingerprint, Encoder, Fingerprint},
    model::Covariance,
    Result,
};
/// Mandatory provenance for an identified run. The implementation digest should
/// cover the executable or complete source, lockfile and build configuration.
/// The backend digest must describe precision, CPU/GPU and relevant arithmetic
/// policy. Mean callbacks require their code AND external-data identity.
#[derive(Clone, Copy, Debug)]
pub struct RunProvenance {
    pub implementation: Fingerprint,
    pub backend: Fingerprint,
    pub mean: Option<Fingerprint>,
}
impl RunProvenance {
    pub(crate) fn validate_mean(self, present: bool) -> Result<()> {
        if present != self.mean.is_some() {
            return Err(invalid(
                "known-mean callback and explicit identity must both be present or absent",
            ));
        }
        Ok(())
    }
    /// Hash the running executable as implementation provenance. Dynamically
    /// loaded libraries, drivers and external assets are not part of that file;
    /// include their relevant identities in `backend` or a custom implementation
    /// manifest instead. This returns an error where current_exe is unavailable.
    pub fn current_executable(
        backend: Fingerprint,
        mean: Option<Fingerprint>,
        cancel: &Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        let implementation = reader(std::fs::File::open(std::env::current_exe()?)?, cancel)?;
        Ok(Self {
            implementation,
            backend,
            mean,
        })
    }
}

/// Build a checkpoint identity from structured components and precomputed input
/// digests. Callers caching digests must verify their inputs have not changed.
/// Resource budgets/cancellation are excluded because they control admission,
/// not accepted numerical output. This targets the sequential checkpoint driver.
pub fn gaussian_run(
    model: &(impl Covariance + CanonicalFingerprint),
    drift: &(impl Drift + CanonicalFingerprint),
    hard: Fingerprint,
    full_path: Fingerprint,
    options: &SimulationOptions<'_>,
    provenance: RunProvenance,
) -> Result<Fingerprint> {
    options.cancellation.check()?;
    provenance.validate_mean(options.mean.is_some())?;
    let mut out = Encoder::new(8);
    out.field(env!("CARGO_PKG_VERSION").as_bytes())?;
    out.u64(1); // Sequential checkpoint-driver contract revision.
    for id in [
        provenance.implementation,
        provenance.backend,
        hard,
        full_path,
        model.fingerprint()?,
        drift.fingerprint()?,
    ] {
        out.write(&id.to_bytes());
    }
    match provenance.mean {
        None => out.write(&[0]),
        Some(id) => {
            out.write(&[1]);
            out.write(&id.to_bytes());
        }
    }
    out.u64(options.seed);
    out.u64(options.realization);
    out.neighborhood(options.hard_neighborhood)?;
    out.neighborhood(options.simulated_neighborhood)?;
    let tolerance = NumericalTolerance::default();
    out.f64(tolerance.absolute);
    out.f64(tolerance.relative);
    options.cancellation.check()?;
    Ok(out.finish())
}

/// Domain separation binds opt-in residual checking and its caller-defined policy.
pub(super) fn checked_run(run: [u8; 32], policy: Fingerprint) -> [u8; 32] {
    let mut out = Encoder::new(9);
    out.u64(1);
    out.write(&run);
    out.write(&policy.to_bytes());
    out.finish().to_bytes()
}
