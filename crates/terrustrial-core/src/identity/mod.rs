//! Versioned, domain-separated fingerprints of exact structured inputs.
//!
//! Integer fields use little-endian u64, dimensions are checked, and f64 fields
//! retain every bit, including signed zero. Sequence order is significant. Sources
//! must remain immutable throughout fingerprinting and the operation that uses it.
//! SQLite users can retain a `SqliteReadPool::open_snapshot` pool across both.
//! Hashing keeps a fixed-size stack buffer, not a copy of the dataset or path.
//! A digest detects changes relative to a trusted digest; it does not authenticate
//! data, validate a caller's opaque identity, or protect against later mutation.
use crate::{
    data::{Group, Observation, Site, Support},
    error::invalid,
    execution::Cancellation,
    gaussian::{Drift, LinearDrift, Ordinary, Simple},
    model::{Anisotropy, Correlation, Covariance, Lmc},
    spatial::{NeighborhoodQuery, ObservationSource},
    Error, Result,
};
use std::io::Read;

mod opaque;
pub use opaque::Identified;

/// A BLAKE3-256 digest in the canonical encoding version 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint([u8; 32]);
impl Fingerprint {
    /// Import a previously persisted digest. This does not validate its provenance.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
    pub const fn to_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// A complete stable identity of the object's numerical definition and inputs.
/// Custom implementations must include all external tables, callback semantics
/// and configuration. An opaque adapter makes that responsibility explicit.
pub trait CanonicalFingerprint {
    fn fingerprint(&self) -> Result<Fingerprint>;
}

pub(crate) struct Encoder {
    hasher: blake3::Hasher,
    buffer: [u8; 4096],
    used: usize,
}
impl Encoder {
    pub(crate) fn new(kind: u8) -> Self {
        let mut out = Self {
            hasher: blake3::Hasher::new_derive_key("terrustrial canonical identity v1"),
            buffer: [0; 4096],
            used: 0,
        };
        out.write(&[kind]);
        out
    }
    pub(crate) fn write(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            if self.used == 0 && bytes.len() >= self.buffer.len() {
                self.hasher.update(bytes);
                return;
            }
            let n = bytes.len().min(self.buffer.len() - self.used);
            self.buffer[self.used..self.used + n].copy_from_slice(&bytes[..n]);
            self.used += n;
            bytes = &bytes[n..];
            if self.used == self.buffer.len() {
                self.hasher.update(&self.buffer);
                self.used = 0;
            }
        }
    }
    pub(crate) fn u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }
    pub(crate) fn usize(&mut self, value: usize) -> Result<()> {
        self.u64(value.try_into().map_err(|_| Error::SizeOverflow)?);
        Ok(())
    }
    pub(crate) fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }
    pub(crate) fn field(&mut self, bytes: &[u8]) -> Result<()> {
        self.usize(bytes.len())?;
        self.write(bytes);
        Ok(())
    }
    pub(crate) fn finish(mut self) -> Fingerprint {
        self.hasher.update(&self.buffer[..self.used]);
        Fingerprint(*self.hasher.finalize().as_bytes())
    }
    fn support(&mut self, support: &Support) -> Result<()> {
        for x in support
            .center()
            .coordinates()
            .into_iter()
            .chain(support.half_extents())
        {
            self.f64(x);
        }
        for n in support.divisions() {
            self.usize(n)?;
        }
        Ok(())
    }
    fn site(&mut self, site: &Site) -> Result<()> {
        self.support(&site.support)?;
        self.usize(site.variable)
    }
    fn observation(&mut self, o: &Observation) -> Result<()> {
        self.u64(o.id());
        self.site(o.site())?;
        self.f64(o.value());
        self.f64(o.noise_variance());
        Ok(())
    }
    fn geometry(&mut self, geometry: &Anisotropy) {
        for x in geometry
            .frame()
            .into_iter()
            .flatten()
            .chain(geometry.scales())
        {
            self.f64(x);
        }
    }
    pub(crate) fn neighborhood(&mut self, query: NeighborhoodQuery<'_>) -> Result<()> {
        query.validate()?;
        match query {
            NeighborhoodQuery::Spherical(n) => {
                self.write(&[0]);
                self.usize(n.maximum)?;
                self.f64(n.radius);
            }
            NeighborhoodQuery::Constrained(n) => {
                self.write(&[if n.expansion_scales().len() != 0 {
                    3
                } else if n.minimums().is_some() {
                    2
                } else {
                    1
                }]);
                self.usize(n.maximum())?;
                self.geometry(n.geometry());
                match n.octant_maximum() {
                    None => self.write(&[0]),
                    Some(maximum) => {
                        self.write(&[1]);
                        self.usize(maximum)?;
                    }
                }
                self.usize(n.variable_limits().len())?;
                for limit in n.variable_limits() {
                    self.usize(limit.variable)?;
                    self.usize(limit.maximum)?;
                }
                if let Some(minimums) = n.minimums() {
                    self.usize(minimums.total)?;
                    self.usize(minimums.variables.len())?;
                    for minimum in &minimums.variables {
                        self.usize(minimum.variable)?;
                        self.usize(minimum.minimum)?;
                    }
                    for minimum in minimums.octants {
                        self.usize(minimum)?;
                    }
                }
                if n.expansion_scales().len() != 0 {
                    self.usize(n.expansion_scales().len())?;
                    for scales in n.expansion_scales() {
                        for radius in scales {
                            self.f64(radius);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Explicit identity of opaque code/data. Use a versioned namespace and complete
/// immutable definition or content digest; a function address is not an identity.
pub fn opaque(namespace: &str, definition: &[u8]) -> Result<Fingerprint> {
    if namespace.is_empty() || definition.is_empty() {
        return Err(invalid(
            "opaque identities require a namespace and nonempty definition",
        ));
    }
    let mut out = Encoder::new(0);
    out.field(namespace.as_bytes())?;
    out.field(definition)?;
    Ok(out.finish())
}

/// Fingerprint a byte stream, independent of read chunk boundaries. I/O failures
/// and cancellation return no digest. The reader's blocking I/O is not interrupted.
pub fn reader(mut input: impl Read, cancel: &Cancellation) -> Result<Fingerprint> {
    let mut out = Encoder::new(1);
    let mut buffer = [0; 16384];
    let mut length = 0_u64;
    loop {
        cancel.check()?;
        let n = match input.read(&mut buffer) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if n == 0 {
            break;
        }
        length = length.checked_add(n as u64).ok_or(Error::SizeOverflow)?;
        out.write(&buffer[..n]);
    }
    cancel.check()?;
    out.u64(length);
    Ok(out.finish())
}

/// Fingerprint observations in scan order, including IDs, support, variable,
/// value and noise. A different scan order conservatively yields a different
/// identity, even if it represents the same set. No external sorting is hidden.
pub fn observations(
    source: &(impl ObservationSource + ?Sized),
    cancel: &Cancellation,
) -> Result<Fingerprint> {
    cancel.check()?;
    let mut out = Encoder::new(2);
    let mut count = 0_u64;
    source.scan(&mut |o| {
        cancel.check()?;
        count = count.checked_add(1).ok_or(Error::SizeOverflow)?;
        out.write(&[1]);
        out.observation(o)
    })?;
    cancel.check()?;
    out.write(&[0]);
    out.u64(count);
    Ok(out.finish())
}

/// Fingerprint the full ordered path, including group boundaries and target IDs.
/// Groups must be nonempty with strictly increasing group IDs. Hashing does not
/// retain target IDs to check uniqueness across the entire path.
pub fn path(groups: impl IntoIterator<Item = Group>, cancel: &Cancellation) -> Result<Fingerprint> {
    try_path(groups.into_iter().map(Ok), cancel)
}

/// Fallible streaming-path variant. A read error returns no digest, including
/// when it occurs after earlier groups were successfully read and hashed.
pub fn try_path(
    groups: impl IntoIterator<Item = Result<Group>>,
    cancel: &Cancellation,
) -> Result<Fingerprint> {
    cancel.check()?;
    let mut out = Encoder::new(3);
    let mut previous = None;
    let mut count = 0_u64;
    for group in groups {
        cancel.check()?;
        let group = group?;
        if group.targets.is_empty() || previous.is_some_and(|id| group.id <= id) {
            return Err(invalid(
                "fingerprint path needs nonempty groups with increasing IDs",
            ));
        }
        previous = Some(group.id);
        count = count.checked_add(1).ok_or(Error::SizeOverflow)?;
        out.write(&[1]);
        out.u64(group.id);
        out.usize(group.targets.len())?;
        for target in group.targets {
            cancel.check()?;
            out.u64(target.id);
            out.site(&target.site)?;
        }
    }
    cancel.check()?;
    out.write(&[0]);
    out.u64(count);
    Ok(out.finish())
}

impl CanonicalFingerprint for Lmc {
    fn fingerprint(&self) -> Result<Fingerprint> {
        let mut out = Encoder::new(4);
        out.usize(self.variables())?;
        out.usize(self.structures().len())?;
        for s in self.structures() {
            out.write(&[match s.correlation {
                Correlation::Nugget => 0,
                Correlation::Spherical => 1,
                Correlation::Exponential => 2,
                Correlation::Gaussian => 3,
                Correlation::Matern32 => 4,
                Correlation::Matern52 => 5,
            }]);
            out.geometry(&s.anisotropy);
            for i in 0..self.variables() {
                for j in 0..self.variables() {
                    out.f64(s.coregionalization.covariance(i, j)?);
                }
            }
        }
        Ok(out.finish())
    }
}
impl CanonicalFingerprint for Simple {
    fn fingerprint(&self) -> Result<Fingerprint> {
        Ok(Encoder::new(5).finish())
    }
}
impl CanonicalFingerprint for Ordinary {
    fn fingerprint(&self) -> Result<Fingerprint> {
        let mut out = Encoder::new(6);
        out.usize(self.variables)?;
        Ok(out.finish())
    }
}
impl CanonicalFingerprint for LinearDrift {
    fn fingerprint(&self) -> Result<Fingerprint> {
        let mut out = Encoder::new(7);
        out.usize(self.variables())?;
        for x in self.origin() {
            out.f64(x);
        }
        out.f64(self.scale());
        Ok(out.finish())
    }
}
