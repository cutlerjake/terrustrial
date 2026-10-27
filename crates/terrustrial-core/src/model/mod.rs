pub mod quadrature;
mod simd;
use crate::{
    data::{Observation, Point3, Site},
    error::{checked_len, invalid},
    Error, Result,
};

/// Normalized radial correlations. Range is a scale, not a shared practical range.
#[derive(Clone, Copy, Debug)]
pub enum Correlation {
    /// Independent point-scale variation; block behavior follows explicit quadrature.
    Nugget,
    Spherical,
    Exponential,
    Gaussian,
    Matern32,
    Matern52,
}
impl Correlation {
    pub fn evaluate(self, h: f64) -> f64 {
        match self {
            Self::Nugget => {
                if h == 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Spherical => {
                if h >= 1.0 {
                    0.0
                } else {
                    1.0 - 1.5 * h + 0.5 * h * h * h
                }
            }
            Self::Exponential => (-h).exp(),
            Self::Gaussian => (-(h * h)).exp(),
            Self::Matern32 => {
                let x = 3.0_f64.sqrt() * h;
                if x.is_infinite() {
                    0.0
                } else {
                    (1.0 + x) * (-x).exp()
                }
            }
            Self::Matern52 => {
                let x = 5.0_f64.sqrt() * h;
                if x > 700.0 {
                    0.0
                } else {
                    (1.0 + x + x * x / 3.0) * (-x).exp()
                }
            }
        }
    }
}

/// Rows of the orthonormal frame are principal directions; scales use coordinate units.
#[derive(Clone, Debug)]
pub struct Anisotropy {
    frame: [[f64; 3]; 3],
    scales: [f64; 3],
    isotropic: bool,
}
impl Anisotropy {
    pub fn isotropic(scale: f64) -> Result<Self> {
        Self::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [scale; 3],
        )
    }
    pub fn new(frame: [[f64; 3]; 3], scales: [f64; 3]) -> Result<Self> {
        if scales
            .iter()
            .any(|r| !r.is_finite() || *r <= 0.0 || !(1.0 / r).is_finite())
            || frame.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(invalid(
                "finite orthonormal frame and positive representable scales required",
            ));
        }
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| frame[i][k] * frame[j][k]).sum();
                if (dot - if i == j { 1.0 } else { 0.0 }).abs() > 1e-10 {
                    return Err(invalid("anisotropy frame is not orthonormal"));
                }
            }
        }
        Ok(Self {
            frame,
            scales,
            isotropic: scales[0] == scales[1] && scales[1] == scales[2],
        })
    }
    pub fn normalized_distance(&self, a: Point3, b: Point3) -> f64 {
        let a = a.coordinates();
        let b = b.coordinates();
        let delta = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        if self.isotropic {
            // Rotations preserve isotropic distance. The common path needs no
            // matrix-vector product. Scale first only for extreme coordinates.
            let length = crate::data::norm3(delta);
            if length.is_finite() {
                return length / self.scales[0];
            }
            return crate::data::norm3(std::array::from_fn(|i| {
                if delta[i].is_finite() {
                    delta[i] / self.scales[0]
                } else {
                    a[i] / self.scales[0] - b[i] / self.scales[0]
                }
            }));
        }
        crate::data::norm3(self.projected_displacement(a, b).0)
    }
    /// Normalized principal-frame displacement and its negative-coordinate mask.
    /// Search uses the same overflow handling as covariance, including when an
    /// unscaled subtraction or projection overflows but the scaled result fits.
    pub(crate) fn projected_displacement(&self, a: [f64; 3], b: [f64; 3]) -> ([f64; 3], usize) {
        let delta: [f64; 3] = std::array::from_fn(|i| a[i] - b[i]);
        let mut negative = 0;
        let normalized = std::array::from_fn(|i| {
            let row = self.frame[i];
            let projected: f64 = (0..3).map(|j| row[j] * delta[j]).sum();
            let (value, sign) = if projected.is_finite() {
                (projected / self.scales[i], projected)
            } else {
                self.scaled_projection(row, delta, a, b, self.scales[i])
            };
            negative |= usize::from(sign < 0.0) << i;
            value
        });
        (normalized, negative)
    }
    #[cold]
    fn scaled_projection(
        &self,
        row: [f64; 3],
        delta: [f64; 3],
        a: [f64; 3],
        b: [f64; 3],
        radius: f64,
    ) -> (f64, f64) {
        // Ignore exactly zero coefficients before multiplying: 0 * an
        // overflowing difference must not contaminate an unrelated direction.
        let direct: f64 = (0..3)
            .filter(|&j| row[j] != 0.0)
            .map(|j| row[j] * delta[j])
            .sum();
        if direct.is_finite() {
            return (direct / radius, direct);
        }
        // Scale only coordinates contributing to this row. A large coordinate
        // in another direction must not erase a small representable component.
        let magnitude = (0..3)
            .filter(|&j| row[j] != 0.0)
            .map(|j| {
                if delta[j].is_finite() {
                    delta[j].abs()
                } else {
                    a[j].abs().max(b[j].abs())
                }
            })
            .fold(0.0, f64::max);
        let unit: f64 = (0..3)
            .filter(|&j| row[j] != 0.0)
            .map(|j| {
                row[j]
                    * if delta[j].is_finite() {
                        delta[j] / magnitude
                    } else {
                        a[j] / magnitude - b[j] / magnitude
                    }
            })
            .sum();
        let projected = unit * magnitude;
        let normalized = if projected.is_finite() {
            projected / radius
        } else {
            (unit / radius) * magnitude
        };
        (normalized, unit)
    }
    pub fn scales(&self) -> [f64; 3] {
        self.scales
    }
    pub fn frame(&self) -> [[f64; 3]; 3] {
        self.frame
    }
}

/// B = A A^T, allowing rank-deficient coregionalization without invalid cross-sills.
#[derive(Clone, Debug)]
pub struct Coregionalization {
    variables: usize,
    matrix: Vec<f64>,
}
impl Coregionalization {
    pub fn from_factor(variables: usize, rank: usize, factor: &[f64]) -> Result<Self> {
        if variables == 0
            || rank == 0
            || factor.len() != checked_len(&[variables, rank])?
            || factor.iter().any(|x| !x.is_finite())
        {
            return Err(invalid("coregionalization factor dimensions or values"));
        }
        let mut matrix = vec![0.0; checked_len(&[variables, variables])?];
        for i in 0..variables {
            for j in 0..variables {
                let value = (0..rank)
                    .map(|k| factor[i * rank + k] * factor[j * rank + k])
                    .sum::<f64>();
                if !value.is_finite() {
                    return Err(invalid("coregionalization product overflow"));
                }
                matrix[i * variables + j] = value;
            }
        }
        Ok(Self { variables, matrix })
    }
    pub fn scalar(variance: f64) -> Result<Self> {
        if !variance.is_finite() || variance < 0.0 {
            return Err(invalid("nonnegative finite variance required"));
        }
        Self::from_factor(1, 1, &[variance.sqrt()])
    }
    pub fn variables(&self) -> usize {
        self.variables
    }
    pub fn covariance(&self, a: usize, b: usize) -> Result<f64> {
        if a >= self.variables || b >= self.variables {
            return Err(invalid("variable index out of range"));
        }
        Ok(self.matrix[a * self.variables + b])
    }
}

#[derive(Clone, Debug)]
pub struct Structure {
    pub correlation: Correlation,
    pub anisotropy: Anisotropy,
    pub coregionalization: Coregionalization,
}

pub trait Covariance: Sync {
    fn variables(&self) -> usize;
    fn point_covariance(
        &self,
        a: Point3,
        variable_a: usize,
        b: Point3,
        variable_b: usize,
    ) -> Result<f64>;
    /// Fill several covariance rows without allocation. Row `i` contains
    /// `cov(sites[i], observations[j].site())` for its observation prefix.
    /// Ragged prefixes permit triangular assembly without extra evaluations.
    /// On error, rows may be partially filled.
    fn covariance_rows(
        &self,
        sites: &[Site],
        observations: &[Observation],
        rows: &mut [&mut [f64]],
    ) -> Result<()> {
        if sites.len() != rows.len() || rows.iter().any(|row| row.len() > observations.len()) {
            return Err(invalid("covariance rows dimensions"));
        }
        for (site, row) in sites.iter().zip(rows) {
            self.covariance_row(site, &observations[..row.len()], row)?;
        }
        Ok(())
    }
    /// Column counterpart of `covariance_rows`, retaining observation-first
    /// quadrature order. Each column selects a prefix of observations.
    fn covariance_columns(
        &self,
        observations: &[Observation],
        sites: &[Site],
        columns: &mut [&mut [f64]],
    ) -> Result<()> {
        if sites.len() != columns.len()
            || columns
                .iter()
                .any(|column| column.len() > observations.len())
        {
            return Err(invalid("covariance columns dimensions"));
        }
        for (site, column) in sites.iter().zip(columns) {
            self.covariance_column(&observations[..column.len()], site, column)?;
        }
        Ok(())
    }
    /// Fill `cov(a, sites[j])` without allocating, preserving the scalar
    /// orientation and support quadrature. Output must match the site count.
    /// On error, output is partial.
    fn covariance_sites_row(&self, a: &Site, sites: &[Site], output: &mut [f64]) -> Result<()> {
        if sites.len() != output.len() {
            return Err(invalid("covariance sites row length"));
        }
        for (site, value) in sites.iter().zip(output) {
            *value = self.covariance(a, site)?;
        }
        Ok(())
    }
    /// Fill `cov(a, observations[j].site())` without allocating. Optimized models
    /// may evaluate lanes concurrently, preserving support quadrature semantics.
    /// Output length must exactly match observations. On error, output is partial.
    fn covariance_row(
        &self,
        a: &Site,
        observations: &[Observation],
        output: &mut [f64],
    ) -> Result<()> {
        if observations.len() != output.len() {
            return Err(invalid("covariance row length"));
        }
        for (observation, value) in observations.iter().zip(output) {
            *value = self.covariance(a, observation.site())?;
        }
        Ok(())
    }
    /// Fill `cov(observations[i].site(), b)` into a contiguous matrix column.
    /// Keep this orientation even for symmetric models: reversing block
    /// quadrature changes floating-point reduction order. No allocation.
    /// Output length must exactly match observations. On error, output is partial.
    fn covariance_column(
        &self,
        observations: &[Observation],
        b: &Site,
        output: &mut [f64],
    ) -> Result<()> {
        if observations.len() != output.len() {
            return Err(invalid("covariance column length"));
        }
        for (observation, value) in observations.iter().zip(output) {
            *value = self.covariance(observation.site(), b)?;
        }
        Ok(())
    }
    fn covariance(&self, a: &Site, b: &Site) -> Result<f64> {
        // A single midpoint node is the center, including one-node block quadrature.
        if a.support.node_count() == 1 && b.support.node_count() == 1 {
            return self.point_covariance(
                a.support.center(),
                a.variable,
                b.support.center(),
                b.variable,
            );
        }
        checked_len(&[a.support.node_count(), b.support.node_count()])?;
        let weight = 1.0 / (a.support.node_count() as f64) / (b.support.node_count() as f64);
        let mut sum = 0.0;
        for p in a.support.nodes() {
            for q in b.support.nodes() {
                sum += self.point_covariance(p, a.variable, q, b.variable)? * weight;
            }
        }
        if !sum.is_finite() {
            return Err(Error::Numerical {
                context: "support covariance",
            });
        }
        Ok(sum)
    }
}

/// Stationary linear model of coregionalization. Measurement noise is on observations.
#[derive(Clone, Debug)]
pub struct Lmc {
    simd: fearless_simd::Level,
    variables: usize,
    structures: Vec<Structure>,
}
impl Lmc {
    pub fn new(structures: Vec<Structure>) -> Result<Self> {
        let variables = structures
            .first()
            .ok_or_else(|| invalid("at least one covariance structure required"))?
            .coregionalization
            .variables;
        if structures
            .iter()
            .any(|s| s.coregionalization.variables != variables)
        {
            return Err(invalid("inconsistent variable counts"));
        }
        Ok(Self {
            variables,
            structures,
            simd: fearless_simd::Level::new(),
        })
    }
    pub fn structures(&self) -> &[Structure] {
        &self.structures
    }
    pub fn component(&self, index: usize) -> Result<Self> {
        Self::new(vec![self
            .structures
            .get(index)
            .ok_or_else(|| invalid("component index out of range"))?
            .clone()])
    }
}
impl Covariance for Lmc {
    fn covariance_rows(
        &self,
        sites: &[Site],
        observations: &[Observation],
        rows: &mut [&mut [f64]],
    ) -> Result<()> {
        simd::covariance_lines(self, sites, observations, rows, false)
    }
    fn covariance_columns(
        &self,
        observations: &[Observation],
        sites: &[Site],
        columns: &mut [&mut [f64]],
    ) -> Result<()> {
        simd::covariance_lines(self, sites, observations, columns, true)
    }
    fn covariance_sites_row(&self, a: &Site, sites: &[Site], output: &mut [f64]) -> Result<()> {
        simd::covariance_line(self, a, sites, output, false)
    }
    fn covariance_row(
        &self,
        a: &Site,
        observations: &[Observation],
        output: &mut [f64],
    ) -> Result<()> {
        simd::covariance_line(self, a, observations, output, false)
    }
    fn covariance_column(
        &self,
        observations: &[Observation],
        b: &Site,
        output: &mut [f64],
    ) -> Result<()> {
        simd::covariance_line(self, b, observations, output, true)
    }
    fn variables(&self) -> usize {
        self.variables
    }
    fn point_covariance(&self, a: Point3, va: usize, b: Point3, vb: usize) -> Result<f64> {
        let mut sum = 0.0;
        for s in &self.structures {
            let correlation = match s.correlation {
                Correlation::Nugget => {
                    if a == b {
                        1.0
                    } else {
                        0.0
                    }
                }
                kernel => kernel.evaluate(s.anisotropy.normalized_distance(a, b)),
            };
            sum += s.coregionalization.covariance(va, vb)? * correlation;
        }
        if !sum.is_finite() {
            return Err(Error::Numerical {
                context: "LMC covariance",
            });
        }
        Ok(sum)
    }
}

#[derive(Clone, Debug)]
pub struct SpatialKernel {
    pub correlation: Correlation,
    pub anisotropy: Anisotropy,
}

pub mod components;
