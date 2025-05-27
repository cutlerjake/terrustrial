use ultraviolet::{DRotor3, DVec3, DVec3x4};
use wide::f64x4;

use super::{nugget::Nugget, spherical::SphericalVariogram, VariogramModel};

#[derive(Clone, Debug)]
pub enum VariogramType {
    Nugget(Nugget),
    Spherical(SphericalVariogram),
    //TODO
    // Exponential,
    // Gaussian,
    // Linear,
    // Power,
    // HoleEffect,
    // Custom,
}

impl VariogramModel for VariogramType {
    fn c_0(&self) -> f64 {
        match self {
            VariogramType::Spherical(v) => v.c_0(),
            &VariogramType::Nugget(v) => v.c_0(),
        }
    }

    fn variogram(&self, h: DVec3) -> f64 {
        match self {
            VariogramType::Spherical(v) => v.variogram(h),
            VariogramType::Nugget(v) => v.variogram(h),
        }
    }

    fn covariogram(&self, h: DVec3) -> f64 {
        match self {
            VariogramType::Spherical(v) => v.covariogram(h),
            VariogramType::Nugget(v) => v.covariogram(h),
        }
    }

    fn variogram_simd(&self, h: DVec3x4) -> f64x4 {
        match self {
            VariogramType::Spherical(v) => v.variogram_simd(h),
            VariogramType::Nugget(v) => v.variogram_simd(h),
        }
    }

    fn covariogram_simd(&self, h: DVec3x4) -> f64x4 {
        match self {
            VariogramType::Spherical(v) => v.covariogram_simd(h),
            VariogramType::Nugget(v) => v.covariogram_simd(h),
        }
    }

    fn set_orientation(&mut self, orientation: DRotor3) {
        match self {
            VariogramType::Spherical(v) => v.set_orientation(orientation),
            VariogramType::Nugget(_) => {}
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompositeVariogram {
    pub variograms: Vec<VariogramType>,
}

impl CompositeVariogram {
    pub fn new(variograms: Vec<VariogramType>) -> Self {
        Self { variograms }
    }

    pub fn set_orientation(&mut self, orientation: DRotor3) {
        for v in self.variograms.iter_mut() {
            v.set_orientation(orientation);
        }
    }
}

impl VariogramModel for CompositeVariogram {
    fn c_0(&self) -> f64 {
        self.variograms.iter().map(VariogramModel::c_0).sum()
    }

    fn variogram(&self, h: DVec3) -> f64 {
        self.variograms
            .iter()
            .fold(0.0, |acc, v| acc + v.variogram(h))
    }

    fn covariogram(&self, h: DVec3) -> f64 {
        self.variograms
            .iter()
            .fold(0.0, |acc, v| acc + v.covariogram(h))
    }

    fn variogram_simd(&self, h: DVec3x4) -> f64x4 {
        self.variograms
            .iter()
            .fold(f64x4::splat(0.0), |acc, v| acc + v.variogram_simd(h))
    }

    fn covariogram_simd(&self, h: DVec3x4) -> f64x4 {
        self.variograms
            .iter()
            .fold(f64x4::splat(0.0), |acc, v| acc + v.covariogram_simd(h))
    }

    fn set_orientation(&mut self, orientation: DRotor3) {
        for v in self.variograms.iter_mut() {
            v.set_orientation(orientation);
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn composite_variogram() {
        let spherical = VariogramType::Spherical(SphericalVariogram::new(
            DVec3::new(1.0, 1.0, 1.0),
            1.0,
            DRotor3::identity(),
        ));

        let composite = CompositeVariogram::new(vec![spherical.clone()]);

        //check if composite variogram is equal to spherical variogram
        assert!(composite.c_0() == spherical.c_0());

        assert!(
            composite.covariogram(DVec3::new(0.5, 0.5, 0.5))
                == spherical.covariogram(DVec3::new(0.5, 0.5, 0.5))
        );

        assert!(
            composite.variogram(DVec3::new(1.0, 1.0, 1.0))
                == spherical.variogram(DVec3::new(1.0, 1.0, 1.0))
        );
        assert!(
            composite.covariogram(DVec3::new(1.0, 1.0, 1.0))
                == spherical.covariogram(DVec3::new(1.0, 1.0, 1.0))
        );
    }
}
