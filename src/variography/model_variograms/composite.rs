use nalgebra::{SimdRealField, SimdValue};

use super::{spherical::SphericalVariogram, VariogramModel};

pub enum VariogramType<T>
where
    T: SimdValue<Element = f32> + Copy,
{
    Spherical(SphericalVariogram<T>),
    //TODO
    // Exponential,
    // Gaussian,
    // Linear,
    // Power,
    // HoleEffect,
    // Custom,
}

impl<T> VariogramModel<T> for VariogramType<T>
where
    T: SimdValue<Element = f32> + SimdRealField + Copy,
{
    fn c_0(&self) -> <T as SimdValue>::Element {
        match self {
            VariogramType::Spherical(v) => v.c_0(),
        }
    }

    fn variogram(&self, h: nalgebra::Vector3<T>) -> T {
        match self {
            VariogramType::Spherical(v) => v.variogram(h),
        }
    }

    fn covariogram(&self, h: nalgebra::Vector3<T>) -> T {
        match self {
            VariogramType::Spherical(v) => v.covariogram(h),
        }
    }
}

pub struct CompositeVariogram<T>
where
    T: SimdValue<Element = f32> + Copy,
{
    pub variograms: Vec<VariogramType<T>>,
}

impl<T> CompositeVariogram<T>
where
    T: SimdValue<Element = f32> + Copy,
{
    pub fn new(variograms: Vec<VariogramType<T>>) -> Self {
        Self {
            variograms: variograms,
        }
    }
}

impl<T> VariogramModel<T> for CompositeVariogram<T>
where
    T: SimdValue<Element = f32> + SimdRealField + std::iter::Sum + Copy,
{
    fn c_0(&self) -> <T as SimdValue>::Element {
        self.variograms.iter().map(|v| v.c_0()).sum()
    }

    fn variogram(&self, h: nalgebra::Vector3<T>) -> T {
        self.variograms.iter().map(|v| v.variogram(h)).sum()
    }

    fn covariogram(&self, h: nalgebra::Vector3<T>) -> T {
        self.variograms.iter().map(|v| v.covariogram(h)).sum()
    }
}
