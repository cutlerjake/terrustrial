use nalgebra::{SimdRealField, SimdValue, UnitQuaternion};

use super::{nugget::Nugget, spherical::SphericalVariogram, VariogramModel};

#[derive(Clone, Debug)]
pub enum VariogramType<T>
where
    T: SimdValue<Element = f32> + Copy,
{
    Nugget(Nugget<T>),
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
            &VariogramType::Nugget(v) => v.c_0(),
        }
    }

    fn variogram(&self, h: nalgebra::Vector3<T>) -> T {
        match self {
            VariogramType::Spherical(v) => v.variogram(h),
            VariogramType::Nugget(v) => v.variogram(h),
        }
    }

    fn covariogram(&self, h: nalgebra::Vector3<T>) -> T {
        match self {
            VariogramType::Spherical(v) => v.covariogram(h),
            VariogramType::Nugget(v) => v.covariogram(h),
        }
    }
}

#[derive(Clone, Debug, Default)]
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
        Self { variograms }
    }

    pub fn set_orientation(&mut self, orientation: UnitQuaternion<T>)
    where
        T: SimdRealField,
    {
        for v in self.variograms.iter_mut() {
            if let VariogramType::Spherical(s) = v {
                s.rotation = orientation.inverse();
            }
        }
    }
}

impl<T> VariogramModel<T> for CompositeVariogram<T>
where
    T: SimdValue<Element = f32> + SimdRealField + Copy,
{
    fn c_0(&self) -> <T as SimdValue>::Element {
        self.variograms.iter().map(VariogramModel::c_0).sum()
    }

    fn variogram(&self, h: nalgebra::Vector3<T>) -> T {
        self.variograms
            .iter()
            .fold(T::splat(0.0), |acc, v| acc + v.variogram(h))
    }

    fn covariogram(&self, h: nalgebra::Vector3<T>) -> T {
        self.variograms
            .iter()
            .fold(T::splat(0.0), |acc, v| acc + v.covariogram(h))
    }
}
