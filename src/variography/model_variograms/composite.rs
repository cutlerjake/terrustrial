use nalgebra::{SimdRealField, SimdValue, UnitQuaternion};
use simba::simd::WideF32x8;

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

impl VariogramType<f32> {
    pub fn to_f32x8(&self) -> VariogramType<WideF32x8> {
        match self {
            VariogramType::Nugget(n) => VariogramType::Nugget(n.to_f32x8()),
            VariogramType::Spherical(s) => VariogramType::Spherical(s.to_f32x8()),
        }
    }
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

    fn set_orientation(&mut self, orientation: UnitQuaternion<T>) {
        match self {
            VariogramType::Spherical(v) => v.set_orientation(orientation),
            VariogramType::Nugget(_) => {}
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

impl CompositeVariogram<f32> {
    pub fn to_f32x8(&self) -> CompositeVariogram<WideF32x8> {
        CompositeVariogram {
            variograms: self.variograms.iter().map(|v| v.to_f32x8()).collect(),
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

    fn set_orientation(&mut self, orientation: UnitQuaternion<T>) {
        for v in self.variograms.iter_mut() {
            v.set_orientation(orientation);
        }
    }
}

#[cfg(test)]
mod test {
    use nalgebra::Vector3;

    use super::*;

    #[test]
    fn composite_variogram() {
        let spherical = VariogramType::Spherical(SphericalVariogram::<f32>::new(
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
            UnitQuaternion::identity(),
        ));

        let composite = CompositeVariogram::new(vec![spherical.clone()]);

        //check if composite variogram is equal to spherical variogram
        assert!(composite.c_0() == spherical.c_0());

        assert!(
            composite.covariogram(Vector3::new(0.5, 0.5, 0.5))
                == spherical.covariogram(Vector3::new(0.5, 0.5, 0.5))
        );

        assert!(
            composite.variogram(Vector3::new(1.0, 1.0, 1.0))
                == spherical.variogram(Vector3::new(1.0, 1.0, 1.0))
        );
        assert!(
            composite.covariogram(Vector3::new(1.0, 1.0, 1.0))
                == spherical.covariogram(Vector3::new(1.0, 1.0, 1.0))
        );
    }
}
