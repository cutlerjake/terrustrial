//! Implicit scalar-field interfaces and domain classification.
//! Gaussian derivative observations encode normals/gradients; a general implicit
//! field is not guaranteed to be an exact signed-distance function.
use crate::{data::Point3, error::invalid, Result};

pub mod evidence;
pub mod validation;

pub trait ImplicitField: Sync {
    fn value(&self, point: Point3) -> Result<f64>;
    fn gradient(&self, _: Point3) -> Result<Option<[f64; 3]>> {
        Ok(None)
    }
    fn evaluate_into(&self, points: &[Point3], out: &mut [f64]) -> Result<()> {
        if out.len() != points.len() {
            return Err(invalid("field output length"));
        }
        for (point, value) in points.iter().zip(out) {
            *value = self.value(*point)?;
        }
        Ok(())
    }
}
impl<F: Fn(Point3) -> Result<f64> + Sync> ImplicitField for F {
    fn value(&self, point: Point3) -> Result<f64> {
        self(point)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Classification {
    Inside,
    Boundary,
    Outside,
}
pub fn classify(
    field: &impl ImplicitField,
    point: Point3,
    tolerance: f64,
) -> Result<Classification> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(invalid("domain tolerance"));
    }
    let value = field.value(point)?;
    if !value.is_finite() {
        return Err(invalid("implicit field returned nonfinite value"));
    }
    Ok(if value.abs() <= tolerance {
        Classification::Boundary
    } else if value < 0.0 {
        Classification::Inside
    } else {
        Classification::Outside
    })
}
