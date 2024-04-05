use std::ops::{Add, Sub};

use super::ValueTransform;

pub struct MeanTransfrom<T> {
    mean: T,
}

impl<T> ValueTransform<T> for MeanTransfrom<T>
where
    T: Copy,
    for<'a> &'a T: Sub<T, Output = T> + Add<T, Output = T>,
{
    fn forward_transform(&self, value: &T) -> T {
        value - self.mean
    }

    fn backward_transform(&self, value: &T) -> T {
        value + self.mean
    }
}
