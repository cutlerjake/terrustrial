use std::ops::{Add, Sub};

use super::ValueTransform;

/// Value transform that subtracts a mean from the conditioning data before estimation and simulation.
///
/// The mean is added back to the estimated and simulated values.
/// This is required for SimpleKriging as the implementation assumes a zero mean.
#[derive(Clone)]
pub struct MeanTransfrom<T> {
    pub mean: T,
}

impl<T> MeanTransfrom<T> {
    pub fn new(mean: T) -> Self {
        Self { mean }
    }
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
