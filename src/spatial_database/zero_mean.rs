use std::iter;

use num_traits::Float;

pub struct ZeroMeanTransform<T>
where
    T: Float,
{
    mean: T,
}

impl<T> ZeroMeanTransform<T>
where
    T: Float,
{
    pub fn new(mean: T) -> Self {
        Self { mean }
    }

    pub fn mean(&self) -> T {
        self.mean
    }
    pub fn transform(&self, data: T) -> T {
        data - self.mean
    }

    pub fn back_transform(&self, data: T) -> T {
        data + self.mean
    }
}

impl<T> From<&[T]> for ZeroMeanTransform<T>
where
    T: Float + iter::Sum + Copy,
{
    fn from(data: &[T]) -> Self {
        let mean = data.iter().copied().sum::<T>() / T::from(data.len()).unwrap();
        Self::new(mean)
    }
}
