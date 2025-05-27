use std::{
    iter,
    ops::{Deref, DerefMut},
};

use num_traits::Float;

use super::SpatialDataBase;

pub struct Normalzer {
    mean: f64,
    std_dev: f64,
}

impl Normalize<f64> for Normalzer {
    fn normalize(&mut self) -> (f64, f64) {
        (self.mean, self.std_dev)
    }

    fn back_transform(&mut self, mean: f64, std_dev: f64) {
        self.mean = mean;
        self.std_dev = std_dev;
    }
}

pub trait Normalize<T> {
    fn normalize(&mut self) -> (T, T);
    fn back_transform(&mut self, mean: T, std_dev: T);
}

impl<SDB, T> Normalize<T> for SDB
where
    SDB: SpatialDataBase<T>,
    T: Float + iter::Sum,
{
    fn normalize(&mut self) -> (T, T) {
        let (data, inds) = self.data_and_inds();
        let mean = data.iter().copied().sum::<T>() / T::from(data.len()).unwrap();
        let variance =
            data.iter().map(|d| (*d - mean).powi(2)).sum::<T>() / T::from(data.len()).unwrap();

        let std_dev = T::sqrt(variance);

        let normalized_data = data
            .iter()
            .map(|d| (*d - mean) / std_dev)
            .collect::<Vec<_>>();

        inds.iter().zip(normalized_data).for_each(|(ind, data)| {
            self.set_data_at_ind(ind, data);
        });

        (mean, std_dev)
    }

    fn back_transform(&mut self, mean: T, std_dev: T) {
        let (data, inds) = self.data_and_inds();

        let unnormalized_data = data.iter().map(|d| *d * std_dev + mean).collect::<Vec<_>>();

        inds.iter().zip(unnormalized_data).for_each(|(ind, data)| {
            self.set_data_at_ind(ind, data);
        });
    }
}

// Would prefer to use a normalized wrapper type to have compile time
// correctnes checks for algorithms that require normalized input checks
// but auto implementing inner trait methods is not possible
// will use Normalize trait instead for now
pub struct NormalizedSpatialDataBase<SDB, T>
where
    SDB: SpatialDataBase<T>,
{
    db: SDB,
    mean: f32,
    variance: f32,
    phantom: std::marker::PhantomData<T>,
}

impl<SDB, T> NormalizedSpatialDataBase<SDB, T>
where
    SDB: SpatialDataBase<T>,
    T: Float,
{
    pub fn new(db: SDB) -> Self {
        let mut db = db;
        let (data, inds) = db.data_and_inds();
        let mean = data.iter().map(|v| v.to_f32().unwrap()).sum::<f32>() / data.len() as f32;
        let variance = data
            .iter()
            .map(|d| (d.to_f32().unwrap() - mean).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        let std_dev = variance.sqrt();

        let normalized_data = data
            .iter()
            .map(|d| T::from((d.to_f32().unwrap() - mean) / std_dev).unwrap())
            .collect::<Vec<_>>();

        inds.iter().zip(normalized_data).for_each(|(ind, data)| {
            db.set_data_at_ind(ind, data);
        });

        Self {
            db,
            mean,
            variance,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn back_transform(self) -> SDB {
        let mut db = self.db;
        let (data, inds) = db.data_and_inds();
        let std_dev = self.variance.sqrt();

        let unnormalized_data = data
            .iter()
            .map(|d| T::from(d.to_f32().unwrap() * std_dev + self.mean).unwrap())
            .collect::<Vec<_>>();

        inds.iter().zip(unnormalized_data).for_each(|(ind, data)| {
            db.set_data_at_ind(ind, data);
        });

        db
    }
}

impl<SDB, T> Deref for NormalizedSpatialDataBase<SDB, T>
where
    SDB: SpatialDataBase<T>,
{
    type Target = SDB;
    fn deref(&self) -> &Self::Target {
        &self.db
    }
}
impl<SDB, T> DerefMut for NormalizedSpatialDataBase<SDB, T>
where
    SDB: SpatialDataBase<T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.db
    }
}
