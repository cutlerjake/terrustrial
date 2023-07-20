use std::ops::{Deref, DerefMut};

use num_traits::Float;

use super::SpatialDataBase;

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
