use itertools::Itertools;
use nalgebra::Point3;
use parry3d::partitioning::Qbvh;
use parry3d::{bounding_volume::Aabb, query};
use rstar::Point;
use std::collections::HashMap;
use std::error;
use std::str::FromStr;

use crate::{
    geometry::ellipsoid::Ellipsoid,
    spatial_database::{ConditioningProvider, SpatialDataBase},
};

use super::conditioning_data_collector::ConditioningDataCollector;
use super::n_best_first::NBestFirst;

#[derive(Debug, Clone)]
pub struct PointSet<T> {
    pub points: Vec<Point3<f32>>,
    pub data: Vec<T>,
    pub tree: Qbvh<u32>,
}

impl<T> PointSet<T> {
    pub fn new(points: Vec<Point3<f32>>, data: Vec<T>) -> Self {
        let mut tree = Qbvh::new();

        tree.clear_and_rebuild(
            points
                .iter()
                .enumerate()
                .map(|(i, point)| (i as u32, Aabb::new(point.clone(), point.clone()))),
            0.0,
        );
        PointSet { points, data, tree }
    }
}

impl<T> PointSet<T>
where
    T: FromStr,
    <T as FromStr>::Err: std::error::Error + 'static,
{
    pub fn from_csv_index(
        csv_path: &str,
        x_col: &str,
        y_col: &str,
        z_col: &str,
        value_col: &str,
    ) -> Result<Self, Box<dyn error::Error>> {
        //storage for data
        let mut point_vec = Vec::new();
        let mut value_vec = Vec::new();

        //read data from csv
        let mut rdr = csv::Reader::from_path(csv_path)?;
        for result in rdr.deserialize() {
            let record: HashMap<String, String> = result?;

            let x = record[x_col].parse::<f32>()?;
            let y = record[y_col].parse::<f32>()?;
            let z = record[z_col].parse::<f32>()?;
            let value = record[value_col].parse::<T>()?;

            point_vec.push(Point3::new(x as f32, y as f32, z as f32));

            value_vec.push(value);
        }

        Ok(Self::new(point_vec, value_vec))
    }
}

impl<T> SpatialDataBase<T> for PointSet<T>
where
    T: Clone,
{
    type INDEX = usize;

    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<Self::INDEX> {
        let mut out = Vec::new();
        self.tree.intersect_aabb(bounding_box, &mut out);
        out.iter().map(|i| *i as usize).collect()
    }

    fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f32> {
        self.points[*inds]
    }

    fn data_at_ind(&self, ind: &Self::INDEX) -> Option<T> {
        self.data.get(*ind).cloned()
    }

    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>) {
        (self.data.clone(), self.points.clone())
    }

    fn data_and_inds(&self) -> (Vec<T>, Vec<Self::INDEX>) {
        (self.data.clone(), (0..self.data.len()).collect())
    }

    fn set_data_at_ind(&mut self, ind: &Self::INDEX, data: T) {
        self.data[*ind] = data;
    }
}

pub struct ConditioningParams {
    pub max_n_cond: usize,
}

impl ConditioningParams {
    pub fn new(max_n_cond: usize) -> Self {
        Self { max_n_cond }
    }
}

impl<T> ConditioningProvider<Ellipsoid, T, ConditioningParams> for PointSet<T>
where
    T: Clone,
{
    fn query(
        &self,
        point: &Point3<f32>,
        ellipsoid: &Ellipsoid,
        params: &ConditioningParams,
    ) -> (Vec<usize>, Vec<T>, Vec<Point3<f32>>) {
        let mut cond_points =
            ConditioningDataCollector::new(*point, ellipsoid, params.max_n_cond, &self);

        let _ = self.tree.traverse_n_best_first(&mut cond_points);

        let inds = cond_points
            .octant_inds
            .into_iter()
            .flatten()
            .map(|ind| ind as usize)
            .collect_vec();
        let points = cond_points.octant_points.into_iter().flatten().collect();
        let data = inds.iter().map(|ind| self.data[*ind].clone()).collect();

        (inds, data, points)
    }
}
