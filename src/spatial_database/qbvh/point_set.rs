use nalgebra::Point3;
use parry3d::bounding_volume::Aabb;
use parry3d::partitioning::Qbvh;

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
