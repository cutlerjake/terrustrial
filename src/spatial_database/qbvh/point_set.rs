use nalgebra::Point3;
use parry3d::bounding_volume::Aabb;
use parry3d::partitioning::Qbvh;

pub struct PointSet {
    pub points: Vec<Point3<f32>>,
    pub data: Vec<f32>,
    pub tree: Qbvh<u32>,
}

impl PointSet {
    pub fn new(points: Vec<Point3<f32>>, data: Vec<f32>) -> Self {
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
