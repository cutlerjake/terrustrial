use std::fmt::Debug;

use coordinate_system::octant;
use nalgebra::{Point3, SimdRealField, SimdValue, UnitQuaternion, Vector3};
use ordered_float::OrderedFloat;
use parry3d::bounding_volume::Aabb;
use permutation::Permutation;

use crate::{estimators::ConditioningParams, geometry::ellipsoid::Ellipsoid};

pub mod coordinate_system;
pub mod gridded_databases;
pub mod normalized;
pub mod rtree_point_set;
pub mod zero_mean;

pub trait SupportInterface {
    fn center(&self) -> Point3<f32>;
}

pub trait SupportTransform<T>: SupportInterface {
    fn transform(self) -> T;
}

impl<T> SupportTransform<T> for T
where
    T: SupportInterface,
{
    fn transform(self) -> Self {
        self
    }
}

impl SupportInterface for Point3<f32> {
    fn center(&self) -> Point3<f32> {
        *self
    }
}

impl SupportTransform<Vec<Point3<f32>>> for Point3<f32> {
    fn transform(self) -> Vec<Point3<f32>> {
        vec![self]
    }
}

pub struct ConditioningDataCollector<'b, S> {
    pub cond_params: &'b ConditioningParams,
    pub max_accepted_dist: f32,
    pub ellipsoid: &'b Ellipsoid,
    pub octant_shapes: Vec<Vec<S>>,
    pub octant_norm_dists: Vec<Vec<f32>>,
    pub octant_values: Vec<Vec<f32>>,
    pub octant_inds: Vec<Vec<u32>>,
    pub octant_tags: Vec<Vec<u32>>,
    pub octant_counts: Vec<u32>,
    pub full_octants: u8,
    pub conditioned_octants: u8,
    pub source_tag: Vec<u32>,
    pub source_count: Vec<u32>,
    pub stop: bool,
}

impl<'b, S: SupportInterface> ConditioningDataCollector<'b, S> {
    pub fn new(ellipsoid: &'b Ellipsoid, cond_params: &'b ConditioningParams) -> Self {
        let octant_max = cond_params.max_octant;
        Self {
            cond_params,
            max_accepted_dist: f32::MAX,
            ellipsoid,
            octant_shapes: (0..8)
                .map(|_| Vec::with_capacity(octant_max))
                .collect::<Vec<_>>(),
            octant_norm_dists: (0..8)
                .map(|_| Vec::with_capacity(octant_max))
                .collect::<Vec<_>>(),
            octant_values: (0..8)
                .map(|_| Vec::with_capacity(octant_max))
                .collect::<Vec<_>>(),
            octant_inds: (0..8)
                .map(|_| Vec::with_capacity(octant_max))
                .collect::<Vec<_>>(),
            octant_tags: (0..8)
                .map(|_| Vec::with_capacity(octant_max))
                .collect::<Vec<_>>(),
            octant_counts: vec![0; 8],
            full_octants: 0,
            conditioned_octants: 0,
            source_tag: Vec::new(),
            source_count: Vec::new(),
            stop: false,
        }
    }

    #[inline(always)]
    pub fn all_octants_full(&self) -> bool {
        self.full_octants == 8
    }

    #[inline(always)]
    pub fn increment_or_insert_tag(&mut self, tag: u32) -> bool {
        if let Some(ind) = self.source_tag.iter().position(|&x| x == tag) {
            if self.source_count[ind] < self.cond_params.same_source_group_limit as u32 {
                self.source_count[ind] += 1;
                true
            } else {
                false
            }
        } else {
            self.source_tag.push(tag);
            self.source_count.push(1);
            true
        }
    }

    #[inline(always)]
    pub fn decrement_tag(&mut self, tag: u32) {
        if let Some(ind) = self.source_tag.iter().position(|&x| x == tag) {
            self.source_count[ind] -= 1;
        }
    }

    #[inline(always)]
    pub fn max_octant_dist(&self, octant: usize) -> Option<(usize, f32)> {
        self.octant_norm_dists[octant]
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|(_, dist)| OrderedFloat(*dist))
    }

    #[inline(always)]
    pub fn insert_shape(
        &mut self,
        octant: usize,
        shape: S,
        value: f32,
        dist: f32,
        ind: u32,
        tag: u32,
    ) {
        // println!("tag: {:?}", tag);
        if !self.increment_or_insert_tag(tag) {
            return;
        }

        let clipped_value = self.cond_params.clipped_value(value, dist);
        self.octant_shapes[octant].push(shape);
        self.octant_inds[octant].push(ind);
        self.octant_norm_dists[octant].push(dist);
        self.octant_values[octant].push(clipped_value);
        self.octant_tags[octant].push(tag);
        self.octant_counts[octant] += 1;

        if self.octant_shapes[octant].len() == 1 {
            self.conditioned_octants += 1;
        }
    }

    #[inline(always)]
    pub fn remove_shape(&mut self, octant: usize, ind: usize) {
        let tag = self.octant_tags[octant][ind];
        self.octant_shapes[octant].swap_remove(ind);
        self.octant_inds[octant].swap_remove(ind);
        self.octant_norm_dists[octant].swap_remove(ind);
        self.octant_values[octant].swap_remove(ind);
        self.octant_counts[octant] -= 1;
        self.decrement_tag(tag);
    }

    #[inline(always)]
    pub fn can_swap_insert(&self, octant: usize, ind: usize, tag: u32) -> bool {
        let old_tag = self.octant_tags[octant][ind];

        // If tag is the same as the old tag we can insert.
        // Gauranteed to not violate same_source_group_limit since we are swapping.
        if old_tag == tag {
            return true;
        }

        // If tag is different we can insert if the new tag is not at the limit.
        // If new tag does not exist we can insert.
        let Some(tag_ind) = self.source_tag.iter().position(|&x| x == tag) else {
            return true;
        };

        self.source_count[tag_ind] < self.cond_params.same_source_group_limit as u32
    }

    #[inline(always)]
    pub fn try_insert_shape(&mut self, shape: S, value: f32, dist: f32, ind: u32, tag: u32) {
        let point = shape.center();
        // if point is further away the primary ellipsoid axis then it cannot be in the ellipsoid
        // and no further points can be in the ellipsoid
        if self.ellipsoid.a * self.ellipsoid.a < dist {
            self.stop = true;
            return;
        }

        //point is not in valid value range -> ignore
        if value < self.cond_params.valid_value_range[0]
            || value > self.cond_params.valid_value_range[1]
        {
            return;
        }

        let local_point = self
            .ellipsoid
            .coordinate_system
            .world_to_local_point(&point);

        //check if point in ellipsoid
        let h = self.ellipsoid.normalized_local_distance_sq(&local_point);

        if h > 1.0 {
            return;
        }

        //determine octant of point in ellispoid coordinate system
        let octant = octant(&local_point);

        //if octant is not full we can insert point
        if self.octant_shapes[octant as usize].len() < self.cond_params.max_octant {
            self.insert_shape(octant as usize, shape, value, h, ind, tag);
            return;
        }

        if let Some((ind, max_dist)) = self.max_octant_dist(octant as usize) {
            if h < max_dist && self.can_swap_insert(octant as usize, ind, tag) {
                self.remove_shape(octant as usize, ind);
                self.insert_shape(octant as usize, shape, value, h, ind as u32, tag);
                return;
            }

            let h_major = local_point.coords.norm();

            if h_major > max_dist {
                self.full_octants += 1;
                if self.all_octants_full() {
                    self.stop = true;
                }
            }
        }
    }
}

pub struct IterNearestElem<S, T> {
    pub shape: S,
    pub dist: f32,
    pub data: T,
    pub tag: u32,
    pub idx: u32,
}
pub trait IterNearest {
    type Shape;
    type Data;

    fn iter_nearest(
        &self,
        location: &Point3<f32>,
    ) -> impl Iterator<Item = IterNearestElem<Self::Shape, Self::Data>> + '_;
}

pub struct FilteredIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(&IterNearestElem<IN::Shape, IN::Data>) -> bool,
{
    pub iter_nearest: &'a IN,
    pub filter: F,
}

impl<'a, IN, F> FilteredIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(&IterNearestElem<IN::Shape, IN::Data>) -> bool,
{
    pub fn new(iter_nearest: &'a IN, filter: F) -> Self {
        Self {
            iter_nearest,
            filter,
        }
    }
}

impl<IN, F> IterNearest for FilteredIterNearest<'_, IN, F>
where
    IN: IterNearest,
    F: Fn(&IterNearestElem<IN::Shape, IN::Data>) -> bool,
{
    type Shape = IN::Shape;
    type Data = IN::Data;

    fn iter_nearest(
        &self,
        location: &Point3<f32>,
    ) -> impl Iterator<Item = IterNearestElem<Self::Shape, Self::Data>> + '_ {
        self.iter_nearest
            .iter_nearest(location)
            .filter(|elem| (self.filter)(elem))
    }
}

pub struct MappedIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(IterNearestElem<IN::Shape, IN::Data>) -> IterNearestElem<IN::Shape, IN::Data>,
{
    pub iter_nearest: &'a IN,
    pub map: F,
}

impl<'a, IN, F> MappedIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(IterNearestElem<IN::Shape, IN::Data>) -> IterNearestElem<IN::Shape, IN::Data>,
{
    pub fn new(iter_nearest: &'a IN, map: F) -> Self {
        Self { iter_nearest, map }
    }
}

impl<'a, IN, F> IterNearest for MappedIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(IterNearestElem<IN::Shape, IN::Data>) -> IterNearestElem<IN::Shape, IN::Data>,
{
    type Shape = IN::Shape;
    type Data = IN::Data;

    fn iter_nearest(
        &self,
        location: &Point3<f32>,
    ) -> impl Iterator<Item = IterNearestElem<Self::Shape, Self::Data>> + '_ {
        self.iter_nearest
            .iter_nearest(location)
            .map(|elem| (self.map)(elem))
    }
}

pub trait ConditioningProvider<G, T, P>: IterNearest {
    // type Shape;
    fn query(
        &self,
        point: &Point3<f32>,
        ellipsoid: &G,
        params: &P,
    ) -> (Vec<usize>, Vec<T>, Vec<Self::Shape>, bool);
}

impl<IN> ConditioningProvider<Ellipsoid, f32, ConditioningParams> for IN
where
    IN: IterNearest<Shape: SupportInterface, Data = f32>,
{
    // type Shape = IN::Shape;

    fn query(
        &self,
        point: &Point3<f32>,
        ellipsoid: &Ellipsoid,
        params: &ConditioningParams,
    ) -> (Vec<usize>, Vec<f32>, Vec<Self::Shape>, bool) {
        let mut cond_points = ConditioningDataCollector::new(ellipsoid, params);

        for IterNearestElem {
            shape,
            dist,
            data,
            tag,
            idx,
        } in self.iter_nearest(point)
        {
            cond_points.try_insert_shape(shape, data, dist, idx, tag as u32);
            if cond_points.stop {
                break;
            }
        }

        let mut inds: Vec<usize> = cond_points
            .octant_inds
            .into_iter()
            .flatten()
            .map(|i| i as usize)
            .collect();
        let mut points: Vec<_> = cond_points.octant_shapes.into_iter().flatten().collect();
        // let mut data: Vec<f32> = inds.iter().map(|ind| self.data[*ind].clone()).collect();
        let mut data = cond_points
            .octant_values
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>();

        if data.len() > params.max_n_cond {
            let mut octant_counts = cond_points.octant_counts;
            let mut can_remove_flag =
                if cond_points.conditioned_octants > params.min_conditioned_octants as u8 {
                    vec![true; 8]
                } else {
                    octant_counts.iter().map(|&count| count > 1).collect()
                };

            let mut octant_inds = cond_points
                .octant_norm_dists
                .iter()
                .enumerate()
                .flat_map(|(i, d)| vec![i; d.len()])
                .collect::<Vec<_>>();

            // println!("octant_inds: {:?}", octant_inds);

            let mut dists: Vec<f32> = cond_points
                .octant_norm_dists
                .into_iter()
                .flatten()
                .collect();

            //sort data, inds, points and dists by distance
            let mut sorted_inds = (0..inds.len()).collect::<Vec<_>>();
            sorted_inds.sort_by_key(|i| OrderedFloat(dists[*i]));

            let mut permutation = Permutation::oneline(sorted_inds).inverse();

            permutation.apply_slice_in_place(&mut inds);
            permutation.apply_slice_in_place(&mut points);
            permutation.apply_slice_in_place(&mut dists);
            permutation.apply_slice_in_place(&mut data);
            permutation.apply_slice_in_place(&mut octant_inds);

            let mut end = octant_inds.len();

            while data.len() > params.max_n_cond {
                let Some(r_ind) = octant_inds[0..end]
                    .iter()
                    .rev()
                    .position(|oct| can_remove_flag[*oct])
                else {
                    break;
                };

                let ind = end - r_ind - 1;

                end = ind;

                let octant = octant_inds[ind];

                //remove value
                inds.swap_remove(ind);
                points.swap_remove(ind);
                dists.swap_remove(ind);
                data.swap_remove(ind);
                octant_inds.swap_remove(ind);

                //update octant counts
                octant_counts[octant] -= 1;

                //update conditioned octants as needed
                if octant_counts[octant] == 0 {
                    cond_points.conditioned_octants -= 1;
                }

                //update can remove flag
                if cond_points.conditioned_octants < params.min_conditioned_octants as u8 {
                    can_remove_flag = octant_counts.iter().map(|&count| count > 1).collect();
                }
            }
        }

        let res = cond_points.conditioned_octants >= params.min_conditioned_octants as u8
            && data.len() >= params.min_n_cond;

        (inds, data, points, res)
    }
}

pub trait SpatialDataBase<T> {
    type INDEX: Debug;
    fn inds_in_bounding_box(&self, bounding_box: &Aabb) -> Vec<Self::INDEX>;
    fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f32>;
    fn data_at_ind(&self, ind: &Self::INDEX) -> Option<T>;
    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f32>>);
    fn data_and_inds(&self) -> (Vec<T>, Vec<Self::INDEX>);
    fn set_data_at_ind(&mut self, ind: &Self::INDEX, data: T);
}

pub enum RoationType {
    Extrinsic,
    Intrinsic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationAxis {
    X,
    Y,
    Z,
}

impl RotationAxis {
    pub fn from_char(c: char) -> Self {
        match c {
            'x' => Self::X,
            'y' => Self::Y,
            'z' => Self::Z,
            _ => panic!("Invalid rotation axis"),
        }
    }

    pub fn as_char(&self) -> char {
        match self {
            Self::X => 'x',
            Self::Y => 'y',
            Self::Z => 'z',
        }
    }
}

pub trait FromAxisAngles
where
    Self: Sized,
{
    fn from_axis_angles(rotations: &[(RotationAxis, f32)]) -> Self;
}

impl<T> FromAxisAngles for UnitQuaternion<T>
where
    T: SimdValue<Element = f32> + SimdRealField + Copy,
{
    fn from_axis_angles(rotations: &[(RotationAxis, f32)]) -> Self {
        rotations
            .iter()
            .map(|(axis, ang)| match axis {
                RotationAxis::X => {
                    let axis = Vector3::x_axis();
                    let angle = T::splat(*ang);
                    UnitQuaternion::from_axis_angle(&axis, angle)
                }
                RotationAxis::Y => {
                    let axis = Vector3::y_axis();
                    let angle = T::splat(*ang);
                    UnitQuaternion::from_axis_angle(&axis, angle)
                }
                RotationAxis::Z => {
                    let axis = Vector3::z_axis();
                    let angle = T::splat(*ang);
                    UnitQuaternion::from_axis_angle(&axis, angle)
                }
            })
            .fold(UnitQuaternion::identity(), |acc, x| acc * x)
    }
}

pub trait DiscretiveVolume {
    fn discretize(&self, dx: f32, dy: f32, dz: f32) -> Vec<Point3<f32>>;
}

impl DiscretiveVolume for Aabb {
    fn discretize(&self, dx: f32, dy: f32, dz: f32) -> Vec<Point3<f32>> {
        //ceil gaurantees that the resulting discretization will have dimensions upperbounded by dx, dy, dz
        let nx = ((self.maxs.x - self.mins.x) / dx).ceil() as usize;
        let ny = ((self.maxs.y - self.mins.y) / dy).ceil() as usize;
        let nz = ((self.maxs.z - self.mins.z) / dz).ceil() as usize;

        //step size in each direction
        let step_x = (self.maxs.x - self.mins.x) / (nx as f32);
        let step_y = (self.maxs.y - self.mins.y) / (ny as f32);
        let step_z = (self.maxs.z - self.mins.z) / (nz as f32);

        //contains the discretized points
        let mut points = Vec::new();

        let mut x = self.mins.x + step_x / 2.0;
        while x <= self.maxs.x {
            let mut y = self.mins.y + step_y / 2.0;
            while y <= self.maxs.y {
                let mut z = self.mins.z + step_z / 2.0;
                while z <= self.maxs.z {
                    points.push(Point3::new(x, y, z));
                    z += step_z;
                }
                y += step_y;
            }
            x += step_x;
        }

        points
    }
}

#[cfg(test)]
mod test {

    use num_traits::real::Real;

    use super::*;

    #[test]
    fn zyz() {
        let rotations = vec![
            (RotationAxis::Z, 10.0.to_radians()),
            (RotationAxis::Y, -20.0.to_radians()),
            (RotationAxis::Z, 30.0.to_radians()),
        ];

        let quat = UnitQuaternion::<f32>::from_axis_angles(rotations.as_slice());

        let vec = Vector3::x_axis();

        let vec = quat.transform_vector(&vec);
        println!("{:?}", vec);
    }
}
