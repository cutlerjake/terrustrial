use std::{collections::HashMap, error, fmt::Debug, str::FromStr};

use coordinate_system::octant;
use nalgebra::{Point3, SimdRealField, SimdValue, UnitQuaternion, Vector3};
use ordered_float::OrderedFloat;
use permutation::Permutation;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use ultraviolet::DVec3;

use crate::{
    geometry::{aabb::Aabb, ellipsoid::Ellipsoid, support::Support},
    group_operators::ConditioningParams,
};

pub mod coordinate_system;
pub mod group_provider;
pub mod normalized;
pub mod zero_mean;

#[derive(Copy, Clone)]
pub struct SpatialData {
    pub(crate) support: Support,
    pub(crate) data_idx: u32,
}

impl SpatialData {
    pub fn support(&self) -> Support {
        self.support
    }

    pub fn data_idx(&self) -> u32 {
        self.data_idx
    }
}

impl RTreeObject for SpatialData {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        let (mins, maxs) = match &self.support {
            Support::Point(p) => (*p, *p),
            Support::Aabb { aabb, disc: _ } => {
                let mins = aabb.mins();
                let maxs = aabb.maxs();

                (mins, maxs)
            }
        };

        AABB::from_corners(*mins.as_array(), *maxs.as_array())
    }
}

impl PointDistance for SpatialData {
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        self.support.sq_dist_to_point(DVec3::from(*point))
    }
}

pub struct NeighboringElement<'a, T> {
    pub idx: u32,
    pub data: &'a T,
    pub support: Support,
    pub sq_dist: f64,
}
#[derive(Clone)]
pub struct SpatialAcceleratedDB<T> {
    pub tree: RTree<SpatialData>,
    pub supports: Vec<Support>,
    pub data: Vec<T>,
}

impl<T> SpatialAcceleratedDB<T> {
    pub fn new(supports: Vec<Support>, data: Vec<T>) -> Self {
        let spatial_data = supports
            .clone()
            .into_iter()
            .enumerate()
            .map(|(data_idx, support)| SpatialData {
                support,
                data_idx: data_idx as u32,
            })
            .collect::<Vec<_>>();

        let tree = RTree::bulk_load(spatial_data);

        Self {
            tree,
            supports,
            data,
        }
    }

    fn _iter_nearest(&self, location: DVec3) -> impl Iterator<Item = NeighboringElement<T>> {
        self.tree
            .nearest_neighbor_iter_with_distance_2(&[location.x, location.y, location.z])
            .map(|(sd, sq_dist)| NeighboringElement {
                idx: sd.data_idx,
                data: &self.data[sd.data_idx as usize],
                support: sd.support,
                sq_dist,
            })
    }
}

impl<T> SpatialAcceleratedDB<T>
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

            let x = record[x_col].parse::<f64>()?;
            let y = record[y_col].parse::<f64>()?;
            let z = record[z_col].parse::<f64>()?;
            let value = record[value_col].parse::<T>()?;

            point_vec.push(Support::Point(DVec3::new(x, y, z)));

            value_vec.push(value);
        }

        //no source tag -> give all points a unique tag

        Ok(Self::new(point_vec, value_vec))
    }
}

impl<T: Copy> IterNearest for SpatialAcceleratedDB<T> {
    type Data = T;

    fn iter_nearest(
        &self,
        location: &DVec3,
    ) -> impl Iterator<Item = IterNearestElem<Self::Data>> + '_ {
        self._iter_nearest(*location).map(|elem| IterNearestElem {
            shape: elem.support,
            sq_dist: elem.sq_dist,
            data: *elem.data,
            tag: 0,
            idx: elem.idx,
        })
    }
}

pub trait SupportInterface {
    fn center(&self) -> Point3<f64>;
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

impl SupportInterface for Point3<f64> {
    fn center(&self) -> Point3<f64> {
        *self
    }
}

impl SupportTransform<Vec<Point3<f64>>> for Point3<f64> {
    fn transform(self) -> Vec<Point3<f64>> {
        vec![self]
    }
}

pub struct ConditioningDataCollector<'b, T> {
    pub cond_params: &'b ConditioningParams,
    pub max_accepted_dist: f64,
    pub ellipsoid: &'b Ellipsoid,
    pub octant_shapes: Vec<Vec<Support>>,
    pub octant_norm_dists: Vec<Vec<f64>>,
    pub octant_values: Vec<Vec<T>>,
    pub octant_inds: Vec<Vec<u32>>,
    pub octant_tags: Vec<Vec<u32>>,
    pub octant_counts: Vec<u32>,
    pub full_octants: u8,
    pub conditioned_octants: u8,
    pub source_tag: Vec<u32>,
    pub source_count: Vec<u32>,
    pub stop: bool,
}

impl<'b, T> ConditioningDataCollector<'b, T> {
    pub fn new(ellipsoid: &'b Ellipsoid, cond_params: &'b ConditioningParams) -> Self {
        let octant_max = cond_params.max_octant;
        Self {
            cond_params,
            max_accepted_dist: f64::MAX,
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
    pub fn max_octant_dist(&self, octant: usize) -> Option<(usize, f64)> {
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
        shape: Support,
        value: T,
        dist: f64,
        ind: u32,
        tag: u32,
    ) {
        // println!("tag: {:?}", tag);
        if !self.increment_or_insert_tag(tag) {
            return;
        }

        self.octant_shapes[octant].push(shape);
        self.octant_inds[octant].push(ind);
        self.octant_norm_dists[octant].push(dist);
        self.octant_values[octant].push(value);
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
    pub fn try_insert_shape(&mut self, shape: Support, value: T, sq_dist: f64, ind: u32, tag: u32) {
        let point = shape.center();
        // if point is further away the primary ellipsoid axis then it cannot be in the ellipsoid
        // and no further points can be in the ellipsoid
        if !self.ellipsoid.may_contain_local_point_at_sq_dist(sq_dist) {
            self.stop = true;
            return;
        }

        let local_point = self
            .ellipsoid
            .coordinate_system
            .into_local()
            .transform_vec(point);

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

        if let Some((_ind, max_dist)) = self.max_octant_dist(octant as usize) {
            if h < max_dist && self.can_swap_insert(octant as usize, _ind, tag) {
                self.remove_shape(octant as usize, _ind);
                self.insert_shape(octant as usize, shape, value, h, ind, tag);
                return;
            }

            let h_major = local_point.mag();

            if h_major > max_dist {
                self.full_octants += 1;
                if self.all_octants_full() {
                    self.stop = true;
                }
            }
        }
    }
}

pub struct IterNearestElem<T> {
    pub shape: Support,
    pub sq_dist: f64,
    pub data: T,
    pub tag: u32,
    pub idx: u32,
}
pub trait IterNearest {
    type Data;

    fn iter_nearest(
        &self,
        location: &DVec3,
    ) -> impl Iterator<Item = IterNearestElem<Self::Data>> + '_;
}

pub struct FilteredIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(&IterNearestElem<IN::Data>) -> bool,
{
    pub iter_nearest: &'a IN,
    pub filter: F,
}

impl<'a, IN, F> FilteredIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(&IterNearestElem<IN::Data>) -> bool,
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
    F: Fn(&IterNearestElem<IN::Data>) -> bool,
{
    type Data = IN::Data;

    fn iter_nearest(
        &self,
        location: &DVec3,
    ) -> impl Iterator<Item = IterNearestElem<Self::Data>> + '_ {
        self.iter_nearest
            .iter_nearest(location)
            .filter(|elem| (self.filter)(elem))
    }
}

pub struct MappedIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(IterNearestElem<IN::Data>) -> IterNearestElem<IN::Data>,
{
    pub iter_nearest: &'a IN,
    pub map: F,
}

impl<'a, IN, F> MappedIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(IterNearestElem<IN::Data>) -> IterNearestElem<IN::Data>,
{
    pub fn new(iter_nearest: &'a IN, map: F) -> Self {
        Self { iter_nearest, map }
    }
}

impl<'a, IN, F> IterNearest for MappedIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(IterNearestElem<IN::Data>) -> IterNearestElem<IN::Data>,
{
    type Data = IN::Data;

    fn iter_nearest(
        &self,
        location: &DVec3,
    ) -> impl Iterator<Item = IterNearestElem<Self::Data>> + '_ {
        self.iter_nearest
            .iter_nearest(location)
            .map(|elem| (self.map)(elem))
    }
}

pub trait ConditioningProvider: IterNearest + Sync + Send {
    // type Shape;
    fn query(
        &self,
        point: &DVec3,
        ellipsoid: &Ellipsoid,
        params: &ConditioningParams,
    ) -> (Vec<usize>, Vec<Self::Data>, Vec<Support>, bool);
}

pub enum FilterMapResult<T> {
    Mapped(T),
    Ignore,
    ExitEarly,
}
pub struct FilterMappedIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(IterNearestElem<IN::Data>) -> FilterMapResult<IterNearestElem<IN::Data>>,
{
    pub iter_nearest: &'a IN,
    pub map: F,
}

impl<'a, IN, F> FilterMappedIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(IterNearestElem<IN::Data>) -> FilterMapResult<IterNearestElem<IN::Data>>,
{
    pub fn new(iter_nearest: &'a IN, map: F) -> Self {
        Self { iter_nearest, map }
    }
}

impl<'a, IN, F> IterNearest for FilterMappedIterNearest<'a, IN, F>
where
    IN: IterNearest,
    F: Fn(IterNearestElem<IN::Data>) -> FilterMapResult<IterNearestElem<IN::Data>>,
{
    type Data = IN::Data;

    fn iter_nearest(
        &self,
        location: &DVec3,
    ) -> impl Iterator<Item = IterNearestElem<Self::Data>> + '_ {
        self.iter_nearest
            .iter_nearest(location)
            .map(|elem| (self.map)(elem))
            .take_while(|res| !matches!(res, FilterMapResult::ExitEarly))
            .filter_map(|res| {
                if let FilterMapResult::Mapped(val) = res {
                    Some(val)
                } else {
                    None
                }
            })
    }
}

impl<IN> ConditioningProvider for IN
where
    IN: IterNearest + Sync + Send,
{
    fn query(
        &self,
        point: &DVec3,
        ellipsoid: &Ellipsoid,
        params: &ConditioningParams,
    ) -> (Vec<usize>, Vec<IN::Data>, Vec<Support>, bool) {
        let mut cond_points = ConditioningDataCollector::new(ellipsoid, params);

        for IterNearestElem {
            shape,
            sq_dist,
            data,
            tag,
            idx,
        } in self.iter_nearest(point)
        {
            cond_points.try_insert_shape(shape, data, sq_dist, idx, tag);
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
            .collect::<Vec<IN::Data>>();

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

            let mut dists: Vec<f64> = cond_points
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
    fn inds_in_bounding_box(
        &self,
        bounding_box: &parry3d_f64::bounding_volume::Aabb,
    ) -> Vec<Self::INDEX>;
    fn point_at_ind(&self, inds: &Self::INDEX) -> Point3<f64>;
    fn data_at_ind(&self, ind: &Self::INDEX) -> Option<T>;
    fn data_and_points(&self) -> (Vec<T>, Vec<Point3<f64>>);
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
    fn discretize(&self, dx: f64, dy: f64, dz: f64) -> Vec<DVec3>;
}

impl DiscretiveVolume for Aabb {
    fn discretize(&self, dx: f64, dy: f64, dz: f64) -> Vec<DVec3> {
        //ceil gaurantees that the resulting discretization will have dimensions upperbounded by dx, dy, dz
        let nx = ((self.maxs().x - self.mins().x) / dx).ceil() as usize;
        let ny = ((self.maxs().y - self.mins().y) / dy).ceil() as usize;
        let nz = ((self.maxs().z - self.mins().z) / dz).ceil() as usize;

        //step size in each direction
        let step_x = (self.maxs().x - self.mins().x) / (nx as f64);
        let step_y = (self.maxs().y - self.mins().y) / (ny as f64);
        let step_z = (self.maxs().z - self.mins().z) / (nz as f64);

        //contains the discretized points
        let mut points = Vec::new();

        let mut x = self.mins().x + step_x / 2.0;
        while x <= self.maxs().x {
            let mut y = self.mins().y + step_y / 2.0;
            while y <= self.maxs().y {
                let mut z = self.mins().z + step_z / 2.0;
                while z <= self.maxs().z {
                    points.push(DVec3::new(x, y, z));
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
