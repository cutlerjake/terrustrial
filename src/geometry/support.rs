use ultraviolet::DVec3;

use super::aabb::Aabb;

#[derive(Clone, Copy, Debug)]
pub enum Support {
    Point(DVec3),
    Aabb { aabb: Aabb, disc: DVec3 },
}

impl Support {
    #[inline(always)]
    pub fn sq_dist_to_point(&self, point: DVec3) -> f64 {
        match self {
            Support::Point(other_point) => {
                let delta = point - *other_point;
                delta.mag_sq()
            }
            Support::Aabb { aabb, disc: _ } => aabb.distance_to_point_squared(point),
        }
    }

    #[inline(always)]
    pub fn center(&self) -> DVec3 {
        match self {
            Support::Point(p) => *p,
            Support::Aabb { aabb, disc: _ } => aabb.center,
        }
    }

    #[inline(always)]
    pub fn num_nodes(&self) -> usize {
        match self {
            Support::Point(_) => 1,
            Support::Aabb { aabb, disc } => {
                let full_extents = aabb.half_extents * 2.0;
                let dim_cnts = full_extents / *disc;
                (dim_cnts.x.ceil() * dim_cnts.y.ceil() * dim_cnts.z.ceil()) as usize
            }
        }
    }

    #[inline(always)]
    fn discretize_in(&self, out: &mut Vec<DVec3>) {
        match self {
            Support::Point(p) => out.push(*p),
            Support::Aabb { aabb, disc } => aabb.discretize_in(*disc, out),
        }
    }

    #[inline(always)]
    pub fn dists_to_other(&self, other: &Self, out: &mut Vec<DVec3>, pt_buffer: &mut Vec<DVec3>) {
        pt_buffer.clear();

        self.discretize_in(pt_buffer);
        let split = pt_buffer.len();
        other.discretize_in(pt_buffer);

        for p1 in pt_buffer[0..split].iter() {
            for p2 in pt_buffer[split..].iter() {
                out.push(*p2 - *p1)
            }
        }
    }

    #[must_use]
    pub fn is_aabb(&self) -> bool {
        matches!(self, Self::Aabb { .. })
    }
}
