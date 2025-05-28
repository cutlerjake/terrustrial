use ultraviolet::{DIsometry3, DVec3};

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub center: DVec3,
    pub half_extents: DVec3,
}

impl Aabb {
    #[inline(always)]
    pub fn new(center: DVec3, half_extents: DVec3) -> Self {
        Self {
            center,
            half_extents,
        }
    }

    #[inline(always)]
    pub fn from_min_max(min: DVec3, max: DVec3) -> Self {
        let center = (min + max) / 2.0;
        let half_extents = (max - min) / 2.0;
        Self {
            center,
            half_extents,
        }
    }

    #[inline(always)]
    pub fn mins(&self) -> DVec3 {
        self.center - self.half_extents
    }

    #[inline(always)]
    pub fn maxs(&self) -> DVec3 {
        self.center + self.half_extents
    }

    #[inline(always)]
    pub fn contains_point(&self, point: DVec3) -> bool {
        let mins = self.mins();
        let maxs = self.maxs();

        mins.x <= point.x
            && mins.y <= point.y
            && mins.z <= point.z
            && maxs.x >= point.x
            && maxs.y >= point.y
            && maxs.z >= point.z
    }

    #[inline(always)]
    pub fn distance_to_point_squared(&self, point: DVec3) -> f64 {
        let inside = self.contains_point(point);
        let clamped = point.clamped(self.mins(), self.maxs());

        let mult = -1.0 * inside as u8 as f64;

        let delta = point - clamped;
        delta.dot(delta) * mult
    }

    #[inline(always)]
    pub fn transformed_by(&self, transform: DIsometry3) -> Self {
        let m = transform.rotation.into_matrix();

        let mut out_c = transform.translation;
        let mut out_r = DVec3::zero();

        out_c += m * self.center;
        out_r += (m * self.half_extents).abs();

        Self {
            center: out_c,
            half_extents: out_r,
        }
    }

    #[inline(always)]
    pub fn discretize_in(&self, disc: DVec3, out: &mut Vec<DVec3>) {
        let dx = disc.x;
        let dy = disc.y;
        let dz = disc.z;
        //ceil gaurantees that the resulting discretization will have dimensions upperbounded by dx, dy, dz
        let nx = ((self.maxs().x - self.mins().x) / dx).ceil() as usize;
        let ny = ((self.maxs().y - self.mins().y) / dy).ceil() as usize;
        let nz = ((self.maxs().z - self.mins().z) / dz).ceil() as usize;

        //step size in each direction
        let step_x = (self.maxs().x - self.mins().x) / (nx as f64);
        let step_y = (self.maxs().y - self.mins().y) / (ny as f64);
        let step_z = (self.maxs().z - self.mins().z) / (nz as f64);

        let mut x = self.mins().x + step_x / 2.0;
        while x <= self.maxs().x {
            let mut y = self.mins().y + step_y / 2.0;
            while y <= self.maxs().y {
                let mut z = self.mins().z + step_z / 2.0;
                while z <= self.maxs().z {
                    out.push(DVec3::new(x, y, z));
                    z += step_z;
                }
                y += step_y;
            }
            x += step_x;
        }
    }

    #[inline(always)]
    pub fn merged(&self, other: &Self) -> Self {
        Self::from_min_max(
            self.mins().min_by_component(other.mins()),
            self.maxs().max_by_component(other.maxs()),
        )
    }
}
