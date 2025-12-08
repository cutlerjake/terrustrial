use ultraviolet::{DIsometry3, DRotor3, DVec3};

#[derive(Clone, Copy, Debug)]
pub struct CoordinateSystem {
    into_local: DIsometry3,
    into_global: DIsometry3,
}

impl CoordinateSystem {
    pub fn new(origin: DVec3, rotation: DRotor3) -> Self {
        let into_global = DIsometry3::new(origin, rotation);
        let into_local = into_global.inversed();

        Self {
            into_local,
            into_global,
        }
    }

    #[inline(always)]
    pub fn into_local(&self) -> &DIsometry3 {
        &self.into_local
    }

    #[inline(always)]
    pub fn into_global(&self) -> &DIsometry3 {
        &self.into_global
    }

    #[inline(always)]
    pub fn set_origin(&mut self, origin: DVec3) {
        self.into_global.translation = origin;
        self.into_local = self.into_global.inversed();
    }

    #[inline(always)]
    pub fn set_rotation(&mut self, rotation: DRotor3) {
        self.into_global.rotation = rotation;
        self.into_local = self.into_global.inversed();
    }
}

/// Octant of a point
pub fn octant(point: &DVec3) -> u8 {
    match (point.x >= 0.0, point.y >= 0.0, point.z >= 0.0) {
        (true, true, true) => 0,
        (false, true, true) => 1,
        (false, false, true) => 2,
        (true, false, true) => 3,
        (true, true, false) => 4,
        (false, true, false) => 5,
        (false, false, false) => 6,
        (true, false, false) => 7,
    }
}
