use std::ops::BitAnd;
use ultraviolet::{f64x4, DRotor3, DVec3};
use wide::CmpGt;

use super::VariogramModel;

#[derive(Clone, Copy, Debug, Default)]
pub struct Nugget {
    pub nugget: f64,
}

impl Nugget {
    pub fn new(nugget: f64) -> Self {
        Self { nugget }
    }
}

impl VariogramModel for Nugget {
    #[inline(always)]
    fn c_0(&self) -> f64 {
        self.nugget
    }

    #[inline(always)]
    fn variogram(&self, h: DVec3) -> f64 {
        let iso_h = h.dot(h);

        if iso_h == 0.0 {
            0.0
        } else {
            self.nugget
        }
    }

    #[inline(always)]
    fn covariogram(&self, h: DVec3) -> f64 {
        self.nugget - self.variogram(h)
    }

    #[inline(always)]
    fn set_orientation(&mut self, _orientation: DRotor3) {
        // Do nothing
    }

    fn variogram_simd(&self, h: ultraviolet::DVec3x4) -> f64x4 {
        let iso_h: f64x4 = h.dot(h);
        let mask = iso_h.cmp_gt(0.0);

        f64x4::splat(self.nugget).bitand(mask)
    }

    fn covariogram_simd(&self, h: ultraviolet::DVec3x4) -> f64x4 {
        self.nugget - self.variogram_simd(h)
    }
}
