use std::ops::DivAssign;
use ultraviolet::DMat3;
use ultraviolet::DMat3x4;
use ultraviolet::DRotor3;
use ultraviolet::DVec3;
use ultraviolet::DVec3x4;
use wide::f64x4;
use wide::CmpEq;
use wide::CmpGt;

use super::VariogramModel;

#[derive(Clone, Copy, Debug)]
pub struct SphericalVariogram {
    pub range: DVec3,
    pub sill: f64,
    pub rotation: DMat3,
}

impl Default for SphericalVariogram {
    fn default() -> Self {
        Self {
            range: DVec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
            sill: 1.0,
            rotation: DRotor3::identity().into_matrix(),
        }
    }
}

impl SphericalVariogram {
    pub fn new(range: DVec3, sill: f64, rotation: DRotor3) -> Self {
        Self {
            range,
            sill,
            rotation: rotation.reversed().into_matrix(),
        }
    }
}

impl VariogramModel for SphericalVariogram {
    #[inline(always)]
    fn c_0(&self) -> f64 {
        self.sill
    }

    #[inline(always)]
    fn variogram(&self, mut h: DVec3) -> f64 {
        h = self.rotation * h;

        h.div_assign(self.range);
        let iso_h = h.mag();

        if iso_h == 0.0 {
            0.0
        } else if iso_h < 1.0 {
            self.sill * (1.5 * iso_h - 0.5 * iso_h * iso_h * iso_h)
        } else {
            self.sill
        }
    }

    #[inline(always)]
    fn covariogram(&self, h: DVec3) -> f64 {
        self.sill - self.variogram(h)
    }

    #[inline(always)]
    fn set_orientation(&mut self, orientation: DRotor3) {
        self.rotation = orientation.reversed().into_matrix();
    }

    fn variogram_simd(&self, mut h: ultraviolet::DVec3x4) -> wide::f64x4 {
        let rotation = DMat3x4 {
            cols: [
                DVec3x4::splat(self.rotation.cols[0]),
                DVec3x4::splat(self.rotation.cols[1]),
                DVec3x4::splat(self.rotation.cols[2]),
            ],
        };

        h = rotation * h;

        h.div_assign(DVec3x4::splat(self.range));

        let iso_h = h.mag();

        let mask_low = iso_h.cmp_eq(0.0);
        let mask_high = iso_h.cmp_gt(1.0);
        let mask_middle = !mask_low & !mask_high;

        let low_vals = f64x4::splat(0.0) & mask_low;
        let high_vals = f64x4::splat(self.sill) & mask_high;
        let middle_vals =
            (f64x4::splat(self.sill) * (1.5 * iso_h - 0.5 * iso_h * iso_h * iso_h)) & mask_middle;

        low_vals | high_vals | middle_vals
    }

    fn covariogram_simd(&self, h: ultraviolet::DVec3x4) -> wide::f64x4 {
        self.sill - self.variogram_simd(h)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn spherical_vgram_var() {
        let sill = 1.0;
        let range = 300.0;

        let vgram =
            SphericalVariogram::new(DVec3::new(range, range, range), sill, DRotor3::identity());

        let dists = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(46.1, 0.0, 0.0),
            DVec3::new(72.8, 0.0, 0.0),
            DVec3::new(68.01, 0.0, 0.0),
        ];

        println!("Variance");
        for d in dists.iter() {
            println!("dist: {} v: {}", d.mag(), vgram.variogram(*d));
        }

        println!("Covariance");
        for d in dists.iter() {
            println!("dist: {} v: {}", d.mag(), vgram.covariogram(*d));
        }
    }
}
