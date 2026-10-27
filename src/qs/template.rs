//! QS template: lag offsets in integer grid steps, with per-lag weights.
//!
//! Lags are signed integer triples `(dx, dy, dz)`. The template's *footprint*
//! is the bounding box of all its lags; this is what determines the FFT
//! padding required when computing mismatch fields.

/// A single template lag with its weight.
#[derive(Clone, Copy, Debug)]
pub struct LagWeight {
    pub dx: i32,
    pub dy: i32,
    pub dz: i32,
    pub weight: f64,
}

/// A spatial template for QS. Lags are anisotropic integer steps. Weights
/// allow distant lags to count for less than near ones; uniform weights are
/// fine for most use cases.
#[derive(Clone, Debug)]
pub struct QsTemplate {
    pub lags: Vec<LagWeight>,
}

impl QsTemplate {
    pub fn new(lags: Vec<LagWeight>) -> Self {
        Self { lags }
    }

    /// Build an anisotropic template covering `[-rx..=rx] x [-ry..=ry] x
    /// [-rz..=rz]` (excluding the origin). Weights decay with normalized
    /// distance so distant lags are less influential.
    pub fn ellipsoidal(rx: i32, ry: i32, rz: i32) -> Self {
        let mut lags = Vec::new();
        for dz in -rz..=rz {
            for dy in -ry..=ry {
                for dx in -rx..=rx {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let fx = if rx == 0 { 0.0 } else { dx as f64 / rx as f64 };
                    let fy = if ry == 0 { 0.0 } else { dy as f64 / ry as f64 };
                    let fz = if rz == 0 { 0.0 } else { dz as f64 / rz as f64 };
                    let d2 = fx * fx + fy * fy + fz * fz;
                    if d2 > 1.0 {
                        continue;
                    }
                    // Linear decay; tune later if needed.
                    let weight = (1.0 - d2.sqrt()).max(1e-3);
                    lags.push(LagWeight { dx, dy, dz, weight });
                }
            }
        }
        Self { lags }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.lags.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lags.is_empty()
    }

    /// Bounding-box extents of all lags (positive integers).
    pub fn extents(&self) -> (i32, i32, i32) {
        let (mut ex, mut ey, mut ez) = (0i32, 0i32, 0i32);
        for l in self.lags.iter() {
            ex = ex.max(l.dx.abs());
            ey = ey.max(l.dy.abs());
            ez = ez.max(l.dz.abs());
        }
        (ex, ey, ez)
    }

    /// Return a template with every lag scaled by an integer factor `s`
    /// along each axis. Used by multi-grid simulation: at coarser grid
    /// levels the template covers physically the same neighbourhood but at
    /// the coarser stride.
    pub fn scaled(&self, sx: i32, sy: i32, sz: i32) -> Self {
        let lags = self
            .lags
            .iter()
            .map(|l| LagWeight {
                dx: l.dx * sx,
                dy: l.dy * sy,
                dz: l.dz * sz,
                weight: l.weight,
            })
            .collect();
        Self { lags }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ellipsoidal_excludes_origin() {
        let t = QsTemplate::ellipsoidal(2, 2, 1);
        for l in t.lags.iter() {
            assert!(!(l.dx == 0 && l.dy == 0 && l.dz == 0));
        }
    }

    #[test]
    fn extents_match_radius() {
        let t = QsTemplate::ellipsoidal(3, 2, 1);
        assert_eq!(t.extents(), (3, 2, 1));
    }
}
