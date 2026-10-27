//! Scaled compensated statistics shared by exact variogram engines.
use crate::{Error, Result};
pub(super) fn half_increment_product(a: f64, b: f64, c: f64, d: f64) -> Result<f64> {
    if a == b || c == d {
        return Ok(0.0);
    }
    let (x, y) = (a - b, c - d);
    let value = if x.is_infinite() {
        (a * 0.5 - b * 0.5) * y
    } else if y.is_infinite() {
        x * (c * 0.5 - d * 0.5)
    } else {
        let product = x * y;
        if product.is_infinite() {
            (x * 0.5) * y
        } else {
            product * 0.5
        }
    };
    if !value.is_finite() {
        return Err(Error::Numerical {
            context: "variogram increment product overflow",
        });
    }
    Ok(value)
}
#[derive(Clone, Default)]
pub(super) struct ScaledMean {
    pub(super) count: u64,
    scale: f64,
    sum: f64,
    correction: f64,
}
impl ScaledMean {
    pub(super) fn add(&mut self, value: f64) -> Result<()> {
        self.count = self.count.checked_add(1).ok_or(Error::SizeOverflow)?;
        if value.abs() > self.scale {
            let factor = self.scale / value.abs();
            self.sum *= factor;
            self.correction *= factor;
            self.scale = value.abs();
        }
        if self.scale == 0.0 {
            return Ok(());
        }
        let value = value / self.scale;
        let total = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - total) + value
        } else {
            (value - total) + self.sum
        };
        self.sum = total;
        Ok(())
    }
    pub(super) fn value(&self) -> Option<f64> {
        (self.count > 0).then(|| {
            let sum = self.sum + self.correction;
            let mean = sum / self.count as f64;
            if mean.abs() < f64::MIN_POSITIVE {
                // A subnormal normalized mean can lose digits before scaling
                // back to physical units. Here |sum| < count*MIN_POSITIVE, so
                // sum*scale cannot overflow even at u64::MAX count and MAX scale.
                ((sum * self.scale) / self.count as f64).clamp(-self.scale, self.scale)
            } else {
                mean.clamp(-1.0, 1.0) * self.scale
            }
        })
    }
}

impl ScaledMean {
    // Merge fixed row blocks in input order, independently of the worker schedule.
    pub(super) fn merge(&mut self, other: &Self) -> Result<()> {
        self.count = self
            .count
            .checked_add(other.count)
            .ok_or(Error::SizeOverflow)?;
        let scale = self.scale.max(other.scale);
        if scale == 0.0 {
            return Ok(());
        }
        self.sum *= self.scale / scale;
        self.correction *= self.scale / scale;
        let factor = other.scale / scale;
        for value in [other.sum * factor, other.correction * factor] {
            let total = self.sum + value;
            self.correction += if self.sum.abs() >= value.abs() {
                (self.sum - total) + value
            } else {
                (value - total) + self.sum
            };
            self.sum = total;
        }
        self.scale = scale;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_scaled_blocks_preserves_counts_cancellation_and_finite_means() {
        let mut positive = ScaledMean::default();
        let mut negative = ScaledMean::default();
        for _ in 0..4096 {
            positive.add(f64::MAX).unwrap();
            negative.add(-f64::MAX).unwrap();
        }
        positive.merge(&negative).unwrap();
        assert_eq!(positive.count, 8192);
        assert_eq!(positive.value(), Some(0.));
        let mut small = ScaledMean::default();
        small.add(1.).unwrap();
        positive.merge(&small).unwrap();
        assert!((positive.value().unwrap() - 1. / 8193.).abs() < 1e-18);
        let mut overflow = ScaledMean {
            count: u64::MAX,
            ..ScaledMean::default()
        };
        assert!(matches!(overflow.merge(&small), Err(Error::SizeOverflow)));
        let mut empty = ScaledMean::default();
        empty.merge(&ScaledMean::default()).unwrap();
        assert_eq!(empty.value(), None);
        for _ in 0..1024 {
            empty.add(f64::MAX).unwrap();
        }
        let copy = empty.clone();
        empty.merge(&copy).unwrap();
        assert_eq!(empty.value(), Some(f64::MAX));
    }
}
