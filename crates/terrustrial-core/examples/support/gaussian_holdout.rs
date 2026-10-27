//! Value-independent geographic split and training-only normalization adapter.
use terrustrial_core::{
    grid::{Layout, Source},
    Error, Result,
};
#[derive(Clone, Copy)]
pub struct Split {
    pub training_end: u64,
    pub evaluation_start: u64,
}
impl Split {
    pub fn training(self, layout: Layout, cell: u64) -> bool {
        cell % layout.dimensions()[0] < self.training_end
    }
    pub fn evaluation(self, layout: Layout, cell: u64) -> bool {
        cell % layout.dimensions()[0] >= self.evaluation_start
    }
    pub fn known(self, layout: Layout, cell: u64, variable: usize) -> bool {
        self.evaluation(layout, cell) && hash(cell * 2 + variable as u64).is_multiple_of(11)
    }
    pub fn target(self, layout: Layout, cell: u64) -> bool {
        self.evaluation(layout, cell) && !(0..2).any(|v| self.known(layout, cell, v))
    }
}
// Fixed SplitMix64 permutation. Used only for value-independent inclusion and
// ranking; this is not a claim of cryptographic randomness or sample uncertainty.
pub fn hash(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
pub fn coordinates(layout: Layout, cell: u64) -> [f64; 3] {
    let [nx, ny, _] = layout.dimensions();
    [
        (cell % nx) as f64,
        ((cell / nx) % ny) as f64,
        (cell / (nx * ny)) as f64,
    ]
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moment {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
}
impl Moment {
    pub fn add(&mut self, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::InvalidInput("nonfinite training value".into()));
        }
        self.count += 1;
        let d = value - self.mean;
        self.mean += d / self.count as f64;
        self.m2 += d * (value - self.mean);
        Ok(())
    }
    pub fn scale(self) -> Result<f64> {
        let s = (self.m2 / self.count as f64).sqrt();
        if self.count < 2 || !s.is_finite() || s <= 0. {
            return Err(Error::InvalidInput("unidentified training variance".into()));
        }
        Ok(s)
    }
}
pub fn moments(source: &mut impl Source, split: Split) -> Result<[Moment; 2]> {
    let mut result = [Moment::default(); 2];
    let mut buffer = vec![None; 8192];
    let layout = source.layout();
    for (v, moment) in result.iter_mut().enumerate() {
        let mut first = 0;
        while first < layout.cells() {
            let n = (layout.cells() - first).min(buffer.len() as u64) as usize;
            source.read(v, first, &mut buffer[..n])?;
            for (i, value) in buffer[..n].iter().enumerate() {
                if split.training(layout, first + i as u64) {
                    moment
                        .add(value.ok_or_else(|| {
                            Error::InvalidInput("missing training value".into())
                        })?)?;
                }
            }
            first += n as u64;
        }
        moment.scale()?;
    }
    Ok(result)
}
pub struct Training<'a, S> {
    pub source: &'a mut S,
    pub split: Split,
    pub moments: [Moment; 2],
}
impl<S: Source> Source for Training<'_, S> {
    fn layout(&self) -> Layout {
        self.source.layout()
    }
    fn variables(&self) -> usize {
        2
    }
    fn read(&mut self, v: usize, first: u64, out: &mut [Option<f64>]) -> Result<()> {
        self.source.read(v, first, out)?;
        let s = self.moments[v].scale()?;
        for (i, value) in out.iter_mut().enumerate() {
            *value = if self.split.training(self.layout(), first + i as u64) {
                value.map(|x| (x - self.moments[v].mean) / s)
            } else {
                None
            };
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture {
        outside: f64,
    }
    impl Source for Fixture {
        fn layout(&self) -> Layout {
            Layout::new([9, 4, 3]).unwrap()
        }
        fn variables(&self) -> usize {
            2
        }
        fn read(&mut self, v: usize, first: u64, out: &mut [Option<f64>]) -> Result<()> {
            for (i, o) in out.iter_mut().enumerate() {
                let c = first + i as u64;
                *o = Some(if c % 9 < 5 {
                    (c + v as u64) as f64
                } else {
                    self.outside
                });
            }
            Ok(())
        }
    }
    #[test]
    fn arbitrary_evaluation_values_cannot_change_training_moments_or_variogram_input() {
        let split = Split {
            training_end: 5,
            evaluation_start: 7,
        };
        let mut reference = None;
        for outside in [-1e200, 7., 1e200] {
            let mut source = Fixture { outside };
            let stats = moments(&mut source, split).unwrap();
            assert_eq!(stats[0].count, 60);
            let mut training = Training {
                source: &mut source,
                split,
                moments: stats,
            };
            let mut values = vec![None; 108];
            training.read(0, 0, &mut values).unwrap();
            for (cell, value) in values.iter().enumerate() {
                assert_eq!(value.is_some(), cell % 9 < 5);
            }
            if let Some((s, v)) = &reference {
                assert_eq!(&stats, s);
                assert_eq!(&values, v);
            } else {
                reference = Some((stats, values));
            }
        }
    }
    #[test]
    fn target_cells_are_disjoint_from_both_known_attributes_and_training() {
        let layout = Layout::new([150, 200, 200]).unwrap();
        let split = Split {
            training_end: 100,
            evaluation_start: 120,
        };
        let mut heterogeneous = 0;
        for cell in (0..layout.cells()).step_by(37) {
            assert!(!(split.training(layout, cell) && split.evaluation(layout, cell)));
            if split.target(layout, cell) {
                assert!(!(0..2).any(|v| split.known(layout, cell, v)));
            }
            if split.known(layout, cell, 0) != split.known(layout, cell, 1) {
                heterogeneous += 1;
            }
            let p = coordinates(layout, cell).map(|x| x as u64);
            assert_eq!(layout.index(p).unwrap(), cell);
        }
        assert!(heterogeneous > 1000);
    }
}
