//! Exact joint-mask counts for every regular-grid lag using integer transforms.
//!
//! This is the count component of accelerated variography, not a floating-point
//! semivariance estimator. A cell is eligible only if both selected attributes
//! exist. Zero lag counts eligible cells, including self-pairs. Nonzero h and -h
//! have the same count and must not both be pooled as distinct unordered pairs.
use super::super::indexed::buffer;
use crate::{
    error::{checked_len, invalid},
    execution::{Cancellation, MemoryBudget, Reservation},
    grid::{Layout, Source},
    Error, Result,
};
use concrete_ntt::prime32::Plan;

/// Smallest axis-aligned source-grid box containing all jointly eligible cells.
/// This is descriptive metadata; the mask may have holes inside the box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    pub origin: [u64; 3],
    pub dimensions: [u64; 3],
}
impl Extent {
    fn cells(self) -> u64 {
        self.dimensions.iter().product()
    }
}

// Bounded adapter from cropped x-fast indices back to original source indices.
// Coalescing whole rows/planes preserves pair order without per-read allocation.
pub(super) struct Cropped<'a, S: Source + ?Sized> {
    source: &'a mut S,
    extent: Extent,
    layout: Layout,
    original: Layout,
    whole: bool,
}
impl<'a, S: Source + ?Sized> Cropped<'a, S> {
    pub(super) fn new(source: &'a mut S, extent: Extent) -> Result<Self> {
        let original = source.layout();
        let dimensions = original.dimensions();
        if (0..3).any(|a| {
            extent.origin[a]
                .checked_add(extent.dimensions[a])
                .is_none_or(|end| end > dimensions[a])
        }) {
            return Err(invalid("eligible extent outside source grid"));
        }
        Ok(Self {
            source,
            extent,
            layout: Layout::new(extent.dimensions)?,
            original,
            whole: extent.dimensions == dimensions,
        })
    }
}
impl<S: Source + ?Sized> Source for Cropped<'_, S> {
    fn layout(&self) -> Layout {
        self.layout
    }
    fn variables(&self) -> usize {
        self.source.variables()
    }
    fn read(&mut self, variable: usize, mut first: u64, mut out: &mut [Option<f64>]) -> Result<()> {
        if variable >= self.variables()
            || first > self.layout.cells()
            || out.len() as u64 > self.layout.cells() - first
        {
            return Err(invalid("cropped grid read range"));
        }
        if out.is_empty() {
            return Ok(());
        }
        if self.whole {
            return self.source.read(variable, first, out);
        }
        let [nx, ny, _] = self.extent.dimensions;
        let original = self.original;
        let full_x = nx == original.dimensions()[0];
        let full_y = ny == original.dimensions()[1];
        while !out.is_empty() {
            let point = [first % nx, first / nx % ny, first / nx / ny];
            let contiguous = if full_x && full_y {
                self.layout.cells() - first
            } else if full_x {
                (ny - point[1]) * nx - point[0]
            } else {
                nx - point[0]
            };
            let n = contiguous.min(out.len() as u64) as usize;
            let global =
                original.index(std::array::from_fn(|a| point[a] + self.extent.origin[a]))?;
            self.source.read(variable, global, &mut out[..n])?;
            out = &mut out[n..];
            first += n as u64;
        }
        Ok(())
    }
}

pub(super) struct Mask {
    layout: Layout,
    pub(super) present: u64,
    pub(super) extent: Option<Extent>,
    reads: u128,
}
impl Mask {
    pub(super) fn transform_length(&self) -> Result<Option<usize>> {
        if self.complete() {
            Ok(None)
        } else {
            Ok(Some(self.geometry()?.length))
        }
    }
    fn complete(&self) -> bool {
        self.extent.is_none_or(|e| e.cells() == self.present)
    }
    pub(super) fn geometry(&self) -> Result<Geometry> {
        Geometry::new(Layout::new(
            self.extent
                .ok_or_else(|| invalid("empty mask geometry"))?
                .dimensions,
        )?)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Geometry {
    layout: Layout,
    padded: [usize; 3],
    pub(super) length: usize,
    origin: usize,
}
impl Geometry {
    pub(super) fn new(layout: Layout) -> Result<Self> {
        let mut padded = [1; 3];
        for (pad, dimension) in padded.iter_mut().zip(layout.dimensions()) {
            *pad = usize::try_from(dimension)
                .ok()
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_sub(1))
                .and_then(usize::checked_next_power_of_two)
                .ok_or(Error::SizeOverflow)?;
        }
        let length = checked_len(&padded)?.max(32);
        // 2013265921 = 15*2^27+1, 3221225473 = 3*2^30+1.
        // Negacyclic transforms need a root of order 2*length.
        if length > 1 << 29 {
            return Err(invalid(
                "masked count transform exceeds supported length; use requested-lag streaming",
            ));
        }
        let mut result = Self {
            layout,
            padded,
            length,
            origin: 0,
        };
        result.origin = result.index(layout.cells() - 1);
        // Each occupied axis lies in the lower half of its padded axis, except
        // singleton axes. Reversing at origin therefore gives a linear product
        // of degree 2*origin < length: no negacyclic wrap or sign correction.
        debug_assert!(2 * result.origin < length);
        Ok(result)
    }
    pub(super) fn index(&self, index: u64) -> usize {
        let [nx, ny, _] = self.layout.dimensions();
        let x = index % nx;
        let y = index / nx % ny;
        let z = index / nx / ny;
        x as usize + self.padded[0] * (y as usize + self.padded[1] * z as usize)
    }
    pub(super) fn cyclic_lag(&self, offset: [i64; 3]) -> usize {
        (offset[0] as i128
            + self.padded[0] as i128
                * (offset[1] as i128 + self.padded[1] as i128 * offset[2] as i128))
            .rem_euclid(self.length as i128) as usize
    }
    fn lag(&self, offset: [i64; 3]) -> usize {
        (self.origin as i128
            + offset[0] as i128
            + self.padded[0] as i128
                * (offset[1] as i128 + self.padded[1] as i128 * offset[2] as i128)) as usize
    }
}
struct Backend {
    plan: Plan,
    left: Vec<u32>,
    right: Vec<u32>,
    _reservation: Reservation,
}
impl Backend {
    fn new(length: usize, budget: &MemoryBudget) -> Result<Self> {
        let (prime, arrays) = if length <= 1 << 26 {
            (2_013_265_921, 6)
        } else {
            (3_221_225_473, 4)
        };
        // Two coefficient arrays and the backend's two or four twiddle arrays.
        // Plan metadata, alignment and allocator overhead are not charged here.
        let reservation = budget.reserve(
            u64::try_from(checked_len(&[length, arrays, 4])?).map_err(|_| Error::SizeOverflow)?,
        )?;
        let left = buffer(length, || 0_u32)?;
        let right = buffer(length, || 0_u32)?;
        let plan =
            Plan::try_new(length, prime).ok_or_else(|| invalid("unsupported count transform"))?;
        Ok(Self {
            plan,
            left,
            right,
            _reservation: reservation,
        })
    }
}

/// Reusable tiled source buffers and a lazily constructed integer-transform plan.
/// Complete rectangular and wholly missing masks need no transform. Masks with
/// holes use O(padded eligible-extent cells) resident storage, admitted before allocation. The backend's
/// twiddle allocations are charged but not fallible; no strict RSS cap is claimed.
/// Cancellation is checked between backend calls; one transform cannot be
/// interrupted internally. Reusing a shape/transform length retains its buffers.
pub struct Workspace {
    first: Vec<Option<f64>>,
    second: Vec<Option<f64>>,
    backend: Option<Backend>,
    budget: MemoryBudget,
    _reservation: Reservation,
}
impl Workspace {
    pub fn new(tile_cells: usize, budget: &MemoryBudget) -> Result<Self> {
        if tile_cells == 0 {
            return Err(invalid("positive mask-count tile size required"));
        }
        let reservation = budget.reserve(
            u64::try_from(checked_len(&[
                tile_cells,
                2,
                std::mem::size_of::<Option<f64>>(),
            ])?)
            .map_err(|_| Error::SizeOverflow)?,
        )?;
        Ok(Self {
            first: buffer(tile_cells, || None)?,
            second: buffer(tile_cells, || None)?,
            backend: None,
            budget: budget.clone(),
            _reservation: reservation,
        })
    }

    /// Sources must remain immutable throughout preparation and queries. The first
    /// scan covers the entire source; a second scan visits only the eligible box
    /// when a transform is needed. Auto counts read one attribute, joint counts
    /// read both. Every selected present
    /// value is checked for finiteness, even when its partner is missing.
    /// Failures expose no partial counts, and the workspace remains reusable.
    pub fn compute<'w>(
        &'w mut self,
        source: &mut (impl Source + ?Sized),
        variables: [usize; 2],
        cancel: &Cancellation,
    ) -> Result<Counts<'w>> {
        let mask = self.prepare(source, variables, cancel, |_, _| {})?;
        self.finish(source, variables, cancel, mask)
    }

    // Share the mandatory first scan with clients that also need value ranges.
    pub(super) fn prepare(
        &mut self,
        source: &mut (impl Source + ?Sized),
        variables: [usize; 2],
        cancel: &Cancellation,
        mut visit: impl FnMut(f64, f64),
    ) -> Result<Mask> {
        cancel.check()?;
        if variables.iter().any(|v| *v >= source.variables()) {
            return Err(invalid("mask-count variable"));
        }
        let layout = source.layout();
        let [nx, ny, _] = layout.dimensions();
        let mut present = 0_u64;
        let mut reads = 0_u128;
        let mut origin = [u64::MAX; 3];
        let mut upper = [0; 3];
        let mut first_present = 0;
        let mut last_present = 0;
        scan(
            source,
            variables,
            &mut self.first,
            &mut self.second,
            cancel,
            |i, a, b| {
                if let (Some(a), Some(b)) = (a, b) {
                    present += 1;
                    if present == 1 {
                        first_present = i;
                    }
                    last_present = i;
                    // X/Y bounds cannot expand once they span their whole axes.
                    // Z is monotone in an x-fast scan, so only first/last indices
                    // need division. Avoid per-cell coordinate decoding after
                    // full X/Y support has been established.
                    if origin[0] != 0 || upper[0] != nx {
                        let x = i % nx;
                        origin[0] = origin[0].min(x);
                        upper[0] = upper[0].max(x + 1);
                    }
                    if origin[1] != 0 || upper[1] != ny {
                        let y = i / nx % ny;
                        origin[1] = origin[1].min(y);
                        upper[1] = upper[1].max(y + 1);
                    }
                    visit(a, b);
                }
                Ok(())
            },
            &mut reads,
        )?;
        let extent = (present > 0).then(|| {
            origin[2] = first_present / nx / ny;
            upper[2] = last_present / nx / ny + 1;
            Extent {
                origin,
                dimensions: std::array::from_fn(|a| upper[a] - origin[a]),
            }
        });
        Ok(Mask {
            layout,
            present,
            extent,
            reads,
        })
    }

    // Release stale integer cache before the floating backend admits a new plan.
    // Paired clients call this before finish to coordinate both cache lifetimes.
    pub(super) fn retain_length(&mut self, length: Option<usize>) {
        if self
            .backend
            .as_ref()
            .is_some_and(|b| Some(b.left.len()) != length)
        {
            self.backend = None;
        }
    }

    pub(super) fn finish<'w>(
        &'w mut self,
        source: &mut (impl Source + ?Sized),
        variables: [usize; 2],
        cancel: &Cancellation,
        mask: Mask,
    ) -> Result<Counts<'w>> {
        cancel.check()?;
        let Mask {
            layout,
            present,
            extent,
            reads: mut source_values_read,
        } = mask;
        if extent.is_none_or(|e| e.cells() == present) {
            self.retain_length(None);
            return Ok(Counts {
                layout,
                present,
                extent,
                source_values_read,
                transformed: None,
            });
        }
        let mut cropped = Cropped::new(source, extent.unwrap())?;
        let geometry = Geometry::new(cropped.layout())?;
        if self
            .backend
            .as_ref()
            .is_none_or(|b| b.left.len() != geometry.length)
        {
            // Release an obsolete plan before admitting its replacement. An
            // error may discard cached scratch but cannot expose stale results.
            self.backend = None;
            self.backend = Some(Backend::new(geometry.length, &self.budget)?);
        }
        let backend = self.backend.as_mut().unwrap();
        debug_assert!(present < u64::from(backend.plan.modulus()));
        backend.left.fill(0);
        backend.right.fill(0);
        let mut second_present = 0;
        scan(
            &mut cropped,
            variables,
            &mut self.first,
            &mut self.second,
            cancel,
            |i, a, b| {
                if a.is_some() && b.is_some() {
                    second_present += 1;
                    let index = geometry.index(i);
                    backend.left[index] = 1;
                    backend.right[geometry.origin - index] = 1;
                }
                Ok(())
            },
            &mut source_values_read,
        )?;
        if second_present != present {
            return Err(invalid("grid mask changed between count passes"));
        }
        cancel.check()?;
        rayon::join(
            || backend.plan.fwd(&mut backend.left),
            || backend.plan.fwd(&mut backend.right),
        );
        cancel.check()?;
        backend
            .plan
            .mul_assign_normalize(&mut backend.left, &backend.right);
        cancel.check()?;
        backend.plan.inv(&mut backend.left);
        cancel.check()?;
        if u64::from(backend.left[geometry.origin]) != present {
            return Err(Error::Numerical {
                context: "integer count transform origin mismatch",
            });
        }
        Ok(Counts {
            layout,
            present,
            source_values_read,
            extent,
            transformed: Some((geometry, &backend.left)),
        })
    }
}

pub(super) fn scan(
    source: &mut (impl Source + ?Sized),
    variables: [usize; 2],
    first: &mut [Option<f64>],
    second: &mut [Option<f64>],
    cancel: &Cancellation,
    mut accept: impl FnMut(u64, Option<f64>, Option<f64>) -> Result<()>,
    reads: &mut u128,
) -> Result<()> {
    let cells = source.layout().cells();
    let mut start = 0;
    while start < cells {
        cancel.check()?;
        let n = (cells - start).min(first.len() as u64) as usize;
        source.read(variables[0], start, &mut first[..n])?;
        *reads += n as u128;
        if variables[0] != variables[1] {
            source.read(variables[1], start, &mut second[..n])?;
            *reads += n as u128;
        }
        for i in 0..n {
            let a = first[i];
            let b = if variables[0] == variables[1] {
                a
            } else {
                second[i]
            };
            if a.is_some_and(|v| !v.is_finite()) || b.is_some_and(|v| !v.is_finite()) {
                return Err(invalid("nonfinite mask-count attribute"));
            }
            accept(start + i as u64, a, b)?;
        }
        start += n as u64;
    }
    cancel.check()
}

/// Borrowed counts for all signed grid offsets. Counts are exact integers, not
/// rounded FFT estimates. Zero lag includes one self-pair per eligible cell;
/// offsets outside the finite grid have zero pairs and never wrap periodically.
pub struct Counts<'a> {
    layout: Layout,
    present: u64,
    pub source_values_read: u128,
    extent: Option<Extent>,
    transformed: Option<(Geometry, &'a [u32])>,
}
impl Counts<'_> {
    pub fn layout(&self) -> Layout {
        self.layout
    }
    pub fn eligible_cells(&self) -> u64 {
        self.present
    }
    pub fn eligible_extent(&self) -> Option<Extent> {
        self.extent
    }
    pub fn uses_transform(&self) -> bool {
        self.transformed.is_some()
    }
    pub fn pairs(&self, offset: [i64; 3]) -> u64 {
        let Some(extent) = self.extent else { return 0 };
        let dimensions = extent.dimensions;
        if (0..3).any(|a| offset[a].unsigned_abs() >= dimensions[a]) {
            return 0;
        }
        if let Some((geometry, values)) = &self.transformed {
            u64::from(values[geometry.lag(offset)])
        } else {
            (0..3)
                .map(|a| dimensions[a] - offset[a].unsigned_abs())
                .product()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cropped_reads_coalesce_contiguous_rows_and_preserve_u64_indices() -> Result<()> {
        struct Recorded {
            layout: Layout,
            calls: Vec<(u64, usize)>,
        }
        impl Source for Recorded {
            fn layout(&self) -> Layout {
                self.layout
            }
            fn variables(&self) -> usize {
                1
            }
            fn read(&mut self, _: usize, first: u64, out: &mut [Option<f64>]) -> Result<()> {
                self.calls.push((first, out.len()));
                out.fill(Some(1.));
                Ok(())
            }
        }
        for (shape, extent, expected) in [
            (
                [3, 2, 2],
                Extent {
                    origin: [0; 3],
                    dimensions: [3, 2, 2],
                },
                vec![(0, 12)],
            ),
            (
                [7, 5, 3],
                Extent {
                    origin: [2, 1, 1],
                    dimensions: [3, 2, 2],
                },
                vec![(44, 3), (51, 3), (79, 3), (86, 3)],
            ),
            (
                [7, 5, 3],
                Extent {
                    origin: [0, 1, 1],
                    dimensions: [7, 2, 2],
                },
                vec![(42, 14), (77, 14)],
            ),
            (
                [7, 5, 3],
                Extent {
                    origin: [0, 0, 1],
                    dimensions: [7, 5, 2],
                },
                vec![(35, 70)],
            ),
            (
                [u64::MAX, 1, 1],
                Extent {
                    origin: [u64::MAX - 5, 0, 0],
                    dimensions: [4, 1, 1],
                },
                vec![(u64::MAX - 5, 4)],
            ),
        ] {
            let mut source = Recorded {
                layout: Layout::new(shape)?,
                calls: vec![],
            };
            {
                let mut crop = Cropped::new(&mut source, extent)?;
                let mut values = vec![None; extent.cells() as usize];
                crop.read(0, 0, &mut values)?;
                assert!(values.iter().all(|v| *v == Some(1.)));
                crop.read(0, extent.cells(), &mut [])?;
                assert!(crop.read(0, extent.cells(), &mut [None]).is_err());
                assert!(crop.read(1, 0, &mut []).is_err());
            }
            assert_eq!(source.calls, expected);
        }
        let mut source = Recorded {
            layout: Layout::new([u64::MAX, 1, 1])?,
            calls: vec![],
        };
        assert!(Cropped::new(
            &mut source,
            Extent {
                origin: [u64::MAX - 1, 0, 0],
                dimensions: [3, 1, 1]
            }
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn checked_geometry_and_wide_prime_convolution() {
        assert!(Geometry::new(Layout::new([u64::MAX, 1, 1]).unwrap()).is_err());
        assert!(Geometry::new(Layout::new([1 << 30, 1, 1]).unwrap()).is_err());
        let geometry = Geometry::new(Layout::new([1 << 27, 1, 1]).unwrap()).unwrap();
        assert_eq!(geometry.length, 1 << 28);
        assert_eq!(geometry.lag([-(1 << 27) + 1, 0, 0]), 0);
        let plan = Plan::try_new(32, 3_221_225_473).unwrap();
        let mut a = [0; 32];
        a[..16].fill(1);
        let mut b = a;
        plan.fwd(&mut a);
        plan.fwd(&mut b);
        plan.mul_assign_normalize(&mut a, &b);
        plan.inv(&mut a);
        for (i, value) in a.into_iter().enumerate() {
            assert_eq!(value, if i < 16 { i as u32 + 1 } else { 31 - i as u32 });
        }
    }
}
