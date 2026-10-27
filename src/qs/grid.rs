//! Regular 3D grid abstraction used throughout the QS module.
//!
//! Grids are stored in row-major / C order over (z, y, x): `index = (z * ny +
//! y) * nx + x`, so consecutive `x` values are contiguous in memory. All FFT
//! shapes are derived from these dimensions.

use std::ops::{Index, IndexMut};

/// 3D regular grid with cell-centred values of type `T`.
#[derive(Clone, Debug)]
pub struct Grid3<T> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub data: Vec<T>,
}

impl<T: Clone + Default> Grid3<T> {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let data = vec![T::default(); nx * ny * nz];
        Self { nx, ny, nz, data }
    }
}

impl<T: Clone> Grid3<T> {
    /// Allocate a grid filled with `value`.
    pub fn filled(nx: usize, ny: usize, nz: usize, value: T) -> Self {
        let data = vec![value; nx * ny * nz];
        Self { nx, ny, nz, data }
    }

    /// Construct from existing row-major data. Panics if `data.len() != nx*ny*nz`.
    pub fn from_data(nx: usize, ny: usize, nz: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), nx * ny * nz, "data length mismatch");
        Self { nx, ny, nz, data }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Linear index for cell `(x, y, z)`. Caller must ensure indices are in
    /// range.
    #[inline]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.ny + y) * self.nx + x
    }

    /// Inverse of `idx`: turns a linear index back into `(x, y, z)`.
    #[inline]
    pub fn coords(&self, idx: usize) -> (usize, usize, usize) {
        let x = idx % self.nx;
        let yz = idx / self.nx;
        let y = yz % self.ny;
        let z = yz / self.ny;
        (x, y, z)
    }

    #[inline]
    pub fn in_bounds(&self, x: i64, y: i64, z: i64) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.nx
            && (y as usize) < self.ny
            && (z as usize) < self.nz
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> &T {
        &self.data[self.idx(x, y, z)]
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut T {
        let i = self.idx(x, y, z);
        &mut self.data[i]
    }
}

impl<T> Index<(usize, usize, usize)> for Grid3<T> {
    type Output = T;
    fn index(&self, (x, y, z): (usize, usize, usize)) -> &T {
        let i = (z * self.ny + y) * self.nx + x;
        &self.data[i]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for Grid3<T> {
    fn index_mut(&mut self, (x, y, z): (usize, usize, usize)) -> &mut T {
        let i = (z * self.ny + y) * self.nx + x;
        &mut self.data[i]
    }
}

/// Smallest power of two ≥ n. Used to pad FFT shapes for efficiency.
#[inline]
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idx_round_trip() {
        let g: Grid3<f64> = Grid3::new(7, 5, 3);
        for z in 0..3 {
            for y in 0..5 {
                for x in 0..7 {
                    let i = g.idx(x, y, z);
                    let (xx, yy, zz) = g.coords(i);
                    assert_eq!((xx, yy, zz), (x, y, z));
                }
            }
        }
    }

    #[test]
    fn next_pow2_is_correct() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(7), 8);
        assert_eq!(next_pow2(8), 8);
        assert_eq!(next_pow2(9), 16);
    }
}
