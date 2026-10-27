//! Benchmark-only allocation-free control. Rejected for production because it
//! is slower than direct faer for wide/larger right-hand sides.
use faer::{MatMut, MatRef, Par};

pub(crate) fn solve_spd(factor: MatRef<'_, f64>, mut rhs: MatMut<'_, f64>) {
    use faer::linalg::{matmul::matmul, triangular_solve};
    use faer::Accum;
    // faer 0.24.4's general GEMM packs into internally allocated buffers above
    // 4096 products. Bound each update to 32*32*4, retaining its portable SIMD
    // small-matrix backend without allocations hidden behind empty solve scratch.
    // Fixed blocking depends only on logical dimensions, never worker/capacity.
    const ROWS: usize = 32;
    const COLS: usize = 4;
    let n = factor.nrows();
    assert_eq!(factor.ncols(), n);
    assert_eq!(rhs.nrows(), n);
    for col in (0..rhs.ncols()).step_by(COLS) {
        let width = COLS.min(rhs.ncols() - col);
        let mut block = rhs.as_mut().submatrix_mut(0, col, n, width);
        for start in (0..n).step_by(ROWS) {
            let rows = ROWS.min(n - start);
            let (previous, remaining) = block.as_mut().split_at_row_mut(start);
            let (mut current, _) = remaining.split_at_row_mut(rows);
            for k in (0..start).step_by(ROWS) {
                matmul(
                    current.as_mut(),
                    Accum::Add,
                    factor.submatrix(start, k, rows, ROWS),
                    previous.as_ref().submatrix(k, 0, ROWS, width),
                    -1.,
                    Par::Seq,
                );
            }
            triangular_solve::solve_lower_triangular_in_place(
                factor.submatrix(start, start, rows, rows),
                current,
                Par::Seq,
            );
        }
        for start in (0..n).step_by(ROWS).rev() {
            let rows = ROWS.min(n - start);
            let end = start + rows;
            let (through, next) = block.as_mut().split_at_row_mut(end);
            let mut current = through.submatrix_mut(start, 0, rows, width);
            for k in (end..n).step_by(ROWS) {
                let count = ROWS.min(n - k);
                matmul(
                    current.as_mut(),
                    Accum::Add,
                    factor.submatrix(k, start, count, rows).transpose(),
                    next.as_ref().submatrix(k - end, 0, count, width),
                    -1.,
                    Par::Seq,
                );
            }
            triangular_solve::solve_upper_triangular_in_place(
                factor.submatrix(start, start, rows, rows).transpose(),
                current,
                Par::Seq,
            );
        }
    }
}
