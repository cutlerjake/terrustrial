//! Validate the benchmark control independently; it is not the production solver.
#[path = "../examples/support/tiled_solve.rs"]
mod backend;
use backend::*;
use faer::Mat;

#[test]
fn solve_residuals_and_zero_allocations_across_tile_boundaries() {
    for n in [
        0, 1, 4, 7, 16, 31, 32, 33, 63, 64, 65, 96, 128, 129, 256, 512,
    ] {
        let factor = Mat::from_fn(n, n, |i, j| {
            if i == j {
                1. + 0.1 * (i as f64).cos()
            } else if i > j {
                0.03 * ((i * 7 + j) as f64).sin()
            } else {
                0.
            }
        });
        for columns in [0, 1, 2, 4, 5, 8, 17] {
            let expected = Mat::from_fn(n, columns, |i, j| ((i * 3 + j) as f64).sin());
            // Independent scalar products construct B=L L^T X from a
            // known solution, without sharing the solver's block arithmetic.
            let transposed = Mat::from_fn(n, columns, |i, j| {
                (i..n)
                    .map(|k| factor[(k, i)] * expected[(k, j)])
                    .sum::<f64>()
            });
            let input = Mat::from_fn(n, columns, |i, j| {
                (0..=i)
                    .map(|k| factor[(i, k)] * transposed[(k, j)])
                    .sum::<f64>()
            });
            let mut solved = input.clone();
            let allocations = allocation_counter::measure(|| {
                for _ in 0..2 {
                    solved.as_mut().copy_from(input.as_ref());
                    solve_spd(factor.as_ref(), solved.as_mut());
                }
            });
            assert_eq!(allocations.count_total, 0, "n={n}, columns={columns}");
            for j in 0..columns {
                for i in 0..n {
                    assert!(
                        (solved[(i, j)] - expected[(i, j)]).abs() < 1e-10,
                        "n={n}, columns={columns}, i={i}, j={j}"
                    );
                }
            }
        }
    }
}
