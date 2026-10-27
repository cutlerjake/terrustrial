use super::FitOptions;
use crate::{Error, Result};

// Pairwise simplex coordinate descent. Each step solves its one-dimensional
// quadratic exactly. It needs no inverse or artificial positive-definite jitter,
// and permits repeated prototypes and zero regularization. A recomputed convex
// optimality gap, not a small step or iteration count, certifies convergence.
pub(super) fn solve(
    h: &[f64],
    q: &[f64],
    options: &FitOptions<'_>,
) -> Result<(Vec<f64>, usize, f64)> {
    let n = q.len();
    let scale = h.iter().chain(q).map(|x| x.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return Err(Error::Numerical {
            context: "high-order objective has no representable scale",
        });
    }
    let start = (0..n)
        .min_by(|&i, &j| {
            (0.5 * (h[i * n + i] / scale) - q[i] / scale)
                .total_cmp(&(0.5 * (h[j * n + j] / scale) - q[j] / scale))
        })
        .unwrap();
    let mut weights = vec![0.0; n];
    weights[start] = 1.0;
    let mut gradient = vec![0.0; n];
    let mut gap = f64::INFINITY;
    for iteration in 0..=options.maximum_iterations {
        options.cancellation.check()?;
        if iteration % 64 == 0 {
            recompute(h, q, &weights, &mut gradient, scale);
        }
        let mut receiver = (0..n)
            .min_by(|&i, &j| gradient[i].total_cmp(&gradient[j]))
            .unwrap();
        gap = weights
            .iter()
            .zip(&gradient)
            .map(|(a, g)| a * g)
            .sum::<f64>()
            - gradient[receiver];
        if gap <= options.tolerance || iteration == options.maximum_iterations {
            // Normalize roundoff before the final certificate, then recompute
            // gradients directly to avoid accepting incremental-update drift.
            let total = weights.iter().sum::<f64>();
            for x in &mut weights {
                *x /= total;
            }
            recompute(h, q, &weights, &mut gradient, scale);
            receiver = (0..n)
                .min_by(|&i, &j| gradient[i].total_cmp(&gradient[j]))
                .unwrap();
            gap = weights
                .iter()
                .zip(&gradient)
                .map(|(a, g)| a * g)
                .sum::<f64>()
                - gradient[receiver];
            if gap.is_finite() && gap <= options.tolerance {
                return Ok((weights, iteration, gap.max(0.0)));
            }
            if iteration == options.maximum_iterations {
                break;
            }
        }
        let donor = (0..n)
            .filter(|&i| weights[i] > 0.0)
            .max_by(|&i, &j| gradient[i].total_cmp(&gradient[j]))
            .unwrap();
        let curvature = h[receiver * n + receiver] / scale + h[donor * n + donor] / scale
            - 2.0 * (h[receiver * n + donor] / scale);
        if !curvature.is_finite() || curvature < -64.0 * f64::EPSILON {
            return Err(Error::Numerical {
                context: "high-order mixture nonconvex coordinate",
            });
        }
        let difference = gradient[donor] - gradient[receiver];
        let step = if curvature <= 0.0 {
            weights[donor]
        } else {
            (difference / curvature).min(weights[donor])
        };
        if !step.is_finite() || step <= 0.0 || receiver == donor {
            return Err(Error::Numerical {
                context: "high-order mixture coordinate stagnation",
            });
        }
        weights[receiver] += step;
        weights[donor] -= step;
        for i in 0..n {
            gradient[i] += step * (h[i * n + receiver] / scale - h[i * n + donor] / scale);
        }
    }
    Err(Error::Convergence {
        context: "high-order simplex mixture",
        iterations: options.maximum_iterations,
        residual: gap,
    })
}
fn recompute(h: &[f64], q: &[f64], weights: &[f64], gradient: &mut [f64], scale: f64) {
    for (row, (g, q)) in h
        .chunks_exact(weights.len())
        .zip(gradient.iter_mut().zip(q))
    {
        *g = row
            .iter()
            .zip(weights)
            .map(|(h, a)| (h / scale) * a)
            .sum::<f64>()
            - q / scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{random_stream, Cancellation, MemoryBudget};
    use rand::RngExt;

    // Independent reference: enumerate every simplex face, solve its augmented
    // equality-constrained system by pivoted elimination, retain KKT-feasible faces.
    fn reference(h: &[f64], q: &[f64]) -> Vec<f64> {
        let n = q.len();
        for mask in 1usize..1 << n {
            let active: Vec<_> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
            let k = active.len();
            let mut a = vec![vec![0.0; k + 2]; k + 1];
            for i in 0..k {
                for j in 0..k {
                    a[i][j] = h[active[i] * n + active[j]];
                }
                a[i][k] = 1.0;
                a[i][k + 1] = q[active[i]];
                a[k][i] = 1.0;
            }
            a[k][k + 1] = 1.0;
            for j in 0..=k {
                let pivot = (j..=k)
                    .max_by(|&i, &l| a[i][j].abs().total_cmp(&a[l][j].abs()))
                    .unwrap();
                a.swap(j, pivot);
                let divisor = a[j][j];
                assert!(divisor.abs() > 1e-14);
                for v in &mut a[j][j..] {
                    *v /= divisor;
                }
                let pivot_row = a[j].clone();
                for (i, row) in a.iter_mut().enumerate() {
                    if i != j {
                        let factor = row[j];
                        for (entry, pivot) in row[j..].iter_mut().zip(&pivot_row[j..]) {
                            *entry -= factor * pivot;
                        }
                    }
                }
            }
            let mut alpha = vec![0.0; n];
            for i in 0..k {
                alpha[active[i]] = a[i][k + 1];
            }
            if alpha.iter().any(|x| *x < -1e-10) {
                continue;
            }
            let lagrange = a[k][k + 1];
            if (0..n).all(|i| {
                (0..n).map(|j| h[i * n + j] * alpha[j]).sum::<f64>() - q[i] + lagrange >= -1e-9
            }) {
                return alpha;
            }
        }
        panic!("no feasible optimum");
    }
    #[test]
    fn coordinate_solver_agrees_with_exhaustive_face_solutions() -> Result<()> {
        let budget = MemoryBudget::new(1 << 20);
        let cancellation = Cancellation::default();
        let options = FitOptions {
            maximum_iterations: 100_000,
            tolerance: 1e-13,
            minimum_normalizer_fraction: 1e-10,
            budget: &budget,
            cancellation: &cancellation,
        };
        for case in 0..50 {
            let n = 2 + (case as usize % 6);
            let mut rng = random_stream(319, case, 0, 0);
            let factor: Vec<f64> = (0..n * n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let h: Vec<f64> = (0..n * n)
                .map(|index| {
                    let i = index / n;
                    let j = index % n;
                    (0..n)
                        .map(|k| factor[k * n + i] * factor[k * n + j])
                        .sum::<f64>()
                        + if i == j { 0.02 } else { 0.0 }
                })
                .collect();
            let q: Vec<f64> = (0..n).map(|_| rng.random_range(-2.0..2.0)).collect();
            let expected = reference(&h, &q);
            let (actual, _, gap) = solve(&h, &q, &options)?;
            assert!(gap <= options.tolerance);
            for (a, b) in actual.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-9, "{case}: {actual:?} != {expected:?}");
            }
            // Scaling the objective cannot change the solution. Exercise very
            // small/large but representable magnitudes without overflowing steps.
            for scale in [1e-200, 1e200] {
                let (large, _, _) = solve(
                    &h.iter().map(|x| x * scale).collect::<Vec<_>>(),
                    &q.iter().map(|x| x * scale).collect::<Vec<_>>(),
                    &options,
                )?;
                for (a, b) in large.iter().zip(&expected) {
                    assert!((a - b).abs() < 1e-9);
                }
            }
        }
        Ok(())
    }
}
