use super::*;

fn plan(n: usize, m: usize, capacity: usize, budget: &MemoryBudget) -> GaussianPlan {
    let mut weights = Mat::with_capacity(capacity, m);
    weights.resize_with(n, m, |i, j| {
        let sign = if (i + j) % 3 == 0 { -1.0 } else { 1.0 };
        sign * (((i * 7 + j * 11) % 29) as f64 + 1.0) / 32.0
    });
    GaussianPlan {
        weights,
        covariance: Mat::new(),
        factor: Mat::new(),
        joint: false,
        roundoff_corrections: 0,
        reservation: budget.reserve(0).unwrap(),
    }
}

#[test]
fn prediction_is_allocation_free_reproducible_and_matches_exact_dyadic_reference() {
    let budget = MemoryBudget::new(1 << 20);
    for n in [0, 1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 512] {
        for m in [0, 1, 2, 7, 8, 9, 16, 64, 129] {
            let compact = plan(n, m, n, &budget);
            let expanded = plan(n, m, 1024, &budget);
            // Products and sums are exactly representable, including cancellation.
            // Vary slice alignment without changing the mathematical input.
            for offset in 0..8 {
                let mut storage = vec![f64::NAN; n + offset];
                for (i, value) in storage[offset..].iter_mut().enumerate() {
                    *value = (i as i64 % 41 - 20) as f64 / 16.0;
                }
                let values = &storage[offset..];
                let expected: Vec<_> = (0..m)
                    .map(|j| {
                        (0..n)
                            .map(|i| compact.weights[(i, j)] * values[i])
                            .sum::<f64>()
                    })
                    .collect();
                let mut output = vec![f64::NAN; m];
                let counts = allocation_counter::measure(|| {
                    for selected in [&compact, &expanded, &compact] {
                        selected.estimate_into(values, &mut output).unwrap();
                        assert_eq!(output, expected, "shape {n} x {m}, offset {offset}");
                    }
                });
                assert_eq!(counts.count_total, 0, "shape {n} x {m}");
            }
        }
    }
}

#[test]
fn prediction_non_dyadic_reductions_ignore_capacity_and_slice_alignment() {
    let budget = MemoryBudget::new(1 << 20);
    for n in [7, 31, 32, 33, 63, 64, 65, 128, 511] {
        let compact = plan(n, 9, n, &budget);
        let expanded = plan(n, 9, 1024, &budget);
        let mut reference = [0.0; 9];
        let mut output = [0.0; 9];
        let values: Vec<_> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        compact.estimate_into(&values, &mut reference).unwrap();
        for offset in 0..8 {
            let mut storage = vec![f64::NAN; offset];
            storage.extend_from_slice(&values);
            for selected in [&compact, &expanded] {
                selected
                    .estimate_into(&storage[offset..], &mut output)
                    .unwrap();
                assert_eq!(output.map(f64::to_bits), reference.map(f64::to_bits));
            }
        }
    }
}

#[test]
fn prediction_rejects_bad_inputs_and_overflow_then_recovers() {
    let budget = MemoryBudget::new(1 << 20);
    let mut selected = plan(2, 1, 2, &budget);
    selected.weights.fill(1.0);
    let mut output = [0.0];
    for values in [&[1.0][..], &[f64::NAN, 1.0], &[1.0, f64::INFINITY]] {
        assert!(selected.estimate_into(values, &mut output).is_err());
    }
    assert!(selected.estimate_into(&[1.0, 2.0], &mut []).is_err());
    assert!(matches!(
        selected.estimate_into(&[f64::MAX, f64::MAX], &mut output),
        Err(Error::Numerical {
            context: "kriging prediction"
        })
    ));
    selected.estimate_into(&[1.0, 2.0], &mut output).unwrap();
    assert_eq!(output, [3.0]);
}

#[test]
fn known_mean_prediction_subtracts_before_multiplying_large_common_offsets() {
    let budget = MemoryBudget::new(1 << 20);
    let mut selected = plan(2, 1, 2, &budget);
    selected.weights.fill(1.0);
    let mut output = [f64::NAN];
    // Separate W^T values and W^T means both overflow. The actual residual is
    // exactly zero, so an allocation-free implementation must retain it.
    let allocations = allocation_counter::measure(|| {
        selected
            .estimate_with_mean_into(&[f64::MAX; 2], &[f64::MAX; 2], &[3.0], &mut output)
            .unwrap();
    });
    assert_eq!(output, [3.0]);
    assert_eq!(allocations.count_total, 0);
}
