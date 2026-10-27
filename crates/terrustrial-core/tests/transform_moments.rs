use statrs::distribution::{ContinuousCDF, Normal};
use terrustrial_core::{
    execution::{Cancellation, MemoryBudget},
    transforms::{MomentQuadrature, NormalScore, TailPolicy},
    Error, Result,
};

// Independent copy of the previous evaluation order, including inverse CDF in
// the loop, to guard against a shared error in the cached and uncached paths.
fn reference(t: &NormalScore, mean: f64, variance: f64, points: usize) -> Result<(f64, f64)> {
    if variance == 0.0 {
        return Ok((t.inverse(mean)?, 0.0));
    }
    let normal = Normal::new(0.0, 1.0).unwrap();
    let (mut average, mut m2) = (0., 0.);
    for i in 0..points {
        let probability = (i as f64 + 0.5) / points as f64;
        let value = t.inverse(mean + variance.sqrt() * normal.inverse_cdf(probability))?;
        let delta = value - average;
        average += delta / (i + 1) as f64;
        m2 += delta * (value - average);
        if !average.is_finite() || !m2.is_finite() {
            return Err(Error::Numerical {
                context: "back-transformed moments",
            });
        }
    }
    Ok((average, (m2 / points as f64).max(0.)))
}
fn same(a: Result<(f64, f64)>, b: &Result<(f64, f64)>) {
    match (a, b) {
        (Ok(a), Ok(b)) => assert_eq!(
            (a.0.to_bits(), a.1.to_bits()),
            (b.0.to_bits(), b.1.to_bits())
        ),
        (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
        (a, b) => panic!("different outcomes: {a:?}, {b:?}"),
    }
}
#[test]
fn cached_and_uncached_match_previous_arithmetic_and_tail_failures() {
    let budget = MemoryBudget::new(1 << 20);
    for samples in [
        &[(1., 1.)][..],
        &[(0., 1.), (0., 2.), (3., 1.), (100., 0.5)][..],
    ] {
        for tail in [TailPolicy::Clamp, TailPolicy::Linear, TailPolicy::Reject] {
            let t = NormalScore::fit(samples, tail, &budget).unwrap();
            for points in [1, 17, 256, 1024] {
                let q = MomentQuadrature::new(points, &budget, &Cancellation::default()).unwrap();
                for mean in [-10., -0.1, 0., 0.5, 10.] {
                    for variance in [0., 1e-12, 0.01, 1., 100.] {
                        let expected = reference(&t, mean, variance, points);
                        same(t.conditional_moments(mean, variance, points), &expected);
                        same(t.conditional_moments_cached(mean, variance, &q), &expected);
                    }
                }
            }
        }
    }
}
#[test]
fn resource_admission_cancellation_invalid_inputs_and_allocation_free_prediction() {
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(1 << 20);
    assert!(MomentQuadrature::new(0, &budget, &cancel).is_err());
    assert!(matches!(
        MomentQuadrature::new(usize::MAX, &budget, &cancel),
        Err(Error::SizeOverflow)
    ));
    assert!(matches!(
        MomentQuadrature::new(256, &MemoryBudget::new(2047), &cancel),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(budget.used(), 0);
    let q = MomentQuadrature::new(256, &budget, &cancel).unwrap();
    assert_eq!(q.points(), 256);
    assert_eq!(budget.used(), q.accounted_bytes());
    drop(q);
    assert_eq!(budget.used(), 0);
    cancel.cancel();
    assert!(matches!(
        MomentQuadrature::new(256, &budget, &cancel),
        Err(Error::Cancelled)
    ));
    assert_eq!(budget.used(), 0);
    let q = MomentQuadrature::new(256, &budget, &Cancellation::default()).unwrap();
    let t = NormalScore::fit(
        &[(0., 1.), (2., 1.), (100., 1.)],
        TailPolicy::Clamp,
        &budget,
    )
    .unwrap();
    for (mean, variance) in [
        (f64::NAN, 0.),
        (f64::INFINITY, 1.),
        (0., -1.),
        (0., f64::INFINITY),
        (0., f64::NAN),
    ] {
        assert!(t.conditional_moments_cached(mean, variance, &q).is_err());
        assert!(t.conditional_moments(mean, variance, 256).is_err());
    }
    assert!(t.conditional_moments(0., 0., 0).is_err());
    let expected = t.conditional_moments(0.1, 0.5, 256).unwrap();
    let measured = allocation_counter::measure(|| {
        for _ in 0..128 {
            let actual = t.conditional_moments_cached(0.1, 0.5, &q).unwrap();
            assert_eq!(actual, expected);
        }
    });
    assert_eq!(measured.count_total, 0);
}
