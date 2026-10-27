use terrustrial_core::{
    anisotropy::{AnchorField, NonstationaryGaussian, PreparedField, Tensor, TensorField},
    data::{Point3, Site, Support},
    execution::{Cancellation, MemoryBudget},
    model::{Anisotropy, Coregionalization, Covariance},
    Error,
};
fn p(x: f64, y: f64, z: f64) -> Point3 {
    Point3::new([x, y, z]).unwrap()
}
fn field() -> AnchorField {
    AnchorField::new(
        (0..20)
            .map(|i| {
                (
                    p(i as f64, 0., 0.),
                    Tensor::from_anisotropy(&Anisotropy::isotropic(1. + i as f64 / 20.).unwrap())
                        .unwrap(),
                )
            })
            .collect(),
        4,
        2.,
    )
    .unwrap()
}
#[test]
fn reusable_interpolation_and_prepared_joint_block_covariance_are_exact_and_allocation_free() {
    let field = field();
    let mut workspace = field.workspace().unwrap();
    let sites = [
        Site::point(p(0., 0., 0.), 0),
        Site {
            support: Support::block(p(1., 0.1, 0.2), [0.1; 3], [2; 3]).unwrap(),
            variable: 1,
        },
        Site::point(p(4.3, 0.2, -0.1), 0),
    ];
    let mut points: Vec<_> = sites.iter().flat_map(|s| s.support.nodes()).collect();
    points.extend([p(-0., 0., 0.), p(0., 0., 0.)]);
    let expected: Vec<_> = points
        .iter()
        .map(|p| field.at(*p).unwrap().matrix())
        .collect();
    let measured = allocation_counter::measure(|| {
        for (p, m) in points.iter().zip(&expected) {
            assert_eq!(
                field
                    .at_with_workspace(*p, &mut workspace)
                    .unwrap()
                    .matrix(),
                *m
            );
        }
    });
    assert_eq!(measured.count_total, 0);
    let budget = MemoryBudget::new(1 << 20);
    let cancel = Cancellation::default();
    let cached = PreparedField::new(&field, &points, &budget, &cancel).unwrap();
    assert_eq!(cached.len(), 10);
    assert_eq!(budget.used(), cached.accounted_bytes());
    let sill = Coregionalization::from_factor(2, 2, &[1., 0., -0.4, 0.8]).unwrap();
    let direct = NonstationaryGaussian {
        field,
        coregionalization: sill.clone(),
    };
    let cached = NonstationaryGaussian {
        field: cached,
        coregionalization: sill,
    };
    let expected: Vec<_> = sites
        .iter()
        .flat_map(|a| sites.iter().map(|b| direct.covariance(a, b).unwrap()))
        .collect();
    let measured = allocation_counter::measure(|| {
        for (i, a) in sites.iter().enumerate() {
            for (j, b) in sites.iter().enumerate() {
                assert_eq!(
                    cached.covariance(a, b).unwrap().to_bits(),
                    expected[3 * i + j].to_bits()
                );
            }
        }
    });
    assert_eq!(measured.count_total, 0);
    assert!(cached.field.at(p(100., 0., 0.)).is_err());
    drop(cached);
    assert_eq!(budget.used(), 0);
}
#[test]
fn preparation_budget_cancellation_empty_and_workspace_capacity() {
    let field = field();
    let tiny = MemoryBudget::new(1);
    let cancel = Cancellation::default();
    assert!(matches!(
        PreparedField::new(&field, &[p(0., 0., 0.)], &tiny, &cancel),
        Err(Error::MemoryBudget { .. })
    ));
    let budget = MemoryBudget::new(1 << 20);
    let empty = PreparedField::new(&field, &[], &budget, &cancel).unwrap();
    assert!(empty.is_empty());
    drop(empty);
    let small = AnchorField::new(
        vec![(
            p(0., 0., 0.),
            Tensor::from_anisotropy(&Anisotropy::isotropic(1.).unwrap()).unwrap(),
        )],
        1,
        2.,
    )
    .unwrap();
    assert!(field
        .at_with_workspace(p(1., 0., 0.), &mut small.workspace().unwrap())
        .is_err());
    cancel.cancel();
    assert!(matches!(
        PreparedField::new(&field, &[], &budget, &cancel),
        Err(Error::Cancelled)
    ));
    assert_eq!(budget.used(), 0);
}
