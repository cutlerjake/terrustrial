//! Small independent oracles for the legacy implementation and the rewrite.
//! The reference solves use pivoted elimination, not the production faer path.

use rand::{rngs::StdRng, SeedableRng};
use terrustrial::{
    geometry::support::Support,
    systems::{
        lu::LUSystem,
        solved_systems::{
            ok_system::SolvedLUOKSystemBuilder, sk_system::SolvedLUSKSystemBuilder, SolvedLUSystem,
            SolvedSystemBuilder,
        },
    },
    variography::model_variograms::{
        composite::{CompositeVariogram, VariogramType},
        spherical::SphericalVariogram,
    },
};
use ultraviolet::{DRotor3, DVec3};

const RANGE: f64 = 8.0;
const SILL: f64 = 1.7;
const DATA: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
const TARGETS: [[f64; 3]; 2] = [[0.5, 0.75, 0.25], [1.5, 1.25, 0.5]];
const VALUES: [f64; 3] = [1.2, -0.7, 2.1];

fn covariance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let distance = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let h = distance / RANGE;
    if h >= 1.0 {
        0.0
    } else {
        SILL * (1.0 - 1.5 * h + 0.5 * h.powi(3))
    }
}

fn solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    for col in 0..rhs.len() {
        let pivot = (col..rhs.len())
            .max_by(|&a, &b| matrix[a][col].abs().total_cmp(&matrix[b][col].abs()))
            .unwrap();
        assert!(
            matrix[pivot][col].abs() > 1e-12,
            "singular reference fixture"
        );
        matrix.swap(col, pivot);
        rhs.swap(col, pivot);
        let scale = matrix[col][col];
        for value in &mut matrix[col][col..] {
            *value /= scale;
        }
        rhs[col] /= scale;
        for row in 0..rhs.len() {
            if row == col {
                continue;
            }
            let factor = matrix[row][col];
            for j in col..rhs.len() {
                matrix[row][j] -= factor * matrix[col][j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    rhs
}

fn weights(target: [f64; 3], ordinary: bool) -> Vec<f64> {
    let n = DATA.len();
    let size = n + usize::from(ordinary);
    let mut matrix = vec![vec![0.0; size]; size];
    let mut rhs = vec![0.0; size];
    for i in 0..n {
        rhs[i] = covariance(DATA[i], target);
        for j in 0..n {
            matrix[i][j] = covariance(DATA[i], DATA[j]);
        }
        if ordinary {
            matrix[i][n] = 1.0;
            matrix[n][i] = 1.0;
        }
    }
    if ordinary {
        rhs[n] = 1.0;
    }
    solve(matrix, rhs)[..n].to_vec()
}

fn system(data: &[[f64; 3]], targets: &[[f64; 3]]) -> LUSystem {
    let supports: Vec<_> = data
        .iter()
        .chain(targets)
        .map(|p| Support::Point(DVec3::from(*p)))
        .collect();
    let model = CompositeVariogram::new(vec![VariogramType::Spherical(SphericalVariogram::new(
        DVec3::new(RANGE, RANGE, RANGE),
        SILL,
        DRotor3::identity(),
    ))]);
    let mut system = LUSystem::new(targets.len(), data.len());
    system.build_cov_matrix(
        data.len(),
        targets.len(),
        &supports,
        &model,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut Vec::new(),
        &mut Vec::new(),
    );
    system
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

#[test]
fn grouped_simple_kriging_matches_independent_reference() {
    let mut solved = SolvedLUSKSystemBuilder
        .build(&mut system(&DATA, &TARGETS))
        .unwrap();
    solved.populate_cond_values_est(VALUES);
    let estimates = solved.estimate();
    for (i, target) in TARGETS.iter().enumerate() {
        let reference = weights(*target, false);
        for (j, expected) in reference.iter().enumerate() {
            assert_close(solved.weights()[(i, j)], *expected, 1e-12);
        }
        assert_close(
            estimates[i],
            reference.iter().zip(VALUES).map(|(w, v)| w * v).sum(),
            1e-12,
        );
    }
}

#[test]
fn grouped_ordinary_kriging_matches_augmented_reference_and_preserves_constants() {
    let mut solved = SolvedLUOKSystemBuilder
        .build(&mut system(&DATA, &TARGETS))
        .unwrap();
    solved.populate_cond_values_est(VALUES);
    let estimates = solved.estimate();
    for (i, target) in TARGETS.iter().enumerate() {
        let reference = weights(*target, true);
        for (j, expected) in reference.iter().enumerate() {
            assert_close(solved.weights()[(i, j)], *expected, 1e-12);
        }
        assert_close(
            estimates[i],
            reference.iter().zip(VALUES).map(|(w, v)| w * v).sum(),
            1e-12,
        );
        assert_close(
            (0..DATA.len()).map(|j| solved.weights()[(i, j)]).sum(),
            1.0,
            1e-12,
        );
    }
    solved.populate_cond_values_est([4.25; 3]);
    for value in solved.estimate() {
        assert_close(value, 4.25, 1e-12);
    }
}

#[test]
fn grouped_simulation_has_the_reference_conditional_covariance() {
    let solved = SolvedLUSKSystemBuilder
        .build(&mut system(&DATA, &TARGETS))
        .unwrap();
    for i in 0..TARGETS.len() {
        for j in 0..TARGETS.len() {
            let expected = covariance(TARGETS[i], TARGETS[j])
                - weights(TARGETS[i], false)
                    .iter()
                    .zip(DATA)
                    .map(|(w, d)| w * covariance(d, TARGETS[j]))
                    .sum::<f64>();
            let actual = (0..=i.min(j))
                .map(|k| solved.l_gg[(i, k)] * solved.l_gg[(j, k)])
                .sum();
            assert_close(actual, expected, 1e-12);
        }
    }
}

#[test]
fn seeded_draws_repeat_and_reproduce_joint_conditional_moments() {
    let mut solved = SolvedLUSKSystemBuilder
        .build(&mut system(&DATA, &TARGETS))
        .unwrap();
    let mut first = StdRng::seed_from_u64(731);
    let mut second = StdRng::seed_from_u64(731);
    solved.populate_cond_values_sim(VALUES, &mut first);
    let draw = solved.simulate();
    solved.populate_cond_values_sim(VALUES, &mut second);
    assert_eq!(draw, solved.simulate());

    let means: Vec<f64> = TARGETS
        .iter()
        .map(|t| {
            weights(*t, false)
                .iter()
                .zip(VALUES)
                .map(|(w, v)| w * v)
                .sum()
        })
        .collect();
    let covariance_ref: Vec<Vec<f64>> = TARGETS
        .iter()
        .map(|a| {
            TARGETS
                .iter()
                .map(|b| {
                    covariance(*a, *b)
                        - weights(*a, false)
                            .iter()
                            .zip(DATA)
                            .map(|(w, d)| w * covariance(d, *b))
                            .sum::<f64>()
                })
                .collect()
        })
        .collect();
    let draws = 20_000;
    let mut sum = [0.0; 2];
    let mut products = [[0.0; 2]; 2];
    for _ in 0..draws {
        solved.populate_cond_values_sim(VALUES, &mut first);
        let draw = solved.simulate();
        for i in 0..2 {
            sum[i] += draw[i];
            for j in 0..2 {
                products[i][j] += (draw[i] - means[i]) * (draw[j] - means[j]);
            }
        }
    }
    for i in 0..2 {
        // Six analytic standard errors, using known means in covariance estimates.
        assert_close(
            sum[i] / draws as f64,
            means[i],
            6.0 * (covariance_ref[i][i] / draws as f64).sqrt(),
        );
        for j in 0..2 {
            let se = ((covariance_ref[i][i] * covariance_ref[j][j] + covariance_ref[i][j].powi(2))
                / draws as f64)
                .sqrt();
            assert_close(
                products[i][j] / draws as f64,
                covariance_ref[i][j],
                6.0 * se,
            );
        }
    }
}

#[test]
fn duplicate_zero_noise_observations_return_a_factorization_error_for_sk() {
    let duplicate = [DATA[0], DATA[0]];
    assert!(SolvedLUSKSystemBuilder
        .build(&mut system(&duplicate, &TARGETS))
        .is_err());
}
