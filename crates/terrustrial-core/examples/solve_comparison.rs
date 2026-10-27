//! Direct production faer solve versus an allocation-free experimental control.
#[path = "support/tiled_solve.rs"]
mod backend;
use dyn_stack::MemStack;
use faer::{linalg::cholesky::llt, Mat, Par};
use std::{hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations: usize = std::env::args()
        .nth(2)
        .ok_or("iterations required")?
        .parse()?;
    if iterations == 0 {
        return Err("positive iterations required".into());
    }
    let mut csv = csv::Writer::from_writer(std::io::stdout().lock());
    csv.write_record(["metric", "value"])?;
    csv.write_record(["iterations", &iterations.to_string()])?;
    for n in [16, 32, 48, 64, 96, 128, 256, 512] {
        for columns in [1, 4, 6, 16] {
            let factor = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    1. + 0.1 * (i as f64).cos()
                } else if i > j {
                    0.03 * ((i * 7 + j) as f64).sin()
                } else {
                    0.
                }
            });
            let input = Mat::from_fn(n, columns, |i, j| ((i * 3 + j) as f64).sin());
            let mut reference = input.clone();
            let mut candidate = input.clone();
            let mut apply = |tiled: bool| {
                if tiled {
                    candidate.as_mut().copy_from(input.as_ref());
                    backend::solve_spd(black_box(factor.as_ref()), candidate.as_mut());
                    black_box(candidate[(0, 0)])
                } else {
                    reference.as_mut().copy_from(input.as_ref());
                    llt::solve::solve_in_place(
                        black_box(factor.as_ref()),
                        reference.as_mut(),
                        Par::Seq,
                        MemStack::new(&mut []),
                    );
                    black_box(reference[(0, 0)])
                }
            };
            apply(false);
            apply(true);
            let mut seconds = [0.; 2];
            for pair in 0..4 {
                for tiled in if pair % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    for _ in 0..5 {
                        black_box(apply(tiled));
                    }
                    let start = Instant::now();
                    for _ in 0..iterations {
                        black_box(apply(tiled));
                    }
                    seconds[tiled as usize] += start.elapsed().as_secs_f64();
                }
            }
            let mut allocations = [0; 2];
            for tiled in [false, true] {
                allocations[tiled as usize] = allocation_counter::measure(|| {
                    for _ in 0..10 {
                        black_box(apply(tiled));
                    }
                })
                .count_total;
            }
            let difference = (0..columns)
                .flat_map(|j| (0..n).map(move |i| (i, j)))
                .map(|(i, j)| {
                    assert!(reference[(i, j)].is_finite() && candidate[(i, j)].is_finite());
                    (reference[(i, j)] - candidate[(i, j)]).abs()
                })
                .fold(0_f64, f64::max);
            assert!(
                difference < 1e-10,
                "n={n}, columns={columns}, difference={difference}"
            );
            let prefix = format!("n{n}_rhs{columns}");
            for (name, value) in [
                ("faer_seconds", seconds[0].to_string()),
                ("tiled_seconds", seconds[1].to_string()),
                ("faer_allocations", allocations[0].to_string()),
                ("tiled_allocations", allocations[1].to_string()),
                ("maximum_difference", difference.to_string()),
            ] {
                csv.write_record([format!("{prefix}_{name}"), value])?;
            }
        }
    }
    csv.flush()?;
    Ok(())
}
