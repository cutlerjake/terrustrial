//! Isolated faer kernel audit. Timings exclude fixture generation and allocation counting.
use dyn_stack::{MemBuffer, MemStack};
use faer::{linalg::cholesky::llt, Mat, Par};
use std::{hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: gaussian_kernels OBSERVATIONS TARGETS ITERATIONS".into());
    }
    let n: usize = args[1].parse()?;
    let m: usize = args[2].parse()?;
    let iterations: usize = args[3].parse()?;
    if !(2..=2048).contains(&n) || !(1..=256).contains(&m) || !(1..=100000).contains(&iterations) {
        return Err("require 2<=observations<=2048, 1<=targets<=256, 1<=iterations<=100000".into());
    }
    // A well-conditioned correlated SPD matrix; the diagonal noise is explicit.
    let original = Mat::from_fn(n, n, |i, j| {
        (-(i.abs_diff(j) as f64) / 16.).exp() + if i == j { 0.03 } else { 0. }
    });
    let rhs = Mat::from_fn(n, m, |i, j| ((i * 17 + j * 7) as f64 * 0.01).sin());
    let mut factor = original.clone();
    let mut scratch = MemBuffer::try_new(llt::factor::cholesky_in_place_scratch::<f64>(
        n,
        Par::Seq,
        Default::default(),
    ))?;
    llt::factor::cholesky_in_place(
        factor.as_mut(),
        Default::default(),
        Par::Seq,
        MemStack::new(&mut scratch),
        Default::default(),
    )?;
    let mut expected = rhs.clone();
    llt::solve::solve_in_place(
        factor.as_ref(),
        expected.as_mut(),
        Par::Seq,
        MemStack::new(&mut []),
    );
    let mut output = rhs.clone();
    let mut csv = csv::Writer::from_writer(std::io::stdout().lock());
    csv.write_record(["metric", "value"])?;
    for chunk in [0, 1, 2, 4, 8, 16, 32, 64] {
        if chunk >= m && chunk != 0 {
            continue;
        }
        let label = format!("solve_chunk_{chunk}");
        let mut run = || {
            output.as_mut().copy_from(rhs.as_ref());
            if chunk == 0 {
                llt::solve::solve_in_place(
                    factor.as_ref(),
                    output.as_mut(),
                    Par::Seq,
                    MemStack::new(&mut []),
                );
            } else {
                for j in (0..m).step_by(chunk) {
                    llt::solve::solve_in_place(
                        factor.as_ref(),
                        output.as_mut().subcols_mut(j, chunk.min(m - j)),
                        Par::Seq,
                        MemStack::new(&mut []),
                    );
                }
            }
            black_box(output.as_ref());
        };
        report(&mut csv, &label, iterations, &mut run)?;
        let mut max_error = 0_f64;
        let mut mismatches = 0;
        for j in 0..m {
            for i in 0..n {
                let a = output[(i, j)];
                let b = expected[(i, j)];
                max_error = max_error.max((a - b).abs() / b.abs().max(1.));
                mismatches += usize::from(a.to_bits() != b.to_bits());
            }
        }
        csv.write_record([format!("{label}_max_scaled_error"), max_error.to_string()])?;
        csv.write_record([format!("{label}_bit_mismatches"), mismatches.to_string()])?;
        if max_error > 1e-11 {
            return Err("batched solve deviates from reference".into());
        }
    }
    // Tune only documented faer factorization parameters. The default remains
    // the baseline; no timing result alone establishes numerical equivalence.
    for threshold in [64, 32, 128, 256] {
        for block_size in [128, 64, 256] {
            let params = llt::factor::LltParams {
                recursion_threshold: threshold,
                block_size,
                ..faer::auto!(f64)
            };
            let label = format!("factor_threshold_{threshold}_block_{block_size}");
            let mut run = || {
                factor.as_mut().copy_from(original.as_ref());
                llt::factor::cholesky_in_place(
                    factor.as_mut(),
                    Default::default(),
                    Par::Seq,
                    MemStack::new(&mut scratch),
                    params.into(),
                )
                .unwrap();
                black_box(factor.as_ref());
            };
            report(&mut csv, &label, iterations, &mut run)?;
            let mut max_error = 0_f64;
            for j in 0..n {
                for i in j..n {
                    let reconstructed: f64 = (0..=j).map(|k| factor[(i, k)] * factor[(j, k)]).sum();
                    max_error = max_error.max((reconstructed - original[(i, j)]).abs());
                }
            }
            csv.write_record([format!("{label}_max_absolute_error"), max_error.to_string()])?;
            if max_error > 1e-11 {
                return Err("factorization reconstruction failed".into());
            }
        }
    }
    csv.flush()?;
    Ok(())
}
fn report<W: std::io::Write>(
    csv: &mut csv::Writer<W>,
    label: &str,
    iterations: usize,
    run: &mut impl FnMut(),
) -> Result<(), Box<dyn std::error::Error>> {
    for _ in 0..4 {
        run();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        run();
    }
    let seconds = start.elapsed().as_secs_f64();
    let allocations = allocation_counter::measure(run);
    for (key, value) in [
        ("seconds", seconds.to_string()),
        ("allocations", allocations.count_total.to_string()),
        ("allocated_bytes", allocations.bytes_total.to_string()),
    ] {
        csv.write_record([format!("{label}_{key}"), value])?;
    }
    Ok(())
}
