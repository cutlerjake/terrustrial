//! Strebelle 2002 binary channel TI benchmark.
//!
//! Runs unconditional categorical QS on the canonical Strebelle channel
//! training image (250x250x1, 2 facies; ~26.7% channel proportion). This is
//! the de-facto MPS benchmark used in the SNESIM / DEESSE / QuickSampling
//! literature, so it lets us check three things:
//!
//! 1. Does the algorithm produce visually plausible channelised realizations?
//! 2. Are global statistics (facies proportions) reproduced?
//! 3. How long does a single 250x250 realization take, end-to-end?
//!
//! Run with: `cargo run --release --example strebelle_bench`
//!
//! Environment overrides:
//!   STREB_TI         path to Strebelle TI in SGeMS / GSLIB format (defaults
//!                    to data/ti_strebelle.sgems).
//!   STREB_NX, STREB_NY    sim-grid size (default 250, 250). Stay at the TI's
//!                    native size for the canonical benchmark.
//!   STREB_RX, STREB_RY    template radii in cells (default 6, 6 — same scale
//!                    as the channels and the standard SNESIM template).
//!   STREB_TOPK       top-K replicate pick (default 5).
//!   STREB_LEVELS     multi-grid levels (default 4 — Strebelle is large
//!                    enough for proper hierarchy).
//!   STREB_CLUSTERS   auto-clusters; 0 disables (default 0 — TI is
//!                    stationary by design).
//!   STREB_SEED       random seed (default 0xC0FFEE).
//!
//! Outputs:
//!   strebelle_realization.csv      one realization with `x,y,z,dx,dy,dz,sim`
//!   strebelle_ti.csv               the TI itself in the same format

use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use terrustrial::qs::{simulate_categorical, Grid3, HardSample, QsParams, QsTemplate};

fn main() -> Result<(), Box<dyn Error>> {
    let ti_path = std::env::var("STREB_TI")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let here = std::env::current_dir().unwrap();
            here.join("../../../data/ti_strebelle.sgems")
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from("data/ti_strebelle.sgems"))
        });

    let nx: usize = parse_env("STREB_NX", 250);
    let ny: usize = parse_env("STREB_NY", 250);
    let rx: i32 = parse_env("STREB_RX", 6);
    let ry: i32 = parse_env("STREB_RY", 6);
    let top_k: usize = parse_env("STREB_TOPK", 5);
    let n_levels: usize = parse_env("STREB_LEVELS", 4);
    let n_clusters: usize = parse_env("STREB_CLUSTERS", 0);
    let seed: u64 = parse_env("STREB_SEED", 0xC0FFEE);

    println!("Loading Strebelle TI from {}", ti_path.display());
    let ti = load_sgems_categorical(&ti_path)?;
    let n_classes = (ti.data.iter().copied().max().unwrap_or(0) + 1) as u32;
    let proportions = facies_proportions(&ti, n_classes);
    println!(
        "TI: {} x {} x {}, {} facies, proportions = {:?}",
        ti.nx, ti.ny, ti.nz, n_classes, proportions
    );

    // Unconditional simulation: no hard data.
    let hard_data: Vec<HardSample<u32>> = Vec::new();
    let sim_shape = (nx, ny, 1);

    // 2D template (rz = 0 -> only z=0 lag, which equals the centre cell;
    // the template constructor excludes the origin so we end up with a
    // proper 2D set).
    let template = QsTemplate::ellipsoidal(rx, ry, 0);
    println!(
        "Template: radii ({}, {}, 0) -> {} lags. n_levels={}, top_k={}, clusters={}",
        rx,
        ry,
        template.len(),
        n_levels,
        top_k,
        n_clusters
    );

    let params = QsParams {
        template: template.clone(),
        top_k,
        ti_sample_fraction: 1.0,
        seed,
        column_sequential: true,
        n_grid_levels: n_levels,
        n_clusters,
        cluster_max_iter: 20,
    };

    let t = std::time::Instant::now();
    let realization = simulate_categorical(&ti, n_classes, sim_shape, &hard_data, &params);
    let dt = t.elapsed().as_secs_f64();

    let sim_proportions = facies_proportions(&realization, n_classes);

    println!("\n=== Strebelle QS benchmark ===");
    println!("sim grid:      {} x {} x 1 = {} cells", nx, ny, nx * ny);
    println!("time:          {:.2} s", dt);
    println!("cells / s:     {:.0}", (nx * ny) as f64 / dt.max(1e-9));
    println!("facies prop (TI / sim):");
    for k in 0..n_classes as usize {
        println!(
            "  {:>2}: {:.4} / {:.4}",
            k, proportions[k], sim_proportions[k]
        );
    }

    write_categorical_csv("strebelle_realization.csv", &realization)?;
    write_categorical_csv("strebelle_ti.csv", &ti)?;
    println!("\nWrote strebelle_realization.csv and strebelle_ti.csv");

    Ok(())
}

fn load_sgems_categorical(path: &PathBuf) -> Result<Grid3<u32>, Box<dyn Error>> {
    // SGeMS-style header:
    //   line 1: "nx ny nz"
    //   line 2: n_vars (typically 1)
    //   next n_vars lines: variable names
    //   then nx*ny*nz values, one per line.
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut lines = reader.lines();

    let header = lines.next().ok_or("missing header line")??;
    let dims: Vec<usize> = header
        .split_whitespace()
        .map(|s| s.parse::<usize>())
        .collect::<Result<_, _>>()?;
    if dims.len() < 2 {
        return Err("header must specify at least nx ny".into());
    }
    let nx = dims[0];
    let ny = dims[1];
    let nz = if dims.len() >= 3 { dims[2] } else { 1 };

    let n_vars_line = lines.next().ok_or("missing variable count line")??;
    let n_vars: usize = n_vars_line.trim().parse()?;
    for _ in 0..n_vars {
        let _name = lines.next().ok_or("missing variable name line")??;
    }

    let total = nx * ny * nz;
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        let line = lines.next().ok_or("data truncated")??;
        // Parse as f64 then cast to u32 to handle e.g. "0" or "1.0" or "0.0".
        let v: f64 = line.trim().parse()?;
        data.push(v.round() as u32);
    }
    if data.len() != total {
        return Err(format!("expected {} values, got {}", total, data.len()).into());
    }

    Ok(Grid3::from_data(nx, ny, nz, data))
}

fn facies_proportions(g: &Grid3<u32>, n_classes: u32) -> Vec<f64> {
    let mut counts = vec![0u64; n_classes as usize];
    for &v in g.data.iter() {
        if v == u32::MAX {
            continue;
        }
        if (v as usize) < counts.len() {
            counts[v as usize] += 1;
        }
    }
    let total: u64 = counts.iter().sum();
    counts
        .iter()
        .map(|c| {
            if total > 0 {
                *c as f64 / total as f64
            } else {
                0.0
            }
        })
        .collect()
}

fn write_categorical_csv(path: &str, g: &Grid3<u32>) -> Result<(), Box<dyn Error>> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "x,y,z,dx,dy,dz,sim")?;
    for z in 0..g.nz {
        for y in 0..g.ny {
            for x in 0..g.nx {
                let v = *g.get(x, y, z);
                let s = if v == u32::MAX {
                    String::new()
                } else {
                    v.to_string()
                };
                writeln!(w, "{},{},{},1,1,1,{}", x, y, z, s)?;
            }
        }
    }
    Ok(())
}

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
