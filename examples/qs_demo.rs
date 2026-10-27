//! Quick Sampling demo on a drill-hole-style dataset.
//!
//! Runs three QS configurations:
//!   1. Continuous QS on `porosity.dat`.
//!   2. Categorical QS on `facies.dat` (4 classes).
//!   3. Multivariate QS jointly simulating both.
//!
//! All three use the same spatial split: a slice of `data/merged_str.csv` as
//! the training image, drill-hole hard data sampled from the remaining
//! simulation region.
//!
//! Run with: `cargo run --release --example qs_demo`
//!
//! Environment overrides (defaults in parentheses):
//!   QS_DATA         path to merged_str.csv
//!   QS_NX/NY/NZ     bounding-box extents (40, 30, 30)
//!   QS_NHOLES       number of drill holes in the sim region (40)
//!   QS_DH_STEP      downhole sample spacing in cells (1)
//!   QS_RX/RY/RZ     template radii in cells (3, 3, 3)
//!   QS_TOPK         top-K replicates for the random pick (5)
//!   QS_SPLIT        fraction of x reserved for TI; 0 = no TI (0.5)
//!   QS_LEVELS       number of multi-grid levels (1 = single grid; 3 typical)
//!   QS_CLUSTERS     number of auto-derived domain clusters (0 = disabled)

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use terrustrial::qs::{
    simulate_categorical, simulate_continuous, simulate_multivariate, Grid3, HardSample,
    MultivariateOutput, MultivariateVariable, QsParams, QsTemplate,
};

#[derive(Clone, Copy)]
struct GridCell {
    x: i32,
    y: i32,
    z: i32,
    porosity: f64,
    facies: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_path = std::env::var("QS_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let here = std::env::current_dir().unwrap();
            here.join("../../../data/merged_str.csv")
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from("data/merged_str.csv"))
        });

    let nx: i32 = parse_env("QS_NX", 40);
    let ny: i32 = parse_env("QS_NY", 30);
    let nz: i32 = parse_env("QS_NZ", 30);
    let n_holes: usize = parse_env("QS_NHOLES", 40);
    let dh_step: i32 = parse_env("QS_DH_STEP", 1);
    let rx: i32 = parse_env("QS_RX", 3);
    let ry: i32 = parse_env("QS_RY", 3);
    let rz: i32 = parse_env("QS_RZ", 3);
    let top_k: usize = parse_env("QS_TOPK", 5);
    let split_frac: f64 = parse_env("QS_SPLIT", 0.5);
    let n_levels: usize = parse_env("QS_LEVELS", 3);
    let n_clusters: usize = parse_env("QS_CLUSTERS", 0);

    println!("Loading data from {}", data_path.display());
    println!("Bounding box: 0..{} x 0..{} x 0..{}", nx, ny, nz);

    let cells = load_subset(&data_path, nx, ny, nz)?;
    println!("Loaded {} cells.", cells.len());

    // Spatial split.
    let ti_start: i32 = ((nx as f64) * (1.0 - split_frac)).round() as i32;
    let sim_cells: Vec<GridCell> = cells.iter().filter(|c| c.x < ti_start).cloned().collect();
    let ti_cells: Vec<GridCell> = cells.iter().filter(|c| c.x >= ti_start).cloned().collect();
    println!(
        "Split: sim region x < {} ({} cells), TI x >= {} ({} cells).",
        ti_start,
        sim_cells.len(),
        ti_start,
        ti_cells.len()
    );

    if ti_cells.is_empty() {
        return Err("QS requires a non-empty training image (set QS_SPLIT > 0)".into());
    }

    // Build the TIs as dense regular grids.
    let ti_nx = nx - ti_start;
    let ti_ny = ny;
    let ti_nz = nz;
    let mut ti_porosity = Grid3::<f64>::filled(ti_nx as usize, ti_ny as usize, ti_nz as usize, 0.0);
    let mut ti_facies = Grid3::<u32>::filled(ti_nx as usize, ti_ny as usize, ti_nz as usize, 0);
    for c in ti_cells.iter() {
        let lx = (c.x - ti_start) as usize;
        let ly = c.y as usize;
        let lz = c.z as usize;
        *ti_porosity.get_mut(lx, ly, lz) = c.porosity;
        *ti_facies.get_mut(lx, ly, lz) = c.facies;
    }

    // Build the drillhole hard data.
    let mut by_xy: HashMap<(i32, i32), Vec<GridCell>> = HashMap::new();
    for c in sim_cells.iter() {
        by_xy.entry((c.x, c.y)).or_default().push(*c);
    }
    for v in by_xy.values_mut() {
        v.sort_by_key(|c| c.z);
    }
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let mut all_xys: Vec<(i32, i32)> = by_xy.keys().cloned().collect();
    all_xys.shuffle(&mut rng);
    let collars = &all_xys[..n_holes.min(all_xys.len())];

    let mut hard_porosity: Vec<HardSample<f64>> = Vec::new();
    let mut hard_facies: Vec<HardSample<u32>> = Vec::new();
    for &(cx, cy) in collars.iter() {
        if let Some(column) = by_xy.get(&(cx, cy)) {
            for c in column.iter().step_by(dh_step as usize) {
                hard_porosity.push(HardSample {
                    x: c.x,
                    y: c.y,
                    z: c.z,
                    value: c.porosity,
                });
                hard_facies.push(HardSample {
                    x: c.x,
                    y: c.y,
                    z: c.z,
                    value: c.facies,
                });
            }
        }
    }
    println!(
        "Drillholes: {} collars -> {} hard samples.",
        collars.len(),
        hard_porosity.len()
    );

    // Truth lookup for diagnostics.
    let mut truth_map: HashMap<(i32, i32, i32), GridCell> = HashMap::new();
    for c in sim_cells.iter() {
        truth_map.insert((c.x, c.y, c.z), *c);
    }

    // Simulation grid is the sim region (the lower-x slab).
    let sim_shape = (ti_start as usize, ny as usize, nz as usize);

    // Template - ellipsoidal in cells.
    let template = QsTemplate::ellipsoidal(rx, ry, rz);
    println!(
        "Template: radii ({}, {}, {}) -> {} lags.",
        rx,
        ry,
        rz,
        template.len()
    );

    let params = QsParams {
        template: template.clone(),
        top_k,
        ti_sample_fraction: 1.0,
        seed: 0xABCDEF01,
        column_sequential: true,
        n_grid_levels: n_levels,
        n_clusters,
        cluster_max_iter: 20,
    };

    // ---- 1. Continuous QS on porosity ----
    println!("\n=== Continuous QS (porosity) ===");
    let t0 = std::time::Instant::now();
    let sim_porosity = simulate_continuous(&ti_porosity, sim_shape, &hard_porosity, &params);
    println!(
        "Done in {:.1}s ({} cells).",
        t0.elapsed().as_secs_f64(),
        sim_porosity.len()
    );
    print_continuous_stats(&sim_porosity, &truth_map);

    // ---- 2. Categorical QS on facies ----
    println!("\n=== Categorical QS (facies, 4 classes) ===");
    let t1 = std::time::Instant::now();
    let sim_facies = simulate_categorical(&ti_facies, 4, sim_shape, &hard_facies, &params);
    println!(
        "Done in {:.1}s ({} cells).",
        t1.elapsed().as_secs_f64(),
        sim_facies.len()
    );
    print_categorical_stats(&sim_facies, &truth_map, 4);

    // ---- 3. Multivariate QS (joint porosity + facies) ----
    println!("\n=== Multivariate QS (porosity + facies, co-located) ===");
    let mv_vars = vec![
        MultivariateVariable::Continuous {
            ti: ti_porosity.clone(),
            hard_data: hard_porosity.clone(),
            weight: 1.0,
        },
        MultivariateVariable::Categorical {
            ti: ti_facies.clone(),
            n_classes: 4,
            hard_data: hard_facies.clone(),
            weight: 1.0,
        },
    ];
    let t2 = std::time::Instant::now();
    let mv_out = simulate_multivariate(mv_vars, sim_shape, &params);
    println!("Done in {:.1}s.", t2.elapsed().as_secs_f64());
    if let MultivariateOutput::Continuous(g) = &mv_out[0] {
        println!("[multivariate continuous]");
        print_continuous_stats(g, &truth_map);
    }
    if let MultivariateOutput::Categorical(g) = &mv_out[1] {
        println!("[multivariate categorical]");
        print_categorical_stats(g, &truth_map, 4);
    }

    write_continuous_csv("qs_porosity.csv", &sim_porosity, &truth_map)?;
    write_categorical_csv("qs_facies.csv", &sim_facies, &truth_map)?;
    if let MultivariateOutput::Continuous(g) = &mv_out[0] {
        write_continuous_csv("qs_mv_porosity.csv", g, &truth_map)?;
    }
    if let MultivariateOutput::Categorical(g) = &mv_out[1] {
        write_categorical_csv("qs_mv_facies.csv", g, &truth_map)?;
    }
    write_hard_csv("qs_hard_drillholes.csv", &hard_porosity, &hard_facies)?;
    println!(
        "\nWrote qs_porosity.csv, qs_facies.csv, qs_mv_porosity.csv, qs_mv_facies.csv, \
         qs_hard_drillholes.csv (all with dx=dy=dz=1)"
    );

    Ok(())
}

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn load_subset(path: &PathBuf, nx: i32, ny: i32, nz: i32) -> Result<Vec<GridCell>, Box<dyn Error>> {
    let f = File::open(path)?;
    let mut reader = BufReader::with_capacity(1 << 20, f);
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let cols: Vec<&str> = header.trim_end().split(',').collect();
    let cx = idx_of(&cols, "x")?;
    let cy = idx_of(&cols, "y")?;
    let cz = idx_of(&cols, "z")?;
    let cporo = idx_of(&cols, "porosity.dat")?;
    let cfacies = idx_of(&cols, "facies.dat")?;

    let mut out = Vec::new();
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let trimmed = line.trim_end();
        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() <= cporo {
            line.clear();
            continue;
        }
        let x: i32 = fields[cx].parse()?;
        let y: i32 = fields[cy].parse()?;
        let z: i32 = fields[cz].parse()?;
        if x >= nx || y >= ny || z >= nz {
            line.clear();
            continue;
        }
        let porosity: f64 = fields[cporo].parse()?;
        let facies: u32 = fields[cfacies].parse::<f64>()? as u32;
        out.push(GridCell {
            x,
            y,
            z,
            porosity,
            facies,
        });
        line.clear();
    }
    Ok(out)
}

fn idx_of(cols: &[&str], name: &str) -> Result<usize, Box<dyn Error>> {
    cols.iter()
        .position(|c| *c == name)
        .ok_or_else(|| format!("column '{}' not found", name).into())
}

fn print_continuous_stats(sim: &Grid3<f64>, truth: &HashMap<(i32, i32, i32), GridCell>) {
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for z in 0..sim.nz {
        for y in 0..sim.ny {
            for x in 0..sim.nx {
                let s = *sim.get(x, y, z);
                if s.is_nan() {
                    continue;
                }
                if let Some(c) = truth.get(&(x as i32, y as i32, z as i32)) {
                    pairs.push((s, c.porosity));
                }
            }
        }
    }
    if pairs.is_empty() {
        println!("(no valid pairs)");
        return;
    }
    let n = pairs.len() as f64;
    let (mut ms, mut mt) = (0.0, 0.0);
    for &(s, t) in pairs.iter() {
        ms += s;
        mt += t;
    }
    ms /= n;
    mt /= n;
    let (mut ss, mut st, mut cov) = (0.0, 0.0, 0.0);
    let mut sse = 0.0;
    for &(s, t) in pairs.iter() {
        let ds = s - ms;
        let dt = t - mt;
        ss += ds * ds;
        st += dt * dt;
        cov += ds * dt;
        sse += (s - t).powi(2);
    }
    let rmse = (sse / n).sqrt();
    let corr = if ss > 0.0 && st > 0.0 {
        cov / (ss * st).sqrt()
    } else {
        f64::NAN
    };
    println!(
        "mean(sim) = {:.4}, mean(truth) = {:.4}, RMSE = {:.4}, corr = {:.3} over {} cells",
        ms,
        mt,
        rmse,
        corr,
        pairs.len()
    );
}

fn print_categorical_stats(sim: &Grid3<u32>, truth: &HashMap<(i32, i32, i32), GridCell>, k: u32) {
    let mut hist_sim = vec![0u64; k as usize];
    let mut hist_truth = vec![0u64; k as usize];
    let mut correct = 0u64;
    let mut total = 0u64;
    for z in 0..sim.nz {
        for y in 0..sim.ny {
            for x in 0..sim.nx {
                let s = *sim.get(x, y, z);
                if s == u32::MAX {
                    continue;
                }
                if let Some(c) = truth.get(&(x as i32, y as i32, z as i32)) {
                    if (s as usize) < hist_sim.len() {
                        hist_sim[s as usize] += 1;
                    }
                    if (c.facies as usize) < hist_truth.len() {
                        hist_truth[c.facies as usize] += 1;
                    }
                    if s == c.facies {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }
    }
    if total == 0 {
        println!("(no valid pairs)");
        return;
    }
    let n = total as f64;
    println!("category | sim%   | truth%");
    for c in 0..k as usize {
        let sp = 100.0 * hist_sim[c] as f64 / n;
        let tp = 100.0 * hist_truth[c] as f64 / n;
        println!("    {:>4} | {:>5.2} | {:>5.2}", c, sp, tp);
    }
    println!(
        "exact-match rate = {:.2}% over {} cells",
        100.0 * correct as f64 / n,
        total
    );
}

fn write_continuous_csv(
    path: &str,
    sim: &Grid3<f64>,
    truth: &HashMap<(i32, i32, i32), GridCell>,
) -> Result<(), Box<dyn Error>> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "x,y,z,dx,dy,dz,truth,sim")?;
    for z in 0..sim.nz {
        for y in 0..sim.ny {
            for x in 0..sim.nx {
                let s = *sim.get(x, y, z);
                let t = truth
                    .get(&(x as i32, y as i32, z as i32))
                    .map(|c| c.porosity.to_string())
                    .unwrap_or_default();
                let s_str = if s.is_nan() {
                    String::new()
                } else {
                    s.to_string()
                };
                writeln!(w, "{},{},{},1,1,1,{},{}", x, y, z, t, s_str)?;
            }
        }
    }
    Ok(())
}

fn write_categorical_csv(
    path: &str,
    sim: &Grid3<u32>,
    truth: &HashMap<(i32, i32, i32), GridCell>,
) -> Result<(), Box<dyn Error>> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "x,y,z,dx,dy,dz,truth,sim")?;
    for z in 0..sim.nz {
        for y in 0..sim.ny {
            for x in 0..sim.nx {
                let s = *sim.get(x, y, z);
                let t = truth
                    .get(&(x as i32, y as i32, z as i32))
                    .map(|c| c.facies.to_string())
                    .unwrap_or_default();
                let s_str = if s == u32::MAX {
                    String::new()
                } else {
                    s.to_string()
                };
                writeln!(w, "{},{},{},1,1,1,{},{}", x, y, z, t, s_str)?;
            }
        }
    }
    Ok(())
}

fn write_hard_csv(
    path: &str,
    porosity: &[HardSample<f64>],
    facies: &[HardSample<u32>],
) -> Result<(), Box<dyn Error>> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "x,y,z,dx,dy,dz,porosity,facies")?;
    for (p, fa) in porosity.iter().zip(facies.iter()) {
        writeln!(w, "{},{},{},1,1,1,{},{}", p.x, p.y, p.z, p.value, fa.value)?;
    }
    Ok(())
}
