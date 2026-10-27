//! Reusable constrained queries against the complete reservoir SQLite point store.
use std::time::Instant;
use terrustrial_core::{
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    model::Anisotropy,
    spatial::{ObservationSource, SearchNeighborhood, SearchWorkspace, VariableLimit},
    storage::SqliteStore,
    Error,
};
fn query_point(i: usize) -> [f64; 3] {
    [
        ((i % 600 * 71) % 150) as f64 + 0.25,
        ((i % 600 * 113) % 200) as f64 + 0.5,
        ((i % 600 * 137) % 200) as f64 + 0.125,
    ]
}
// Independent coordinate enumeration, complete sorting and greedy selection.
// It does not call SearchNeighborhood for candidate scoring or selection.
fn reference(p: [f64; 3]) -> Vec<u64> {
    let (s, c) = 0.37_f64.sin_cos();
    let mut candidates = Vec::new();
    let max = [150, 200, 200];
    let lower: [i32; 3] = std::array::from_fn(|i| ((p[i] - 9.0).floor() as i32).max(0));
    let upper: [i32; 3] = std::array::from_fn(|i| ((p[i] + 9.0).ceil() as i32).min(max[i] - 1));
    for x in lower[0]..=upper[0] {
        for y in lower[1]..=upper[1] {
            for z in lower[2]..=upper[2] {
                let [dx, dy, dz] = [x as f64 - p[0], y as f64 - p[1], z as f64 - p[2]];
                let [u, v, w] = [c * dx + s * dy, -s * dx + c * dy, dz];
                let d = ((u / 8.0).powi(2) + (v / 3.0).powi(2) + (w / 1.5).powi(2)).sqrt();
                if d <= 1.0 {
                    let sector = (u < 0.0) as usize
                        | (((v < 0.0) as usize) << 1)
                        | (((w < 0.0) as usize) << 2);
                    candidates.push((d, x as u64 * 40000 + y as u64 * 200 + z as u64, sector));
                }
            }
        }
    }
    candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut sectors = [0; 8];
    let mut result = Vec::new();
    for (_, id, sector) in candidates {
        if sectors[sector] == 4 {
            continue;
        }
        result.push(id);
        sectors[sector] += 1;
        if result.len() == 24 {
            break;
        }
    }
    result
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 || args[1] != "neighborhood-sqlite" {
        return Err(
            "usage: neighborhood_benchmark neighborhood-sqlite RESERVOIR.sqlite QUERIES".into(),
        );
    }
    let count: usize = args[3].parse()?;
    if count == 0 {
        return Err("positive query count required".into());
    }
    let budget = MemoryBudget::new(32 << 20);
    let cancel = Cancellation::default();
    let preparation = Instant::now();
    let store = SqliteStore::open(&args[2], 16 << 20, &budget)?;
    // Verify the complete lattice and row-ID mapping before relying on a
    // generated-coordinate oracle. Values are irrelevant to neighbor selection.
    let mut rows = 0_u64;
    store.scan(&mut |o| {
        let p = o.site().support.center().coordinates();
        let id = o.id();
        let expected = [
            (id / 40000) as f64,
            ((id / 200) % 200) as f64,
            (id % 200) as f64,
        ];
        if id >= 6000000 || p != expected || o.site().variable != 0 {
            return Err(Error::InvalidInput(
                "database is not the complete reservoir lattice with expected row IDs".into(),
            ));
        }
        rows += 1;
        Ok(())
    })?;
    if rows != 6000000 {
        return Err("incomplete reservoir database".into());
    }
    let (s, c) = 0.37_f64.sin_cos();
    let query = SearchNeighborhood::new(
        32,
        Anisotropy::new(
            [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
            [8.0, 3.0, 1.5],
        )?,
        Some(4),
        vec![VariableLimit {
            variable: 0,
            maximum: 24,
        }],
    )?;
    let mut workspace = SearchWorkspace::new(&query, &budget)?;
    // The deterministic location sequence has period 600. Check every distinct
    // location before timing, so later repetitions cannot conceal a wrong query.
    let checked = count.min(600);
    for i in 0..checked {
        let p = query_point(i);
        let found: Vec<_> = store
            .search(Point3::new(p)?, &mut workspace, &cancel)?
            .iter()
            .map(|o| o.id())
            .collect();
        if found != reference(p) {
            return Err(format!("independent neighborhood mismatch at query {i}").into());
        }
    }
    let preparation_seconds = preparation.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut checksum = 0_u64;
    let mut returned = 0_u64;
    for i in 0..count {
        for o in store.search(Point3::new(query_point(i))?, &mut workspace, &cancel)? {
            returned = returned.checked_add(1).ok_or(Error::SizeOverflow)?;
            checksum = checksum.rotate_left(7) ^ o.id();
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let mut output = csv::Writer::from_writer(std::io::stdout().lock());
    output.write_record(["metric", "value"])?;
    for (key, value) in [
        ("records_verified", rows.to_string()),
        ("queries", count.to_string()),
        ("oracle_queries", checked.to_string()),
        ("workspace_bytes", workspace.accounted_bytes().to_string()),
        ("returned", returned.to_string()),
        ("id_checksum", checksum.to_string()),
        ("preparation_seconds", preparation_seconds.to_string()),
        ("seconds", seconds.to_string()),
        ("queries_per_second", (count as f64 / seconds).to_string()),
    ] {
        output.write_record([key, &value])?;
    }
    output.flush()?;
    Ok(())
}
