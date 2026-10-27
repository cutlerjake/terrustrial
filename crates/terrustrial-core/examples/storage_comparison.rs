//! Direct current-backend comparison, with non-repeating query coordinates and
//! seven-variable, eight-point block workloads. Generated inputs stay local.
//! Optional Windows placement: `--placement=physical` or `physical-writer`,
//! also accepted through TERRUSTRIAL_PLACEMENT. The CLI overrides the environment;
//! the default is inherited scheduling. A publisher core requires packed simulation.
//! Add `--publication-decode` to packed simulation with `physical-writer` to use
//! a bounded decoding helper. Placement requires two service cores in addition
//! to `min(WORKERS, WINDOW)` compute cores. Decoding is serial by default.
//! Enable `mi-allocator` for mimalloc in this executable; defaults use the system allocator.
#[cfg(feature = "mi-allocator")]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[path = "support/nisk/blocks.rs"]
mod blocks;
#[path = "support/nisk/persistence.rs"]
#[allow(dead_code)]
mod persistence;
#[path = "support/thread_placement.rs"]
mod thread_placement;
mod fitting {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
}
use fitting::Result;
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::{path::Path, time::Instant};
use terrustrial_core::{
    data::{Observation, Point3, Site},
    estimation::{estimate_groups, EstimationOptions},
    execution::{Cancellation, ExecutionConfig, ExecutionOptions, MemoryBudget},
    gaussian::{NumericalTolerance, Ordinary, Simple, Uncertainty},
    model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
    simulation::{parallel::simulate_pipelined as simulate_groups, SimulationOptions},
    spatial::{
        NeighborhoodQuery, ObservationSource, SearchNeighborhood, SearchWorkspace, SpatialIndex,
        VariableLimit,
    },
    storage::packed::{Options, PackedBuilder, ScheduledOptions},
};
// Native Windows process CPU time, sampled only around the measured operation.
// Includes all process threads; it is distinct from summed stage wall times.
#[cfg(windows)]
fn process_cpu_seconds() -> Option<f64> {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetProcessTimes(
            process: *mut std::ffi::c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
    let (mut creation, mut exit, mut kernel, mut user) = (
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
    );
    // SAFETY: -1 is the current-process pseudo handle; all output pointers are
    // valid FILETIME layouts and remain live for this synchronous call.
    let ok = unsafe {
        GetProcessTimes(
            -1isize as *mut _,
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };
    let ticks = |v: FileTime| (u64::from(v.high) << 32) | u64::from(v.low);
    (ok != 0).then(|| (ticks(kernel) + ticks(user)) as f64 / 10_000_000.)
}
#[cfg(not(windows))]
fn process_cpu_seconds() -> Option<f64> {
    None
}
#[cfg(windows)]
fn process_cycles() -> Option<u64> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn QueryProcessCycleTime(process: *mut std::ffi::c_void, cycles: *mut u64) -> i32;
    }
    let mut cycles = 0;
    // SAFETY: current-process pseudo handle and a valid writable u64 output.
    let ok = unsafe { QueryProcessCycleTime(-1isize as *mut _, &mut cycles) };
    (ok != 0).then_some(cycles)
}
#[cfg(not(windows))]
fn process_cycles() -> Option<u64> {
    None
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn unit(x: u64) -> f64 {
    (mix(x) >> 11) as f64 / (1u64 << 53) as f64
}
fn point(i: usize) -> Point3 {
    Point3::new(std::array::from_fn(|a| {
        unit(i as u64 * 3 + a as u64) * [1000., 1000., 300.][a]
    }))
    .unwrap()
}
fn target(i: usize, n: usize) -> Point3 {
    let p = point(mix(i as u64) as usize % n).coordinates();
    Point3::new(std::array::from_fn(|a| {
        p[a] + (unit(i as u64 * 7 + a as u64 + 99) - 0.5) * 3.
    }))
    .unwrap()
}
fn observation(i: usize) -> Observation {
    Observation::new(
        i as u64,
        Site::point(point(i / 7), i % 7),
        unit(i as u64 * 11),
        0.01,
    )
    .unwrap()
}
fn query(n: usize) -> terrustrial_core::Result<SearchNeighborhood> {
    SearchNeighborhood::new(
        n * 7,
        Anisotropy::isotropic(400.)?,
        None,
        (0..7)
            .map(|variable| VariableLimit {
                variable,
                maximum: n,
            })
            .collect(),
    )
}
fn model() -> terrustrial_core::Result<Lmc> {
    let mut factor = vec![0.; 49];
    for i in 0..7 {
        factor[i * 7 + i] = 1.;
        if i > 0 {
            factor[i * 7] = 0.3;
        }
    }
    Lmc::new(vec![Structure {
        correlation: Correlation::Exponential,
        anisotropy: Anisotropy::isotropic(120.)?,
        coregionalization: Coregionalization::from_factor(7, 7, &factor)?,
    }])
}
fn reads(
    source: &(impl ObservationSource + Sync),
    n: usize,
    count: usize,
    workers: usize,
    budget: &MemoryBudget,
) -> Result<String> {
    let q = query(32)?;
    let cancel = Cancellation::default();
    let mut scratch = (0..workers)
        .map(|_| SearchWorkspace::new(&q, budget))
        .collect::<terrustrial_core::Result<Vec<_>>>()?;
    let hashes = scratch
        .par_iter_mut()
        .enumerate()
        .map(|(w, workspace)| {
            let mut hash = blake3::Hasher::new();
            for i in (w..count).step_by(workers) {
                hash.update(&(i as u64).to_le_bytes());
                for o in source.search(target(i, n), workspace, &cancel)? {
                    hash.update(&o.id().to_le_bytes());
                    hash.update(&o.value().to_bits().to_le_bytes());
                }
            }
            Ok(hash.finalize())
        })
        .collect::<terrustrial_core::Result<Vec<_>>>()?;
    let mut hash = blake3::Hasher::new();
    for h in hashes {
        hash.update(h.as_bytes());
    }
    Ok(hash.finalize().to_hex().to_string())
}
static PUBLICATION: std::sync::OnceLock<ScheduledOptions> = std::sync::OnceLock::new();
static WINDOW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static PLACEMENT: std::sync::OnceLock<thread_placement::Placement> = std::sync::OnceLock::new();
static PUBLICATION_DECODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn operation_unix_ns() -> Option<u64> {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok()
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
}

fn publication_decode_argument(args: &mut Vec<String>) -> Result<bool> {
    let mut enabled = false;
    for index in (1..args.len()).rev() {
        if args[index] == "--publication-decode" {
            if enabled {
                return Err("specify --publication-decode only once".into());
            }
            enabled = true;
            args.remove(index);
        } else if args[index].starts_with("--publication-decode=") {
            return Err("--publication-decode is a flag and takes no value".into());
        }
    }
    Ok(enabled)
}

fn validate_publication_decode(
    enabled: bool,
    backend: &str,
    mode: &str,
    placement: &str,
) -> Result<()> {
    if enabled && !(backend == "packed" && mode == "simulate" && placement == "physical-writer") {
        return Err("--publication-decode requires packed simulate and --placement=physical-writer".into());
    }
    Ok(())
}
#[cfg(feature = "experimental-two-phase")]
static TWO_PHASE: std::sync::Mutex<Option<serde_json::Value>> = std::sync::Mutex::new(None);
fn execution_options() -> ExecutionOptions {
    ExecutionOptions {
        batch_groups: *WINDOW.get().expect("benchmark window configured"),
        maximum_targets_per_group: 56,
        bytes_per_group: 64 << 20,
    }
}
fn execution() -> ExecutionConfig {
    let options = execution_options();
    match PLACEMENT.get() {
        Some(placement) => options.with_thread_start(placement.thread_start()),
        None => options.into(),
    }
}
// Preserve the original ordered little-endian byte stream, while feeding BLAKE3
// blocks instead of making millions of tiny calls on the output coordinator.
struct OutputHash {
    hasher: blake3::Hasher,
    buffer: [u8; 4096],
    len: usize,
}
impl OutputHash {
    fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
            buffer: [0; 4096],
            len: 0,
        }
    }
    fn word(&mut self, value: u64) {
        self.buffer[self.len..self.len + 8].copy_from_slice(&value.to_le_bytes());
        self.len += 8;
        if self.len == self.buffer.len() {
            self.hasher.update(&self.buffer);
            self.len = 0;
        }
    }
    fn finish(mut self) -> String {
        self.hasher.update(&self.buffer[..self.len]);
        self.hasher.finalize().to_hex().to_string()
    }
}
fn simulate(
    source: &(impl ObservationSource + Sync),
    store: &mut impl terrustrial_core::simulation::parallel::SnapshotStore,
    blocks: &[blocks::Block],
    path: &[usize],
    budget: &MemoryBudget,
) -> Result<String> {
    let m = model()?;
    let q = query(32)?;
    let sq = query(16)?;
    let cancel = Cancellation::default();
    let mut hash = OutputHash::new();
    #[cfg(feature = "experimental-two-phase")]
    if let Ok(order) = std::env::var("TERRUSTRIAL_TWO_PHASE") {
        use terrustrial_core::simulation::parallel::two_phase::{self, ReadyOrder};
        let ready_order = match order.as_str() {
            "path" => ReadyOrder::Path,
            "small" => ReadyOrder::SmallFirst,
            "large" => ReadyOrder::LargeFirst,
            _ => return Err("TERRUSTRIAL_TWO_PHASE must be path, small or large".into()),
        };
        let groups: Vec<_> = path.iter().enumerate()
            .map(|(rank, &cell)| blocks[cell].group(rank, cell, 2.)).collect();
        let stats = two_phase::simulate(&m, source, store, &groups, &Simple,
            SimulationOptions { seed: 20261006, realization: 1, mean: None,
                hard_neighborhood: NeighborhoodQuery::Constrained(&q),
                simulated_neighborhood: NeighborhoodQuery::Constrained(&sq), budget, cancellation: &cancel },
            execution_options(), ready_order, |os| {
                for o in os { hash.word(o.id()); hash.word(o.value().to_bits()); }
                Ok(())
            })?;
        *TWO_PHASE.lock().unwrap() = Some(serde_json::json!({
            "order": order, "geometry_seconds":stats.geometry_seconds,
            "selection_seconds":stats.selection_seconds,"preparation_seconds":stats.preparation_seconds,
            "graph_seconds":stats.graph_seconds,"drawing_seconds":stats.drawing_seconds,
            "publication_seconds":stats.publication_seconds,"groups":stats.groups,
            "dependency_edges":stats.dependency_edges,"dependency_depth":stats.dependency_depth,
            "initial_ready":stats.initial_ready,"maximum_ready":stats.maximum_ready,
            "retained_plan_bytes":stats.retained_plan_bytes,
            "budget_bytes_at_preparation_barrier":stats.budget_bytes_at_preparation_barrier
        }));
        return Ok(hash.finish());
    }
    simulate_groups(
        &m,
        source,
        store,
        path.iter()
            .enumerate()
            .map(|(rank, &cell)| blocks[cell].group(rank, cell, 2.)),
        &Simple,
        SimulationOptions {
            seed: 20261006,
            realization: 1,
            mean: None,
            hard_neighborhood: NeighborhoodQuery::Constrained(&q),
            simulated_neighborhood: NeighborhoodQuery::Constrained(&sq),
            budget,
            cancellation: &cancel,
        },
        execution(),
        |os| {
            for o in os {
                hash.word(o.id());
                hash.word(o.value().to_bits());
            }
            Ok(())
        },
    )?;
    Ok(hash.finish())
}
fn estimate(
    source: &(impl ObservationSource + Sync),
    blocks: &[blocks::Block],
    budget: &MemoryBudget,
) -> Result<String> {
    let m = model()?;
    let q = query(64)?;
    let cancel = Cancellation::default();
    let mut hash = OutputHash::new();
    estimate_groups(
        &m,
        source,
        blocks.iter().enumerate().map(|(i, b)| b.group(i, i, 2.)),
        &Ordinary { variables: 7 },
        EstimationOptions {
            neighborhood: NeighborhoodQuery::Constrained(&q),
            mean: None,
            uncertainty: Uncertainty::Joint,
            tolerance: NumericalTolerance::default(),
            budget,
            cancellation: &cancel,
        },
        execution(),
        |e| {
            for v in &e.values {
                hash.word(v.to_bits());
            }
            let c = e.covariance().unwrap();
            for j in 0..c.ncols() {
                for i in 0..c.nrows() {
                    hash.word(c[(i, j)].to_bits());
                }
            }
            Ok(())
        },
    )?;
    Ok(hash.finish())
}
#[allow(clippy::too_many_arguments)]
fn work(
    source: &(impl ObservationSource + Sync),
    backend: &str,
    mode: &str,
    n: usize,
    count: usize,
    workers: usize,
    budget: &MemoryBudget,
    root: &Path,
    engine: Option<&terrustrial_core::storage::packed::PackedStore>,
) -> Result<serde_json::Value> {
    if mode == "query" {
        let start = Instant::now();
        reads(source, n, count, workers, budget)?;
        let warm = start.elapsed().as_secs_f64();
        let before = engine.map(|e| e.statistics());
        let operation_start_unix_ns = operation_unix_ns();
        let start = Instant::now();
        let hash = reads(source, n, count, workers, budget)?;
        let seconds = start.elapsed().as_secs_f64();
        let operation_end_unix_ns = operation_unix_ns();
        let delta=engine.zip(before).map(|(e,b)|{let a=e.statistics();serde_json::json!({"page_reads":a.page_reads-b.page_reads,"page_bytes":a.bytes_read-b.bytes_read,"cache_hits":a.cache_hits-b.cache_hits})});
        return Ok(
            serde_json::json!({"seconds":seconds,"operation_start_unix_ns":operation_start_unix_ns,"operation_end_unix_ns":operation_end_unix_ns,"warmup_seconds":warm,"hash":hash,"timed_cache":delta}),
        );
    }
    let blocks: Vec<_> = (0..count)
        .map(|i| blocks::Block {
            center: target(i, n),
            mask: 255,
        })
        .collect();
    if mode == "estimate" {
        estimate(source, &blocks[..blocks.len().min(32)], budget)?;
        #[cfg(feature = "profiling")]
        terrustrial_core::execution::profiling::reset();
        let operation_start_unix_ns = operation_unix_ns();
        let start = Instant::now();
        let cpu_start = process_cpu_seconds();
        let cycles_start = process_cycles();
        let hash = estimate(source, &blocks, budget)?;
        let seconds = start.elapsed().as_secs_f64();
        let operation_end_unix_ns = operation_unix_ns();
        let cpu_seconds = cpu_start.zip(process_cpu_seconds()).map(|(a, b)| b - a);
        let cycles = cycles_start.zip(process_cycles()).map(|(a, b)| b - a);
        return Ok(
            serde_json::json!({"seconds":seconds,"operation_start_unix_ns":operation_start_unix_ns,"operation_end_unix_ns":operation_end_unix_ns,"operation_cpu_seconds":cpu_seconds,"operation_cycles":cycles,"hash":hash}),
        );
    }
    let mut path: Vec<_> = (0..count).collect();
    path.shuffle(&mut ChaCha8Rng::seed_from_u64(20261006));
    let start = Instant::now();
    if backend == "packed" {
        let store = persistence::Scheduled::build(
            &root.join("simulation.sqlite"),
            &blocks,
            &path,
            2.,
            budget,
            &Cancellation::default(),
        )?;
        let publication = *PUBLICATION.get().expect("publication configured");
        let mut store = match PLACEMENT.get() {
            Some(placement) if *PUBLICATION_DECODE.get().expect("publication decode configured") => {
                store.into_async_with_publication_decode(publication, budget, placement.thread_start())?
            }
            Some(placement) => store.into_async_with_thread_start(
                publication, budget, placement.thread_start(),
            )?,
            None => store.into_async_with_options(publication, budget)?,
        };
        let build = start.elapsed().as_secs_f64();
        let initial_bytes = std::fs::metadata(root.join("simulation.sqlite"))?.len();
        if std::env::var_os("TERRUSTRIAL_PROFILE_PHASE").is_some() {
            eprintln!("simulation-ready");
        }
        #[cfg(feature = "profiling")]
        terrustrial_core::execution::profiling::reset();
        let operation_start_unix_ns = operation_unix_ns();
        let start = Instant::now();
        let cpu_start = process_cpu_seconds();
        let cycles_start = process_cycles();
        let hash = simulate(source, &mut store, &blocks, &path, budget)?;
        let seconds = start.elapsed().as_secs_f64();
        let operation_end_unix_ns = operation_unix_ns();
        let cpu_seconds = cpu_start.zip(process_cpu_seconds()).map(|(a, b)| b - a);
        let cycles = cycles_start.zip(process_cycles()).map(|(a, b)| b - a);
        if !store.complete() {
            return Err("incomplete packed simulation".into());
        }
        Ok(
            serde_json::json!({"seconds":seconds,"operation_start_unix_ns":operation_start_unix_ns,"operation_end_unix_ns":operation_end_unix_ns,"simulation_cpu_seconds":cpu_seconds,"operation_cycles":cycles,"simulation_build_seconds":build,"initial_simulation_file_bytes":initial_bytes,"hash":hash,"storage":store.statistics()}),
        )
    } else {
        let mut store = SpatialIndex::with_budget(vec![], budget)?;
        let build = start.elapsed().as_secs_f64();
        if std::env::var_os("TERRUSTRIAL_PROFILE_PHASE").is_some() {
            eprintln!("simulation-ready");
        }
        #[cfg(feature = "profiling")]
        terrustrial_core::execution::profiling::reset();
        let operation_start_unix_ns = operation_unix_ns();
        let start = Instant::now();
        let cpu_start = process_cpu_seconds();
        let cycles_start = process_cycles();
        let hash = simulate(source, &mut store, &blocks, &path, budget)?;
        let seconds = start.elapsed().as_secs_f64();
        let operation_end_unix_ns = operation_unix_ns();
        let cpu_seconds = cpu_start.zip(process_cpu_seconds()).map(|(a, b)| b - a);
        let cycles = cycles_start.zip(process_cycles()).map(|(a, b)| b - a);
        Ok(
            serde_json::json!({"seconds":seconds,"operation_start_unix_ns":operation_start_unix_ns,"operation_end_unix_ns":operation_end_unix_ns,"simulation_cpu_seconds":cpu_seconds,"operation_cycles":cycles,"simulation_build_seconds":build,"hash":hash,"stored_values":store.len()}),
        )
    }
}
fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args().collect();
    let publication_decode = publication_decode_argument(&mut args)?;
    PUBLICATION_DECODE.set(publication_decode).expect("single benchmark run");
    let mut placement_arg = None;
    for index in (1..args.len()).rev() {
        if let Some(value) = args[index].strip_prefix("--placement=") {
            if placement_arg.replace(value.to_owned()).is_some() {
                return Err("specify --placement only once".into());
            }
            args.remove(index);
        }
    }
    let placement = match placement_arg {
        Some(value) => value,
        None => match std::env::var("TERRUSTRIAL_PLACEMENT") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "inherit".to_owned(),
            Err(error) => return Err(error.into()),
        },
    };
    if !(7..=10).contains(&args.len()) {
        return Err("storage_comparison resident|packed query|estimate|simulate POINTS OPERATIONS WORKERS CACHE_PAGES [WINDOW [COMMIT_GROUPS [QUEUED_GROUPS]]] [--placement=inherit|physical|physical-writer] [--publication-decode]; placement may also be set by TERRUSTRIAL_PLACEMENT; publication decoding requires packed simulate and physical-writer placement with two separate service cores".into());
    }
    let backend = &args[1];
    let mode = &args[2];
    let n: usize = args[3].parse()?;
    let count: usize = args[4].parse()?;
    let workers: usize = args[5].parse()?;
    let cache: usize = args[6].parse()?;
    let window: usize = args
        .get(7)
        .map(|arg| arg.parse())
        .transpose()?
        .unwrap_or(16);
    if !(1..=128).contains(&window) {
        return Err("benchmark window must be in 1..=128".into());
    }
    WINDOW.set(window).expect("single benchmark run");
    let commit_groups: usize = args
        .get(8)
        .map(|arg| arg.parse())
        .transpose()?
        .unwrap_or(persistence::DEFAULT_COMMIT_GROUPS);
    let queued_groups: usize = args
        .get(9)
        .map(|arg| arg.parse())
        .transpose()?
        .unwrap_or(persistence::DEFAULT_QUEUED_GROUPS);
    if commit_groups == 0
        || commit_groups > 8192
        || queued_groups < commit_groups
        || queued_groups > 16384
    {
        return Err(
            "publication requires 1..=8192 commit groups and commit..=16384 queued groups".into(),
        );
    }
    PUBLICATION
        .set(ScheduledOptions {
            maximum_group_observations: 56,
            commit_groups,
            queued_groups,
            retained_groups: (queued_groups + window + 1).max(128),
        })
        .expect("single benchmark run");
    if !["resident", "packed"].contains(&backend.as_str())
        || !["query", "estimate", "simulate"].contains(&mode.as_str())
        || n == 0
        || count == 0
        || workers == 0
        || n >= 1_000_000 / 7
    {
        return Err("invalid benchmark arguments or overlapping hard/target IDs".into());
    }
    validate_publication_decode(publication_decode, backend, mode, &placement)?;
    match placement.as_str() {
        "inherit" => {}
        "physical" | "physical-writer" => {
            if mode == "query" {
                return Err("physical placement applies to estimate/simulate worker startup, not query mode".into());
            }
            #[cfg(feature = "experimental-two-phase")]
            if std::env::var_os("TERRUSTRIAL_TWO_PHASE").is_some() {
                return Err("physical placement is unsupported by the experimental two-phase path".into());
            }
            let reserve_writer = placement == "physical-writer";
            if reserve_writer && !(backend == "packed" && mode == "simulate") {
                return Err("physical-writer placement requires packed simulation".into());
            }
            // Topology/restriction/capacity validation precedes any store build.
            let plan = if publication_decode {
                thread_placement::Placement::discover_with_publication_prepare(workers.min(window))?
            } else {
                thread_placement::Placement::discover(workers.min(window), reserve_writer)?
            };
            PLACEMENT.set(plan).expect("single benchmark placement plan");
        }
        _ => return Err("placement must be inherit, physical or physical-writer".into()),
    }
    let dir = tempfile::tempdir()?;
    // Large admission windows plus compute and bounded I/O workspaces can
    // reserve more than 8 GiB. This is a ceiling, not an eager allocation.
    let budget = MemoryBudget::new(16u64 << 30);
    let cancel = Cancellation::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let result = pool.install(|| -> std::result::Result<serde_json::Value, String> {
        let run = || -> Result<serde_json::Value> {
            let start = Instant::now();
            let (build, mut result) = if backend == "packed" {
                let mut b = PackedBuilder::create(
                    dir.path().join("hard.sqlite"),
                    Options {
                        leaf_records: 256,
                        cached_pages: cache,
                        build_window_leaves: 4096,
                        readers: workers,
                        directory_pages: 4096,
                        sqlite_cache_bytes: 16 << 20,
                        ..Options::default()
                    },
                    &budget,
                )?;
                for i in 0..n * 7 {
                    b.push(observation(i), 0, &cancel)?;
                }
                let engine = b.finish(&cancel)?;
                let source = engine.snapshot()?;
                let build = start.elapsed().as_secs_f64();
                let mut result = work(
                    &source,
                    backend,
                    mode,
                    n,
                    count,
                    workers,
                    &budget,
                    dir.path(),
                    Some(&engine),
                )?;
                result["conditioning_file_bytes"] =
                    serde_json::json!(std::fs::metadata(engine.path())?.len());
                result["conditioning_statistics"] =
                    serde_json::json!(format!("{:?}", engine.statistics()));
                (build, result)
            } else {
                let source =
                    SpatialIndex::with_budget((0..n * 7).map(observation).collect(), &budget)?;
                let build = start.elapsed().as_secs_f64();
                (
                    build,
                    work(
                        &source,
                        backend,
                        mode,
                        n,
                        count,
                        workers,
                        &budget,
                        dir.path(),
                        None,
                    )?,
                )
            };
            #[cfg(feature = "profiling")]
            {
                result["profile"] =
                    serde_json::json!(terrustrial_core::execution::profiling::snapshot());
            }
            result["conditioning_build_seconds"] = serde_json::json!(build);
            #[cfg(feature = "experimental-two-phase")]
            if let Some(stats) = TWO_PHASE.lock().unwrap().take() {
                result["two_phase"] = stats;
            }
            result["backend"] = serde_json::json!(backend);
            result["mode"] = serde_json::json!(mode);
            result["points"] = serde_json::json!(n);
            result["operations"] = serde_json::json!(count);
            result["workers"] = serde_json::json!(workers);
            result["placement"] = PLACEMENT.get().map_or_else(
                || serde_json::json!({"policy":"inherit"}),
                thread_placement::Placement::description,
            );
            result["window"] = serde_json::json!(window);
            if backend == "packed" && mode == "simulate" {
                result["publication"] = serde_json::json!({
                    "commit_groups": commit_groups, "queued_groups": queued_groups,
                    "retained_groups": PUBLICATION.get().unwrap().retained_groups,
                    "decode_helper_requested": publication_decode,
                });
            }
            result["cache_pages"] = serde_json::json!(cache);
            result["budget_bytes"] = serde_json::json!(budget.limit());
            Ok(result)
        };
        run().map_err(|e| e.to_string())
    })?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

#[cfg(test)]
mod publication_decode_configuration_tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn publication_decode_flag_is_opt_in_and_preserves_other_arguments() {
        let mut args = arguments(&["comparison", "packed", "simulate", "--placement=physical-writer"]);
        let original = args.clone();
        assert!(!publication_decode_argument(&mut args).unwrap());
        assert_eq!(args, original);
        args.insert(2, "--publication-decode".into());
        assert!(publication_decode_argument(&mut args).unwrap());
        assert_eq!(args, original);
    }

    #[test]
    fn publication_decode_flag_rejects_duplicates_and_values() {
        for values in [
            vec!["comparison", "--publication-decode", "--publication-decode"],
            vec!["comparison", "--publication-decode=1"],
            vec!["comparison", "--publication-decode=0"],
            vec!["comparison", "--publication-decode="],
        ] {
            assert!(publication_decode_argument(&mut arguments(&values)).is_err());
        }
    }

    #[test]
    fn publication_decode_requires_packed_simulation_and_reserved_placement() {
        assert!(validate_publication_decode(true, "packed", "simulate", "physical-writer").is_ok());
        for (backend, mode, placement) in [
            ("resident", "simulate", "physical-writer"),
            ("packed", "estimate", "physical-writer"),
            ("packed", "query", "physical-writer"),
            ("packed", "simulate", "inherit"),
            ("packed", "simulate", "physical"),
        ] {
            assert!(validate_publication_decode(true, backend, mode, placement).is_err());
            assert!(validate_publication_decode(false, backend, mode, placement).is_ok());
        }
    }
}
