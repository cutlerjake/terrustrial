//! Fixed-budget checked simulation with durable SQLite output and streamed input.
//! Rust requested allocations and process RSS are separate measurements. SQLite's
//! C allocations and allocator overhead are represented only by process RSS.
#[cfg(feature = "sqlite")]
mod run {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        sync::atomic::{AtomicU64, Ordering},
        time::Instant,
    };
    use terrustrial_core::{
        data::{Group, Observation, Point3, Site, Target},
        execution::{Cancellation, MemoryBudget},
        gaussian::Simple,
        identity,
        model::{Anisotropy, Coregionalization, Correlation, Lmc, Structure},
        simulation::{self, CheckpointedOptions, SimulationOptions},
        spatial::{Neighborhood, ObservationSource},
        storage::SqliteStore,
        Error,
    };
    pub struct Counting;
    static LIVE: AtomicU64 = AtomicU64::new(0);
    static PEAK: AtomicU64 = AtomicU64::new(0);
    static ALLOCS: AtomicU64 = AtomicU64::new(0);
    fn add(n: usize) {
        let live = LIVE.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
        PEAK.fetch_max(live, Ordering::Relaxed);
        ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
    // Delegate unchanged layouts and pointers to System; account only successful allocations.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(l) };
            if !p.is_null() {
                add(l.size());
            }
            p
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc_zeroed(l) };
            if !p.is_null() {
                add(l.size());
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) };
            LIVE.fetch_sub(l.size() as u64, Ordering::Relaxed);
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            let p = unsafe { System.realloc(p, l, n) };
            if !p.is_null() {
                LIVE.fetch_sub(l.size() as u64, Ordering::Relaxed);
                add(n);
            }
            p
        }
    }
    fn checksum(acc: &mut [u8; 32], o: &Observation) {
        let mut h = blake3::Hasher::new();
        h.update(&o.id().to_le_bytes());
        h.update(&o.value().to_bits().to_le_bytes());
        for (a, b) in acc.iter_mut().zip(h.finalize().as_bytes()) {
            *a ^= b;
        }
    }
    pub fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let args: Vec<_> = std::env::args().collect();
        if args.len() != 3 {
            return Err("usage: simulation_resources GROUPS NEW_SQLITE_PATH".into());
        }
        let count: u64 = args[1].parse()?;
        if !(1..=1_000_000).contains(&count) {
            return Err("groups must be 1..=1000000".into());
        }
        let budget = MemoryBudget::new(16 << 20);
        let storage_budget = MemoryBudget::new(8 << 20);
        let cancel = Cancellation::default();
        let model = Lmc::new(vec![Structure {
            correlation: Correlation::Exponential,
            anisotropy: Anisotropy::isotropic(2.)?,
            coregionalization: Coregionalization::from_factor(2, 2, &[1., 0., 0.4, 0.9])?,
        }])?;
        let mut store = SqliteStore::create(&args[2], 4 << 20, &storage_budget)?;
        let hard: Vec<_> = (0..2)
            .map(|v| {
                Observation::new(
                    v as u64,
                    Site::point(Point3::new([-1., 0., 0.]).unwrap(), v),
                    v as f64,
                    0.,
                )
                .unwrap()
            })
            .collect();
        let groups = (0..count).map(|i| Group {
            id: i,
            targets: (0..2)
                .map(|v| Target {
                    id: 10 + i * 2 + v as u64,
                    site: Site::point(Point3::new([i as f64 / 16., 0., 0.]).unwrap(), v),
                })
                .collect(),
        });
        let policy = identity::opaque(
            "resource acceptance policy",
            b"componentwise residual <=1e-10; no lost products",
        )?;
        let options = SimulationOptions {
            seed: 821,
            realization: 0,
            mean: None,
            hard_neighborhood: Neighborhood {
                maximum: 2,
                radius: 1.1,
            }
            .into(),
            simulated_neighborhood: Neighborhood {
                maximum: 32,
                radius: 1.1,
            }
            .into(),
            budget: &budget,
            cancellation: &cancel,
        };
        let mut output_hash = [0u8; 32];
        let mut outputs = 0u64;
        let mut maximum_residual = 0f64;
        let mut peak_budget = 0;
        let started = Instant::now();
        simulation::simulate_checkpointed_with_residuals(
            &model,
            hard.as_slice(),
            &mut store,
            groups,
            &Simple,
            CheckpointedOptions {
                simulation: options,
                identity: [67; 32],
            },
            policy,
            |group, report| {
                let r = report.equations;
                for block in [
                    Some(r.covariance_equations),
                    Some(r.drift_constraints),
                    Some(r.original_drift_constraints),
                    r.conditional_covariance,
                    r.sampling_factor,
                ]
                .into_iter()
                .flatten()
                {
                    maximum_residual = maximum_residual.max(block.max_componentwise_backward_error);
                    if block.max_componentwise_backward_error > 1e-10
                        || block.scaled_product_underflows != 0
                    {
                        eprintln!("group {} rejected: {:?}", group.id, block);
                        return Err(Error::Numerical {
                            context: "resource residual gate",
                        });
                    }
                }
                peak_budget = peak_budget.max(budget.used());
                Ok(())
            },
            |rows| {
                for o in rows {
                    checksum(&mut output_hash, o);
                    outputs += 1;
                }
                if outputs.is_multiple_of(20_000) {
                    eprintln!("{} of {count} groups committed", outputs / 2);
                }
                Ok(())
            },
        )?;
        let seconds = started.elapsed().as_secs_f64();
        assert_eq!(budget.used(), 0);
        assert_eq!(outputs, count * 2);
        assert_eq!(store.checkpoint()?.unwrap().completed_group, count - 1);
        drop(store);
        assert_eq!(storage_budget.used(), 0);
        let store = SqliteStore::open(&args[2], 4 << 20, &storage_budget)?;
        let mut read_hash = [0u8; 32];
        let mut read_count = 0;
        store.scan(&mut |o| {
            checksum(&mut read_hash, o);
            read_count += 1;
            Ok(())
        })?;
        assert_eq!(output_hash, read_hash);
        assert_eq!(read_count, outputs);
        drop(store);
        assert_eq!(storage_budget.used(), 0);
        println!("metric,value\ngroups,{count}\nobservations,{outputs}\nseconds,{seconds}\ndatabase_bytes,{}\nmaximum_report_residual,{maximum_residual}\npeak_driver_reservation,{peak_budget}\ndriver_limit,{}\nstorage_limit,{}\nrust_requested_peak_bytes,{}\nrust_requested_live_bytes,{}\nrust_allocation_count,{}\nreopened_payload_match,1",std::fs::metadata(&args[2])?.len(),budget.limit(),storage_budget.limit(),PEAK.load(Ordering::Relaxed),LIVE.load(Ordering::Relaxed),ALLOCS.load(Ordering::Relaxed));
        Ok(())
    }
}
#[cfg(feature = "sqlite")]
#[global_allocator]
static ALLOCATOR: run::Counting = run::Counting;
#[cfg(feature = "sqlite")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    run::main()
}
#[cfg(not(feature = "sqlite"))]
fn main() {
    panic!("simulation_resources requires sqlite feature")
}
