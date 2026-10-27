//! Adapter for the existing packed, disk-backed engine. Geometry is staged once
//! with future ranks. A complete simulated group becomes visible atomically.
use super::{blocks::Block, fitting::Result};
use std::path::Path;
use terrustrial_core::{
    data::{Observation, Point3, Site},
    execution::{Cancellation, MemoryBudget, Reservation},
    spatial::{NearestWorkspace, ObservationSink, ObservationSource},
    storage::packed::{Options, PackedBuilder, PackedStore, Snapshot},
    Error,
};
// Throughput defaults for the bounded asynchronous examples. Commits remain
// FULL synchronous; finish acknowledges the final suffix. Callers can override.
pub const DEFAULT_COMMIT_GROUPS: usize = 512;
pub const DEFAULT_QUEUED_GROUPS: usize = 1024;

pub struct Scheduled {
    engine: PackedStore,
    snapshot: Snapshot,
    counts: Vec<usize>,
    rank: usize,
    pending: Vec<(u64, f64)>,
    cancel: Cancellation,
    _reservation: Reservation,
}
impl terrustrial_core::simulation::parallel::SnapshotStore for Scheduled {
    type View = Snapshot;
    fn deferred_page_reads(&self) -> bool {
        true
    }
    fn read_view(&mut self) -> terrustrial_core::Result<Snapshot> {
        Ok(self.snapshot.clone())
    }
    fn append_group(
        &mut self,
        _: &terrustrial_core::data::Group,
        observations: &[Observation],
    ) -> terrustrial_core::Result<()> {
        for o in observations {
            self.append(*o)?;
        }
        Ok(())
    }
}
fn invalid(s: &str) -> Error {
    Error::InvalidInput(s.into())
}
impl Scheduled {
    pub fn into_async(self, budget: &MemoryBudget) -> Result<AsyncScheduled> {
        self.into_async_with_options(
            terrustrial_core::storage::packed::ScheduledOptions {
                maximum_group_observations: 56,
                queued_groups: DEFAULT_QUEUED_GROUPS,
                commit_groups: DEFAULT_COMMIT_GROUPS,
                // Covers the examples' preparation windows of up to 64 groups.
                retained_groups: DEFAULT_QUEUED_GROUPS + 64 + 1,
            },
            budget,
        )
    }
    pub fn into_async_with_options(
        self,
        options: terrustrial_core::storage::packed::ScheduledOptions,
        budget: &MemoryBudget,
    ) -> Result<AsyncScheduled> {
        self.into_async_config(options, budget, None, false)
    }
    pub fn into_async_with_thread_start(
        self,
        options: terrustrial_core::storage::packed::ScheduledOptions,
        budget: &MemoryBudget,
        thread_start: terrustrial_core::execution::ThreadStart,
    ) -> Result<AsyncScheduled> {
        self.into_async_config(options, budget, Some(thread_start), false)
    }
    /// Use a bounded helper to decode cold publication pages before commit.
    /// The startup hook must place its preparation role on a separate core.
    pub fn into_async_with_publication_decode(
        self,
        options: terrustrial_core::storage::packed::ScheduledOptions,
        budget: &MemoryBudget,
        thread_start: terrustrial_core::execution::ThreadStart,
    ) -> Result<AsyncScheduled> {
        self.into_async_config(options, budget, Some(thread_start), true)
    }
    fn into_async_config(
        self,
        options: terrustrial_core::storage::packed::ScheduledOptions,
        budget: &MemoryBudget,
        thread_start: Option<terrustrial_core::execution::ThreadStart>,
        publication_decode: bool,
    ) -> Result<AsyncScheduled> {
        let expected = self.counts.len() as u64;
        let engine = self.engine.clone();
        let writer = if publication_decode {
            terrustrial_core::storage::packed::ScheduledWriter::new_with_publication_decode(
                self.engine,
                options,
                budget,
                &self.cancel,
                thread_start.ok_or_else(|| invalid("publication decode requires thread placement"))?,
            )?
        } else if let Some(thread_start) = thread_start {
            terrustrial_core::storage::packed::ScheduledWriter::new_with_thread_start(
                self.engine,
                options,
                budget,
                &self.cancel,
                thread_start,
            )?
        } else {
            terrustrial_core::storage::packed::ScheduledWriter::new(
                self.engine,
                options,
                budget,
                &self.cancel,
            )?
        };
        Ok(AsyncScheduled {
            writer,
            engine,
            expected,
        })
    }
    pub fn build(
        path: &Path,
        blocks: &[Block],
        order: &[usize],
        spacing: f64,
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        let reservation = budget.reserve((std::mem::size_of_val(order) + 56 * 16) as u64)?;
        let options = Options {
            leaf_records: 256,
            cached_pages: 131072,
            build_window_leaves: 4096,
            readers: 14,
            directory_pages: 4096,
            sqlite_cache_bytes: 16 << 20,
            wal_checkpoint_bytes: 64 << 20,
            max_database_bytes: 30u64 << 30,
        };
        let mut builder = PackedBuilder::create(path, options, budget)?;
        let mut counts = Vec::with_capacity(order.len());
        for (rank, &cell) in order.iter().enumerate() {
            let group = blocks[cell].group(rank, cell, spacing);
            counts.push(group.targets.len());
            for target in group.targets {
                builder.push(
                    Observation::new(target.id, target.site, 0., 0.)?,
                    rank as u64 + 1,
                    cancel,
                )?;
            }
        }
        let engine = builder.finish(cancel)?;
        let snapshot = engine.snapshot()?;
        Ok(Self {
            engine,
            snapshot,
            counts,
            rank: 0,
            pending: Vec::with_capacity(56),
            cancel: cancel.clone(),
            _reservation: reservation,
        })
    }
    #[cfg(test)]
    pub fn complete(&self) -> bool {
        self.rank == self.counts.len() && self.pending.is_empty()
    }
}
pub struct AsyncScheduled {
    writer: terrustrial_core::storage::packed::ScheduledWriter,
    engine: PackedStore,
    expected: u64,
}
impl AsyncScheduled {
    pub fn complete(&mut self) -> bool {
        self.writer
            .durable_group()
            .is_ok_and(|rank| rank == self.expected)
    }
    pub fn statistics(&mut self) -> serde_json::Value {
        let decode = self.writer.publication_decode_statistics();
        let mut wal = self.engine.path().as_os_str().to_os_string();
        wal.push("-wal");
        let wal_bytes = match std::fs::metadata(wal) {
            Ok(metadata) => Some(metadata.len()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some(0),
            Err(_) => None,
        };
        serde_json::json!({"drawn_groups": self.writer.drawn_group(), "completed_groups": self.writer.durable_group().ok(),
            "file_bytes": std::fs::metadata(self.engine.path()).ok().map(|m| m.len()),
            "wal_bytes": wal_bytes,
            "statistics": format!("{:?}", self.engine.statistics()),
            "publication_decode": {
                "admitted_current": decode.admitted_current,
                "decoded_pages": decode.decoded_pages,
                "decode_elapsed_ns": decode.decode_elapsed_ns,
                "collect_wait_elapsed_ns": decode.collect_wait_elapsed_ns,
                "max_inflight": decode.max_inflight,
            }})
    }
}
impl terrustrial_core::simulation::parallel::SnapshotStore for AsyncScheduled {
    type View = terrustrial_core::storage::packed::ScheduledView;
    fn deferred_page_reads(&self) -> bool {
        true
    }
    fn read_view(&mut self) -> terrustrial_core::Result<Self::View> {
        self.writer.read_view()
    }
    fn append_group(
        &mut self,
        group: &terrustrial_core::data::Group,
        observations: &[Observation],
    ) -> terrustrial_core::Result<()> {
        self.writer.append_group(group, observations)
    }
    fn finish(&mut self) -> terrustrial_core::Result<()> {
        self.writer.finish()
    }
}
impl ObservationSink for Scheduled {
    fn append(&mut self, o: Observation) -> terrustrial_core::Result<()> {
        let count = *self
            .counts
            .get(self.rank)
            .ok_or_else(|| invalid("extra scheduled simulation output"))?;
        self.pending.push((o.id(), o.value()));
        if self.pending.len() == count {
            self.snapshot =
                self.engine
                    .publish_group(self.rank as u64 + 1, &self.pending, &self.cancel)?;
            self.pending.clear();
            self.rank += 1;
        }
        Ok(())
    }
}
impl ObservationSource for Scheduled {
    fn scan(
        &self,
        visitor: &mut dyn FnMut(&Observation) -> terrustrial_core::Result<()>,
    ) -> terrustrial_core::Result<()> {
        self.snapshot.scan(visitor)
    }
    fn visit_bounds(
        &self,
        low: [f64; 3],
        high: [f64; 3],
        cancel: &Cancellation,
        visitor: &mut dyn FnMut(&Observation) -> terrustrial_core::Result<()>,
    ) -> terrustrial_core::Result<()> {
        self.snapshot.visit_bounds(low, high, cancel, visitor)
    }
    fn at_site(&self, site: &Site) -> terrustrial_core::Result<Vec<Observation>> {
        self.snapshot.at_site(site)
    }
    fn nearest_into<'a>(
        &self,
        point: Point3,
        workspace: &'a mut NearestWorkspace,
        cancel: &Cancellation,
    ) -> terrustrial_core::Result<&'a [Observation]> {
        self.snapshot.nearest_into(point, workspace, cancel)
    }
}

/// Measure the existing file layout on a deterministic, spatially distributed
/// subset before committing to the full run. This is a lower-bound-style disk
/// planning estimate: staging, journals and retained update versions need more.
pub fn admission(
    out: &Path,
    blocks: &[Block],
    spacing: f64,
    budget: &MemoryBudget,
    cancel: &Cancellation,
) -> Result<serde_json::Value> {
    let count = blocks.len().min(2048);
    let mut builder = PackedBuilder::create(
        out.join("storage-layout-sample.sqlite"),
        Options {
            leaf_records: 256,
            cached_pages: 256,
            build_window_leaves: 256,
            readers: 1,
            ..Options::default()
        },
        budget,
    )?;
    let mut records = 0u64;
    for rank in 0..count {
        let cell = rank * blocks.len() / count;
        for t in blocks[cell].group(rank, cell, spacing).targets {
            builder.push(
                Observation::new(t.id, t.site, 0., 0.)?,
                rank as u64 + 1,
                cancel,
            )?;
            records += 1;
        }
    }
    let store = builder.finish(cancel)?;
    let bytes = std::fs::metadata(store.path())?.len();
    let total: u64 = blocks.iter().map(|b| b.mask.count_ones() as u64 * 7).sum();
    let projected = (bytes as f64 / records as f64 * total as f64).ceil() as u64;
    let report = serde_json::json!({"engine":"storage::packed::PackedStore","blocks":blocks.len(),"scalar_values_per_realization":total,"sample_blocks":count,"sample_records":records,"sample_database_bytes":bytes,"projected_initial_database_bytes":projected,"excludes":"Additional build staging/journal peaks and retained payload generations during updates. This is an extrapolation, not measured full-run disk use.","maximum_database_bytes":30u64<<30,"admitted":projected<20u64<<30});
    std::fs::write(
        out.join("storage-admission.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use terrustrial_core::{
        model::Anisotropy,
        spatial::{SearchNeighborhood, SearchWorkspace, SpatialIndex, VariableLimit},
    };
    #[test]
    fn scheduled_packed_groups_match_resident_visibility_and_queries() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(1 << 30);
        let cancel = Cancellation::default();
        let blocks = vec![
            Block {
                center: Point3::new([0.; 3]).unwrap(),
                mask: 255,
            },
            Block {
                center: Point3::new([2., 0., 0.]).unwrap(),
                mask: 5,
            },
            Block {
                center: Point3::new([4., 0., 0.]).unwrap(),
                mask: 127,
            },
        ];
        let path = [2, 0, 1];
        let mut packed = Scheduled::build(
            &dir.path().join("store.sqlite"),
            &blocks,
            &path,
            2.,
            &budget,
            &cancel,
        )
        .unwrap();
        let mut resident = SpatialIndex::new(vec![]).unwrap();
        let q = SearchNeighborhood::new(
            49,
            Anisotropy::isotropic(20.).unwrap(),
            None,
            (0..7)
                .map(|variable| VariableLimit {
                    variable,
                    maximum: 7,
                })
                .collect(),
        )
        .unwrap();
        let mut a = SearchWorkspace::new(&q, &budget).unwrap();
        let mut b = SearchWorkspace::new(&q, &budget).unwrap();
        let keys = |os: &[Observation]| {
            os.iter()
                .map(|o| (o.id(), *o.site(), o.value().to_bits()))
                .collect::<Vec<_>>()
        };
        for (rank, &cell) in path.iter().enumerate() {
            let group = blocks[cell].group(rank, cell, 2.);
            let old = packed.snapshot.len();
            for (i, t) in group.targets.iter().enumerate() {
                let o = Observation::new(t.id, t.site, (t.id as f64).sin(), 0.).unwrap();
                packed.append(o).unwrap();
                resident.append(o).unwrap();
                if i + 1 < group.targets.len() {
                    assert_eq!(packed.snapshot.len(), old);
                }
            }
            assert_eq!(packed.snapshot.len(), resident.len() as u64);
            for p in [[0.; 3], [1., 0., 0.], [4., 0., 0.]] {
                let p = Point3::new(p).unwrap();
                assert_eq!(
                    keys(packed.search(p, &mut a, &cancel).unwrap()),
                    keys(resident.search(p, &mut b, &cancel).unwrap())
                );
            }
        }
        assert!(packed.complete());
        let reopened = PackedStore::open(
            dir.path().join("store.sqlite"),
            Options {
                leaf_records: 256,
                ..Options::default()
            },
            &budget,
        )
        .unwrap();
        assert_eq!(reopened.snapshot().unwrap().len(), resident.len() as u64);
    }
}
