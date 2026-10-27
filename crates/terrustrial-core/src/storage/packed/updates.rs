use super::*;
use builder::{build_segment, page_scratch_bytes, prepare_staging, scratch_bytes, stage};

pub(super) struct ValueUpdate {
    id: u64,
    value: f64,
    page: u64,
    slot: usize,
}

#[derive(Clone, Copy)]
struct Identity {
    rank: u64,
    id: u64,
    page: u64,
    slot: usize,
}
impl Identity {
    fn read(row: &rusqlite::Row<'_>) -> Result<Self> {
        let id = row
            .get_ref(1)?
            .as_blob()
            .map_err(|_| invalid("packed identity"))?;
        Ok(Self {
            rank: sql_u64(row, 0)?,
            id: u64::from_be_bytes(
                id.try_into()
                    .map_err(|_| invalid("packed identity length"))?,
            ),
            page: sql_u64(row, 2)?,
            slot: sql_usize(row, 3)?,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IdentityStamp {
    generation: u64,
    visible: u64,
    external_changes: i64,
    local_changes: u64,
}
impl IdentityStamp {
    // Compare only on this store's writer connection, inside a transaction.
    // data_version detects other connections, including external edits which
    // do not update our manifest. total_changes detects this connection's DML.
    fn read(c: &Connection, generation: u64, visible: u64) -> Result<Self> {
        Ok(Self {
            generation,
            visible,
            external_changes: c.pragma_query_value(None, "data_version", |r| r.get(0))?,
            local_changes: c.total_changes(),
        })
    }
}
pub(super) struct PreparedIdentities {
    entries: Vec<Identity>,
    pages: Vec<u64>,
    capacity: usize,
    through: u64,
    capacity_limited_at: Option<u64>,
    stamp: Option<IdentityStamp>,
    owner: Weak<Inner>,
    _memory: Reservation,
    #[cfg(test)]
    scanned_rows: usize,
}
impl PreparedIdentities {
    pub(super) fn new(capacity: usize, budget: &MemoryBudget) -> Result<Self> {
        let capacity = capacity.min(65536);
        let memory = budget.reserve(
            (capacity * std::mem::size_of::<Identity>()
                + capacity.min(4096) * 8
                + std::mem::size_of::<Self>()) as u64,
        )?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(capacity)
            .map_err(|_| invalid("publication identities allocation"))?;
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(capacity.min(4096))
            .map_err(|_| invalid("publication page preparation allocation"))?;
        Ok(Self {
            entries,
            pages,
            capacity,
            through: 0,
            capacity_limited_at: None,
            stamp: None,
            owner: Weak::new(),
            _memory: memory,
            #[cfg(test)]
            scanned_rows: 0,
        })
    }
}

fn latest_payload(
    c: &Connection,
    page: u64,
    generation: u64,
    capacity: usize,
    compact: bool,
) -> Result<Vec<u8>> {
    payload::fetch(c, page, generation, capacity, compact)?.materialize(capacity)
}
fn latest_payload_into(
    inner: &Inner,
    c: &Connection,
    page: u64,
    generation: u64,
    capacity: usize,
    payload: &mut Vec<u8>,
) -> Result<()> {
    // Published payloads are immutable. A proven cached version is also a valid
    // read/modify/write base, including when unrelated pages advanced the store.
    let cached = {
        let cache = inner.cache[inner.partition(page)]
            .read()
            .map_err(|_| invalid("packed cache lock poisoned"))?;
        cache
            .get(Key {
                page,
                generation,
                payload: true,
            })
            .map(|entry| entry.value.clone())
    };
    if let Some(cached) = cached {
        let Page::Payload(observations) = &cached.page else {
            return Err(invalid("packed payload cache kind"));
        };
        payload.clear();
        for slot in 0..observations.len() {
            payload.extend_from_slice(&encode_record(&observations.observation(slot)?));
        }
        return Ok(());
    }
    let body = latest_payload(c, page, generation, capacity, inner.compact_values)?;
    payload.clear();
    payload.extend_from_slice(&body);
    Ok(())
}
// Share the checksum-validated immutable base and copy only values. The old
// cached version stays resident for readers while the new version is staged.
// Charge the values copy first; if it cannot be admitted, use the serialized path.
#[allow(clippy::too_many_arguments)]
fn publish_cached(
    inner: &Inner,
    c: &Connection,
    page: u64,
    generation: u64,
    next_generation: u64,
    updates: &[ValueUpdate],
    warmed: &mut Vec<(Key, Arc<Cached>)>,
    values: &mut Vec<u8>,
) -> Result<bool> {
    if !inner.compact_values || warmed.len() == warmed.capacity() {
        return Ok(false);
    }
    let cached = {
        let cache = inner.cache[inner.partition(page)]
            .read()
            .map_err(|_| invalid("packed cache lock poisoned"))?;
        cache
            .get(Key {
                page,
                generation,
                payload: true,
            })
            .map(|entry| entry.value.clone())
    };
    let Some(cached) = cached else {
        return Ok(false);
    };
    let _span = Span::new(Metric::PublishCached);
    let encode = Stamp::now();
    let Page::Payload(base) = &cached.page else {
        return Err(invalid("packed payload cache kind"));
    };
    let Ok(reservation) = inner.budget.reserve(inner.value_charge()) else {
        return Ok(false);
    };
    let mut copied_values = Vec::new();
    copied_values
        .try_reserve_exact(base.len())
        .map_err(|_| invalid("packed publication copy allocation"))?;
    copied_values.extend_from_slice(&base.values);
    let mut payload = Payload {
        base: base.base.clone(),
        values: copied_values,
    };
    for update in updates {
        payload.update(update.slot, update.id, update.value)?;
    }
    values.clear();
    for value in &payload.values {
        values.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    encode.record(Metric::PublishEncode);
    payload::write_value_bytes(c, page, next_generation, values)?;
    warmed.push((
        Key {
            page,
            generation: next_generation,
            payload: true,
        },
        Arc::new(Cached {
            page: Page::Payload(payload),
            _reservation: reservation,
        }),
    ));
    Ok(true)
}
// Optional write-through residency never makes an otherwise admissible commit
// fail for lack of cache memory. Decode before commit so corrupt payloads cannot
// become acknowledged publications. Staged pages are invisible to readers.
fn warm_payload(
    inner: &Inner,
    warmed: &mut Vec<(Key, Arc<Cached>)>,
    page: u64,
    generation: u64,
    payload: &[u8],
) -> Result<()> {
    let _span = Span::new(Metric::PublishDecode);
    if warmed.len() == warmed.capacity() {
        return Ok(());
    }
    let Ok(reservation) = inner.budget.reserve(inner.page_charge(true)) else {
        return Ok(());
    };
    let value = Arc::new(Cached::payload(
        payload,
        inner.options.leaf_records,
        reservation,
    )?);
    warmed.push((
        Key {
            page,
            generation,
            payload: true,
        },
        value,
    ));
    Ok(())
}
pub(super) fn manifest(c: &Connection) -> Result<(u64, u64, u64, Vec<Node>)> {
    let (g,v,n,b,crc): (u64,u64,u64,Vec<u8>,u32) = c.query_row("SELECT generation,visible,eligible,CASE WHEN length(roots)<=?1 THEN roots ELSE NULL END,roots_crc FROM manifest WHERE singleton=1",[(8+MAX_ROOTS*72) as i64],|r| Ok((sql_u64(r,0)?,sql_u64(r,1)?,sql_u64(r,2)?,r.get(3)?,r.get(4)?)))?;
    if crc32fast::hash(&b) != crc {
        return Err(invalid("packed manifest checksum mismatch"));
    }
    Ok((g, v, n, decode_nodes(&b, MAX_ROOTS)?))
}
fn publish(
    c: &Connection,
    generation: u64,
    visible: u64,
    eligible: u64,
    roots: &[Node],
) -> Result<()> {
    let body = encode_nodes(roots);
    c.execute("UPDATE manifest SET generation=?1,visible=?2,eligible=?3,roots=?4,roots_crc=?5 WHERE singleton=1",params![sql_id(generation)?,sql_id(visible)?,sql_id(eligible)?,body,crc32fast::hash(&body)])?;
    Ok(())
}
fn stage_existing(
    c: &Connection,
    generation: u64,
    capacity: usize,
    compact: bool,
    cancel: &Cancellation,
) -> Result<()> {
    // Identities describes the current generation only. Old readers use their
    // immutable geometry roots and versioned payloads, never this update map.
    let mut s = c.prepare("SELECT page FROM identities GROUP BY page")?;
    let mut rows = s.query([])?;
    while let Some(r) = rows.next()? {
        cancel.check()?;
        let page = sql_u64(r, 0)?;
        let payload = latest_payload(c, page, generation, capacity, compact)?;
        let (body, crc): (Vec<u8>, u32) = c.query_row(
            "SELECT body,crc FROM geometry WHERE id=?1 AND kind=2 AND length(body)<=?2",
            params![sql_id(page)?, (capacity * 56 + 16) as i64],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        if crc32fast::hash(&body) != crc {
            return Err(invalid("packed compaction geometry integrity"));
        }
        let Page::Leaf(leaf) = decode_page(2, &body, capacity)? else {
            unreachable!()
        };
        if leaf.ranks.len() * RECORD_BYTES != payload.len() {
            return Err(invalid("packed compaction association"));
        }
        for (record, rank) in payload.as_chunks::<RECORD_BYTES>().0.iter().zip(leaf.ranks) {
            stage(c, decode_record(record)?, rank)?;
        }
    }
    Ok(())
}
impl PackedStore {
    pub(super) fn prepare_identities(
        &self,
        plan: &mut PreparedIdentities,
        through: u64,
        cancel: &Cancellation,
    ) -> Result<()> {
        let _span = Span::new(Metric::PublishPrepare);
        cancel.check()?;
        let mut c = lock(&self.inner.writer)?;
        // A short read transaction pins a consistent source without retaining
        // a writer lock while waiting for more simulation groups to arrive.
        let tx = c.transaction()?;
        let (generation, visible, _, _) = manifest(&tx)?;
        let stamp = IdentityStamp::read(&tx, generation, visible)?;
        let owner = Arc::downgrade(&self.inner);
        if plan.stamp != Some(stamp) || !Weak::ptr_eq(&plan.owner, &owner) {
            plan.entries.clear();
            plan.through = visible;
            plan.capacity_limited_at = None;
            plan.stamp = Some(stamp);
            plan.owner = owner;
        }
        let through_sql = sql_id(through)?;
        // Under the same identity stamp, a range that exceeded capacity still
        // cannot fit when extended. Smaller ranges may succeed, so remember
        // the failed endpoint separately from the retained complete prefix.
        if through <= plan.through
            || plan
                .capacity_limited_at
                .is_some_and(|failed| through >= failed)
        {
            return Ok(());
        }
        let before = plan.entries.len();
        let result = (|| {
            let mut statement = tx.prepare_cached("SELECT rank,id,page,slot FROM identities WHERE rank>?1 AND rank<=?2 ORDER BY rank,id LIMIT ?3")?;
            let mut rows = statement.query(params![
                sql_id(plan.through)?,
                through_sql,
                (plan.capacity - before + 1) as i64
            ])?;
            while let Some(row) = rows.next()? {
                cancel.check()?;
                #[cfg(test)]
                {
                    plan.scanned_rows += 1;
                }
                if plan.entries.len() == plan.capacity {
                    return Ok(false);
                }
                plan.entries.push(Identity::read(row)?);
            }
            Ok(true)
        })();
        match result {
            Ok(true) => {
                plan.through = through;
                // Reuse the identity scan and this short reader transaction to
                // stage validated immutable payloads. This is optional cache
                // work, independent of the final drawn values and commit.
                plan.pages.clear();
                plan.pages.extend(
                    plan.entries[before..]
                        .iter()
                        .take(plan.capacity.min(4096))
                        .map(|identity| identity.page),
                );
                plan.pages.sort_unstable();
                plan.pages.dedup();
                let _ = self.warm_prepared_pages(&tx, generation, &plan.pages, cancel);
                Ok(())
            }
            Ok(false) => {
                plan.capacity_limited_at = Some(through);
                plan.entries.truncate(before);
                Ok(())
            }
            Err(error) => {
                plan.entries.truncate(before);
                Err(error)
            }
        }
    }
    fn warm_prepared_pages(
        &self,
        c: &Connection,
        generation: u64,
        pages: &[u64],
        cancel: &Cancellation,
    ) -> Result<()> {
        let _span = Span::new(Metric::PublishWarm);
        if pages.len() > self.inner.options.cached_pages / 4
            || self
                .inner
                .page_charge(true)
                .saturating_mul(pages.len() as u64)
                > self.inner.budget.limit() - self.inner.budget.used()
        {
            return Ok(());
        }
        for &page in pages {
            cancel.check()?;
            let partition = self.inner.partition(page);
            let key = Key {
                page,
                generation,
                payload: true,
            };
            if self.inner.cache[partition]
                .read()
                .map_err(|_| invalid("packed cache lock poisoned"))?
                .get(key)
                .is_some()
            {
                continue;
            }
            // Do not evict pages to satisfy speculative memory admission.
            let reservation = self.inner.budget.reserve(self.inner.page_charge(true))?;
            let encoded = payload::fetch(
                c,
                page,
                generation,
                self.inner.options.leaf_records,
                self.inner.compact_values,
            )?;
            let stored = encoded.generation;
            let bytes = encoded.bytes();
            let blobs = encoded.blobs();
            let body = encoded.materialize(self.inner.options.leaf_records)?;
            let value = Arc::new(Cached::payload(
                &body,
                self.inner.options.leaf_records,
                reservation,
            )?);
            self.inner
                .counters
                .reads
                .fetch_add(blobs, Ordering::Relaxed);
            self.inner
                .counters
                .bytes
                .fetch_add(bytes as u64, Ordering::Relaxed);
            let mut cache = self.inner.cache[partition]
                .write()
                .map_err(|_| invalid("packed cache lock poisoned"))?;
            let key = Key {
                generation: stored,
                ..key
            };
            if let Some(existing) = cache.entries.get(&key) {
                existing
                    .valid_through
                    .fetch_max(generation, Ordering::Relaxed);
            } else {
                cache.insert(key, value, generation);
            }
        }
        Ok(())
    }
    /// Publish every value in the next scheduled group in one transaction.
    /// Geometry and noise remain unchanged. Duplicate/missing IDs, wrong ranks,
    /// cancellation and nonfinite values abort the entire publication.
    pub fn publish_group(
        &self,
        group: u64,
        values: &[(u64, f64)],
        cancel: &Cancellation,
    ) -> Result<Snapshot> {
        self.publish_groups(&[(group, values)], cancel)
    }

    /// Atomically publish a contiguous sequence of complete scheduled groups.
    /// Each affected payload page is rewritten once, even when several groups
    /// share it. A rejected group rolls back the complete transaction.
    pub fn publish_groups(
        &self,
        groups: &[(u64, &[(u64, f64)])],
        cancel: &Cancellation,
    ) -> Result<Snapshot> {
        self.publish_prepared(groups, cancel, None)
    }
    #[cfg(test)]
    pub(super) fn publish_staged(
        &self,
        groups: &[(u64, &[(u64, f64)])],
        cancel: &Cancellation,
        prepared: &mut Option<PreparedIdentities>,
    ) -> Result<Snapshot> {
        self.publish_staged_with_decode(groups, cancel, prepared, &mut None)
    }
    pub(super) fn publish_staged_with_decode(
        &self,
        groups: &[(u64, &[(u64, f64)])],
        cancel: &Cancellation,
        prepared: &mut Option<PreparedIdentities>,
        decoder: &mut Option<publication_decode::Decoder>,
    ) -> Result<Snapshot> {
        match self.publish_prepared_with_decode(groups, cancel, prepared.as_ref(), decoder.as_mut())
        {
            Err(Error::MemoryBudget { .. }) if prepared.is_some() || decoder.is_some() => {
                // Every fallible reservation in publication precedes commit.
                // Reclaim optional identity storage, then retry the rolled-back
                // transaction with its ordinary live validation cursor.
                *prepared = None;
                // Stop/join and release every optional helper allocation before
                // retrying. Merely marking it stopped retains its fixed arena.
                *decoder = None;
                self.publish_groups(groups, cancel)
            }
            result => result,
        }
    }
    pub(super) fn publish_prepared(
        &self,
        groups: &[(u64, &[(u64, f64)])],
        cancel: &Cancellation,
        prepared: Option<&PreparedIdentities>,
    ) -> Result<Snapshot> {
        self.publish_prepared_with_decode(groups, cancel, prepared, None)
    }
    fn publish_prepared_with_decode(
        &self,
        groups: &[(u64, &[(u64, f64)])],
        cancel: &Cancellation,
        prepared: Option<&PreparedIdentities>,
        mut decoder: Option<&mut publication_decode::Decoder>,
    ) -> Result<Snapshot> {
        if groups.is_empty() {
            return Err(invalid("empty simulation publication"));
        }
        cancel.check()?;
        let _scratch = self
            .inner
            .reserve_work(page_scratch_bytes(self.inner.options))?;
        let mut c = lock(&self.inner.writer)?;
        let validation = Stamp::now();
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let (generation, visible, eligible, roots) = manifest(&tx)?;
        let prepared = if let Some(plan) = prepared {
            (Weak::ptr_eq(&plan.owner, &Arc::downgrade(&self.inner))
                && plan.stamp == Some(IdentityStamp::read(&tx, generation, visible)?))
            .then_some(plan)
        } else {
            None
        };
        let value_count = groups.iter().try_fold(0usize, |count, (_, values)| {
            count.checked_add(values.len()).ok_or(Error::SizeOverflow)
        })?;
        let mut reusable = lock(&self.inner.publication)?;
        let result = (|| {
            let PublicationScratch {
                changed,
                reservation,
                updates,
                update_reservation,
                warmed,
                warm_reservation,
            } = &mut *reusable;
            self.inner
                .admit(|| crate::execution::buffer::grow(changed, value_count, reservation))?;
            changed.clear();
            warmed.clear();
            self.inner.admit(|| {
                crate::execution::buffer::grow(updates, value_count, update_reservation)
            })?;
            updates.clear();
            let mut last = visible;
            let mut added = 0u64;
            {
                // One covering-index scan validates the complete requested prefix.
                // Advancing exactly once per supplied ID also detects skipped ranks
                // and incomplete preceding groups without a query for each group.
                let mut statement = tx.prepare_cached(
                    "SELECT rank,id,page,slot FROM identities WHERE rank>?1 ORDER BY rank,id",
                )?;
                let mut rows =
                    statement.query([sql_id(prepared.map_or(visible, |plan| plan.through))?])?;
                let mut cached = prepared
                    .map_or(&[][..], |plan| plan.entries.as_slice())
                    .iter();
                let mut next = || -> Result<Option<Identity>> {
                    if let Some(identity) = cached.next() {
                        return Ok(Some(*identity));
                    }
                    rows.next()?.map(Identity::read).transpose()
                };
                for (group, values) in groups {
                    if *group <= last || values.is_empty() {
                        return Err(invalid("publish the next scheduled simulation group"));
                    }
                    let start = updates.len();
                    for &(id, value) in *values {
                        if !value.is_finite() {
                            return Err(invalid("nonfinite simulation value"));
                        }
                        updates.push(ValueUpdate {
                            id,
                            value,
                            page: 0,
                            slot: 0,
                        });
                    }
                    let selected = &mut updates[start..];
                    selected.sort_unstable_by_key(|update| update.id);
                    for update in selected {
                        cancel.check()?;
                        let identity =
                            next()?.ok_or_else(|| invalid("incomplete simulation group"))?;
                        if identity.rank != *group {
                            return Err(invalid(
                                "publish the next complete scheduled simulation group",
                            ));
                        }
                        if identity.id != update.id {
                            return Err(invalid("simulation group IDs do not match schedule"));
                        }
                        update.page = identity.page;
                        update.slot = identity.slot;
                    }
                    let expected = values.len() as u64;
                    last = *group;
                    added = added.checked_add(expected).ok_or(Error::SizeOverflow)?;
                }
                let following_rank = if let Some(identity) = cached.next() {
                    Some(identity.rank)
                } else {
                    rows.next()?.map(|row| sql_u64(row, 0)).transpose()?
                };
                if let Some(rank) = following_rank {
                    if rank <= last {
                        return Err(invalid("incomplete simulation group"));
                    }
                }
            }
            let next_generation = generation.checked_add(1).ok_or(Error::SizeOverflow)?;
            publish(
                &tx,
                next_generation,
                last,
                eligible.checked_add(added).ok_or(Error::SizeOverflow)?,
                &roots,
            )?;
            cancel.check()?;
            let snapshot = self.snapshot_at(&tx)?;
            let warm_capacity = value_count.min(self.inner.options.cached_pages / 2);
            // Cache staging is optional and cannot consume the reservations
            // required for the transaction's update buffers or result snapshot.
            let _ = crate::execution::buffer::grow(warmed, warm_capacity, warm_reservation);
            validation.record(Metric::PublishValidate);
            let payload_time = Stamp::now();
            // Slots are unique after identity validation; writes to distinct
            // slots commute. Only page grouping is needed, not slot ordering.
            updates.sort_unstable_by_key(|update| update.page);
            let mut payload = Vec::new();
            payload
                .try_reserve_exact(self.inner.options.leaf_records * RECORD_BYTES)
                .map_err(|_| invalid("packed publication payload allocation"))?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(self.inner.options.leaf_records * 8)
                .map_err(|_| invalid("packed value serialization allocation"))?;
            for page_updates in updates.chunk_by(|a, b| a.page == b.page) {
                cancel.check()?;
                if let Some(decoder) = decoder.as_deref_mut().filter(|d| d.active()) {
                    // Cached publication also consumes warmed capacity. Do not
                    // let it take slots already promised to queued decode work.
                    if warmed.len() + decoder.inflight() >= warmed.capacity() {
                        decoder.drain(warmed, cancel)?;
                    }
                }
                let page = page_updates[0].page;
                changed.push(page);
                if publish_cached(
                    &self.inner,
                    &tx,
                    page,
                    generation,
                    next_generation,
                    page_updates,
                    warmed,
                    &mut values,
                )? {
                    continue;
                }
                let _cold_span = Span::new(Metric::PublishCold);
                let fetch = Stamp::now();
                latest_payload_into(
                    &self.inner,
                    &tx,
                    page,
                    generation,
                    self.inner.options.leaf_records,
                    &mut payload,
                )?;
                fetch.record(Metric::PublishFetch);
                for update in page_updates {
                    let first = update
                        .slot
                        .checked_mul(RECORD_BYTES)
                        .and_then(|x| x.checked_add(88))
                        .ok_or(Error::SizeOverflow)?;
                    payload
                        .get_mut(first..first + 8)
                        .ok_or_else(|| invalid("packed value slot"))?
                        .copy_from_slice(&update.value.to_bits().to_le_bytes());
                }
                payload::write_values(
                    &tx,
                    page,
                    next_generation,
                    &payload,
                    self.inner.compact_values,
                    &mut values,
                )?;
                if let Some(decoder) = decoder.as_deref_mut().filter(|d| d.active()) {
                    decoder.stage(
                        &self.inner,
                        page,
                        next_generation,
                        &mut payload,
                        warmed,
                        cancel,
                    )?;
                } else {
                    warm_payload(&self.inner, warmed, page, next_generation, &payload)?;
                }
            }
            if let Some(decoder) = decoder.as_deref_mut().filter(|d| d.active()) {
                decoder.drain(warmed, cancel)?;
                // Cache staging interleaves cached pages with decoded pages.
                // Reestablish the original increasing-page installation order.
                warmed.sort_unstable_by_key(|(key, _)| key.page);
            }
            cancel.check()?;
            payload_time.record(Metric::PublishPayload);
            let commit = Stamp::now();
            tx.commit()?;
            commit.record(Metric::PublishCommit);
            let cache = Stamp::now();
            self.inner.published(generation, next_generation, changed)?;
            self.inner.install_published(warmed)?;
            cache.record(Metric::PublishCache);
            Ok(snapshot)
        })();
        if result.is_err() {
            if let Some(decoder) = decoder {
                decoder.stop_and_join();
            }
        }
        reusable.warmed.clear();
        result
    }

    /// Stream an atomic batch of new conditioning observations. At most 32
    /// immutable segments are searched. When full, a bounded-memory rebuild
    /// consolidates current geometry and the incoming batch before publication.
    /// A consolidation can be expensive; it is synchronous and cancellable.
    pub fn append_batch(
        &self,
        records: impl IntoIterator<Item = Result<Observation>>,
        cancel: &Cancellation,
    ) -> Result<Snapshot> {
        self.rebuild_or_append(Some(records.into_iter()), false, cancel)
    }
    /// Repack all current observations into one segment while preserving old
    /// snapshots. This improves layout, but retains historical pages on disk.
    pub fn compact(&self, cancel: &Cancellation) -> Result<Snapshot> {
        self.rebuild_or_append(None::<std::iter::Empty<Result<Observation>>>, true, cancel)
    }
    fn rebuild_or_append<I: Iterator<Item = Result<Observation>>>(
        &self,
        records: Option<I>,
        force: bool,
        cancel: &Cancellation,
    ) -> Result<Snapshot> {
        cancel.check()?;
        let _scratch = self.inner.reserve_work(scratch_bytes(self.inner.options))?;
        let mut c = lock(&self.inner.writer)?;
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let (generation, visible, eligible, mut roots) = manifest(&tx)?;
        let next_generation = generation.checked_add(1).ok_or(Error::SizeOverflow)?;
        let consolidate = force || roots.len() == MAX_ROOTS;
        prepare_staging(&tx)?;
        if consolidate {
            stage_existing(
                &tx,
                generation,
                self.inner.options.leaf_records,
                self.inner.compact_values,
                cancel,
            )?;
        }
        let mut added = 0u64;
        if let Some(records) = records {
            for o in records {
                cancel.check()?;
                let o = o?;
                // Check before stage construction, including IDs in other
                // segments. This never merges independent colocated records.
                let exists: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM identities WHERE id=?1)",
                    [o.id().to_be_bytes().as_slice()],
                    |r| r.get(0),
                )?;
                if exists {
                    return Err(invalid("duplicate observation ID"));
                }
                stage(&tx, o, 0)?;
                added = added.checked_add(1).ok_or(Error::SizeOverflow)?;
            }
        }
        if consolidate {
            tx.execute("DELETE FROM identities", [])?;
            roots.clear();
        }
        if let Some(root) = build_segment(&tx, self.inner.options, next_generation, cancel)? {
            roots.push(root);
        }
        publish(
            &tx,
            next_generation,
            visible,
            eligible.checked_add(added).ok_or(Error::SizeOverflow)?,
            &roots,
        )?;
        cancel.check()?;
        let snapshot = self.snapshot_at(&tx)?;
        cancel.check()?;
        tx.commit()?;
        self.inner.observe_generation(next_generation)?;
        Ok(snapshot)
    }
    /// Copy the current geometry and latest values to a new file, preserving
    /// scheduled groups. The source and all its snapshots remain untouched.
    /// Historical versions are omitted. The destination is never overwritten.
    pub fn export_compacted(
        &self,
        path: impl AsRef<Path>,
        options: Options,
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<PackedStore> {
        let c = lock(&self.inner.writer)?;
        // Hold a read transaction so other processes cannot change the current
        // identity map while this export streams it.
        let tx = c.unchecked_transaction()?;
        let (generation, visible, _, _) = manifest(&tx)?;
        let mut builder = PackedBuilder::create(path, options, budget)?;
        let mut s = tx.prepare("SELECT id,page,slot,rank FROM identities ORDER BY page,slot")?;
        let mut rows = s.query([])?;
        let mut previous = None;
        let mut payload = Vec::new();
        let _scratch = budget.reserve(scratch_bytes(options))?;
        while let Some(r) = rows.next()? {
            cancel.check()?;
            let page = sql_u64(r, 1)?;
            if previous != Some(page) {
                payload = latest_payload(
                    &tx,
                    page,
                    generation,
                    self.inner.options.leaf_records,
                    self.inner.compact_values,
                )?;
                previous = Some(page);
            }
            let slot = sql_usize(r, 2)?;
            let start = slot.checked_mul(RECORD_BYTES).ok_or(Error::SizeOverflow)?;
            let o = decode_record(
                payload
                    .get(start..start + RECORD_BYTES)
                    .ok_or_else(|| invalid("packed export slot"))?,
            )?;
            let rank = sql_u64(r, 3)?;
            builder.push(o, rank, cancel)?;
        }
        drop(rows);
        drop(s);
        tx.commit()?;
        builder.finish_visible(visible, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Site;

    fn fixture(path: &Path, budget: &MemoryBudget) -> PackedStore {
        let cancel = Cancellation::default();
        let mut builder = PackedBuilder::create(path, Options::default(), budget).unwrap();
        for id in 0..8 {
            builder
                .push(
                    Observation::new(
                        id,
                        Site::point(Point3::new([id as f64, 0., 0.]).unwrap(), 0),
                        0.,
                        0.,
                    )
                    .unwrap(),
                    id / 2 + 1,
                    &cancel,
                )
                .unwrap();
        }
        builder.finish(&cancel).unwrap()
    }

    fn identity_location(store: &PackedStore, id: u64) -> (u64, usize) {
        lock(&store.inner.writer)
            .unwrap()
            .query_row(
                "SELECT page,slot FROM identities WHERE id=?1",
                [id.to_be_bytes().as_slice()],
                |row| Ok((sql_u64(row, 0)?, sql_usize(row, 1)?)),
            )
            .unwrap()
    }

    fn cached_payload(cached: &Cached) -> &Payload {
        let Page::Payload(payload) = &cached.page else {
            panic!("expected a cached payload");
        };
        payload
    }

    #[test]
    fn shared_payload_base_keeps_its_charge_after_original_version_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let store = fixture(&dir.path().join("shared-charge"), &budget);
        let cancel = Cancellation::default();
        let (page, slot) = identity_location(&store, 0);
        let original = store.inner.page(page, 0, true, &cancel).unwrap();
        assert_eq!(
            original._reservation.bytes() + cached_payload(&original).base._reservation.bytes(),
            store.inner.page_charge(true)
        );
        let snapshot = store
            .publish_prepared(&[(1, &[(0, -0.0), (1, 11.)])], &cancel, None)
            .unwrap();
        let current = store
            .inner
            .page(page, snapshot.generation(), true, &cancel)
            .unwrap();
        assert!(Arc::ptr_eq(
            &cached_payload(&original).base,
            &cached_payload(&current).base
        ));
        assert_ne!(
            cached_payload(&original).values.as_ptr(),
            cached_payload(&current).values.as_ptr()
        );
        let reads = store.statistics().page_reads;
        let still_resident = store.inner.page(page, 0, true, &cancel).unwrap();
        assert!(Arc::ptr_eq(&original, &still_resident));
        assert_eq!(store.statistics().page_reads, reads);
        assert_eq!(
            cached_payload(&original)
                .observation(slot)
                .unwrap()
                .value()
                .to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(
            cached_payload(&current)
                .observation(slot)
                .unwrap()
                .value()
                .to_bits(),
            (-0.0f64).to_bits()
        );
        drop(still_resident);
        store.clear_cache().unwrap();
        let base_charge = cached_payload(&current).base._reservation.bytes();
        let original_charge = original._reservation.bytes();
        let current_charge = current._reservation.bytes();
        assert_eq!(current_charge, store.inner.value_charge());
        let used = budget.used();
        drop(original);
        assert_eq!(budget.used(), used - original_charge);
        assert_eq!(Arc::strong_count(&cached_payload(&current).base), 1);
        assert_eq!(cached_payload(&current).observation(slot).unwrap().id(), 0);
        drop(current);
        assert_eq!(
            budget.used(),
            used - original_charge - current_charge - base_charge
        );
        drop((snapshot, store));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn shared_payload_versions_preserve_history_and_warm_cold_query_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared-history");
        let budget = MemoryBudget::new(16 << 20);
        let store = fixture(&path, &budget);
        let cancel = Cancellation::default();
        let old = store
            .publish_prepared(&[(1, &[(0, -0.0), (1, 11.)])], &cancel, None)
            .unwrap();
        let (page, _) = identity_location(&store, 0);
        let pinned = store
            .inner
            .page(page, old.generation(), true, &cancel)
            .unwrap();
        let before = (0..cached_payload(&pinned).len())
            .map(|slot| cached_payload(&pinned).observation(slot).unwrap())
            .collect::<Vec<_>>();
        let current = store
            .publish_prepared(&[(2, &[(2, 12.), (3, 13.)])], &cancel, None)
            .unwrap();
        let current_page = store
            .inner
            .page(page, current.generation(), true, &cancel)
            .unwrap();
        assert!(Arc::ptr_eq(
            &cached_payload(&pinned).base,
            &cached_payload(&current_page).base
        ));
        assert!(before.iter().enumerate().all(|(slot, observation)| {
            cached_payload(&pinned)
                .observation(slot)
                .unwrap()
                .same_bits(observation)
        }));
        let query = Neighborhood {
            maximum: 8,
            radius: 100.,
        };
        let point = Point3::new([1.5, 0., 0.]).unwrap();
        let old_values = old.nearest(point, query, &cancel).unwrap();
        let current_values = current.nearest(point, query, &cancel).unwrap();
        assert_eq!(
            old_values.iter().map(Observation::id).collect::<Vec<_>>(),
            [1, 0]
        );
        assert_eq!(
            current_values
                .iter()
                .map(Observation::id)
                .collect::<Vec<_>>(),
            [1, 2, 0, 3]
        );
        store.clear_cache().unwrap();
        for (snapshot, expected) in [(&old, &old_values), (&current, &current_values)] {
            let reloaded = snapshot.nearest(point, query, &cancel).unwrap();
            assert_eq!(reloaded.len(), expected.len());
            assert!(reloaded.iter().zip(expected).all(|(a, b)| a.same_bits(b)));
        }
        drop((pinned, current_page, old, current, store));
        assert_eq!(budget.used(), 0);
        let reopened = PackedStore::open(&path, Options::default(), &budget).unwrap();
        let reopened_values = reopened
            .snapshot()
            .unwrap()
            .nearest(point, query, &cancel)
            .unwrap();
        assert!(
            reopened_values
                .iter()
                .zip(&current_values)
                .all(|(a, b)| a.same_bits(b))
        );
        drop(reopened);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn shared_payload_copy_needs_only_value_charge_and_cancellation_discards_it() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let store = fixture(&dir.path().join("shared-budget"), &budget);
        let cancel = Cancellation::default();
        let (page, slot) = identity_location(&store, 0);
        let original = store.inner.page(page, 0, true, &cancel).unwrap();
        let mut warmed = Vec::with_capacity(1);
        let mut values = Vec::with_capacity(store.inner.options.leaf_records * 8);
        let held = budget
            .reserve(budget.limit() - budget.used() - store.inner.value_charge())
            .unwrap();
        assert!(budget.reserve(store.inner.page_charge(true)).is_err());
        let before = budget.used();
        let mut connection = lock(&store.inner.writer).unwrap();
        let tx = connection.transaction().unwrap();
        assert!(
            publish_cached(
                &store.inner,
                &tx,
                page,
                0,
                1,
                &[ValueUpdate {
                    id: 0,
                    value: 10.,
                    page,
                    slot
                }],
                &mut warmed,
                &mut values,
            )
            .unwrap()
        );
        assert_eq!(budget.used(), budget.limit());
        assert!(Arc::ptr_eq(
            &cached_payload(&original).base,
            &cached_payload(&warmed[0].1).base
        ));
        let reads = store.statistics().page_reads;
        let reader = store.inner.page(page, 0, true, &cancel).unwrap();
        assert!(Arc::ptr_eq(&reader, &original));
        assert_eq!(store.statistics().page_reads, reads);
        assert_eq!(
            cached_payload(&reader).observation(slot).unwrap().value(),
            0.
        );
        cancel.cancel();
        assert!(matches!(cancel.check(), Err(Error::Cancelled)));
        tx.rollback().unwrap();
        drop(connection);
        warmed.clear();
        assert_eq!(budget.used(), before);
        drop((held, reader, original, store));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn shared_payload_publication_failure_keeps_cached_and_durable_values() {
        for corrupt_identity in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let budget = MemoryBudget::new(16 << 20);
            let store = fixture(&dir.path().join("shared-rollback"), &budget);
            let cancel = Cancellation::default();
            let old = store
                .publish_prepared(&[(1, &[(0, 10.), (1, 11.)])], &cancel, None)
                .unwrap();
            let (page, _) = identity_location(&store, 2);
            let original = store
                .inner
                .page(page, old.generation(), true, &cancel)
                .unwrap();
            let before = (0..cached_payload(&original).len())
                .map(|slot| cached_payload(&original).observation(slot).unwrap())
                .collect::<Vec<_>>();
            if corrupt_identity {
                lock(&store.inner.writer)
                    .unwrap()
                    .execute(
                        "UPDATE identities SET slot=999 WHERE id=?1",
                        [3u64.to_be_bytes().as_slice()],
                    )
                    .unwrap();
            } else {
                lock(&store.inner.writer).unwrap().execute_batch(
                    "CREATE TRIGGER reject_payload BEFORE INSERT ON payloads BEGIN SELECT RAISE(ABORT, 'test publication failure'); END;",
                ).unwrap();
            }
            assert!(
                store
                    .publish_prepared(&[(2, &[(2, 12.), (3, 13.)])], &cancel, None)
                    .is_err()
            );
            let current = store.snapshot().unwrap();
            assert_eq!(current.generation(), old.generation());
            assert_eq!(current.completed_group(), 1);
            let still_cached = store
                .inner
                .page(page, old.generation(), true, &cancel)
                .unwrap();
            assert!(Arc::ptr_eq(&original, &still_cached));
            assert!(before.iter().enumerate().all(|(slot, observation)| {
                cached_payload(&still_cached)
                    .observation(slot)
                    .unwrap()
                    .same_bits(observation)
            }));
            let unpublished = lock(&store.inner.writer)
                .unwrap()
                .query_row(
                    "SELECT count(*) FROM payloads WHERE generation>1",
                    [],
                    |row| sql_u64(row, 0),
                )
                .unwrap();
            assert_eq!(unpublished, 0);
            drop((original, still_cached, old, current, store));
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn prepared_identity_capacity_includes_metadata_and_keeps_its_bound() {
        let capacity = 65536;
        let charge = (capacity * std::mem::size_of::<Identity>()
            + 4096 * std::mem::size_of::<u64>()
            + std::mem::size_of::<PreparedIdentities>()) as u64;
        let budget = MemoryBudget::new(charge);
        let plan = PreparedIdentities::new(usize::MAX, &budget).unwrap();
        assert_eq!(plan.capacity, capacity);
        assert_eq!(budget.used(), charge);
        drop(plan);
        assert_eq!(budget.used(), 0);
        let short = MemoryBudget::new(charge - 1);
        assert!(PreparedIdentities::new(capacity, &short).is_err());
        assert_eq!(short.used(), 0);
    }

    #[test]
    fn saturated_identity_preparation_skips_rescans_but_allows_smaller_prefixes() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let store = fixture(&dir.path().join("saturated"), &budget);
        let cancel = Cancellation::default();
        let mut plan = PreparedIdentities::new(6, &budget).unwrap();
        store.prepare_identities(&mut plan, 4, &cancel).unwrap();
        assert_eq!(plan.scanned_rows, 7);
        assert!(plan.entries.is_empty());
        assert_eq!(plan.through, 0);
        assert_eq!(plan.capacity_limited_at, Some(4));
        for through in [4, 5, 4] {
            store
                .prepare_identities(&mut plan, through, &cancel)
                .unwrap();
        }
        assert_eq!(plan.scanned_rows, 7, "saturated ranges are not rescanned");
        store.prepare_identities(&mut plan, 2, &cancel).unwrap();
        assert_eq!(plan.scanned_rows, 11);
        assert_eq!(plan.entries.len(), 4);
        assert_eq!(plan.through, 2);
        assert_eq!(plan.capacity_limited_at, Some(4));
        store.prepare_identities(&mut plan, 3, &cancel).unwrap();
        assert_eq!(plan.scanned_rows, 13);
        assert_eq!(plan.entries.len(), 6);
        assert_eq!(plan.through, 3);
        assert_eq!(plan.capacity_limited_at, Some(4));
        store.prepare_identities(&mut plan, 4, &cancel).unwrap();
        assert_eq!(plan.scanned_rows, 13);
        assert!(
            store
                .prepare_identities(&mut plan, u64::MAX, &cancel)
                .is_err()
        );
        let stopped = Cancellation::default();
        stopped.cancel();
        assert!(matches!(
            store.prepare_identities(&mut plan, 4, &stopped),
            Err(Error::Cancelled)
        ));
        assert_eq!(plan.scanned_rows, 13);
        // A real publication advances the stamp and discards the old proof.
        let values = [(0, 10.), (1, 11.), (2, 12.), (3, 13.), (4, 14.), (5, 15.)];
        let snapshot = store
            .publish_prepared(
                &[(1, &values[..2]), (2, &values[2..4]), (3, &values[4..])],
                &cancel,
                Some(&plan),
            )
            .unwrap();
        assert_eq!(snapshot.completed_group(), 3);
        store.prepare_identities(&mut plan, 4, &cancel).unwrap();
        assert_eq!(plan.scanned_rows, 15);
        assert_eq!(plan.entries.len(), 2);
        assert_eq!(plan.capacity_limited_at, None);
        drop((snapshot, plan, store));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn capacity_limited_identity_prefix_rechecks_local_and_external_changes() {
        for external in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("capacity-limited-identities");
            let budget = MemoryBudget::new(16 << 20);
            let store = fixture(&path, &budget);
            let cancel = Cancellation::default();
            let mut plan = PreparedIdentities::new(6, &budget).unwrap();
            store.prepare_identities(&mut plan, 3, &cancel).unwrap();
            store.prepare_identities(&mut plan, 4, &cancel).unwrap();
            assert_eq!(plan.entries.len(), 6);
            assert_eq!(plan.through, 3);
            assert_eq!(plan.capacity_limited_at, Some(4));
            let scanned = plan.scanned_rows;
            store.prepare_identities(&mut plan, 5, &cancel).unwrap();
            assert_eq!(plan.scanned_rows, scanned);
            if external {
                Connection::open(&path)
                    .unwrap()
                    .execute("UPDATE identities SET rank=2 WHERE rank=1", [])
                    .unwrap();
            } else {
                lock(&store.inner.writer)
                    .unwrap()
                    .execute("UPDATE identities SET rank=2 WHERE rank=1", [])
                    .unwrap();
            }
            // Test the previously saturated endpoint first: freshness must be
            // checked before its memo, even if the new map also exceeds capacity.
            store.prepare_identities(&mut plan, 4, &cancel).unwrap();
            assert_eq!(plan.scanned_rows, scanned + 7);
            assert!(plan.entries.is_empty());
            store.prepare_identities(&mut plan, 3, &cancel).unwrap();
            assert!(plan.entries.iter().all(|identity| identity.rank != 1));
            assert!(
                store
                    .publish_prepared(&[(1, &[(0, 10.), (1, 11.)])], &cancel, Some(&plan))
                    .is_err()
            );
            assert_eq!(store.snapshot().unwrap().completed_group(), 0);
            drop((plan, store));
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn prepared_prefix_and_sql_suffix_validate_complete_groups_and_release_budget() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let store = fixture(&dir.path().join("prefix"), &budget);
        let cancel = Cancellation::default();
        let mut plan = PreparedIdentities::new(6, &budget).unwrap();
        store.prepare_identities(&mut plan, 1, &cancel).unwrap();
        store.prepare_identities(&mut plan, 3, &cancel).unwrap();
        assert_eq!(plan.entries.len(), 6);
        store.prepare_identities(&mut plan, 4, &cancel).unwrap();
        assert_eq!(
            plan.entries.len(),
            6,
            "failed extension retains only complete prefix"
        );
        assert_eq!(plan.through, 3);
        let values = [
            (0, -0.0),
            (1, 1.0),
            (2, 2.0),
            (3, 3.0),
            (4, 4.0),
            (5, 5.0),
            (6, 6.0),
            (7, 7.0),
        ];
        assert!(
            store
                .publish_prepared(
                    &[(1, &values[..2]), (2, &values[2..3])],
                    &cancel,
                    Some(&plan)
                )
                .is_err()
        );
        assert!(
            store
                .publish_prepared(&[(1, &[(0, 0.), (0, 0.)])], &cancel, Some(&plan))
                .is_err()
        );
        assert_eq!(store.snapshot().unwrap().completed_group(), 0);
        let snapshot = store
            .publish_prepared(
                &[
                    (1, &values[..2]),
                    (2, &values[2..4]),
                    (3, &values[4..6]),
                    (4, &values[6..]),
                ],
                &cancel,
                Some(&plan),
            )
            .unwrap();
        assert_eq!(snapshot.completed_group(), 4);
        snapshot
            .scan(&mut |o| {
                assert_eq!(o.value().to_bits(), values[o.id() as usize].1.to_bits());
                Ok(())
            })
            .unwrap();
        drop((snapshot, plan, store));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn prepared_identities_reject_external_and_local_edits_without_manifest_changes() {
        for external in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("changed");
            let budget = MemoryBudget::new(16 << 20);
            let store = fixture(&path, &budget);
            let cancel = Cancellation::default();
            let mut plan = PreparedIdentities::new(8, &budget).unwrap();
            store.prepare_identities(&mut plan, 4, &cancel).unwrap();
            if external {
                let other = Connection::open(&path).unwrap();
                other
                    .execute("UPDATE identities SET rank=2 WHERE rank=1", [])
                    .unwrap();
            } else {
                lock(&store.inner.writer)
                    .unwrap()
                    .execute("UPDATE identities SET rank=2 WHERE rank=1", [])
                    .unwrap();
            }
            // A stale map would wrongly accept rank one, which no longer exists.
            assert!(
                store
                    .publish_prepared(&[(1, &[(0, 10.), (1, 11.)])], &cancel, Some(&plan))
                    .is_err()
            );
            assert_eq!(store.snapshot().unwrap().generation(), 0);
            assert_eq!(store.snapshot().unwrap().completed_group(), 0);
            drop((plan, store));
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn identity_preparation_is_optional_bounded_cancelled_and_store_specific() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let store = fixture(&dir.path().join("first"), &budget);
        let other = fixture(&dir.path().join("second"), &budget);
        assert!(PreparedIdentities::new(8, &MemoryBudget::new(1)).is_err());
        let cancel = Cancellation::default();
        let mut plan = PreparedIdentities::new(8, &budget).unwrap();
        lock(&store.inner.writer)
            .unwrap()
            .execute("UPDATE identities SET rank=rank WHERE rank=1", [])
            .unwrap();
        store.prepare_identities(&mut plan, 3, &cancel).unwrap();
        // The other store has the same counters but a different identity map.
        lock(&other.inner.writer)
            .unwrap()
            .execute("UPDATE identities SET rank=2 WHERE rank=1", [])
            .unwrap();
        assert_eq!(
            plan.stamp,
            Some(IdentityStamp::read(&lock(&other.inner.writer).unwrap(), 0, 0).unwrap())
        );
        assert!(
            other
                .publish_prepared(&[(1, &[(0, 10.), (1, 11.)])], &cancel, Some(&plan))
                .is_err()
        );
        store
            .publish_prepared(&[(1, &[(0, 10.), (1, 11.)])], &cancel, Some(&plan))
            .unwrap();
        // Own successful publication also invalidates a retained preparation.
        store
            .publish_prepared(&[(2, &[(2, 12.), (3, 13.)])], &cancel, Some(&plan))
            .unwrap();
        cancel.cancel();
        assert!(matches!(
            store.prepare_identities(&mut plan, 4, &cancel),
            Err(Error::Cancelled)
        ));
        drop((plan, store, other));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn publication_reclaims_optional_identity_memory_before_failing_admission() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let store = fixture(&dir.path().join("reclaim"), &budget);
        let cancel = Cancellation::default();
        let mut plan = Some(PreparedIdentities::new(65536, &budget).unwrap());
        store.clear_cache().unwrap();
        let held = budget.reserve(budget.limit() - budget.used() - 8).unwrap();
        let reads = store.statistics().page_reads;
        store
            .prepare_identities(plan.as_mut().unwrap(), 4, &cancel)
            .unwrap();
        assert_eq!(
            store.statistics().page_reads,
            reads,
            "optional warming yields to the memory budget"
        );
        let snapshot = store
            .publish_staged(&[(1, &[(0, 10.), (1, 11.)])], &cancel, &mut plan)
            .unwrap();
        assert!(plan.is_none());
        assert_eq!(snapshot.generation(), 1, "exactly one durable publication");
        assert_eq!(snapshot.completed_group(), 1);
        assert!(budget.used() <= budget.limit());
        drop((snapshot, held, store));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn speculative_warming_checks_integrity_and_cannot_bypass_publication_validation() {
        let dir = tempfile::tempdir().unwrap();
        let budget = MemoryBudget::new(16 << 20);
        let store = fixture(&dir.path().join("corrupt"), &budget);
        let cancel = Cancellation::default();
        let mut plan = PreparedIdentities::new(8, &budget).unwrap();
        lock(&store.inner.writer)
            .unwrap()
            .execute("UPDATE payloads SET crc=crc+1", [])
            .unwrap();
        store.prepare_identities(&mut plan, 4, &cancel).unwrap();
        assert_eq!(
            plan.through, 4,
            "identity staging can survive optional warm failure"
        );
        let error = store
            .publish_prepared(&[(1, &[(0, 10.), (1, 11.)])], &cancel, Some(&plan))
            .err()
            .unwrap();
        assert!(error.to_string().contains("checksum"), "{error}");
        assert_eq!(store.snapshot().unwrap().generation(), 0);
        drop((plan, store));
        assert_eq!(budget.used(), 0);
    }
}
