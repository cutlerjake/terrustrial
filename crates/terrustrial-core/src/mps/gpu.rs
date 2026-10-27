//! Portable WGSL direct pattern scoring. Equal-weight categorical events use exact
//! integer counts and scalar-order f64 reconstruction; other scores use approximate
//! f32 arithmetic. Category identities and validity remain integer data.
//! The TI stays on the device across batches. Batches are independent queries, not
//! permission to run dependent sequential simulation nodes concurrently.
use super::*;
use crate::compute::gpu::Context;
use crate::execution::Reservation;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct Query<'a> {
    pub events: &'a [Event],
    pub requested: &'a [usize],
    pub boundary: Boundary,
}
#[derive(Debug, PartialEq)]
enum Encoding {
    Float,
    /// Prefix sums reproduce the scalar order exactly, including roundoff.
    Counts(Vec<f64>),
}
impl Encoding {
    fn decode(&self, value: u32) -> f64 {
        match self {
            Self::Float => f32::from_bits(value) as f64,
            Self::Counts(table) => table[value as usize],
        }
    }
}
/// Query-major raw scores, decoded through a borrowed view without a second full
/// output allocation. Invalid candidates remain distinct from every finite score.
pub struct Scores {
    cells: usize,
    encodings: Vec<Encoding>,
    values: Vec<u8>,
    _reservation: Reservation,
}
#[derive(Debug, PartialEq)]
pub struct QueryScores<'a> {
    values: &'a [u8],
    encoding: &'a Encoding,
}
impl QueryScores<'_> {
    /// True for empty events or equally weighted categorical conditioning. These
    /// values reproduce scalar f64 accumulation; other scores are promoted f32.
    pub fn is_exact(&self) -> bool {
        matches!(self.encoding, Encoding::Counts(_))
    }
    pub fn len(&self) -> usize {
        self.values.len() / 8
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Option<f64>> + '_ {
        self.values.as_chunks::<8>().0.iter().map(|record| {
            let value = u32::from_ne_bytes(record[..4].try_into().unwrap());
            let valid = u32::from_ne_bytes(record[4..].try_into().unwrap());
            if valid == 0 {
                None
            } else {
                Some(self.encoding.decode(value))
            }
        })
    }
}
impl Scores {
    pub fn query(&self, query: usize) -> Result<QueryScores<'_>> {
        let encoding = self
            .encodings
            .get(query)
            .ok_or_else(|| invalid("GPU score query index"))?;
        Ok(QueryScores {
            values: &self.values[query * self.cells * 8..(query + 1) * self.cells * 8],
            encoding,
        })
    }
}
/// Cached native GPU data and pipeline. The scalar engine remains available if
/// Context::new returns BackendUnavailable or the workload exceeds device limits.
/// No silent fallback is performed, so reported timing always identifies execution.
/// This first backend retains the entire TI and each output batch on the device.
/// Cancellation does not preempt a submitted GPU command.
pub struct Matcher<'a> {
    context: Arc<Context>,
    training: &'a Grid,
    normalization: Vec<Option<(f64, f64)>>,
    training_buffer: wgpu::Buffer,
    kinds_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    _reservation: Reservation,
}
fn bytes(words: &[u32]) -> &[u8] {
    bytemuck::cast_slice(words)
}
fn normalized(value: f64, center: f64, scale: f64) -> Result<f32> {
    let difference = value - center;
    let scaled = if difference.is_finite() {
        difference / scale
    } else {
        value / scale - center / scale
    };
    if !scaled.is_finite() || scaled.abs() > f32::MAX.sqrt() as f64 * 0.25 {
        return Err(Error::Numerical {
            context: "GPU continuous mismatch exceeds supported f32 range; use CPU",
        });
    }
    Ok(scaled as f32)
}
fn storage_size(context: &Context, size: usize) -> Result<()> {
    if size as u128 > context.limits().max_storage_buffer_binding_size as u128
        || size as u128 > context.limits().max_buffer_size as u128
    {
        return Err(invalid(
            "GPU storage binding exceeds device limits; reduce the batch or use CPU",
        ));
    }
    Ok(())
}
fn reserve(budget: &MemoryBudget, size: usize) -> Result<Reservation> {
    budget.reserve(u64::try_from(size).map_err(|_| Error::SizeOverflow)?)
}
impl<'a> Matcher<'a> {
    pub fn new(
        training: &'a Grid,
        context: Arc<Context>,
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        let n = training.shape.cells;
        let variables = training.variables.len();
        if n > u32::MAX as usize
            || variables > u32::MAX as usize
            || training
                .shape
                .dimensions
                .iter()
                .any(|n| *n > (i32::MAX / 2) as usize)
            || training.values.len() > u32::MAX as usize
        {
            return Err(invalid("GPU grid exceeds shader addressing range; use CPU"));
        }
        let data_bytes = checked_len(&[training.values.len(), 8])?;
        storage_size(&context, data_bytes)?;
        storage_size(&context, checked_len(&[variables, 4])?)?;
        let resident_bytes = data_bytes
            .checked_add(checked_len(&[variables, 64])?)
            .ok_or(Error::SizeOverflow)?;
        let reservation = reserve(budget, resident_bytes)?;
        let _temporary = reserve(budget, data_bytes)?;
        let mut normalization = Vec::with_capacity(variables);
        let mut kinds = Vec::with_capacity(variables);
        for (v, kind) in training.variables.iter().enumerate() {
            match kind {
                Variable::Continuous { scale } => {
                    let mut minimum = f64::INFINITY;
                    let mut maximum = f64::NEG_INFINITY;
                    for cell in training.values.chunks_exact(variables) {
                        if let Some(value) = cell[v] {
                            minimum = minimum.min(value);
                            maximum = maximum.max(value);
                        }
                    }
                    let center = if minimum.is_finite() {
                        minimum * 0.5 + maximum * 0.5
                    } else {
                        0.0
                    };
                    normalization.push(Some((center, *scale)));
                    kinds.push(0);
                }
                Variable::Categorical { .. } => {
                    normalization.push(None);
                    kinds.push(1);
                }
            }
        }
        let mut packed = Vec::with_capacity(training.values.len() * 2);
        for (index, value) in training.values.iter().enumerate() {
            if index % 4096 == 0 {
                cancel.check()?;
            }
            let encoded = match (value, normalization[index % variables]) {
                (Some(value), Some((center, scale))) => {
                    normalized(*value, center, scale)?.to_bits()
                }
                (Some(value), None) => *value as u32,
                (None, _) => 0,
            };
            packed.extend_from_slice(&[encoded, u32::from(value.is_some())]);
        }
        let (training_buffer, kinds_buffer, pipeline) = context.checked(|| {
            let training_buffer =
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MPS training"),
                        contents: bytes(&packed),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            let kinds_buffer =
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MPS kinds"),
                        contents: bytes(&kinds),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            let shader = context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("MPS direct WGSL"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("pattern.wgsl").into()),
                });
            let pipeline =
                context
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("MPS direct scoring"),
                        layout: None,
                        module: &shader,
                        entry_point: Some("score_patterns"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
            (training_buffer, kinds_buffer, pipeline)
        })?;
        cancel.check()?;
        Ok(Self {
            context,
            training,
            normalization,
            training_buffer,
            kinds_buffer,
            pipeline,
            _reservation: reservation,
        })
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.context.adapter_info()
    }

    /// Exact GPU ranking for equal-weight categorical events. Requested target
    /// attributes may be mixed. Empty events use the seeded CPU marginal sampler.
    pub fn best_matches_categorical(
        &mut self,
        events: &[Event],
        requested: &[usize],
        options: &MatchOptions<'_>,
    ) -> Result<Vec<Match>> {
        options.cancellation.check()?;
        if events.iter().any(|event| {
            !matches!(
                self.training.variables.get(event.variable),
                Some(Variable::Categorical { .. })
            )
        }) || events
            .windows(2)
            .any(|pair| pair[0].weight != pair[1].weight)
        {
            return Err(invalid(
                "exact GPU ranking requires equal-weight categorical events",
            ));
        }
        self.best_matches_approximate(events, requested, options, 0.0)
    }

    /// Automatically uses exact integer ranking for equal-weight categorical events.
    /// Otherwise uses approximate ranking with an absolute score band, followed by f64
    /// refinement and complete CPU validity checks. This does not promise the
    /// global scalar top-k. Use `mps::best_matches` for that exact contract.
    pub fn best_matches_approximate(
        &mut self,
        events: &[Event],
        requested: &[usize],
        options: &MatchOptions<'_>,
        refinement_tolerance: f64,
    ) -> Result<Vec<Match>> {
        options.cancellation.check()?;
        if !refinement_tolerance.is_finite() || refinement_tolerance < 0.0 {
            return Err(invalid("GPU refinement tolerance"));
        }
        if events.is_empty() {
            return super::best_matches(self.training, events, requested, options);
        }
        let count = options.neighbors.min(self.training.shape.cells);
        let _reservation = reserve(
            options.budget,
            checked_len(&[count, 64])?
                .checked_add(checked_len(&[events.len(), 128])?)
                .and_then(|bytes| bytes.checked_add(checked_len(&[requested.len(), 64]).ok()?))
                .ok_or(Error::SizeOverflow)?,
        )?;
        let weights = validate_query(self.training, events, requested, options.neighbors)?;
        let scores = self.score_batch(
            &[Query {
                events,
                requested,
                boundary: options.boundary,
            }],
            options.budget,
            options.cancellation,
        )?;
        let scores = scores.query(0)?;
        let mut heap = BinaryHeap::<Candidate>::new();
        if scores.is_exact() {
            let mut rng = random_stream(options.seed, options.realization, options.query_id, 2);
            for (index, score) in scores.iter().enumerate() {
                if index % 4096 == 0 {
                    options.cancellation.check()?;
                }
                let tie = rng.random::<u128>();
                if let Some(mismatch) = score {
                    push_candidate(
                        &mut heap,
                        count,
                        Candidate {
                            matched: Match { index, mismatch },
                            tie,
                        },
                    );
                }
            }
            if heap.is_empty() {
                return Err(invalid(
                    "no GPU training candidate supports the full conditioning event",
                ));
            }
            let matches: Vec<_> = heap
                .into_sorted_vec()
                .into_iter()
                .map(|entry| entry.matched)
                .collect();
            for entry in &matches {
                let verified = score_candidate(
                    self.training,
                    events,
                    &weights,
                    requested,
                    entry.index,
                    options.boundary,
                )?;
                if verified != Some(entry.mismatch) {
                    return Err(Error::Backend(
                        "GPU integer score failed direct CPU verification".into(),
                    ));
                }
            }
            return Ok(matches);
        }
        for (index, score) in scores.iter().enumerate() {
            if let Some(score) = score {
                let candidate = Candidate {
                    matched: Match {
                        index,
                        mismatch: score,
                    },
                    tie: 0,
                };
                push_candidate(&mut heap, count, candidate);
            }
        }
        let threshold = heap
            .peek()
            .ok_or_else(|| {
                invalid("no GPU training candidate supports the full conditioning event")
            })?
            .matched
            .mismatch
            + refinement_tolerance;
        if !threshold.is_finite() {
            return Err(invalid("GPU refinement threshold overflow"));
        }
        heap.clear();
        let mut rng = random_stream(options.seed, options.realization, options.query_id, 2);
        for (index, score) in scores.iter().enumerate() {
            if index % 4096 == 0 {
                options.cancellation.check()?;
            }
            let tie = rng.random::<u128>();
            if score.is_none_or(|score| score > threshold)
                || requested.iter().any(|&v| {
                    self.training.values[index * self.training.variables.len() + v].is_none()
                })
            {
                continue;
            }
            let Some(score) = score_candidate(
                self.training,
                events,
                &weights,
                requested,
                index,
                options.boundary,
            )?
            else {
                continue;
            };
            let candidate = Candidate {
                matched: Match {
                    index,
                    mismatch: score,
                },
                tie,
            };
            push_candidate(&mut heap, count, candidate);
        }
        if heap.len() < count {
            return super::best_matches(self.training, events, requested, options);
        }
        Ok(heap
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| candidate.matched)
            .collect())
    }

    pub fn draw_approximate(
        &mut self,
        events: &[Event],
        requested: &[usize],
        options: &MatchOptions<'_>,
        refinement_tolerance: f64,
    ) -> Result<JointDraw> {
        let matches =
            self.best_matches_approximate(events, requested, options, refinement_tolerance)?;
        select_match(self.training, &matches, requested, options)
    }

    /// Immutable training data used by this reusable matching engine.
    pub fn training(&self) -> &'a Grid {
        self.training
    }

    /// Scores independent events together to amortize submission and transfer cost.
    /// Each query view reports whether its scores are exact. Equal-weight categorical
    /// queries reproduce scalar f64 sums; other scores are promoted from f32.
    pub fn score_batch(
        &mut self,
        queries: &[Query<'_>],
        budget: &MemoryBudget,
        cancel: &Cancellation,
    ) -> Result<Scores> {
        cancel.check()?;
        if queries.is_empty() {
            return Err(invalid("nonempty GPU query batch required"));
        }
        let limits = self.context.limits();
        if queries.len() > limits.max_compute_workgroups_per_dimension as usize {
            return Err(invalid("GPU query dispatch dimension"));
        }
        let pairs = checked_len(&[queries.len(), self.training.shape.cells])?;
        if pairs > u32::MAX as usize {
            return Err(invalid("GPU output addressing range"));
        }
        let output_bytes = checked_len(&[pairs, 8])?;
        storage_size(&self.context, output_bytes)?;
        let events_count = queries.iter().try_fold(0_usize, |n, q| {
            n.checked_add(q.events.len()).ok_or(Error::SizeOverflow)
        })?;
        let requests_count = queries.iter().try_fold(0_usize, |n, q| {
            n.checked_add(q.requested.len()).ok_or(Error::SizeOverflow)
        })?;
        if events_count > u32::MAX as usize || requests_count > u32::MAX as usize {
            return Err(invalid("GPU event/request addressing range"));
        }
        let event_bytes = checked_len(&[events_count.max(1), 32])?;
        let request_bytes = checked_len(&[requests_count.max(1), 4])?;
        let query_bytes = checked_len(&[queries.len(), 32])?;
        for size in [event_bytes, request_bytes, query_bytes] {
            storage_size(&self.context, size)?;
        }
        let working_bytes = output_bytes
            .checked_mul(3)
            .and_then(|n| n.checked_add(event_bytes.checked_mul(6)?))
            .and_then(|n| n.checked_add(request_bytes.checked_mul(18)?))
            .and_then(|n| n.checked_add(query_bytes.checked_mul(2)?))
            .and_then(|n| n.checked_add(64))
            .ok_or(Error::SizeOverflow)?;
        let retained_bytes = output_bytes
            .checked_add(checked_len(&[
                events_count
                    .checked_add(queries.len())
                    .ok_or(Error::SizeOverflow)?,
                8,
            ])?)
            .and_then(|n| n.checked_add(query_bytes))
            .ok_or(Error::SizeOverflow)?;
        let output_reservation = reserve(budget, retained_bytes)?;
        let _workspace = reserve(budget, working_bytes)?;
        let mut encodings = Vec::with_capacity(queries.len());
        let mut packed_events = Vec::with_capacity(events_count.max(1) * 8);
        let mut packed_queries = Vec::with_capacity(queries.len() * 8);
        let mut packed_requests = Vec::with_capacity(requests_count.max(1));
        for query in queries {
            cancel.check()?;
            let weights = validate_query(self.training, query.events, query.requested, 1)?;
            let integer = query.events.iter().all(|event| {
                matches!(
                    self.training.variables[event.variable],
                    Variable::Categorical { .. }
                )
            }) && weights.windows(2).all(|pair| pair[0] == pair[1]);
            if integer {
                let mut table = Vec::with_capacity(weights.len() + 1);
                let mut score = 0.0;
                table.push(score);
                for &weight in &weights {
                    score += weight;
                    table.push(score);
                }
                encodings.push(Encoding::Counts(table));
            } else {
                encodings.push(Encoding::Float);
            }
            let start = (packed_events.len() / 8) as u32;
            let mut impossible = false;
            for (event, weight) in query.events.iter().zip(weights) {
                let mut offset = [0_u32; 3];
                for (axis, encoded) in offset.iter_mut().enumerate() {
                    let dimension = self.training.shape.dimensions[axis] as i64;
                    let value = match query.boundary {
                        Boundary::Periodic => event.offset[axis].rem_euclid(dimension),
                        Boundary::Bounded => {
                            if event.offset[axis] <= -dimension || event.offset[axis] >= dimension {
                                impossible = true;
                                0
                            } else {
                                event.offset[axis]
                            }
                        }
                    };
                    *encoded = value as i32 as u32;
                }
                let value = if let Some((center, scale)) = self.normalization[event.variable] {
                    normalized(event.value, center, scale)?.to_bits()
                } else {
                    event.value as u32
                };
                let weight = weight as f32;
                if weight < f32::MIN_POSITIVE {
                    return Err(invalid("GPU event weight is below f32 resolution; use CPU"));
                }
                packed_events.extend_from_slice(&[
                    offset[0],
                    offset[1],
                    offset[2],
                    event.variable as u32,
                    value,
                    weight.to_bits(),
                    0,
                    0,
                ]);
            }
            packed_queries.extend_from_slice(&[
                start,
                query.events.len() as u32,
                packed_requests.len() as u32,
                query.requested.len() as u32,
                u32::from(query.boundary == Boundary::Bounded),
                u32::from(impossible),
                u32::from(integer),
                0,
            ]);
            packed_requests.extend(query.requested.iter().map(|v| *v as u32));
        }
        if packed_events.is_empty() {
            packed_events.resize(8, 0);
        }
        let groups = (self.training.shape.cells as u32).div_ceil(64);
        let group_x = groups.min(limits.max_compute_workgroups_per_dimension);
        let group_y = groups.div_ceil(group_x);
        if group_y > limits.max_compute_workgroups_per_dimension {
            return Err(invalid("GPU candidate dispatch dimension"));
        }
        let [x, y, z] = self.training.shape.dimensions.map(|n| n as u32);
        let parameters = [
            x,
            y,
            z,
            self.training.variables.len() as u32,
            self.training.shape.cells as u32,
            queries.len() as u32,
            group_x * 64,
            0,
        ];
        let readback = self.context.checked(|| {
            let input = |name, contents, usage| {
                self.context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(name),
                        contents,
                        usage,
                    })
            };
            let parameter_buffer = input(
                "MPS dimensions",
                bytes(&parameters),
                wgpu::BufferUsages::UNIFORM,
            );
            let event_buffer = input(
                "MPS events",
                bytes(&packed_events),
                wgpu::BufferUsages::STORAGE,
            );
            let request_buffer = input(
                "MPS requests",
                bytes(&packed_requests),
                wgpu::BufferUsages::STORAGE,
            );
            let query_buffer = input(
                "MPS queries",
                bytes(&packed_queries),
                wgpu::BufferUsages::STORAGE,
            );
            let output_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MPS scores"),
                size: output_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MPS readback"),
                size: output_bytes as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let buffers = [
                &parameter_buffer,
                &self.training_buffer,
                &self.kinds_buffer,
                &event_buffer,
                &request_buffer,
                &query_buffer,
                &output_buffer,
            ];
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let bind_group = self
                .context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MPS scoring inputs"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("MPS score batch"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MPS scores"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(group_x, group_y, queries.len() as u32);
            }
            encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_bytes as u64);
            self.context.queue.submit([encoder.finish()]);
            readback
        })?;
        let output = self.context.readback(&readback, cancel)?;
        for (query, records) in output
            .chunks_exact(self.training.shape.cells * 8)
            .enumerate()
        {
            for record in records.as_chunks::<8>().0 {
                let raw = u32::from_ne_bytes(record[..4].try_into().unwrap());
                let valid = u32::from_ne_bytes(record[4..].try_into().unwrap());
                let value_valid = match &encodings[query] {
                    Encoding::Float => {
                        let value = f32::from_bits(raw);
                        value.is_finite() && value >= 0.0
                    }
                    Encoding::Counts(table) => (raw as usize) < table.len(),
                };
                if valid > 1 || (valid == 1 && !value_valid) {
                    return Err(Error::Numerical {
                        context: "GPU pattern score or mismatch count",
                    });
                }
            }
        }
        cancel.check()?;
        Ok(Scores {
            cells: self.training.shape.cells,
            encodings,
            values: output,
            _reservation: output_reservation,
        })
    }
}
