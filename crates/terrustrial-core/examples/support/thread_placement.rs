//! Explicit Windows benchmark placement. No library policy or process affinity
//! is changed. `physical-writer` excludes the publisher's whole core from the
//! library's compute and blocking-read threads; the calling thread is untouched.
//! Publication decoding can also reserve a separate preparation core.
use std::io;
use terrustrial_core::execution::{ThreadInfo, ThreadRole, ThreadStart};

#[derive(Clone, Debug)]
pub struct Placement {
    compute: Vec<usize>,
    process_mask: usize,
    writer_core: Option<usize>,
    prepare_core: Option<usize>,
    available_cores: usize,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

impl Placement {
    /// Discover the current process's allowed physical cores before constructing
    /// a store. CPU-set selections and multiple processor groups are rejected,
    /// so this adapter never silently broadens an unsupported restriction.
    pub fn discover(compute_count: usize, reserve_writer: bool) -> io::Result<Self> {
        let (cores, mask) = platform::discover()?;
        Self::from_cores(cores, mask, compute_count, reserve_writer)
    }

    pub fn discover_with_publication_prepare(compute_count: usize) -> io::Result<Self> {
        let (cores, mask) = platform::discover()?;
        Self::from_cores_with_publication_prepare(cores, mask, compute_count, true, true)
    }

    fn from_cores(
        cores: Vec<usize>,
        process_mask: usize,
        compute_count: usize,
        reserve_writer: bool,
    ) -> io::Result<Self> {
        Self::from_cores_with_publication_prepare(
            cores, process_mask, compute_count, reserve_writer, false,
        )
    }

    fn from_cores_with_publication_prepare(
        mut cores: Vec<usize>,
        process_mask: usize,
        compute_count: usize,
        reserve_writer: bool,
        reserve_prepare: bool,
    ) -> io::Result<Self> {
        if reserve_prepare && !reserve_writer {
            return Err(invalid("preparation placement requires a separate publisher core"));
        }
        if compute_count == 0 || process_mask == 0 {
            return Err(invalid(
                "placement requires workers and a nonempty process affinity",
            ));
        }
        let mut covered = 0;
        for &core in &cores {
            if core == 0 || covered & core != 0 {
                return Err(invalid("invalid or overlapping physical-core masks"));
            }
            covered |= core;
        }
        if process_mask & !covered != 0 {
            return Err(invalid(
                "process affinity contains processors absent from topology",
            ));
        }
        cores.retain(|core| core & process_mask != 0);
        cores.sort_unstable_by_key(|core| (core & process_mask).trailing_zeros());
        let available_cores = cores.len();
        let service_cores = usize::from(reserve_writer) + usize::from(reserve_prepare);
        if compute_count.saturating_add(service_cores) > available_cores {
            return Err(invalid(format!(
                "placement needs {} compute cores and {service_cores} service cores; the process may use {available_cores} physical cores",
                compute_count,
            )));
        }
        let writer_core = reserve_writer.then(|| cores.pop().unwrap());
        // Sorted by the lowest allowed logical processor: publisher takes the
        // highest physical core and preparation takes the second highest.
        let prepare_core = reserve_prepare.then(|| cores.pop().unwrap());
        let compute = cores
            .into_iter()
            .take(compute_count)
            .map(|core| lowest_bit(core & process_mask))
            .collect();
        Ok(Self {
            compute,
            process_mask,
            writer_core,
            prepare_core,
            available_cores,
        })
    }

    pub fn thread_start(&self) -> ThreadStart {
        let plan = self.clone();
        ThreadStart::new(move |info| {
            plan.apply(info).map_err(|error| {
                terrustrial_core::Error::InvalidInput(format!("Windows thread placement: {error}"))
            })
        })
    }

    fn blocking_read_mask(&self) -> Option<usize> {
        self.writer_core.map(|writer| {
            self.process_mask & !(writer | self.prepare_core.unwrap_or(0))
        })
    }

    fn thread_mask(&self, info: ThreadInfo) -> io::Result<Option<usize>> {
        if info.count == 0 || info.index >= info.count {
            return Err(invalid("invalid startup worker index/count"));
        }
        let mask = match info.role {
            ThreadRole::Compute => {
                if info.count > self.compute.len() {
                    return Err(invalid(
                        "execution created more compute workers than the placement plan",
                    ));
                }
                Some(self.compute[info.index])
            }
            ThreadRole::BlockingRead => self.blocking_read_mask(),
            ThreadRole::Publish => {
                if info.count != 1 || info.index != 0 {
                    return Err(invalid("this placement plan supports one publisher"));
                }
                self.writer_core
                    .map(|core| lowest_bit(core & self.process_mask))
            }
            ThreadRole::PublicationPrepare => {
                if info.count != 1 || info.index != 0 {
                    return Err(invalid("this placement plan supports one preparation helper"));
                }
                Some(lowest_bit(self.prepare_core.ok_or_else(|| {
                    invalid("publication preparation requires a reserved helper core")
                })? & self.process_mask))
            }
        };
        Ok(mask)
    }

    fn apply(&self, info: ThreadInfo) -> io::Result<()> {
        let mask = self.thread_mask(info)?;
        if let Some(mask) = mask {
            platform::apply(mask, matches!(info.role, ThreadRole::BlockingRead))?;
        }
        Ok(())
    }

    pub fn description(&self) -> serde_json::Value {
        serde_json::json!({
            "policy": if self.writer_core.is_some() { "physical-writer" } else { "physical" },
            "topology_source": "GetLogicalProcessorInformationEx(RelationProcessorCore)",
            "available_physical_cores": self.available_cores,
            "process_affinity_mask": format!("{:#x}", self.process_mask),
            "compute_logical_processors": self.compute.iter().map(|mask| mask.trailing_zeros()).collect::<Vec<_>>(),
            "publisher_logical_processor": self.writer_core.map(|core| (core & self.process_mask).trailing_zeros()),
            "publisher_physical_core_mask": self.writer_core.map(|core| format!("{core:#x}")),
            "publication_prepare_logical_processor": self.prepare_core.map(|core| (core & self.process_mask).trailing_zeros()),
            "publication_prepare_physical_core_mask": self.prepare_core.map(|core| format!("{core:#x}")),
            "blocking_read_affinity_mask": self.blocking_read_mask().map(|mask| format!("{mask:#x}")),
        })
    }
}

fn lowest_bit(mask: usize) -> usize {
    mask & mask.wrapping_neg()
}

#[cfg(all(windows, target_pointer_width = "64"))]
mod platform {
    use super::*;
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetActiveProcessorGroupCount() -> u16;
        fn GetLogicalProcessorInformationEx(
            relation: i32,
            buffer: *mut c_void,
            bytes: *mut u32,
        ) -> i32;
        fn GetProcessAffinityMask(
            process: *mut c_void,
            mask: *mut usize,
            system: *mut usize,
        ) -> i32;
        fn GetThreadGroupAffinity(thread: *mut c_void, affinity: *mut GroupAffinity) -> i32;
        fn GetProcessDefaultCpuSets(
            process: *mut c_void,
            ids: *mut u32,
            count: u32,
            required: *mut u32,
        ) -> i32;
        fn GetThreadSelectedCpuSets(
            thread: *mut c_void,
            ids: *mut u32,
            count: u32,
            required: *mut u32,
        ) -> i32;
        fn SetThreadAffinityMask(thread: *mut c_void, mask: usize) -> usize;
    }
    const PROCESS: *mut c_void = -1isize as *mut c_void;
    const THREAD: *mut c_void = -2isize as *mut c_void;

    fn restrictions() -> io::Result<(usize, usize)> {
        // SAFETY: no pointers; the query does not change processor configuration.
        if unsafe { GetActiveProcessorGroupCount() } != 1 {
            return Err(invalid(
                "physical placement supports exactly one Windows processor group",
            ));
        }
        let mut required = 0;
        // SAFETY: the null buffer with zero capacity queries only the required
        // count. Pseudo handles identify this process/thread without ownership.
        let ok =
            unsafe { GetProcessDefaultCpuSets(PROCESS, std::ptr::null_mut(), 0, &mut required) };
        if required != 0 {
            return Err(invalid(
                "physical placement does not support process CPU-set selections",
            ));
        }
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: same count-only query contract for the current thread.
        let ok =
            unsafe { GetThreadSelectedCpuSets(THREAD, std::ptr::null_mut(), 0, &mut required) };
        if required != 0 {
            return Err(invalid(
                "physical placement does not support thread CPU-set selections",
            ));
        }
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let (mut process, mut system) = (0, 0);
        // SAFETY: valid writable pointer-sized masks and a process pseudo handle.
        if unsafe { GetProcessAffinityMask(PROCESS, &mut process, &mut system) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut thread = GroupAffinity::default();
        // SAFETY: GroupAffinity is the Win64 GROUP_AFFINITY layout.
        if unsafe { GetThreadGroupAffinity(THREAD, &mut thread) } == 0 {
            return Err(io::Error::last_os_error());
        }
        if process == 0 || process & !system != 0 || thread.group != 0 || thread.mask == 0 {
            return Err(invalid(
                "cannot establish a supported process/thread affinity domain",
            ));
        }
        Ok((process, thread.mask))
    }

    pub fn discover() -> io::Result<(Vec<usize>, usize)> {
        let (process, _) = restrictions()?;
        let mut bytes = 0;
        // SAFETY: a null buffer requests the required size only.
        let ok = unsafe { GetLogicalProcessorInformationEx(0, std::ptr::null_mut(), &mut bytes) };
        let error = io::Error::last_os_error();
        if ok != 0 || error.raw_os_error() != Some(122) || bytes == 0 || bytes > 1 << 20 {
            return Err(invalid(format!(
                "invalid physical-core topology size query: {error}"
            )));
        }
        let mut buffer = vec![0u64; (bytes as usize).div_ceil(8)];
        let capacity = buffer.len() * 8;
        // SAFETY: the u64 buffer is aligned for Win64 topology records and has
        // at least the requested writable byte capacity; bytes is an in/out size.
        if unsafe { GetLogicalProcessorInformationEx(0, buffer.as_mut_ptr().cast(), &mut bytes) }
            == 0
        {
            return Err(io::Error::last_os_error());
        }
        if bytes as usize > capacity {
            return Err(invalid("topology changed during discovery"));
        }
        // SAFETY: the returned size was checked against the allocated buffer.
        let raw =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), bytes as usize) };
        Ok((parse_cores(raw)?, process))
    }

    fn parse_cores(mut raw: &[u8]) -> io::Result<Vec<usize>> {
        let mut cores = Vec::new();
        while !raw.is_empty() {
            // Win64 SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX header is 8 bytes;
            // PROCESSOR_RELATIONSHIP has GroupCount at 30 and GROUP_AFFINITY at 32.
            if raw.len() < 48 {
                return Err(invalid("truncated physical-core topology"));
            }
            let relation = u32::from_le_bytes(raw[..4].try_into().unwrap());
            let size = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
            let groups = u16::from_le_bytes(raw[30..32].try_into().unwrap());
            let group = u16::from_le_bytes(raw[40..42].try_into().unwrap());
            if relation != 0 || size < 48 || size > raw.len() || groups != 1 || group != 0 {
                return Err(invalid(
                    "unsupported physical-core topology record or processor group",
                ));
            }
            cores.push(u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize);
            raw = &raw[size..];
        }
        if cores.is_empty() {
            return Err(invalid("no physical cores reported"));
        }
        Ok(cores)
    }

    pub fn apply(requested: usize, allow_narrower: bool) -> io::Result<()> {
        let (process, inherited_thread) = restrictions()?;
        let available = process & inherited_thread;
        let mask = if allow_narrower {
            requested & available
        } else {
            requested
        };
        if mask == 0 || mask & !available != 0 {
            return Err(invalid(
                "planned processor is excluded by current process/thread restrictions",
            ));
        }
        // SAFETY: only the current library-owned worker is changed, and its
        // nonzero mask is a subset of both existing process and thread affinity.
        if unsafe { SetThreadAffinityMask(THREAD, mask) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn topology_parser_rejects_truncation_and_other_processor_groups() {
            let mut record = [0u8; 48];
            record[4..8].copy_from_slice(&48u32.to_le_bytes());
            record[30..32].copy_from_slice(&1u16.to_le_bytes());
            record[32..40].copy_from_slice(&0x41u64.to_le_bytes());
            assert_eq!(parse_cores(&record).unwrap(), [0x41]);
            assert!(parse_cores(&record[..47]).is_err());
            record[40..42].copy_from_slice(&1u16.to_le_bytes());
            assert!(parse_cores(&record).is_err());
        }
    }
}

#[cfg(not(all(windows, target_pointer_width = "64")))]
mod platform {
    use super::*;
    pub fn discover() -> io::Result<(Vec<usize>, usize)> {
        Err(invalid("physical thread placement requires 64-bit Windows"))
    }
    pub fn apply(_: usize, _: bool) -> io::Result<()> {
        Err(invalid("physical thread placement requires 64-bit Windows"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn planning_uses_topology_and_process_restrictions_instead_of_smt_numbering() {
        let plan = Placement::from_cores(vec![0x88, 0x41, 0x22, 0x14], 0xfe, 3, true).unwrap();
        assert_eq!(plan.compute, [0x02, 0x04, 0x08]);
        assert_eq!(plan.writer_core, Some(0x41));
        assert!(plan.compute.iter().all(|mask| mask & 0x41 == 0));
        assert_eq!(plan.process_mask & !plan.writer_core.unwrap(), 0xbe);
    }
    #[test]
    fn insufficient_or_invalid_core_domains_are_rejected_before_startup() {
        assert!(Placement::from_cores(vec![3, 12], 15, 2, true).is_err());
        assert!(Placement::from_cores(vec![3, 6], 7, 1, false).is_err());
        assert!(Placement::from_cores(vec![3], 7, 1, false).is_err());
        assert!(Placement::from_cores(vec![3, 12], 3, 2, false).is_err());
    }
    #[test]
    fn helper_and_publisher_have_separate_cores_excluded_from_reads() {
        let plan = Placement::from_cores_with_publication_prepare(
            vec![0x88, 0x41, 0x22, 0x14], 0xfe, 2, true, true,
        ).unwrap();
        assert_eq!(plan.compute, [0x02, 0x04]);
        assert_eq!(plan.writer_core, Some(0x41));
        assert_eq!(plan.prepare_core, Some(0x88));
        assert_eq!(plan.blocking_read_mask(), Some(0x36));
        assert_eq!(plan.thread_mask(ThreadInfo {
            role: ThreadRole::PublicationPrepare, index: 0, count: 1,
        }).unwrap(), Some(0x08));
        assert_eq!(plan.thread_mask(ThreadInfo {
            role: ThreadRole::Publish, index: 0, count: 1,
        }).unwrap(), Some(0x40));
    }
    #[test]
    fn helper_reservation_rejects_insufficient_cores_and_unplanned_startup() {
        assert!(Placement::from_cores_with_publication_prepare(
            vec![3, 12], 15, 1, true, true,
        ).is_err());
        assert!(Placement::from_cores_with_publication_prepare(
            vec![3, 12, 48], 63, 1, false, true,
        ).is_err());
        let plan = Placement::from_cores(vec![3, 12, 48], 63, 2, true).unwrap();
        assert!(plan.thread_mask(ThreadInfo {
            role: ThreadRole::PublicationPrepare, index: 0, count: 1,
        }).is_err());
    }
    #[test]
    fn helper_capacity_counts_allowed_physical_cores_and_excludes_both_siblings() {
        let cores = vec![0x03, 0x0c, 0x30, 0xc0];
        let plan = Placement::from_cores_with_publication_prepare(
            cores.clone(), 0xab, 2, true, true,
        ).unwrap();
        assert_eq!(plan.available_cores, 4);
        assert_eq!(plan.compute, [0x01, 0x08]);
        assert_eq!(plan.prepare_core, Some(0x30));
        assert_eq!(plan.writer_core, Some(0xc0));
        assert_eq!(plan.blocking_read_mask(), Some(0x0b));
        assert_eq!(plan.thread_mask(ThreadInfo {
            role: ThreadRole::PublicationPrepare, index: 0, count: 1,
        }).unwrap(), Some(0x20));
        // Four allowed logical processors can still represent only two cores.
        assert!(Placement::from_cores_with_publication_prepare(
            cores.clone(), 0x0f, 1, true, true,
        ).is_err());
        assert!(Placement::from_cores_with_publication_prepare(
            cores.clone(), 0x3f, 1, true, true,
        ).is_ok());
        assert!(Placement::from_cores_with_publication_prepare(
            cores, 0x3f, 2, true, true,
        ).is_err());
    }
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn publisher_only_defaults_keep_thirty_and_thirty_one_worker_mapping() {
        let cores: Vec<_> = (0..32).map(|core| 3usize << (core * 2)).collect();
        for count in [30, 31] {
            let plan = Placement::from_cores(cores.clone(), usize::MAX, count, true).unwrap();
            let expected: Vec<_> = (0..count).map(|core| 1usize << (core * 2)).collect();
            assert_eq!(plan.compute, expected);
            assert_eq!(plan.writer_core, Some(3usize << 62));
            assert_eq!(plan.prepare_core, None);
            assert_eq!(plan.blocking_read_mask(), Some(usize::MAX & !(3usize << 62)));
            // With no helper requested, reads retain access to the spare core.
            assert_eq!(plan.blocking_read_mask().unwrap() & (3usize << 60), 3usize << 60);
        }
    }
}
