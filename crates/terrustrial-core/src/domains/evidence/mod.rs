//! Geological interval evidence. Gaps are unknown, never implied exterior.
use crate::{error::invalid, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Membership {
    Interior,
    Exterior,
    Unknown,
}
#[derive(Clone, Debug)]
pub struct Record {
    pub from: f64,
    pub to: f64,
    pub code: String,
    /// Source line identifiers retained through merging.
    pub source_rows: Vec<u64>,
}
#[derive(Clone, Debug)]
pub struct Interval {
    pub from: f64,
    pub to: f64,
    pub membership: Membership,
    pub code: Option<String>,
    pub source_rows: Vec<u64>,
}
#[derive(Clone, Debug)]
pub struct Contact {
    pub depth: f64,
    pub source_rows: Vec<u64>,
    /// A different interpreted unit touches this boundary. Otherwise it borders
    /// unknown coverage; the endpoint remains an interpreted target limit.
    pub known_exterior: bool,
}
/// An interval endpoint beside unknown coverage is not automatically an
/// observed interface. Admit it only under an explicit interpretation policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactPolicy {
    KnownTransitions,
    InterpretedLimits,
}
#[derive(Clone, Debug)]
pub struct Hole {
    pub id: String,
    pub length: f64,
    pub intervals: Vec<Interval>,
    pub contacts: Vec<Contact>,
}
impl Hole {
    pub fn fitting_contacts(&self, policy: ContactPolicy) -> impl Iterator<Item = &Contact> {
        self.contacts
            .iter()
            .filter(move |c| c.known_exterior || policy == ContactPolicy::InterpretedLimits)
    }
    /// Different codes are mutually exclusive geological units in this input
    /// contract. Overlapping different codes must be reconciled by the caller.
    pub fn new(id: String, length: f64, target: &str, mut records: Vec<Record>) -> Result<Self> {
        if id.is_empty() || target.is_empty() || !length.is_finite() || length <= 0. {
            return Err(invalid("geological hole identity, target or length"));
        }
        records.sort_by(|a, b| {
            a.from
                .total_cmp(&b.from)
                .then(a.to.total_cmp(&b.to))
                .then(a.code.cmp(&b.code))
        });
        let mut merged: Vec<Record> = Vec::new();
        for r in records {
            if !r.from.is_finite()
                || !r.to.is_finite()
                || r.from < 0.
                || r.to <= r.from
                || r.to > length
                || r.code.is_empty()
                || r.source_rows.is_empty()
            {
                return Err(invalid("invalid geological interval or missing provenance"));
            }
            if let Some(previous) = merged.last_mut() {
                if r.from < previous.to {
                    return Err(invalid(
                        "overlapping geological intervals require reconciliation",
                    ));
                }
                if r.from == previous.to && r.code == previous.code {
                    previous.to = r.to;
                    previous.source_rows.extend(r.source_rows);
                    continue;
                }
            }
            merged.push(r);
        }
        let mut intervals = Vec::new();
        let mut start = 0.;
        for r in merged {
            if start < r.from {
                intervals.push(Interval {
                    from: start,
                    to: r.from,
                    membership: Membership::Unknown,
                    code: None,
                    source_rows: vec![],
                });
            }
            start = r.to;
            intervals.push(Interval {
                from: r.from,
                to: r.to,
                membership: if r.code == target {
                    Membership::Interior
                } else {
                    Membership::Exterior
                },
                code: Some(r.code),
                source_rows: r.source_rows,
            });
        }
        if start < length {
            intervals.push(Interval {
                from: start,
                to: length,
                membership: Membership::Unknown,
                code: None,
                source_rows: vec![],
            });
        }
        let mut contacts = Vec::new();
        for (i, r) in intervals
            .iter()
            .enumerate()
            .filter(|(_, r)| r.membership == Membership::Interior)
        {
            if r.from > 0. {
                let mut source_rows = r.source_rows.clone();
                if i > 0 && intervals[i - 1].membership == Membership::Exterior {
                    source_rows.extend_from_slice(&intervals[i - 1].source_rows);
                }
                contacts.push(Contact {
                    depth: r.from,
                    source_rows,
                    known_exterior: i > 0 && intervals[i - 1].membership == Membership::Exterior,
                });
            }
            if r.to < length {
                let mut source_rows = r.source_rows.clone();
                if intervals
                    .get(i + 1)
                    .is_some_and(|r| r.membership == Membership::Exterior)
                {
                    source_rows.extend_from_slice(&intervals[i + 1].source_rows);
                }
                contacts.push(Contact {
                    depth: r.to,
                    source_rows,
                    known_exterior: intervals
                        .get(i + 1)
                        .is_some_and(|r| r.membership == Membership::Exterior),
                });
            }
        }
        Ok(Self {
            id,
            length,
            intervals,
            contacts,
        })
    }
    /// Interval midpoints at spacing no greater than `maximum_step`, including
    /// short units. Unknown coverage is returned as unknown for auditing.
    pub fn sample(
        &self,
        maximum_step: f64,
        maximum_points: usize,
    ) -> Result<Vec<(f64, Membership)>> {
        if !maximum_step.is_finite() || maximum_step <= 0. {
            return Err(invalid("geological evidence sampling step"));
        }
        let mut out = Vec::new();
        for r in &self.intervals {
            let n = ((r.to - r.from) / maximum_step).ceil().max(1.);
            if n > maximum_points.saturating_sub(out.len()) as f64 {
                return Err(invalid("geological evidence sample limit exceeded"));
            }
            for i in 0..n as usize {
                out.push((
                    r.from + (i as f64 + 0.5) * (r.to - r.from) / n,
                    r.membership,
                ));
            }
        }
        Ok(out)
    }
}
