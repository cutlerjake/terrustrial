//! One-entry exact-center miss reuse within one immutable selection view.
use super::ObservationSource;
use crate::{
    data::{Observation, Site},
    Result,
};
use std::borrow::Cow;

/// Fixed metadata only: positive results and incomplete/erroring queries are
/// never cached. The owner charges this inline storage and resets it before
/// every selection attempt, including retries against a different source view.
#[derive(Default)]
pub(crate) struct ExactWorkspace {
    empty_center: Option<[f64; 3]>,
}

impl ExactWorkspace {
    pub(crate) fn reset(&mut self) {
        self.empty_center = None;
    }

    pub(crate) fn at_site<S: ObservationSource + ?Sized>(
        &mut self,
        source: &S,
        site: &Site,
    ) -> Result<Cow<'_, [Observation]>> {
        let center = site.support.center().coordinates();
        // Numeric equality intentionally identifies positive and negative zero,
        // exactly as the source's geometric predicates do.
        if self.empty_center == Some(center) {
            return Ok(Cow::Borrowed(&[]));
        }
        self.empty_center = None;
        let (found, center_empty) = source.at_site_with_center_status(site)?;
        if center_empty && found.is_empty() {
            self.empty_center = Some(center);
        }
        Ok(Cow::Owned(found))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{data::Point3, error::invalid};
    use std::cell::Cell;

    struct Source {
        calls: Cell<usize>,
        empty: Cell<bool>,
        fail: Cell<bool>,
    }
    impl ObservationSource for Source {
        fn scan(&self, _: &mut dyn FnMut(&Observation) -> Result<()>) -> Result<()> {
            panic!("use exact source override")
        }
        fn at_site_with_center_status(&self, _: &Site) -> Result<(Vec<Observation>, bool)> {
            self.calls.set(self.calls.get() + 1);
            if self.fail.get() {
                return Err(invalid("incomplete exact traversal"));
            }
            Ok((Vec::new(), self.empty.get()))
        }
    }
    fn site(x: f64, variable: usize) -> Site {
        Site::point(Point3::new([x, 0., 0.]).unwrap(), variable)
    }
    fn source() -> Source {
        Source {
            calls: Cell::new(0),
            empty: Cell::new(true),
            fail: Cell::new(false),
        }
    }

    #[test]
    fn exact_center_miss_reuses_consecutive_variables_and_signed_zero() {
        let source = source();
        let mut workspace = ExactWorkspace::default();
        for variable in 0..7 {
            assert!(workspace
                .at_site(&source, &site(-0., variable))
                .unwrap()
                .is_empty());
            assert!(workspace
                .at_site(&source, &site(0., variable))
                .unwrap()
                .is_empty());
        }
        assert_eq!(source.calls.get(), 1);
        workspace.at_site(&source, &site(1., 0)).unwrap();
        workspace.at_site(&source, &site(0., 0)).unwrap();
        assert_eq!(source.calls.get(), 3);
    }

    #[test]
    fn exact_center_miss_requires_proof_and_resets_between_views() {
        let source = source();
        let mut workspace = ExactWorkspace::default();
        source.empty.set(false);
        for variable in 0..7 {
            workspace.at_site(&source, &site(0., variable)).unwrap();
        }
        assert_eq!(source.calls.get(), 7);
        source.empty.set(true);
        workspace.at_site(&source, &site(0., 0)).unwrap();
        workspace.reset();
        source.empty.set(false);
        workspace.at_site(&source, &site(0., 1)).unwrap();
        assert_eq!(source.calls.get(), 9);
    }

    #[test]
    fn exact_center_miss_does_not_cache_failed_traversals() {
        let source = source();
        let mut workspace = ExactWorkspace::default();
        source.fail.set(true);
        assert!(workspace.at_site(&source, &site(0., 0)).is_err());
        assert!(workspace.at_site(&source, &site(0., 1)).is_err());
        assert_eq!(source.calls.get(), 2);
        source.fail.set(false);
        workspace.at_site(&source, &site(0., 0)).unwrap();
        workspace.at_site(&source, &site(0., 1)).unwrap();
        assert_eq!(source.calls.get(), 3);
    }
}
