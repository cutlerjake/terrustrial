//! Eight quadrature nodes per block, with stable seven-variable target IDs.
use terrustrial_core::data::{Group, Point3, Site, Target};
#[derive(Clone)]
pub struct Block {
    pub center: Point3,
    pub mask: u8,
}
impl Block {
    pub fn nodes(&self, spacing: f64) -> impl Iterator<Item = (usize, Point3)> + '_ {
        (0..8).filter(|n| self.mask & (1 << n) != 0).map(move |n| {
            (
                n,
                Point3::new(std::array::from_fn(|a| {
                    self.center.coordinates()[a]
                        + if n & (1 << a) == 0 {
                            -spacing / 4.
                        } else {
                            spacing / 4.
                        }
                }))
                .expect("finite admitted lattice"),
            )
        })
    }
    pub fn group(&self, rank: usize, cell: usize, spacing: f64) -> Group {
        let mut targets = Vec::with_capacity(self.mask.count_ones() as usize * 7);
        for (node, p) in self.nodes(spacing) {
            targets.extend((0..7).map(|v| Target {
                id: 1_000_000 + cell as u64 * 56 + node as u64 * 7 + v as u64,
                site: Site::point(p, v),
            }));
        }
        Group {
            id: rank as u64,
            targets,
        }
    }
}
