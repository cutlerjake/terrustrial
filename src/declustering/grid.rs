use nalgebra::Point3;

use crate::spatial_database::DiscretiveVolume;
use parking_lot::RwLock;
use parry3d::bounding_volume::Aabb;
use rand::rngs::ThreadRng;
use rand::Rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashMap;

pub enum DeclusterDirection {
    Minimize,
    Maximize,
}

pub struct GridDecluster {
    pub cell_sizes: Vec<[f32; 3]>,
    pub origins: Vec<[f32; 3]>,
}

impl GridDecluster {
    pub fn new(cell_sizes: Vec<[f32; 3]>, origins: Vec<[f32; 3]>) -> Self {
        Self {
            cell_sizes,
            origins,
        }
    }

    pub fn origins_from_volume(aabb: Aabb, max_n: usize) -> Vec<[f32; 3]> {
        let mut points = Vec::with_capacity(max_n);
        let mut rng = rand::thread_rng();

        for _ in 0..max_n {
            let x = rng.gen_range(aabb.mins.x..aabb.maxs.x);
            let y = rng.gen_range(aabb.mins.y..aabb.maxs.y);
            let z = rng.gen_range(aabb.mins.z..aabb.maxs.z);

            points.push([x, y, z]);
        }
        points
    }

    pub fn sizes_from_steps(&mut self, size: [f32; 3], dx: f32, dy: f32, dz: f32, steps: u32) {
        let mut sizes = Vec::new();
        for i in 0..steps {
            sizes.push([
                size[0] + dx * i as f32,
                size[1] + dy * i as f32,
                size[2] + dz * i as f32,
            ]);
        }
        self.cell_sizes.extend(sizes);
    }

    pub fn decluster(
        &self,
        points: &[Point3<f32>],
        values: &[f32],
        direction: DeclusterDirection,
    ) -> Vec<f32> {
        let curr_best = match direction {
            DeclusterDirection::Minimize => f32::MAX,
            DeclusterDirection::Maximize => f32::MIN,
        };

        //iter over cell sizes
        self.cell_sizes
            .par_iter()
            .fold(
                || (Vec::new(), curr_best),
                |(mut optimal_weights, mut best_weight), cell_size| {
                    //iter over origins

                    let origins = Self::origins_from_volume(
                        Aabb::new(
                            Point3::origin(),
                            Point3::new(cell_size[0], cell_size[1], cell_size[2]),
                        ),
                        1000,
                    );
                    let avg_weights = origins
                        .par_iter()
                        .fold(
                            || vec![0.0; points.len()],
                            |mut weights, origin| {
                                //compute the grid indices for each point
                                let grid_inds = points
                                    .iter()
                                    .map(|point| {
                                        //shift point by origin
                                        let shifted_point = Point3::new(
                                            point.x - origin[0],
                                            point.y - origin[1],
                                            point.z - origin[2],
                                        );

                                        //compute grid origin
                                        let grid_index = [
                                            (shifted_point.x / cell_size[0]) as i32,
                                            (shifted_point.y / cell_size[1]) as i32,
                                            (shifted_point.z / cell_size[2]) as i32,
                                        ];

                                        grid_index
                                    })
                                    .collect::<Vec<_>>();

                                //group points by grid index
                                let grid_groups = grid_inds.iter().zip(points.iter()).fold(
                                    HashMap::new(),
                                    |mut map, (grid_index, point)| {
                                        map.entry(grid_index).or_insert(Vec::new()).push(point);
                                        map
                                    },
                                );

                                //numer of occupied grid cells
                                let n_occupied = grid_groups.len();
                                // let grid_weight = 1.0 / n_occupied as f32;

                                let constant = points.len() as f32 / n_occupied as f32;

                                //construct weight vector
                                let point_weights = grid_inds
                                    .iter()
                                    .map(|ind| constant / grid_groups[ind].len() as f32)
                                    .collect::<Vec<_>>();

                                //update weights
                                weights.iter_mut().zip(point_weights.iter()).for_each(
                                    |(weight, point_weight)| {
                                        *weight += point_weight;
                                    },
                                );

                                weights
                            },
                        )
                        .reduce(
                            || vec![0.0; points.len()],
                            |mut weights, weights2| {
                                weights.iter_mut().zip(weights2.iter()).for_each(
                                    |(weight, weight2)| {
                                        *weight += weight2 / origins.len() as f32;
                                    },
                                );
                                weights
                            },
                        );

                    //compute weighted mean
                    let declustered_mean: f32 = avg_weights
                        .iter()
                        .zip(values.iter())
                        .map(|(w, v)| w * v)
                        .sum::<f32>()
                        / avg_weights.iter().sum::<f32>();

                    match direction {
                        DeclusterDirection::Minimize => {
                            if declustered_mean < best_weight {
                                best_weight = declustered_mean;
                                optimal_weights = avg_weights;
                            }
                        }
                        DeclusterDirection::Maximize => {
                            if declustered_mean > best_weight {
                                best_weight = declustered_mean;
                                optimal_weights = avg_weights;
                            }
                        }
                    }

                    println!("[{},{}],", cell_size[0], declustered_mean);
                    (optimal_weights, best_weight)
                },
            )
            .reduce(
                || (Vec::new(), 0.0),
                |(mut optimal_weights, best_weights), (optimal_weights2, best_weights2)| {
                    match direction {
                        DeclusterDirection::Minimize => {
                            if best_weights2 < best_weights {
                                optimal_weights = optimal_weights2;
                                (optimal_weights, best_weights2)
                            } else {
                                (optimal_weights, best_weights)
                            }
                        }
                        DeclusterDirection::Maximize => {
                            if best_weights2 > best_weights {
                                optimal_weights = optimal_weights2;
                                (optimal_weights, best_weights2)
                            } else {
                                (optimal_weights, best_weights)
                            }
                        }
                    }
                },
            )
            .0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn grid_decluster() {
        println!("Reading Target Data");
        //let mut reader = csv::Reader::from_path("C:\\Users\\2jake\\test.csv").unwrap();
        let mut reader =
            csv::Reader::from_path("C:\\GitRepos\\terrustrial\\data\\point_set_filtered.csv")
                .unwrap();
        let mut points = Vec::new();
        let mut values = Vec::new();
        for record in reader.deserialize() {
            let record: (f32, f32, f32, f32) = record.unwrap();
            points.push(Point3::new(record.0, record.1, record.3));
            values.push(record.3);
        }

        let mut grid_decluster = GridDecluster::new(vec![], vec![]);
        grid_decluster.sizes_from_steps([10.0, 10.0, 10.0], 4.0, 4.0, 4.0, 500);
        // grid_decluster.origins_from_volume(
        //     Aabb::new(
        //         Point3::new(0.0, 0.0, 0.0),
        //         Point3::new(1000.0, 1000.0, 1000.0),
        //     ),
        //     1.0,
        //     1.0,
        //     1.0,
        //     Some(0.1),
        // );

        let weights = grid_decluster.decluster(&points, &values, DeclusterDirection::Minimize);
    }
}
