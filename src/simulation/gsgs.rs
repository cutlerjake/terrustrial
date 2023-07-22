use itertools::{iproduct, izip, Itertools};
use nalgebra::Point3;
use ndarray::Array3;
use rand::{rngs::StdRng, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    geometry::Geometry,
    kriging::simple_kriging::SimpleKrigingSystem,
    spatial_database::{
        gridded_databases::{
            gridded_data_base_query_engine::GriddedDataBaseOctantQueryEngine,
            GriddedDataBaseInterface,
        },
        SpatialQueryable,
    },
    variography::model_variograms::VariogramModel,
};

use super::{lu::LUSystem, sgs::SGSParameters};

pub struct GSGSParameters {
    pub max_octant_cond_data: usize,
    pub max_octant_sim_data: usize,
    pub group_size: [usize; 3],
}

pub struct GSGS<S, V, G>
where
    S: SpatialQueryable<f32, G>,
{
    conditioning_data: S,
    variogram_model: V,
    gsgs_parameters: GSGSParameters,
    phantom: std::marker::PhantomData<G>,
}

impl<S, V, G> GSGS<S, V, G>
where
    S: SpatialQueryable<f32, G> + Sync,
    V: VariogramModel + Sync,
    G: Geometry + Sync + Clone,
{
    /// Create a new simple kriging estimator with the given parameters
    /// # Arguments
    /// * `conditioning_data` - The data to condition the kriging system on
    /// * `variogram_model` - The variogram model to use
    /// * `search_ellipsoid` - The search ellipsoid to use
    /// * `kriging_parameters` - The kriging parameters to use
    /// # Returns
    /// A new simple kriging estimator
    pub fn new(
        conditioning_data: S,
        variogram_model: V,
        kriging_parameters: GSGSParameters,
    ) -> Self {
        Self {
            conditioning_data,
            variogram_model,
            gsgs_parameters: kriging_parameters,
            phantom: std::marker::PhantomData,
        }
    }

    /// Perform simple kriging at all kriging points
    pub fn simulate_grid<GDB>(&self, grid: &mut GDB, rng: &mut StdRng)
    where
        GDB: GriddedDataBaseInterface<f32> + std::marker::Sync,
    {
        // create query engine for simulation grid
        let geometry = self.conditioning_data.geometry().clone();
        let mut sim_qe = GriddedDataBaseOctantQueryEngine::new(
            geometry.clone(),
            grid,
            self.gsgs_parameters.max_octant_sim_data,
        );

        //filter offsets to avoid unnecessary searching
        let step = self.gsgs_parameters.group_size;
        let internal_offsets = iproduct!(0..step[0], 0..step[1], 0..step[2])
            .map(|(i, j, k)| [i, j, k])
            .collect_vec();
        let retainer = |offset: &[isize; 3]| {
            if offset.iter().any(|v| *v < 0) {
                return true;
            }
            let offset = offset.map(|v| v as usize);
            !internal_offsets.iter().any(|v| v == &offset)
        };
        sim_qe.retain_offsets(retainer);

        // Array to store simulation order
        let mut simulation_order = Array3::from_elem(grid.shape(), 0);

        //create path over groups
        let i_steps = (grid.shape()[0] as f32 / step[0] as f32).ceil() as usize;
        let j_steps = (grid.shape()[1] as f32 / step[1] as f32).ceil() as usize;
        let k_steps = (grid.shape()[2] as f32 / step[2] as f32).ceil() as usize;
        let grid_shape = grid.shape();

        let mut path = Vec::new();
        for (group_i, group_j, group_k) in iproduct!(0..i_steps, 0..j_steps, 0..k_steps) {
            //get group bounds
            let i_min = group_i * step[0];
            let i_max = ((group_i + 1) * step[0]).min(grid_shape[0]);
            let j_min = group_j * step[1];
            let j_max = ((group_j + 1) * step[1]).min(grid_shape[1]);
            let k_min = group_k * step[2];
            let k_max = ((group_k + 1) * step[2]).min(grid_shape[2]);

            //get all points in group
            let mut group_points = Vec::new();
            for i in i_min..i_max {
                for j in j_min..j_max {
                    for k in k_min..k_max {
                        group_points.push([i, j, k]);
                    }
                }
            }

            //shuffle group points
            group_points.shuffle(rng);
            path.push(group_points);
        }

        //shuffle path order
        path.shuffle(rng);

        //set simulation order
        for (ind, val) in path.iter().enumerate() {
            for val in val.iter() {
                simulation_order[*val] = ind;
            }
        }

        //println!("Starting loop");
        // Note: We do not know the values so we can't populate the grid
        // But at each location in the grid we now all the points that will be previously simulated and the locations of the conditioning data
        // thus, we can solve for the weights in parrallel, then populate the grid sequentially
        let mut local_system = LUSystem::new(
            self.gsgs_parameters.group_size.iter().product(),
            (self.gsgs_parameters.max_octant_cond_data + self.gsgs_parameters.max_octant_sim_data)
                * 8,
        );

        // let sequential_data = path
        //     .iter()
        //     .map(|inds| {
        //         //let ind = [ind.0 as isize, ind.1 as isize, ind.2 as isize];
        //         //get pointer at center of groud
        //         let mut sim_points = inds
        //             .iter()
        //             .map(|ind| grid.ind_to_point(&ind.map(|i| i as isize)))
        //             .collect::<Vec<_>>();
        //         let point = sim_points.iter().fold(Point3::origin(), |mut acc, p| {
        //             acc.coords += p.coords;
        //             acc
        //         }) / inds.len() as f32;

        //         //get nearest conditioning  points and values
        //         let (cond_values, mut cond_points) = self.conditioning_data.query(&point);

        //         // get nearest simulation points
        //         let (sim_cond_inds, sim_cond_points) =
        //             sim_qe.nearest_inds_and_points_masked(&point, |neighbor_ind| {
        //                 simulation_order[neighbor_ind]
        //                     < simulation_order[inds[0].map(|ind| ind as usize)]
        //             });

        //         // sim_cond_points
        //         //     .retain(|point| cond_points.iter().all(|cond_point| cond_point != point));

        //         //append simulation points to conditioning points
        //         cond_points.extend(sim_cond_points.iter());

        //         //slightly shift points to avoid degenerate systems
        //         let cond_points = cond_points
        //             .iter_mut()
        //             .map(|point| {
        //                 point.coords.x += rng.gen::<f32>() * 0.0001;
        //                 point.coords.y += rng.gen::<f32>() * 0.0001;
        //                 point.coords.z += rng.gen::<f32>() * 0.0001;
        //                 *point
        //             })
        //             .collect_vec();

        //         let mini_system = local_system.create_mini_system(
        //             &cond_points,
        //             &sim_points,
        //             &self.variogram_model,
        //         );

        //         (inds, sim_points, cond_values, sim_cond_inds, mini_system)
        //     })
        //     .collect::<Vec<_>>();

        let sequential_data = path
            .par_iter()
            .map_with(
                (
                    LUSystem::new(
                        self.gsgs_parameters.group_size.iter().product(),
                        (self.gsgs_parameters.max_octant_cond_data
                            + self.gsgs_parameters.max_octant_sim_data)
                            * 8,
                    ),
                    StdRng::from_entropy(),
                ),
                |(local_system, local_rng), inds| {
                    //let ind = [ind.0 as isize, ind.1 as isize, ind.2 as isize];
                    //get pointer at center of groud
                    let mut sim_points = inds
                        .iter()
                        .map(|ind| grid.ind_to_point(&ind.map(|i| i as isize)))
                        .collect::<Vec<_>>();
                    let point = sim_points.iter().fold(Point3::origin(), |mut acc, p| {
                        acc.coords += p.coords;
                        acc
                    }) / inds.len() as f32;

                    //get nearest conditioning  points and values
                    let (cond_values, mut cond_points) = self.conditioning_data.query(&point);

                    // get nearest simulation points
                    let (sim_cond_inds, sim_cond_points) =
                        sim_qe.nearest_inds_and_points_masked(&point, |neighbor_ind| {
                            simulation_order[neighbor_ind]
                                < simulation_order[inds[0].map(|ind| ind as usize)]
                        });

                    // sim_cond_points
                    //     .retain(|point| cond_points.iter().all(|cond_point| cond_point != point));

                    //append simulation points to conditioning points
                    cond_points.extend(sim_cond_points.iter());

                    //slightly shift points to avoid degenerate systems
                    let cond_points = cond_points
                        .iter_mut()
                        .map(|point| {
                            point.coords.x += local_rng.gen::<f32>() * 0.0001;
                            point.coords.y += local_rng.gen::<f32>() * 0.0001;
                            point.coords.z += local_rng.gen::<f32>() * 0.0001;
                            *point
                        })
                        .collect_vec();

                    let mini_system = local_system.create_mini_system(
                        &cond_points,
                        &sim_points,
                        &self.variogram_model,
                    );

                    (inds, sim_points, cond_values, sim_cond_inds, mini_system)
                },
            )
            .collect::<Vec<_>>();

        println!("Finished building systems");

        sequential_data.into_iter().for_each(
            |(inds, sim_points, cond_values, sim_cond_inds, mut mini_system)| {
                //get simulation values
                let sim_values = sim_cond_inds
                    .iter()
                    .map(|ind| grid.data_at_ind(ind).expect("No value at ind"))
                    .collect::<Vec<_>>();

                let values = cond_values
                    .into_iter()
                    .chain(sim_values.into_iter())
                    .collect::<Vec<_>>();

                mini_system.populate_w_vec(values.as_slice(), rng);
                let vals = mini_system.simulate();

                //set values
                for (ind, val) in inds.iter().zip(vals.iter()) {
                    grid.set_data_at_ind(ind, *val);
                }

                // //store values and weights in kriging system
                // unsafe { kriging_system.values.set_dims(values.len(), 1) };
                // for i in 0..values.len() {
                //     unsafe { kriging_system.values.write_unchecked(i, 0, *values[i]) };
                // }
                // kriging_system.weights = weights;
                // kriging_system.krig_point_cov_vec = cov_vec;

                // //compute mean
                // let mean = kriging_system.estimate();

                // //compute variance
                // let variance = kriging_system.variance();

                // let value = mean + rng.gen::<f32>() * variance;

                // //set value
                // grid.set_data_at_ind(&ind.map(|v| v as usize), value);
            },
        );
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write};

    use nalgebra::{UnitQuaternion, Vector3};
    use num_traits::Float;
    use rand::SeedableRng;

    use crate::{
        geometry::ellipsoid::Ellipsoid,
        spatial_database::{
            coordinate_system::{CoordinateSystem, GridSpacing},
            gridded_databases::incomplete_grid::InCompleteGriddedDataBase,
            normalized::Normalize,
        },
        variography::model_variograms::spherical::SphericalVariogram,
    };

    use super::*;
    #[test]
    fn gsgs_test() {
        let coordinate_system = CoordinateSystem::new(
            Point3::new(0.0, 0.0, 0.0).into(),
            UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians()),
        );
        let mut gdb = InCompleteGriddedDataBase::from_csv_index(
            "C:\\GitRepos\\terrustrial_test\\data\\walker.csv",
            "X",
            "Y",
            "Z",
            "V",
            GridSpacing {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
            coordinate_system,
        )
        .unwrap();

        gdb.normalize();

        let sim_grid_arr = Array3::<Option<f32>>::from_elem(gdb.shape(), None);
        let mut sim_db = InCompleteGriddedDataBase::new(
            sim_grid_arr,
            gdb.grid_spacing().clone(),
            CoordinateSystem::new(
                Point3::new(0.0, 0.0, 0.0).into(),
                UnitQuaternion::from_euler_angles(
                    0.0.to_radians(),
                    0.0.to_radians(),
                    0.0.to_radians(),
                ),
            ),
        );

        let vgram_rot =
            UnitQuaternion::from_euler_angles(0.0.to_radians(), 0.0.to_radians(), 0.0.to_radians());
        let vgram_origin = Point3::new(0.0, 0.0, 0.0);
        let vgram_coordinate_system = CoordinateSystem::new(vgram_origin.into(), vgram_rot);
        let range = Vector3::new(450.0, 150.0, 1.0);
        let sill = 1.0;
        let nugget = 0.0;

        let spherical_vgram =
            SphericalVariogram::new(range, sill, nugget, vgram_coordinate_system.clone());

        // create search ellipsoid
        let search_ellipsoid = Ellipsoid::new(450.0, 150.0, 1.0, vgram_coordinate_system.clone());

        let query_engine = GriddedDataBaseOctantQueryEngine::new(search_ellipsoid, &gdb, 16);
        //create simple kriging

        let gsgs = GSGS::new(
            query_engine,
            spherical_vgram,
            GSGSParameters {
                max_octant_cond_data: 16,
                max_octant_sim_data: 16,
                group_size: [4, 4, 1],
            },
        );

        let mut rng = StdRng::seed_from_u64(0);

        gsgs.simulate_grid(&mut sim_db, &mut rng);
        //sgs.simulate_grid(&mut sim_db, &mut rng);

        let (values, points) = sim_db.data_and_points();

        println!("kriging");
        //let values = simple_kriging.krig(kriging_points.as_slice());
        let mut out = File::create("out.txt").unwrap();
        out.write_all(b"surfs\n");
        out.write_all(b"4\n");
        out.write_all(b"x\n");
        out.write_all(b"y\n");
        out.write_all(b"z\n");
        out.write_all(b"value\n");

        println!("GRID");
        for (point, value) in points.iter().zip(values.iter()) {
            //println!("point: {:?}, value: {}", point, value);
            out.write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
        }
    }
}
