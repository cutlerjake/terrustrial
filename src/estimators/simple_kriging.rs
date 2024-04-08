use crate::{spatial_database::SupportInterface, variography::model_variograms::VariogramModel};
use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};
use faer::{
    modules::{cholesky, core::mul::inner_prod::inner_prod_with_conj},
    MatRef, Parallelism,
};
use faer::{Conj, Mat};
use nalgebra::{Point3, SimdRealField, SimdValue, Vector3};
use num_traits::Float;
use simba::simd::SimdPartialOrd;

use super::KrigingSystem;

pub trait SKBuilder {
    type Support: SupportInterface + Send + Sync;

    fn build_cov_mat<'a, I, V, T>(cov_mat: &mut Mat<f32>, cond: I, vgram: &V)
    where
        T: SimdValue<Element = f32> + SimdRealField + Copy,
        V: VariogramModel<T>,
        I: Iterator<Item = &'a Self::Support> + Clone,
        <Self as SKBuilder>::Support: 'a;

    fn build_cov_vec<'a, I, V, T>(
        cov_vec: &mut Mat<f32>,
        cond: I,
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        T: SimdValue<Element = f32> + SimdRealField + Copy,
        V: VariogramModel<T>,
        I: Iterator<Item = &'a Self::Support> + Clone,
        <Self as SKBuilder>::Support: 'a;
}

pub struct SKPointSupportBuilder;

impl SKBuilder for SKPointSupportBuilder {
    type Support = Point3<f32>;
    fn build_cov_mat<'a, I, V, T>(cov_mat: &mut Mat<f32>, cond: I, vgram: &V)
    where
        T: SimdValue<Element = f32> + SimdRealField + Copy,
        V: VariogramModel<T>,
        I: Iterator<Item = &'a Self::Support> + Clone,
        <Self as SKBuilder>::Support: 'a,
    {
        let mut cnt = 0;
        let mut len = 0;

        //compute lower triangle of cov matrix
        let mut last_insert_i = 0;
        let mut last_insert_j = 0;

        //let mut point = Point3::<T>::origin();
        let mut vec = Vector3::<T>::zeros();
        for (i, p1) in cond.clone().enumerate() {
            len += 1;
            for (j, p2) in cond.clone().take(i + 1).enumerate() {
                //distance between points
                let d = p1 - p2;

                //set coords
                vec.x.replace(cnt, d.x);
                vec.y.replace(cnt, d.y);
                vec.z.replace(cnt, d.z);

                //update insertion counts
                cnt += 1;

                //if all lanes populated, compute covariogram and insert into matrix
                if cnt >= T::lanes() {
                    cnt = 0;

                    //compute covariance
                    let cov = vgram.covariogram(vec);
                    let mut extract_ind = 0;
                    //insert covariance into matrix on current row
                    'outer: for insert_i in last_insert_i..=i {
                        if insert_i == last_insert_i {
                            for insert_j in last_insert_j..=insert_i {
                                if extract_ind == T::lanes() {
                                    break 'outer;
                                }
                                cov_mat.write(insert_i, insert_j, cov.extract(extract_ind));
                                extract_ind += 1;
                            }
                        } else {
                            for insert_j in 0..=insert_i {
                                if extract_ind == T::lanes() {
                                    break 'outer;
                                }
                                cov_mat.write(insert_i, insert_j, cov.extract(extract_ind));
                                extract_ind += 1;
                            }
                        }
                    }

                    //update last insert indices
                    if j < i {
                        last_insert_i = i;
                        last_insert_j = j + 1;
                    } else {
                        last_insert_i = i + 1;
                        last_insert_j = 0;
                    }
                }
            }
        }

        //hanlde reamining values
        let cov = vgram.covariogram(vec);
        let mut extract_ind = 0;
        //insert covariance into matrix on current row
        'outer: for insert_i in last_insert_i..len {
            if insert_i == last_insert_i {
                for insert_j in last_insert_j..=insert_i {
                    if extract_ind == T::lanes() {
                        break 'outer;
                    }
                    cov_mat.write(insert_i, insert_j, cov.extract(extract_ind));
                    extract_ind += 1;
                }
            } else {
                for insert_j in 0..=insert_i {
                    if extract_ind == T::lanes() {
                        break 'outer;
                    }
                    cov_mat.write(insert_i, insert_j, cov.extract(extract_ind));
                    extract_ind += 1;
                }
            }
        }
    }

    fn build_cov_vec<'a, I, V, T>(
        cov_vec: &mut Mat<f32>,
        cond: I,
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        T: SimdValue<Element = f32> + SimdRealField + Copy,
        V: VariogramModel<T>,
        I: Iterator<Item = &'a Self::Support> + Clone,
        <Self as SKBuilder>::Support: 'a,
    {
        let mut cnt = 0;
        let mut len = 0;

        //compute lower triangle of cov matrix
        let mut last_insert_i = 0;
        //let mut point = Point3::<T>::origin();
        let mut vec = Vector3::<T>::zeros();
        for (i, p1) in cond.enumerate() {
            len += 1;
            //distance between points
            let d = p1 - kriging_point;

            //set coords
            vec.x.replace(cnt, d.x);
            vec.y.replace(cnt, d.y);
            vec.z.replace(cnt, d.z);

            //update insertion counts
            cnt += 1;

            //if all lanes populated, compute covariogram and insert into matrix
            if cnt >= T::lanes() {
                cnt = 0;

                //compute covariance
                let cov = vgram.covariogram(vec);
                //insert covariance into matrix on current row
                for (extract_ind, insert_i) in (last_insert_i..=i).enumerate() {
                    cov_vec.write(insert_i, 0, cov.extract(extract_ind));
                }

                //update last insert indices
                last_insert_i = i + 1;
            }
        }

        //handle reamining values
        let cov = vgram.covariogram(vec);

        //insert covariance into matrix on current row
        for (extract_ind, insert_i) in (last_insert_i..len).enumerate() {
            cov_vec.write(insert_i, 0, cov.extract(extract_ind));
        }
    }
}

pub struct SKVolumeSupportBuilder;

impl SupportInterface for Vec<Point3<f32>> {
    fn center(&self) -> Point3<f32> {
        let mut center = Point3::origin();
        for p in self.iter() {
            center.coords += p.coords;
        }
        center.coords /= self.len() as f32;
        center
    }
}

impl SKBuilder for SKVolumeSupportBuilder {
    //TODO: If use a slice instread
    type Support = Vec<Point3<f32>>;
    fn build_cov_mat<'a, I, V, T>(cov_mat: &mut Mat<f32>, cond: I, vgram: &V)
    where
        T: SimdValue<Element = f32> + SimdRealField + Copy,
        V: VariogramModel<T>,
        I: Iterator<Item = &'a Self::Support> + Clone,
        <Self as SKBuilder>::Support: 'a,
    {
        //compute lower triangle of cov matrix
        let mut vec = Vector3::<T>::zeros();
        for (i, ps1) in cond.clone().enumerate() {
            for (j, ps2) in cond.clone().take(i + 1).enumerate() {
                let mut n = 1;
                let mut total_cov = T::zero();
                let mut cnt = 0;
                //compute the average variogram value over the points
                for p1 in ps1.iter() {
                    for p2 in ps2.iter() {
                        //distance between points
                        let d = p1 - p2;

                        //set coords
                        vec.x.replace(cnt, d.x);
                        vec.y.replace(cnt, d.y);
                        vec.z.replace(cnt, d.z);

                        //update insertion counts
                        cnt += 1;

                        //if all lanes populated, compute covariogram and insert into matrix
                        if cnt >= T::lanes() {
                            cnt = 0;
                            n += T::lanes();

                            //update covariance
                            total_cov += vgram.covariogram(vec);
                        }
                    }
                }

                //hanlde reamining values
                let mut remaining_cov = vgram.covariogram(vec);
                n += cnt;
                //set unused lanes to 0
                for extract_ind in cnt..T::lanes() {
                    remaining_cov.replace(extract_ind, 0.0);
                }
                total_cov += remaining_cov;

                let avg_cov = total_cov.simd_horizontal_sum() / n as f32;

                cov_mat.write(i, j, avg_cov);
            }
        }
    }

    #[allow(unused_variables)]
    fn build_cov_vec<'a, I, V, T>(
        cov_vec: &mut Mat<f32>,
        cond: I,
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        T: SimdValue<Element = f32> + SimdRealField + Copy,
        V: VariogramModel<T>,
        I: Iterator<Item = &'a Self::Support> + Clone,
        <Self as SKBuilder>::Support: 'a,
    {
        todo!()
    }
}

pub struct SimpleKrigingSystem {
    pub cond_cov_mat: Mat<f32>,
    pub krig_point_cov_vec: Mat<f32>,
    pub weights: Mat<f32>,
    pub values: Mat<f32>,
    pub c_0: f32,
    pub cholesky_compute_mem: GlobalPodBuffer,
    pub cholesky_solve_mem: GlobalPodBuffer,
    pub n_elems: usize,
}

impl Clone for SimpleKrigingSystem {
    fn clone(&self) -> Self {
        let n_elems = self.n_elems;
        Self {
            cond_cov_mat: self.cond_cov_mat.clone(),
            krig_point_cov_vec: self.krig_point_cov_vec.clone(),
            weights: self.weights.clone(),
            values: self.values.clone(),
            c_0: self.c_0,
            cholesky_compute_mem: GlobalPodBuffer::new(
                cholesky::llt::compute::cholesky_in_place_req::<f32>(
                    n_elems,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            ),
            cholesky_solve_mem: GlobalPodBuffer::new(
                cholesky::llt::solve::solve_req::<f32>(n_elems, 1, Parallelism::None).unwrap(),
            ),
            n_elems,
        }
    }
}

impl SimpleKrigingSystem
// where
//     T: SimdValue<Element = f32> + SimdRealField + Copy,
{
    /// Create a new simple kriging system
    /// # Arguments
    /// * `n_elems` - The maximum number of elements in the system
    /// # Returns
    /// * `Self` - The new simple kriging system
    /// * Requires zero mean data
    pub fn new(n_elems: usize) -> Self {
        let cholesky_compute_mem = GlobalPodBuffer::new(
            cholesky::llt::compute::cholesky_in_place_req::<f32>(
                n_elems,
                Parallelism::None,
                Default::default(),
            )
            .unwrap(),
        );

        let cholesky_solve_mem = GlobalPodBuffer::new(
            cholesky::llt::solve::solve_req::<f32>(n_elems, 1, Parallelism::None).unwrap(),
        );

        Self {
            cond_cov_mat: Mat::zeros(n_elems, n_elems),
            krig_point_cov_vec: Mat::zeros(n_elems, 1),
            weights: Mat::zeros(n_elems, 1),
            values: Mat::zeros(n_elems, 1),
            c_0: 0.0,
            cholesky_compute_mem,
            cholesky_solve_mem,
            n_elems,
        }
    }

    /// Set dimensions of the system for the given number of elements
    /// # Arguments
    /// * `n_elems` - The number of elements in the system
    /// # Safety
    /// This function is unsafe because it does not check that the number of elements is less than the maximum number of elements
    /// panics will occur if the number of elements is greater than the maximum number of elements
    #[inline(always)]
    pub fn set_dim(&mut self, n_elems: usize) {
        unsafe { self.cond_cov_mat.set_dims(n_elems, n_elems) };
        unsafe { self.krig_point_cov_vec.set_dims(n_elems, 1) };
        unsafe { self.weights.set_dims(n_elems, 1) };
        unsafe { self.values.set_dims(n_elems, 1) };
    }

    /// Compute SK weights
    #[inline(always)]
    pub fn compute_weights(&mut self) {
        //create dynstack
        let mut cholesky_compute_stack = PodStack::new(&mut self.cholesky_compute_mem);
        let mut cholesky_solve_stack = PodStack::new(&mut self.cholesky_solve_mem);

        //compute cholseky decomposition of covariance matrix
        let _ = cholesky::llt::compute::cholesky_in_place(
            self.cond_cov_mat.as_mut(),
            Default::default(),
            Parallelism::None,
            cholesky_compute_stack.rb_mut(),
            Default::default(),
        );

        //solve SK system
        cholesky::llt::solve::solve_with_conj(
            self.weights.as_mut(),
            self.cond_cov_mat.as_ref(),
            Conj::No,
            self.krig_point_cov_vec.as_ref(),
            Parallelism::None,
            cholesky_solve_stack.rb_mut(),
        );
    }

    /// Build the system for SK and compute the weights
    /// # Arguments
    /// * `cond_points` - The conditioning points for the kriging point
    /// * `cond_values` - The conditioning values for the kriging point
    /// * `kriging_point` - The kriging point
    /// * 'vgram' - The variogram model
    #[inline(always)]
    pub fn build_system<V, VT, SKB>(
        &mut self,
        cond_points: &[SKB::Support],
        cond_values: &[f32],
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) where
        V: VariogramModel<VT>,
        VT: SimdValue<Element = f32> + SimdRealField + Copy,
        SKB: SKBuilder,
    {
        //set dimensions
        self.set_dim(cond_points.len());

        //build covariance matrix and vector
        SKB::build_cov_mat(&mut self.cond_cov_mat, cond_points.iter(), vgram);
        SKB::build_cov_vec(
            &mut self.krig_point_cov_vec,
            cond_points.iter(),
            kriging_point,
            vgram,
        );

        //store values
        unsafe { self.values.set_dims(cond_values.len(), 1) };
        for (i, &value) in cond_values.iter().enumerate() {
            unsafe { self.values.write_unchecked(i, 0, value) };
        }
        self.c_0 = vgram.c_0();

        //compute kriging weights
        self.compute_weights();
    }

    /// SK Estimate
    #[inline(always)]
    pub fn estimate(&self) -> f32 {
        inner_prod_with_conj(
            self.values.as_ref(),
            Conj::No,
            self.weights.as_ref(),
            Conj::No,
        )
    }
    /// SK Variance
    #[inline(always)]
    pub fn variance(&self) -> f32 {
        self.c_0
            - inner_prod_with_conj(
                self.weights.as_ref(),
                Conj::No,
                self.krig_point_cov_vec.as_ref(),
                Conj::No,
            )
    }

    /// Build the system for SK and compute the weights
    /// # Arguments
    /// * `cond_points` - The conditioning points for the kriging point
    /// * `cond_values` - The conditioning values for the kriging point
    /// * `kriging_point` - The kriging point
    /// * 'vgram' - The variogram model
    #[inline(always)]
    pub fn build_mini_system<V, VT, SKB>(
        &mut self,
        cond_points: &[SKB::Support],
        kriging_point: &Point3<f32>,
        vgram: &V,
    ) -> MiniSKSystem
    where
        V: VariogramModel<VT>,
        VT: SimdValue<Element = f32> + SimdRealField + Copy,
        SKB: SKBuilder,
    {
        //set dimensions
        self.set_dim(cond_points.len());

        //build covariance matrix and vector
        SKB::build_cov_mat(&mut self.cond_cov_mat, cond_points.iter(), vgram);
        SKB::build_cov_vec(
            &mut self.krig_point_cov_vec,
            cond_points.iter(),
            kriging_point,
            vgram,
        );
        self.c_0 = vgram.c_0();

        //compute kriging weights
        self.compute_weights();

        MiniSKSystem {
            c_0: self.c_0,
            weights: self.weights.clone(),
            cov_vec: self.krig_point_cov_vec.clone(),
        }
    }
}

pub struct MiniSKSystem {
    c_0: f32,
    weights: Mat<f32>,
    cov_vec: Mat<f32>,
}

impl MiniSKSystem {
    #[inline(always)]
    pub fn new(c_0: f32, weights: Mat<f32>, cov_vec: Mat<f32>) -> Self {
        Self {
            c_0,
            weights,
            cov_vec,
        }
    }

    #[inline(always)]
    pub fn estimate(&self, values: MatRef<f32>) -> f32 {
        inner_prod_with_conj(values, Conj::No, self.weights.as_ref(), Conj::No)
    }

    #[inline(always)]
    pub fn variance(&self) -> f32 {
        self.c_0
            - inner_prod_with_conj(
                self.weights.as_ref(),
                Conj::No,
                self.cov_vec.as_ref(),
                Conj::No,
            )
    }
}

impl<V, T> KrigingSystem<V, T> for SimpleKrigingSystem
where
    T: SimdValue<Element = f32> + SimdRealField + Clone + Copy,
    T: SimdPartialOrd + SimdRealField,
    <T as SimdValue>::Element: SimdRealField + Float,
    V: VariogramModel<T>,
{
    fn new(n_elems: usize) -> Self {
        SimpleKrigingSystem::new(n_elems)
    }

    fn build_system<SKB>(
        &mut self,
        conditioning_points: &[SKB::Support],
        conditioning_values: &[f32],
        kriging_point: &Point3<f32>,
        variogram_model: &V,
    ) where
        SKB: SKBuilder,
    {
        SimpleKrigingSystem::build_system::<V, T, SKB>(
            self,
            conditioning_points,
            conditioning_values,
            kriging_point,
            variogram_model,
        );
    }

    fn estimate(&self) -> f32 {
        SimpleKrigingSystem::estimate(self)
    }

    fn variance(&self) -> f32 {
        SimpleKrigingSystem::variance(self)
    }
}

#[cfg(test)]
mod tests {

    use nalgebra::{UnitQuaternion, Vector3};

    use crate::variography::model_variograms::spherical::SphericalVariogram;

    use super::*;

    #[test]
    fn test_simple_kriging() {
        let cond_points = vec![
            Point3::new(2f32, 2f32, 0f32),
            Point3::new(3f32, 7f32, 0f32),
            Point3::new(9f32, 9f32, 0f32),
            Point3::new(6f32, 5f32, 0f32),
            Point3::new(5f32, 3f32, 0f32),
        ];
        let kriging_point = Point3::new(5f32, 5f32, 0f32);
        let values = vec![3f32, 4f32, 2f32, 4f32, 6f32];

        let rot = UnitQuaternion::identity();
        let vgram = SphericalVariogram::new(Vector3::new(10f32, 10f32, 10f32), 1f32, rot);
        let mut system = SimpleKrigingSystem::new(cond_points.len());
        system.build_system::<_, _, SKPointSupportBuilder>(
            cond_points.as_slice(),
            values.as_slice(),
            &kriging_point,
            &vgram,
        );

        let mean = system.estimate();
        let variance = system.variance();

        assert_eq!(mean, 4.148547);
        assert_eq!(variance, 0.49257421);
    }

    // #[test]
    // fn sk_test() {
    //     // Define the coordinate system for the grid
    //     // origing at x = 0, y = 0, z = 0
    //     // azimuth = 0, dip = 0, plunge = 0

    //     // create a gridded database from a csv file (walker lake)
    //     println!("Reading Cond Data");
    //     let gdb = PointSet::from_csv_index("C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/mineralized_domain_composites.csv", "X", "Y", "Z", "CU")
    //         .expect("Failed to create gdb");

    //     let vgram_rot = UnitQuaternion::identity();
    //     let cs = CoordinateSystem::new(Default::default(), Default::default());
    //     let range = Vector3::new(
    //         WideF32x8::splat(200.0),
    //         WideF32x8::splat(200.0),
    //         WideF32x8::splat(200.0),
    //     );
    //     let sill = WideF32x8::splat(1.0f32);

    //     let spherical_vgram = SphericalVariogram::new(range, sill, vgram_rot);

    //     // create search ellipsoid
    //     let search_ellipsoid = Ellipsoid::new(200f32, 200f32, 200f32, cs.clone());

    //     // create a query engine for the conditioning data
    //     //let query_engine = GriddedDataBaseOctantQueryEngine::new(search_ellipsoid, &gdb, 16);

    //     let mt = ZeroMeanTransform::from(gdb.data());
    //     let mean = mt.mean();

    //     // create a gsgs system
    //     let sk = SimpleKriging::new(
    //         gdb.clone(),
    //         spherical_vgram,
    //         search_ellipsoid,
    //         ConditioningParams::default(),
    //         mean,
    //     );

    //     //simulate values on grid
    //     // let points = krig_db
    //     //     .raw_grid
    //     //     .grid
    //     //     .indexed_iter()
    //     //     .map(|(ind, _)| krig_db.point_at_ind(&[ind.0, ind.1, ind.2]))
    //     //     .collect_vec();
    //     //"C:\Users\2jake\OneDrive - McGill University\Fall2022\MIME525\Project4\points_jacob.txt"
    //     println!("Reading Target Data");
    //     let targ = PointSet::<f32>::from_csv_index(
    //         "C:/Users/2jake/OneDrive - McGill University/Fall2022/MIME525/Project4/target.csv",
    //         "X",
    //         "Y",
    //         "Z",
    //         "V",
    //     )
    //     .unwrap();
    //     let points = targ.points.clone();
    //     // let points = (0..400)
    //     //     .cartesian_product(0..400)
    //     //     .map(|(x, y)| {
    //     //         let point = Point3::new(x as f32, y as f32, 0.0);
    //     //         point
    //     //     })
    //     //     .collect_vec();
    //     let time1 = std::time::Instant::now();
    //     let values = sk.krig::<SKPointSupportBuilder>(points.as_slice());
    //     let time2 = std::time::Instant::now();
    //     println!("Time: {:?}", (time2 - time1).as_secs());
    //     println!(
    //         "Points per minute: {}",
    //         values.len() as f32 / (time2 - time1).as_secs_f32() * 60.0
    //     );

    //     //save values to file for visualization

    //     let mut out = File::create("./test_results/sk.txt").unwrap();
    //     let _ = out.write_all(b"surfs\n");
    //     let _ = out.write_all(b"4\n");
    //     let _ = out.write_all(b"x\n");
    //     let _ = out.write_all(b"y\n");
    //     let _ = out.write_all(b"z\n");
    //     let _ = out.write_all(b"value\n");

    //     for (point, value) in points.iter().zip(values.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out
    //             .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
    //     }

    //     let mut out = File::create("./test_results/sk_cond_data.txt").unwrap();
    //     let _ = out.write_all(b"surfs\n");
    //     let _ = out.write_all(b"4\n");
    //     let _ = out.write_all(b"x\n");
    //     let _ = out.write_all(b"y\n");
    //     let _ = out.write_all(b"z\n");
    //     let _ = out.write_all(b"value\n");

    //     for (point, value) in gdb.points.iter().zip(gdb.data.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out
    //             .write_all(format!("{} {} {} {}\n", point.x, point.y, point.z, value).as_bytes());
    //     }

    //     let mut out = File::create("./test_results/sk.csv").unwrap();
    //     //write header
    //     let _ = out.write_all("X,Y,Z,XS,YS,ZS,V\n".as_bytes());

    //     //write each row

    //     for (point, value) in points.iter().zip(values.iter()) {
    //         //println!("point: {:?}, value: {}", point, value);
    //         let _ = out.write_all(
    //             format!(
    //                 "{},{},{},{},{},{},{}\n",
    //                 point.x, point.y, point.z, 5, 5, 10, value
    //             )
    //             .as_bytes(),
    //         );
    //     }
    // }
}
