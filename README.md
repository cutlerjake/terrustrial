# terrustrial
Turrustrial is a geostats library written entirely in rust.

Terrustrial is under active development 
- breaking changes are expected
- features missing

# Usage

```rust
    //create a random array
    let mut rng = rand::thread_rng();

    // Create grid of conditional data
    let mut arr = Array3::from_shape_fn((100, 100, 100), |(i, j, k)| Some(rng.gen::<f32>()));

    // Set 90% of the data to None
    let mut kriging_inds = Vec::new();
    arr.indexed_iter_mut()
        .filter(|(_, _)| rng.gen::<f32>() > 0.1)
        .for_each(|(ind, v)| {
            *v = None;
            kriging_inds.push([ind.0, ind.1, ind.2]);
        });

    // Create a coordinate system
    let origin = Translation3::new(10.0, 100.0, 115.0);
    let rotation = nalgebra::UnitQuaternion::from_euler_angles(
        30.0.to_radians(),
        30.0.to_radians(),
        30.0.to_radians(),
    );
    let coordinate_system = CoordinateSystem::new(origin, rotation);

    // Create a gridded database
    let gdb = InCompleteGriddedDataBase::new(
        arr,
        GridSpacing {
            x: 10.0,
            y: 10.0,
            z: 10.0,
        },
        coordinate_system,
    );

    let kriging_points = kriging_inds
        .iter()
        .map(|ind| gdb.ind_to_point(&ind.map(|i| i as isize)))
        .collect::<Vec<_>>();

    // Create model variogram
    let vgram_rot =
        UnitQuaternion::from_euler_angles(30.0.to_radians(), 30.0.to_radians(), 30.0.to_radians());
    let vgram_origin = Point3::new(0.0, 0.0, 0.0);
    let vgram_coordinate_system = CoordinateSystem::new(vgram_origin.into(), vgram_rot);
    let range = Vector3::new(300.0, 300.0, 300.0);
    let sill = 1.0;
    let nugget = 0.0;

    let spherical_vgram =
        SphericalVariogram::new(range, sill, nugget, vgram_coordinate_system.clone());

    // Create search ellipsoid
    let search_ellipsoid = Ellipsoid::new(300.0, 300.0, 300.0, vgram_coordinate_system.clone());

    // Create Kriging Parameters
    let kriging_params = KrigingParameters {
        max_cond_data: 100,
        min_cond_data: 0,
        max_octant_data: 9,
        min_octant_data: 0,
    };

    // Create simple kriging
    let simple_kriging = SimpleKriging::new(gdb, spherical_vgram, search_ellipsoid, kriging_params);

    // Compute SK estimate at kriging points
    let values = simple_kriging.krige(kriging_points.as_slice());
```

# Implemented Features
- Experimental variogram computation
- Spherical variogram
- simple kriging

 # Planned Features
 ## Variography
 - Visualization
 - Vectorize experimental variogram
 - Pairwise relative experimental varigram
 - More theoretical varigorams (Exponential, Gaussian, Matern...)
   
 ## Estimation
 - Ordinary Kriging
   
 ## Simulation
 - Gaussian simulation methods (SGS, GSGS, DBSIM)
 - Multi-point simulation methods (SNESIM, FILTERSIM)
 - High-Order simulation methods (HOSIM)

