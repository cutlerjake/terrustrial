![terrustrial logo](assets/logo.svg?raw=true)



Turrustrial is an experimental geostats library written entirely in rust with a focus on performance.

# Example

```rust
    
    // Read point clouds from file.
    let cond = SpatialAcceleratedDB::from_csv_index(FILE_PATH, "X", "Y", "Z", "CU")
            .expect("Failed to create gdb");

    // Some function create the blocks.
    let blocks = get_blocks();

    // Create a group provider
    let groups = GroupProvider::optimized_groups(&all_blocks, 5f64, 5f64, 10f64, 2, 2, 2);

    // Variogram rotation
    let vgram_rot = DRotor3::from_euler_angles(0.00.to_radians(), 0.0, 0.0);

    // Variogram range
    let range = DVec3::new(100.0, 200.0, 100.0);

    // Variogram sill
    let sill = 1.0;

    // Create a composite variogram
    let spherical_vgram = CompositeVariogram::new(vec![VariogramType::Spherical(
            SphericalVariogram::new(range, sill, vgram_rot),
    )]);

    // Create search ellipsoid
    let search_ellipsoid = Ellipsoid::new(
        200f64,
        50f64,
        50f64,
        CoordinateSystem::new(DVec3::zero(), vgram_rot),
    );

    // Create default conditioning params
    let params = ConditioningParams::default();

    // Estimate grades using ordinary kriging
    let values = estimate(
        &cond,
        &params,
        &spherical_vgram,
        search_ellipsoid,
        &groups,
        SolvedLUOKSystemBuilder,
    );
```