use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{Point3, SimdValue, Translation3, UnitQuaternion, Vector3};
use simba::simd::WideF32x8;
use terrustrial::{
    estimators::{
        generalized_sequential_kriging::{GSKSystemParameters, GSK},
        simple_kriging::SKPointSupportBuilder,
        ConditioningParams,
    },
    geometry::ellipsoid::Ellipsoid,
    node_providers::point_group::PointGroupProvider,
    spatial_database::{coordinate_system::CoordinateSystem, rtree_point_set::point_set::PointSet},
    systems::solved_systems::{ok_system::SolvedLUOKSystemBuilder, SolvedSystemBuilder},
    variography::model_variograms::{
        composite::{CompositeVariogram, VariogramType},
        spherical::SphericalVariogram,
    },
};

fn create_point_set(domain: [[f32; 3]; 2], n_points: usize) -> PointSet<f32> {
    let mut points = vec![];
    let mut values = vec![];
    for _ in 0..n_points {
        let x = rand::random::<f32>() * (domain[1][0] - domain[0][0]) + domain[0][0];
        let y = rand::random::<f32>() * (domain[1][1] - domain[0][1]) + domain[0][1];
        let z = rand::random::<f32>() * (domain[1][2] - domain[0][2]) + domain[0][2];
        let v = rand::random::<f32>();
        points.push([x, y, z].into());
        values.push(v);
    }
    let tags = vec![0; n_points];
    PointSet::new(points, values, tags)
}

fn create_grid(
    domain: [[f32; 3]; 2],
    n_groups: [u32; 3],
    group_dim: [u32; 3],
) -> Vec<Vec<Point3<f32>>> {
    let g_x = (domain[1][0] - domain[0][0]) / n_groups[0] as f32;
    let g_y = (domain[1][1] - domain[0][1]) / n_groups[1] as f32;
    let g_z = (domain[1][2] - domain[0][2]) / n_groups[2] as f32;

    let s_x = g_x / group_dim[0] as f32;
    let s_y = g_y / group_dim[1] as f32;
    let s_z = g_z / group_dim[2] as f32;

    let create_group = |i: u32, j: u32, k: u32| {
        let mut group = vec![];
        for x in 0..group_dim[0] {
            for y in 0..group_dim[1] {
                for z in 0..group_dim[2] {
                    let x = domain[0][0] + i as f32 * g_x + x as f32 * s_x;
                    let y = domain[0][1] + j as f32 * g_y + y as f32 * s_y;
                    let z = domain[0][2] + k as f32 * g_z + z as f32 * s_z;
                    group.push(Point3::new(x, y, z));
                }
            }
        }
        group
    };

    let mut grid = vec![];
    for i in 0..n_groups[0] {
        for j in 0..n_groups[1] {
            for k in 0..n_groups[2] {
                let group = create_group(i, j, k);
                grid.push(group);
            }
        }
    }

    grid
}

fn create_vgram(
    unit_quaternion: UnitQuaternion<f32>,
    range: Vector3<f32>,
    sill: f32,
) -> CompositeVariogram<WideF32x8> {
    let unit_quaternion = UnitQuaternion::splat(unit_quaternion);
    let range = Vector3::splat(range);
    let sill = WideF32x8::splat(sill);
    CompositeVariogram::new(vec![VariogramType::Spherical(SphericalVariogram::new(
        range,
        sill,
        unit_quaternion,
    ))])
}

fn create_ellipsoid(x: f32, y: f32, z: f32) -> Ellipsoid {
    Ellipsoid::new(
        x,
        y,
        z,
        CoordinateSystem::new(
            Translation3::new(0.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0f32.to_radians()),
        ),
    )
}

fn create_gsk(max_group_size: usize) -> GSK {
    GSK::new(GSKSystemParameters { max_group_size })
}

fn create_node_provider(groups: Vec<Vec<Point3<f32>>>) -> PointGroupProvider {
    let orientations = vec![UnitQuaternion::identity(); groups.len()];
    PointGroupProvider::from_groups(groups, orientations)
}

fn ordinary_kriging(
    gsk: &GSK,
    cond_points: &PointSet<f32>,
    params: &ConditioningParams,
    vgram: CompositeVariogram<WideF32x8>,
    search_ellipsoid: Ellipsoid,
    node_provider: &PointGroupProvider,
    builder: impl SolvedSystemBuilder,
) -> Vec<f32> {
    gsk.estimate::<SKPointSupportBuilder, _, _, _>(
        cond_points,
        &params,
        vgram,
        search_ellipsoid,
        node_provider,
        builder,
    )
}

fn criterion_benchmark(c: &mut Criterion) {
    let domain = [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]];
    let n_points = 1_000_000;
    let n_groups = [50, 50, 50];
    let group_dim = [2, 2, 2];
    let cond_points = create_point_set(domain, n_points);
    let grid = create_grid(domain, n_groups, group_dim);

    let vgram = create_vgram(
        UnitQuaternion::identity(),
        Vector3::new(10.0, 10.0, 10.0),
        1.0,
    );
    let search_ellipsoid = create_ellipsoid(10.0, 10.0, 10.0);

    let params = ConditioningParams::default();

    let node_provider = create_node_provider(grid);

    let builder = SolvedLUOKSystemBuilder;
    let gsk = create_gsk(group_dim.iter().map(|v| *v as usize).product());
    c.bench_function("fib 20", |b| {
        b.iter(|| {
            ordinary_kriging(
                black_box(&gsk),
                black_box(&cond_points),
                black_box(&params),
                black_box(vgram.clone()),
                black_box(search_ellipsoid.clone()),
                black_box(&node_provider),
                black_box(builder.clone()),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
