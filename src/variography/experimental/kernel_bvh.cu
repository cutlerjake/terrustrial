
struct AABB
{
    float3 min;
    float3 max;
};

struct Quaternion
{
    float w, x, y, z;
};

__device__ Quaternion quaternion_multiply(Quaternion q1, Quaternion q2)
{
    Quaternion result;
    result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return result;
}

__device__ Quaternion quaternion_conjugate(Quaternion q)
{
    Quaternion result;
    result.w = q.w;
    result.x = -q.x;
    result.y = -q.y;
    result.z = -q.z;
    return result;
}

__device__ float3 rotate_point_by_quaternion(float3 p, Quaternion q)
{
    // Convert point to quaternion
    Quaternion point_quat = {0, p.x, p.y, p.z};

    // Rotate the point
    Quaternion temp = quaternion_multiply(q, point_quat);
    Quaternion rotated_quat = quaternion_multiply(temp, quaternion_conjugate(q));

    // Convert back to point
    float3 rotated_point = {rotated_quat.x, rotated_quat.y, rotated_quat.z};
    return rotated_point;
}

struct VgramParameters
{
    // data
    unsigned int n_data; // number of data points

    // bvh
    unsigned int n_bvh; // number of nodes in bvh

    // lags
    unsigned int n_lags; // number of lags

    // rotations
    unsigned int n_rotations; // number of rotations

    // ellipsoid dimensions
    float a;                // major axis length
    float a_tol;            // major axis tolerance (azimuth tolerance)
    float a_dist_threshold; // major axis distance threshold (ellispoid distance threshold)
    float b;                // minor axis length
    float b_tol;            // minor axis tolerance (dip tolerance)
    float b_dist_threshold; // minor axis distance threshold (ellipsoid distance threshold)
};

struct FlatBVHNode
{
    AABB aabb;
    unsigned int entry_index;
    unsigned int exit_index;
    unsigned int shape_index;
};

// returns true if aabb a intersects aabb b
__device__ bool aabb_intersect(AABB a, AABB b)
{
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
           (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
           (a.min.z <= b.max.z && a.max.z >= b.min.z);
}

// returns true if aabb a contains point (x, y, z)
__device__ bool aabb_contains_point(AABB a, float3 point)
{
    return (point.x >= a.min.x && point.x <= a.max.x) &&
           (point.y >= a.min.y && point.y <= a.max.y) &&
           (point.z >= a.min.z && point.z <= a.max.z);
}

// computes the loose aabb of an elliptical cylinder
__device__ AABB elliptical_cylinder_loose_aabb(float3 p1, float3 p2, float major, float minor)
{
    // minimum and maximum x, y, and z values
    float min_x = min(p1.x, p2.x);
    float min_y = min(p1.y, p2.y);
    float min_z = min(p1.z, p2.z);

    float max_x = max(p1.x, p2.x);
    float max_y = max(p1.y, p2.y);
    float max_z = max(p1.z, p2.z);

    // amount to offset the aabb by
    float offset = max(major, minor);

    // create aabb
    AABB aabb;
    aabb.min.x = min_x - offset;
    aabb.min.y = min_y - offset;
    aabb.min.z = min_z - offset;

    aabb.max.x = max_x + offset;
    aabb.max.y = max_y + offset;
    aabb.max.z = max_z + offset;

    return aabb;
}

__device__ bool elliptical_cylinder_contains_point(float3 pt1, float3 pt2, float major, float minor, float major_threshold, float minor_threshold, float major_tol, float minor_tol, float3 testpt, Quaternion rotation)
{
    float dx, dy, dz;    // vector d  from line segment point 1 to point 2
    float pdx, pdy, pdz; // vector pd from point 1 to test point
    float dot;
    float lengthsq;

    dx = pt2.x - pt1.x; // translate so pt1 is origin.  Make vector from
    dy = pt2.y - pt1.y; // pt1 to pt2.  Need for this is easily eliminated
    dz = pt2.z - pt1.z;

    pdx = testpt.x - pt1.x; // vector from pt1 to test point.
    pdy = testpt.y - pt1.y;
    pdz = testpt.z - pt1.z;

    // Dot the d and pd vectors to see if point lies behind the
    // cylinder cap at pt1.x, pt1.y, pt1.z

    dot = pdx * dx + pdy * dy + pdz * dz;

    // sq length of the line segment

    lengthsq = dx * dx + dy * dy + dz * dz;

    // If dot is less than zero the point is behind the pt1 cap.
    // If greater than the cylinder axis line segment length squared
    // then the point is outside the other end cap at pt2.

    if (dot < 0.0f || dot > lengthsq)
    {
        return false;
    }

    // allowable range
    float axial_dist = sqrt(dot);
    float a_range = (axial_dist > major_threshold) ? major : major + axial_dist * tan(major_tol);
    float b_range = (axial_dist > minor_threshold) ? minor : minor + axial_dist * tan(minor_tol);

    // Point lies within the parallel caps, so find
    // position of test point relative to the cylinder

    // vector from pt1 to test point
    float3 test_v = make_float3(pdx, pdy, pdz);

    // rotate test point by quaternion
    test_v = rotate_point_by_quaternion(test_v, rotation);

    // y now aligned with major and z aligned with minor

    if ((test_v.y * test_v.y) / (a_range * a_range) + (test_v.z * test_v.z) / (b_range * b_range) > 1.0f)
    {
        return false;
    }

    return true;
};

struct LagBounds
{
    float lower_bound;
    float upper_bound;
};

extern "C" __global__ void vgram_kernel(float3 *points, FlatBVHNode *flat_bvh, float *v, LagBounds *lag_bounds, Quaternion *rotations, Quaternion *inv_rotations, VgramParameters params, float *global_vgram, unsigned int *global_lag_counts)
{

    // flat index of current point and rotation
    unsigned int flat_ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (flat_ind >= params.n_data * params.n_rotations)
    {
        return;
    }

    // index of current point
    unsigned int i = flat_ind % params.n_data;
    // index of rotation
    unsigned int rotation_i = flat_ind / params.n_data;

    // maximum of 250 lags supported
    float lag_var[250] = {0.0};
    unsigned int lag_count[250] = {0};

    // iterate over lags
    for (unsigned int lag_i = 0; lag_i < params.n_lags; lag_i++)
    {
        // get lag bounds
        float lag_lower_bound = lag_bounds[lag_i].lower_bound;
        float lag_upper_bound = lag_bounds[lag_i].upper_bound;

        float3 lower_bound = make_float3(lag_lower_bound, 0.0, 0.0);
        float3 upper_bound = make_float3(lag_upper_bound, 0.0, 0.0);

        //   rotate lag bounds
        Quaternion rotation = rotations[rotation_i];
        lower_bound = rotate_point_by_quaternion(lower_bound, rotation);
        upper_bound = rotate_point_by_quaternion(upper_bound, rotation);

        // shift lag bounds to current point
        lower_bound.x += points[i].x;
        lower_bound.y += points[i].y;
        lower_bound.z += points[i].z;

        upper_bound.x += points[i].x;
        upper_bound.y += points[i].y;
        upper_bound.z += points[i].z;

        // create (loose) bounding box
        AABB bounding_box = elliptical_cylinder_loose_aabb(lower_bound, upper_bound, params.a, params.b);

        unsigned int index = 0;
        // Iterate while the node index is valid.
        while (index < params.n_bvh)
        {

            FlatBVHNode node = flat_bvh[index];

            if (node
                    .entry_index == 4294967295)
            {
                // If the entry_index is MAX_UINT32, then it's a leaf node.
                float3 shape = points[node.shape_index];
                if (aabb_contains_point(bounding_box, shape))

                {
                    Quaternion inv_rotation = inv_rotations[rotation_i];
                    if (elliptical_cylinder_contains_point(lower_bound, upper_bound, params.a, params.b, params.a_dist_threshold, params.b_dist_threshold, params.a_tol, params.b_tol, shape, inv_rotation))
                    {
                        // printf("points: (%f, %f, %f) (%f, %f, %f)\n", points[i].x, points[i].y, points[i].z, shape.x, shape.y, shape.z);
                        //    increment the lag count
                        lag_count[lag_i] += 1;
                        // add var
                        lag_var[lag_i] += ((v[i] - v[node.shape_index]) * (v[i] - v[node.shape_index]));
                        // printf("lag %d: %f\n", lag_i, (v[i] - v[node.shape_index]));
                    }
                }

                // Exit the current node.
                index = node.exit_index;
            }
            else if (aabb_intersect(bounding_box, node.aabb))
            {
                // If entry_index is not MAX_UINT32 and the AABB test passes, then
                // proceed to the node in entry_index (which goes down the bvh branch).
                index = node.entry_index;
            }
            else
            {
                // If entry_index is not MAX_UINT32 and the AABB test fails, then
                // proceed to the node in exit_index (which defines the next untested partition).
                index = node.exit_index;
            }
        }
    }

    // update global values
    for (unsigned int k = 0; k < params.n_lags; k++)
    {
        unsigned int gvi = k + (rotation_i * 250u);
        atomicAdd(&global_lag_counts[gvi], lag_count[k]);
        atomicAdd(&global_vgram[gvi], lag_var[k]);
        // atomicAdd(&global_lag_counts[k], lag_count[k]);
        // atomicAdd(&global_vgram[k], lag_var[k]);
    }
}