// Each invocation owns one candidate score. No floating-point atomics or parallel
// reductions: event accumulation follows the supplied order for reproducibility.
struct Parameters {
    dimensions: vec3<u32>, variables: u32,
    cells: u32, queries: u32, row_width: u32, padding: u32,
}
struct Event {
    offset: vec3<i32>, variable: u32,
    value: u32, weight: f32, padding: vec2<u32>,
}
struct Query {
    event_start: u32, event_count: u32, request_start: u32, request_count: u32,
    bounded: u32, impossible: u32, integer: u32, padding: u32,
}
struct Score { value: u32, valid: u32, }
@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var<storage, read> training: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> kinds: array<u32>;
@group(0) @binding(3) var<storage, read> events: array<Event>;
@group(0) @binding(4) var<storage, read> requests: array<u32>;
@group(0) @binding(5) var<storage, read> queries: array<Query>;
@group(0) @binding(6) var<storage, read_write> scores: array<Score>;

@compute @workgroup_size(64)
fn score_patterns(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let cell = invocation.x + invocation.y * parameters.row_width;
    let query_index = invocation.z;
    if cell >= parameters.cells || query_index >= parameters.queries { return; }
    let output = query_index * parameters.cells + cell;
    scores[output] = Score(0u, 0u);
    let query = queries[query_index];
    if query.impossible != 0u { return; }
    for (var r = 0u; r < query.request_count; r += 1u) {
        let variable = requests[query.request_start + r];
        if training[cell * parameters.variables + variable].y == 0u { return; }
    }
    let position = vec3<i32>(i32(cell % parameters.dimensions.x),
        i32((cell / parameters.dimensions.x) % parameters.dimensions.y),
        i32(cell / (parameters.dimensions.x * parameters.dimensions.y)));
    let dimensions = vec3<i32>(parameters.dimensions);
    var score = 0.0;
    var mismatches = 0u;
    for (var e = 0u; e < query.event_count; e += 1u) {
        let event = events[query.event_start + e];
        var neighbor = position + event.offset;
        if query.bounded != 0u {
            if any(neighbor < vec3<i32>(0)) || any(neighbor >= dimensions) { return; }
        } else {
            // The host reduces periodic offsets to [0, dimension). Therefore
            // every coordinate is in [0, 2*dimension), needing one subtraction.
            neighbor = select(neighbor, neighbor - dimensions, neighbor >= dimensions);
        }
        let index = u32(neighbor.x) + parameters.dimensions.x * (u32(neighbor.y) + parameters.dimensions.y * u32(neighbor.z));
        let value = training[index * parameters.variables + event.variable];
        if value.y == 0u { return; }
        if query.integer != 0u {
            mismatches += select(1u, 0u, value.x == event.value);
        } else {
            var mismatch = 0.0;
            if kinds[event.variable] == 1u {
                mismatch = select(1.0, 0.0, value.x == event.value);
            } else {
                let difference = bitcast<f32>(value.x) - bitcast<f32>(event.value);
                mismatch = difference * difference;
            }
            score += event.weight * mismatch;
        }
    }
    scores[output] = Score(select(bitcast<u32>(score), mismatches, query.integer != 0u), 1u);
}
