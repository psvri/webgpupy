@group(0) @binding(0)
var<storage, read> input_1 : array<f32>;

@group(0) @binding(1)
var<storage, read> input_2 : array<f32>;

@group(0) @binding(2)
var<storage, read> shape : vec2<u32>;

@group(0) @binding(3)
var<storage, read_write> output : array<f32>;

@compute
@workgroup_size(256)
fn cross_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if shape.x == 3u && shape.y == 3u {
        if global_id.x < (arrayLength(&input_1) / 3u) {
            let base_index = global_id.x * 3u;
            let output_cross = cross(
                vec3(input_1[base_index + 0u], input_1[base_index + 1u], input_1[base_index + 2u]),
                vec3(input_2[base_index + 0u], input_2[base_index + 1u], input_2[base_index + 2u])
            );
            output[base_index + 0u] = output_cross.x;
            output[base_index + 1u] = output_cross.y;
            output[base_index + 2u] = output_cross.z;
        }
    } else if shape.x == 3 && shape.y == 2 {
        if global_id.x < (arrayLength(&input_1) / 3u) {
            let base_index_x = global_id.x * 3u;
            let base_index_y = global_id.x * 2u;
            let output_cross = cross(
                vec3(input_1[base_index_x + 0u], input_1[base_index_x + 1u], input_1[base_index_x + 2u]),
                vec3(input_2[base_index_y + 0u], input_2[base_index_y + 1u], 0)
            );
            output[base_index_x + 0u] = output_cross.x;
            output[base_index_x + 1u] = output_cross.y;
            output[base_index_x + 2u] = output_cross.z;
        }
    } else if shape.x == 2 && shape.y == 3 {
        if global_id.x < (arrayLength(&input_1) / 2u) {
            let base_index_x = global_id.x * 2u;
            let base_index_y = global_id.x * 3u;
            let output_cross = cross(
                vec3(input_1[base_index_x + 0u], input_1[base_index_x + 1u], 0),
                vec3(input_2[base_index_y + 0u], input_2[base_index_y + 1u], input_2[base_index_y + 2u])
            );
            output[base_index_y + 0u] = output_cross.x;
            output[base_index_y + 1u] = output_cross.y;
            output[base_index_y + 2u] = output_cross.z;
        }
    } else {
        if global_id.x < (arrayLength(&input_1) / 2u) {
            let base_index = global_id.x * 2u;
            output[global_id.x] = input_1[base_index + 0u] * input_2[base_index + 1u] - input_1[base_index + 1u] * input_2[base_index + 0u];
        }
    }
}