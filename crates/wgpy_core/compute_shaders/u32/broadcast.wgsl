struct IndexSlice {
    from_shape: u32,
    to_shape: u32,
}

@group(0) @binding(0)
var<storage, read_write> slice : IndexSlice;

@group(0) @binding(1)
var<storage, read_write> input_indexes : array<u32>;

@group(0) @binding(2)
var<storage, read_write> output_indexes : array<u32>;

@compute
@workgroup_size(256)
fn get_item_indexes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < arrayLength(&input_indexes) {
        let start_base = slice.from_shape * input_indexes[global_id.x];
        var start_index = global_id.x * slice.to_shape;
        if slice.from_shape == 1u {
            for (var i = 0u; i < slice.to_shape; i += 1u) {
                output_indexes[start_index + i] = start_base;
            }
        } else {
            for (var i = 0u; i < slice.to_shape; i += 1u) {
                output_indexes[start_index + i] = start_base + i;
            }
        }
    }
}