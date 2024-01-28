struct IndexSlice {
    start: u32,
    stop: u32,
    step: u32,
    length: u32,
    count: u32,
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
        let start_base = slice.length * input_indexes[global_id.x];
        var start_index = global_id.x * slice.count;
        for (var i = slice.start; i < slice.stop; i += slice.step) {
            output_indexes[start_index] = start_base + i;
            start_index += 1u;
        }
    }
}