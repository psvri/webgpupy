struct IndexSlice {
    start: u32,
    stop: u32,
    step: u32,
}

@group(0) @binding(0)
var<storage, read_write> slice : IndexSlice;

@group(0) @binding(1)
var<storage, read_write> indexes : array<u32>;

@compute
@workgroup_size(256)
fn arange(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < arrayLength(&indexes) {
        indexes[global_id.x] = slice.start + (slice.step * global_id.x);
    }
}