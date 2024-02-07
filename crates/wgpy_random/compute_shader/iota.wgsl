
@group(0)
@binding(0)
var<storage, read_write> output: array<u32>;


@compute
@workgroup_size(256)
fn iota(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < arrayLength(&output) {
        output[global_id.x] = global_id.x;
    }
}
    