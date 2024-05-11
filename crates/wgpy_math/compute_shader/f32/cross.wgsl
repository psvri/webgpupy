@group(0) @binding(0)
var<storage, read> input_1 : array<u32>;

@group(0) @binding(1)
var<storage, read> input_2 : array<u32>;

@group(0) @binding(2)
var<storage, read_write> input_2 : array<u32>;

@compute
@workgroup_size(256)
fn cross_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x * 3 < arrayLength(input_1) {
        
    }
}