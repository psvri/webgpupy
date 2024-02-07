use arrow_gpu::gpu_utils::*;
use wgpu::Buffer;

const SHADER: &str = include_str!("../compute_shader/threefry.wgsl");

pub fn threefry2x32_op(
    keys: &Buffer,
    data: &Buffer,
    pipeline: &mut ArrowComputePipeline,
) -> Buffer {
    let entry_point = "threefry";
    let dispatch_size = (data.size() / 4 * 2).div_ceil(256) as u32;

    pipeline.apply_binary_function(keys, data, data.size(), SHADER, entry_point, dispatch_size)
}

#[cfg(test)]
mod test {
    use webgpupy_core::GPU_DEVICE;

    use super::*;

    // Test cases obtained from
    // https://github.com/google/jax/blob/bf0841136f01a8ae7003a9f6543c17db8cd4aef5/tests/random_test.py#L200
    fn test_threefry_(key: &[u32; 2], data: &[u32], expected: &[u32]) {
        let mut pipeline = ArrowComputePipeline::new(GPU_DEVICE.clone(), None);

        let keys = GPU_DEVICE.create_gpu_buffer_with_data(key);
        let data = GPU_DEVICE.create_gpu_buffer_with_data(data);

        let buffer = threefry2x32_op(&keys, &data, &mut pipeline);

        pipeline.finish();
        let values = GPU_DEVICE.as_ref().retrive_data(&buffer);

        let result: Vec<u32> = bytemuck::cast_slice(&values).to_vec();
        assert_eq!(result, expected)
    }

    #[test]
    fn test_threefry() {
        test_threefry_(&[0u32, 0], &[0u32, 0], &[0x6b200159u32, 0x99ba4efe]);
        test_threefry_(
            &[u32::MAX, u32::MAX],
            &[u32::MAX, u32::MAX],
            &[0x1cb996fc, 0xbb002be7],
        );
        test_threefry_(
            &[0x13198a2eu32, 0x03707344],
            &[0x243f6a88u32, 0x85a308d3],
            &[0xc4923a9cu32, 0x483df7a0],
        );
    }

    #[test]
    fn test_threefry_large() {
        //TODO fix for large values like 10**6
        const N: usize = 1000000;
        let keys = [0x13198a2e, 0x03707344];
        let mut data = vec![0; 2 * N];
        data[0..N].copy_from_slice(&[0x243f6a88; N]);
        data[N..].copy_from_slice(&[0x85a308d3; N]);

        let mut expected = vec![0; 2 * N];
        expected[0..N].copy_from_slice(&[0xc4923a9c; N]);
        expected[N..].copy_from_slice(&[0x483df7a0; N]);
        test_threefry_(&keys, &data, &expected);
    }
}
