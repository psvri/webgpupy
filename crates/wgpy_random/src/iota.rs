use arrow_gpu::gpu_utils::*;
use webgpupy_core::{arange_op, IndexSlice};
use wgpu::Buffer;

pub fn iota_op(shape: &[u32], pipeline: &mut ArrowComputePipeline) -> Buffer {
    let total_count: u32 = shape.iter().product();

    arange_op(
        &IndexSlice::new(0, total_count.into(), 1, total_count).unwrap(),
        pipeline,
    )
}

#[cfg(test)]
mod test {
    use super::iota_op;
    use arrow_gpu::gpu_utils::*;
    use webgpupy_core::*;

    #[test]
    fn test_iota() {
        let shape = [10, 20];
        let total = shape.iter().product();
        let mut pipeline = ArrowComputePipeline::new(GPU_DEVICE.clone(), None);
        let buffer = iota_op(&shape, &mut pipeline);
        pipeline.finish();
        let values = GPU_DEVICE.as_ref().retrive_data(&buffer);

        let result: Vec<u32> = bytemuck::cast_slice(&values).to_vec();
        assert_eq!(result, (0..total).collect::<Vec<u32>>())
    }
}
