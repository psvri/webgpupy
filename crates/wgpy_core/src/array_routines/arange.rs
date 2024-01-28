use std::sync::Arc;

use arrow_gpu::array::{ArrowComputePipeline, GpuDevice, UInt32ArrayGPU};
use wgpu::Buffer;

use crate::{Dtype, IndexSlice, NdArray, GPU_DEVICE};

const ARANGE_SHADER: &str = include_str!("../../compute_shaders/u32/slice_to_index.wgsl");

// TODO handle cases like these np.arange(10, 0, -1)
pub fn arange(
    start: Option<u32>,
    stop: u32,
    step: Option<i32>,
    dtype: Option<Dtype>,
    gpu_device: Option<Arc<GpuDevice>>,
) -> NdArray {
    let gpu_device = gpu_device.unwrap_or(GPU_DEVICE.clone());
    let mut pipeline = ArrowComputePipeline::new(gpu_device.clone(), None);
    let start = start.unwrap_or(0);
    let step = step.unwrap_or(1);
    let index_slice = IndexSlice {
        start,
        stop,
        step,
    };
    let data = arange_op(&index_slice, &mut pipeline);

    //TODO handle dtype
    let dtype = match dtype {
        Some(Dtype::UInt32) => Dtype::UInt32,
        None => Dtype::UInt32,
        Some(_) => todo!(),
    };

    pipeline.finish();

    let data = UInt32ArrayGPU {
        data: Arc::new(data),
        gpu_device,
        phantom: std::marker::PhantomData,
        len: index_slice.element_count() as usize,
        null_buffer: None,
    };

    NdArray {
        shape: vec![index_slice.element_count()],
        dims: 1,
        data: data.into(),
        dtype,
    }
}

pub fn arange_op(slice: &IndexSlice, pipeline: &mut ArrowComputePipeline) -> Buffer {
    let input_buffer = pipeline.device.create_gpu_buffer_with_data(&[
        slice.start,
        slice.stop,
        slice.step as u32,
    ]);

    let new_buffer_size = slice.element_count() as u64 * 4;
    let dispatch_size = (slice.element_count()).div_ceil(256);

    pipeline.apply_unary_function(
        &input_buffer,
        new_buffer_size,
        ARANGE_SHADER,
        "arange",
        dispatch_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arange() {
        let input_1 = arange(None, 10, None, None, None);
        let result = (0..10).into_iter().collect::<Vec<u32>>();
        assert_eq!(input_1.data.get_raw_values(), result.into());

        let input_1 = arange(Some(2), 10, None, None, None);
        let result = (2..10).into_iter().collect::<Vec<u32>>();
        assert_eq!(input_1.data.get_raw_values(), result.into());

        let input_1 = arange(Some(3), 10, Some(2), None, None);
        let result = (3..10).into_iter().step_by(2).collect::<Vec<u32>>();
        assert_eq!(input_1.data.get_raw_values(), result.into());
    }
}
