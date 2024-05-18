use std::sync::Arc;

use arrow_gpu::{
    array::{ArrayUtils, ArrowArrayGPU, ArrowPrimitiveType, PrimitiveArrayGpu},
    gpu_utils::ArrowComputePipeline,
    kernels::broadcast,
};
use webgpupy_core::{broadcast_shapes, Dtype, NdArray};

trait CrossType: ArrowPrimitiveType {
    const CROSS_SHADER: &'static str;
}

impl CrossType for f32 {
    const CROSS_SHADER: &'static str = include_str!("../compute_shader/f32/cross.wgsl");
}

trait Cross {
    fn cross(&self, other: &Self, shape: &[u32]) -> Self;
}

impl<T: CrossType> Cross for PrimitiveArrayGpu<T> {
    fn cross(&self, other: &Self, shape: &[u32]) -> Self {
        let mut pipeline = ArrowComputePipeline::new(self.get_gpu_device(), Some("cross"));

        let shape_buffer = self.gpu_device.create_gpu_buffer_with_data(shape);

        let dispatch_size = (self.data.size() / T::ITEM_SIZE) as u32 / shape[0];
        let new_buffer_size = match (shape[0], shape[1]) {
            (3, 3) | (3, 2) => self.data.size(),
            (2, 3) => other.data.size(),
            (2, 2) => self.data.size() / 2,
            _ => panic!("invalid shape for cross"),
        };

        let result_buffer = pipeline.apply_ternary_function(
            &self.data,
            &other.data,
            &shape_buffer,
            new_buffer_size,
            T::CROSS_SHADER,
            "cross_",
            dispatch_size,
        );

        pipeline.finish();

        Self {
            data: Arc::new(result_buffer),
            gpu_device: self.get_gpu_device(),
            len: (new_buffer_size / T::ITEM_SIZE) as usize,
            null_buffer: None,
            phantom: std::marker::PhantomData,
        }
    }
}

pub fn cross(a: &NdArray, b: &NdArray) -> NdArray {
    let shape_last_a = a.shape.last().unwrap();
    let shape_last_b = b.shape.last().unwrap();

    let mut shape = a.shape.clone();
    let mut dims = shape.len() as u16;

    match (shape_last_a, shape_last_b) {
        (3, 3) | (3, 2) => {}
        (2, 3) => {
            shape = b.shape.clone();
        }
        (2, 2) => {
            shape.pop();
            dims = shape.len() as u16;
        }
        _ => panic!(
            "cross not supported for shapes {:?} and {:?}",
            a.shape, b.shape
        ),
    };

    //TODO add broadcast support
    let data = match (&a.data, &b.data) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => {
            x.cross(y, &[*shape_last_a, *shape_last_b])
        }
        _ => panic!(
            "cross not supported for dtype {:?} and {:?}",
            a.dtype, b.dtype
        ),
    }
    .into();

    NdArray {
        shape,
        dims,
        data,
        dtype: a.dtype,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use test_utils::float_slice_eq_in_error;

    fn test_cross_f32_results(
        input1: &[f32],
        input1_shape: Vec<u32>,
        input2: &[f32],
        input2_shape: Vec<u32>,
        output: Vec<f32>,
    ) {
        let array_1 = NdArray::from_slice(input1.into(), input1_shape, None);
        let array_2 = NdArray::from_slice(input2.into(), input2_shape, None);

        if let ArrowArrayGPU::Float32ArrayGPU(x) = cross(&array_1, &array_2).data {
            let new_values = x.raw_values().unwrap();
            float_slice_eq_in_error(output.into(), new_values.into());
        }
    }

    #[test]
    fn test_cross_f32() {
        test_cross_f32_results(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            &[10.0f32, 20.0, 30.0, 11.0, 22.0, 33.0],
            vec![2, 3],
            vec![0.0f32, 0.0, 0.0, 33.0, -66.0, 33.0],
        );

        test_cross_f32_results(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            &[7.0, 8.0, 11.0, 12.0],
            vec![2, 2],
            vec![-24., 21., -6., -72., 66., -7.],
        );

        test_cross_f32_results(
            &[7.0, 8.0, 11.0, 12.0],
            vec![2, 2],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![-24., 21., -6., -72., 66., -7.],
        );

        test_cross_f32_results(
            &[7.0, 8.0, 11.0, 12.0],
            vec![2, 2],
            &[1.0f32, 2.0, 4.0, 5.0],
            vec![2, 2],
            vec![-6., -7.],
        );
    }
}
