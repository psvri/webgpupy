use std::{cmp::max, ops::Div, sync::Arc};

use arrow_gpu::{array::UInt32ArrayGPU, gpu_utils::*, kernels::take_op_dyn};
use wgpu::Buffer;

use crate::{NdArray, NdArrayError, NdArrayResult};

pub fn broadcast_shape(x: &[u32], y: &[u32]) -> NdArrayResult<Vec<u32>> {
    let max_shape_length = max(x.len(), y.len());
    let mut new_shape = vec![1; max_shape_length];

    for i in 0..max_shape_length {
        let v1 = if i < x.len() { x[x.len() - 1 - i] } else { 1 };
        let v2 = if i < y.len() { y[y.len() - 1 - i] } else { 1 };
        if v1 == v2 {
            new_shape[max_shape_length - 1 - i] = v1;
        } else if v1 == 1 {
            new_shape[max_shape_length - 1 - i] = v2;
        } else if v2 == 1 {
            new_shape[max_shape_length - 1 - i] = v1;
        } else {
            return Err(NdArrayError::BroadcastError(format!(
                "Cannot broadcast shapes {:?}, {:?}",
                x, y
            )));
        }
    }

    Ok(new_shape)
}

pub fn broadcast_shapes(shapes: &[&[u32]]) -> NdArrayResult<Vec<u32>> {
    if shapes.len() < 2 {
        Err(NdArrayError::BroadcastError(
            "cannot broadcast shapes array of length less than 2".to_string(),
        ))
    } else {
        let mut new_shape = broadcast_shape(shapes[0], shapes[1])?;

        for i in 2..shapes.len() {
            new_shape = broadcast_shape(&new_shape, shapes[i])?;
        }

        Ok(new_shape)
    }
}

// TODO probably can simpify this
fn shape_to_indexes_buf(
    to_shape: &[u32],
    from_shape: &[u32],
    pipeline: &mut ArrowComputePipeline,
) -> Buffer {
    const SHADER: &str = include_str!("../../compute_shaders/u32/broadcast.wgsl");

    let mut initial_buffer = pipeline.device.create_empty_buffer(4);

    let size_diff = to_shape.len() - from_shape.len();

    for (index, to) in to_shape.iter().enumerate() {
        let from = if index < size_diff {
            1
        } else {
            from_shape[index - size_diff]
        };

        let new_buffer_size = initial_buffer.size() * (*to as u64);
        let dispatch_size = initial_buffer.size().div(4).div_ceil(256) as u32;
        let slice_buffer = pipeline.device.create_gpu_buffer_with_data(&[from, *to]);

        initial_buffer = pipeline.apply_binary_function(
            &slice_buffer,
            &initial_buffer,
            new_buffer_size,
            SHADER,
            "get_item_indexes",
            dispatch_size,
        );
    }

    initial_buffer
}

/// Broadcast an array to a new shape
pub fn broadcast_to(x: &NdArray, shape: &[u32]) -> NdArray {
    let mut pipeline = ArrowComputePipeline::new(x.get_gpu_device(), None);
    let result = broadcast_to_op(x, shape, &mut pipeline);
    pipeline.finish();
    result
}

/// Broadcast an array to a new shape
pub fn broadcast_if_required(
    arr: &NdArray,
    broadcasted_shape: &[u32],
    pipeline: &mut ArrowComputePipeline,
) -> Option<NdArray> {
    if arr.shape != broadcasted_shape {
        Some(broadcast_to_op(arr, broadcasted_shape, pipeline))
    } else {
        None
    }
}

/// Broadcast an array to a new shape
pub fn broadcast_to_op(x: &NdArray, shape: &[u32], pipeline: &mut ArrowComputePipeline) -> NdArray {
    let braodcasted_shape = broadcast_shape(&x.shape, shape).unwrap();

    let buffer = shape_to_indexes_buf(&braodcasted_shape, &x.shape, pipeline);
    let len = (buffer.size() / 4) as usize;
    let indexes = UInt32ArrayGPU {
        data: Arc::new(buffer),
        gpu_device: x.get_gpu_device(),
        phantom: std::marker::PhantomData,
        len,
        null_buffer: None,
    };
    let dims = braodcasted_shape.len() as u16;

    let new_data = take_op_dyn(&x.data, &indexes, pipeline);
    let dtype = *x.get_dtype();

    NdArray {
        shape: braodcasted_shape,
        dims,
        data: new_data,
        dtype,
    }
}

#[cfg(test)]
mod test {
    use arrow_gpu::utils::ScalarArray;

    use super::{broadcast_shape, broadcast_to};
    use crate::{NdArray, ScalarArrayRef};

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(
            broadcast_shape(&vec![1, 1, 1], &vec![3]).unwrap(),
            vec![1, 1, 3]
        );
        assert_eq!(
            broadcast_shape(&vec![1, 2, 1], &vec![3]).unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            broadcast_shape(&vec![4, 1, 3], &vec![2, 3]).unwrap(),
            vec![4, 2, 3]
        );
        assert_eq!(
            broadcast_shape(&vec![1, 3], &vec![2, 1]).unwrap(),
            vec![2, 3]
        );
        assert!(broadcast_shape(&vec![2], &vec![3]).is_err())
    }

    fn test_broadcast(
        values: ScalarArrayRef,
        shape: Vec<u32>,
        new_shape: &[u32],
        results: ScalarArray,
    ) {
        let input = NdArray::from_slice(values, shape, None);
        let new_gpu_array = broadcast_to(&input, new_shape);
        assert_eq!(new_gpu_array.data.get_raw_values(), results);
    }

    #[test]
    fn test_broadcast_to() {
        test_broadcast(
            [1.0f32, 2.0, 3.0].as_ref().into(),
            vec![1, 1, 3],
            &vec![1, 2, 3],
            vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0].into(),
        );

        test_broadcast(
            [1.0f32].as_ref().into(),
            vec![1, 1, 1],
            &vec![1, 2, 3],
            vec![1.0f32; 6].into(),
        );

        test_broadcast(
            [1.0f32].as_ref().into(),
            vec![1],
            &vec![1, 2, 3],
            vec![1.0f32; 6].into(),
        );

        test_broadcast(
            [1.0f32, 2.0, 3.0].as_ref().into(),
            vec![3],
            &vec![1, 2, 3],
            vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0].into(),
        );

        test_broadcast(
            [1.0f32, 20.0, 33.0].as_ref().into(),
            vec![3],
            &vec![10, 50, 3],
            (0..10 * 50)
                .into_iter()
                .flat_map(|_| [1.0f32, 20.0, 33.0].into_iter())
                .collect::<Vec<f32>>()
                .into(),
        );

        test_broadcast(
            [1.0f32, 20.0].as_ref().into(),
            vec![1, 2, 1],
            &vec![1, 2, 3],
            vec![1.0f32, 1.0, 1.0, 20.0, 20.0, 20.0].into(),
        );

        test_broadcast(
            [1.0f32, 20.0, 30.0, 40.0].as_ref().into(),
            vec![2, 2, 1],
            &vec![2, 2, 3],
            vec![
                1.0f32, 1.0, 1.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 40.0, 40.0, 40.0,
            ]
            .into(),
        );
    }
}
