use std::{ops::Div, sync::Arc};

use arrow_gpu::{
    array::{
        broadcast_dyn, ArrowArrayGPU, BooleanArrayGPU, Float32ArrayGPU, Int16ArrayGPU,
        Int32ArrayGPU, Int8ArrayGPU, UInt16ArrayGPU, UInt32ArrayGPU, UInt8ArrayGPU,
    },
    gpu_utils::*,
    kernels::{cast_dyn, neg_dyn, take_dyn, take_op_dyn},
};

use crate::{Dtype, IndexSlice, IndexSliceOp, Operand, ScalarArrayRef, ScalarValue, GPU_DEVICE};

#[derive(Debug)]
pub struct NdArray {
    pub shape: Vec<u32>,
    pub dims: u16,
    pub data: ArrowArrayGPU,
    pub dtype: Dtype,
}

impl NdArray {
    pub fn new(shape: Vec<u32>, dtype: Dtype, gpu_device: Option<Arc<GpuDevice>>) -> Self {
        zeros(shape, Some(dtype), gpu_device)
    }

    pub fn get_gpu_device(&self) -> Arc<GpuDevice> {
        self.data.get_gpu_device()
    }

    pub fn get_dtype(&self) -> &Dtype {
        &self.dtype
    }

    pub fn from_slice(
        values_array: ScalarArrayRef<'_>,
        shape: Vec<u32>,
        gpu_device: Option<Arc<GpuDevice>>,
    ) -> Self {
        let gpu_device = gpu_device.unwrap_or(GPU_DEVICE.clone());
        let dims = shape.len() as u16;
        let (dtype, data) = match values_array {
            ScalarArrayRef::F32ARRAY(x) => (
                Dtype::Float32,
                Float32ArrayGPU::from_slice(x, gpu_device).into(),
            ),
            ScalarArrayRef::U32ARRAY(x) => (
                Dtype::UInt32,
                UInt32ArrayGPU::from_slice(x, gpu_device).into(),
            ),
            ScalarArrayRef::U16ARRAY(x) => (
                Dtype::UInt16,
                UInt16ArrayGPU::from_slice(x, gpu_device).into(),
            ),
            ScalarArrayRef::U8ARRAY(x) => (
                Dtype::UInt8,
                UInt8ArrayGPU::from_slice(x, gpu_device).into(),
            ),
            ScalarArrayRef::I32ARRAY(x) => (
                Dtype::Int32,
                Int32ArrayGPU::from_slice(x, gpu_device).into(),
            ),
            ScalarArrayRef::I16ARRAY(x) => (
                Dtype::Int16,
                Int16ArrayGPU::from_slice(x, gpu_device).into(),
            ),
            ScalarArrayRef::I8ARRAY(x) => {
                (Dtype::Int8, Int8ArrayGPU::from_slice(x, gpu_device).into())
            }
            ScalarArrayRef::BOOLARRAY(x) => (
                Dtype::Bool,
                BooleanArrayGPU::from_slice(x, gpu_device).into(),
            ),
        };

        NdArray {
            shape,
            dims,
            data,
            dtype,
        }
    }

    pub fn astype(&self, dtype: Dtype) -> Self {
        let data = cast_dyn(&self.data, (&dtype).into());

        Self {
            shape: self.shape.clone(),
            dims: self.dims,
            data,
            dtype,
        }
    }

    pub fn len(&self) -> u32 {
        if self.shape.is_empty() {
            0
        } else {
            self.shape.iter().product()
        }
    }

    pub fn clone_array(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            dims: self.dims,
            data: self.data.clone_array(),
            dtype: self.dtype,
        }
    }

    pub fn take(&self, indices: &NdArray, axis: Option<u32>) -> Self {
        if axis.is_some() {
            todo!()
        } else {
            if let ArrowArrayGPU::UInt32ArrayGPU(indices_array) = &indices.data {
                let array = take_dyn(&self.data, indices_array);
                Self {
                    shape: vec![array.len() as u32],
                    dims: 1u16,
                    data: array,
                    dtype: self.dtype,
                }
            } else {
                unreachable!()
            }
        }
    }

    pub fn neg(&self) -> Self {
        let data = neg_dyn(&self.data);
        let dtype = data.get_dtype().into();
        Self {
            shape: self.shape.clone(),
            dims: self.dims,
            data,
            dtype,
        }
    }

    pub fn get_items(&self, slices: &[IndexSliceOp]) -> Self {
        if slices.is_empty() {
            panic!("cant get items from empty slice")
        }

        let mut pipeline = ArrowComputePipeline::new(self.data.get_gpu_device(), None);
        const SHADER: &str = include_str!("../compute_shaders/u32/get_item_index.wgsl");
        let mut new_shape = Vec::with_capacity(self.shape.len());

        let shape_iter = self.shape.iter();
        let mut index_slice_iter = slices.iter();

        let mut initial_buffer = pipeline.device.create_empty_buffer(4);
        let mut temp_slice;

        for shape in shape_iter {
            let slice = if let Some(slice) = index_slice_iter.next() {
                temp_slice = slice.into_index_slice(*shape);
                match slice {
                    IndexSliceOp::Index(_) => {}
                    _ => new_shape.push(temp_slice.element_count()),
                }
                temp_slice
            } else {
                new_shape.push(*shape);
                temp_slice = IndexSlice::new(0, *shape as i64, 1, *shape).unwrap();
                temp_slice
            };

            let count = slice.element_count() as u64;
            let new_buffer_size = initial_buffer.size() * count;
            let dispatch_size = initial_buffer.size().div(4).div_ceil(256) as u32;
            let slice_buffer = pipeline.device.create_gpu_buffer_with_data(&[
                slice.start,
                slice.stop,
                slice.step as u32,
                *shape,
                count as u32,
            ]);

            initial_buffer = pipeline.apply_binary_function(
                &slice_buffer,
                &initial_buffer,
                new_buffer_size,
                SHADER,
                "get_item_indexes",
                dispatch_size,
            );
        }

        let len = (initial_buffer.size() / 4) as usize;
        let indexes = UInt32ArrayGPU {
            data: Arc::new(initial_buffer),
            gpu_device: self.data.get_gpu_device(),
            phantom: std::marker::PhantomData,
            len,
            null_buffer: None,
        };
        let data = take_op_dyn(&self.data, &indexes, &mut pipeline);

        pipeline.finish();

        Self {
            shape: new_shape,
            dims: self.dims,
            data,
            dtype: self.dtype,
        }
    }

    //TODO add flatten
}

pub fn full(
    shape: Vec<u32>,
    value: Operand<'_>,
    dtype: Option<Dtype>,
    gpu_device: Option<Arc<GpuDevice>>,
) -> NdArray {
    let result = match value {
        Operand::Scalar(value) => {
            let gpu_device = gpu_device.unwrap_or(GPU_DEVICE.clone());
            let dims = shape.len() as u16;
            let len = (shape.iter().product::<u32>()) as usize;
            let dtype = (&value).into();
            let data = broadcast_dyn(value.into(), len, gpu_device);

            NdArray {
                shape,
                dims,
                data,
                dtype,
            }
        }
        Operand::ScalarArrayRef(_) => todo!(),
        Operand::NdArrayRef(_) => todo!(),
    };

    match dtype {
        Some(x) => {
            if result.dtype == x {
                result
            } else {
                result.astype(x)
            }
        }
        None => result,
    }
}

pub fn zeros(shape: Vec<u32>, dtype: Option<Dtype>, gpu_device: Option<Arc<GpuDevice>>) -> NdArray {
    let dtype = dtype.unwrap_or(Dtype::Float32);

    let gpu_device = gpu_device.unwrap_or(GPU_DEVICE.clone());
    let dims = shape.len() as u16;
    let len = (shape.iter().product::<u32>()) as usize;
    let data = broadcast_dyn(ScalarValue::zero(&dtype).into(), len, gpu_device);

    NdArray {
        shape,
        dims,
        data,
        dtype,
    }
}

pub fn ones(shape: Vec<u32>, dtype: Option<Dtype>, gpu_device: Option<Arc<GpuDevice>>) -> NdArray {
    let dtype = dtype.unwrap_or(Dtype::Float32);

    let gpu_device = gpu_device.unwrap_or(GPU_DEVICE.clone());
    let dims = shape.len() as u16;
    let len = (shape.iter().product::<u32>()) as usize;
    let data = broadcast_dyn(ScalarValue::one(&dtype).into(), len, gpu_device);

    NdArray {
        shape,
        dims,
        data,
        dtype,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get_items() {
        let values = (0..300).collect::<Vec<i32>>();
        let array = NdArray::from_slice(values.as_slice().into(), vec![10, 10, 3], None);
        let slices = vec![(1..3).into()];

        let result = (30..90).collect::<Vec<i32>>();

        let items = array.get_items(&slices);

        assert_eq!(&items.shape, &[2, 10, 3]);

        assert_eq!(items.data.get_raw_values(), result.into());
    }

    #[test]
    fn test_get_items_mixed() {
        let values = (0..300).map(|x| x as f32).collect::<Vec<f32>>();
        let array = NdArray::from_slice(values.as_slice().into(), vec![10, 10, 3], None);
        let slices = vec![(5..7).into(), (1..2).into()];

        let items = array.get_items(&slices);

        assert_eq!(&items.shape, &[2, 1, 3]);

        assert_eq!(
            items.data.get_raw_values(),
            vec![153.0f32, 154.0, 155.0, 183.0, 184.0, 185.0].into()
        );
    }

    #[test]
    fn test_get_items_all() {
        let values = (0..300).map(|x| x as f32).collect::<Vec<f32>>();
        let array = NdArray::from_slice(values.as_slice().into(), vec![10, 10, 3], None);

        let items = array.get_items(&[(5..7).into(), (1..2).into(), (2..).into()]);

        assert_eq!(&items.shape, &[2, 1, 1]);

        assert_eq!(items.data.get_raw_values(), vec![155.0f32, 185.0].into());
    }

    #[test]
    fn test_get_items_index() {
        let values = (0..300).map(|x| x as f32).collect::<Vec<f32>>();
        let array = NdArray::from_slice(values.as_slice().into(), vec![10, 10, 3], None);

        let items = array.get_items(&[5.into(), 1.into()]);

        assert_eq!(&items.shape, &[3]);

        assert_eq!(
            items.data.get_raw_values(),
            vec![153.0f32, 154.0, 155.0].into()
        );
    }

    #[test]
    fn test_get_items_last() {
        let values = (0..1500).map(|x| x as f32).collect::<Vec<f32>>();
        let array = NdArray::from_slice(values.as_slice().into(), vec![10, 50, 3], None);

        for i in 0..3 {
            let items = array.get_items(&[(0..).into(), (0..).into(), i.into()]);

            let result = (0..500)
                .into_iter()
                .map(|x| ((x * 3) + i) as f32)
                .collect::<Vec<f32>>();

            assert_eq!(&items.shape, &[10, 50]);

            assert_eq!(items.data.get_raw_values(), result.into());
        }
    }
}
