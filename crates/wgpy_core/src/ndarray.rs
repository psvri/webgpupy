use std::sync::Arc;

use arrow_gpu::{
    array::{
        broadcast_dyn, ArrowArrayGPU, BooleanArrayGPU, Float32ArrayGPU, GpuDevice, Int16ArrayGPU,
        Int32ArrayGPU, Int8ArrayGPU, UInt16ArrayGPU, UInt32ArrayGPU, UInt8ArrayGPU,
    },
    kernels::{cast_dyn, take_dyn},
};

use crate::{Dtype, Operand, ScalarArrayRef, ScalarValue, GPU_DEVICE};

#[derive(Debug)]
pub struct NdArray {
    pub shape: Vec<u32>,
    pub dims: u16,
    pub data: ArrowArrayGPU,
    pub dtype: Dtype,
}

impl NdArray {
    pub async fn new(shape: Vec<u32>, dtype: Dtype, gpu_device: Option<Arc<GpuDevice>>) -> Self {
        zeros(shape, Some(dtype), gpu_device).await
    }

    pub async fn from_slice(
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

    pub async fn astype(&self, dtype: Dtype) -> Self {
        let data = cast_dyn(&self.data, (&dtype).into()).await;

        Self {
            shape: self.shape.clone(),
            dims: self.dims,
            data,
            dtype,
        }
    }

    pub fn len(&self) -> u32 {
        if self.shape.len() == 0 {
            0
        } else {
            self.shape.iter().product()
        }
    }

    pub async fn clone_array(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            dims: self.dims,
            data: self.data.clone_array().await,
            dtype: self.dtype,
        }
    }

    pub async fn take(&self, indices: &NdArray, axis: Option<u32>) -> Self {
        if let Some(_) = axis {
            todo!()
        } else {
            if let ArrowArrayGPU::UInt32ArrayGPU(indices_array) = &indices.data {
                let array = take_dyn(&self.data, &indices_array).await;
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

    //TODO add flatten
}

pub async fn full(
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
            let data = broadcast_dyn(value.into(), len, gpu_device).await;

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
                result.astype(x).await
            }
        }
        None => result,
    }
}

pub async fn zeros(
    shape: Vec<u32>,
    dtype: Option<Dtype>,
    gpu_device: Option<Arc<GpuDevice>>,
) -> NdArray {
    let dtype = dtype.unwrap_or(Dtype::Float32);

    let gpu_device = gpu_device.unwrap_or(GPU_DEVICE.clone());
    let dims = shape.len() as u16;
    let len = (shape.iter().product::<u32>()) as usize;
    let data = broadcast_dyn(ScalarValue::zero(&dtype).into(), len, gpu_device).await;

    NdArray {
        shape,
        dims,
        data,
        dtype,
    }
}

pub async fn ones(
    shape: Vec<u32>,
    dtype: Option<Dtype>,
    gpu_device: Option<Arc<GpuDevice>>,
) -> NdArray {
    let dtype = dtype.unwrap_or(Dtype::Float32);

    let gpu_device = gpu_device.unwrap_or(GPU_DEVICE.clone());
    let dims = shape.len() as u16;
    let len = (shape.iter().product::<u32>()) as usize;
    let data = broadcast_dyn(ScalarValue::one(&dtype).into(), len, gpu_device).await;

    NdArray {
        shape,
        dims,
        data,
        dtype,
    }
}
