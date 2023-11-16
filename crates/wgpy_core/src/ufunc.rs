use std::{future::IntoFuture, pin::Pin};

use arrow_gpu::{
    array::{broadcast_dyn, ArrowArrayGPU},
    kernels::{cast_dyn, merge_dyn},
};
use futures::Future;

use crate::{broadcast_shape, broadcast_to, Dtype, NdArray, ScalarValue};

pub async fn ufunc_nin1_nout1<'a, F, T>(
    dyn_function: F,
    ndarray: &'a NdArray,
    where_: Option<&'a NdArray>,
    dtype: Option<Dtype>,
) -> NdArray
where
    F: FnOnce(&'a ArrowArrayGPU) -> T,
    T: IntoFuture<Output = ArrowArrayGPU>,
{
    let mut new_gpu_array = dyn_function(&ndarray.data).await;

    if let Some(mask) = where_ {
        if let ArrowArrayGPU::BooleanArrayGPU(mask) = &mask.data {
            let zero_array = broadcast_dyn(
                ScalarValue::zero(&ndarray.dtype).into(),
                ndarray.len() as usize,
                ndarray.data.get_gpu_device(),
            )
            .await;
            new_gpu_array = merge_dyn(&new_gpu_array, &zero_array, mask).await;
        }
    }

    if let Some(dtype) = dtype {
        new_gpu_array = cast_dyn(&new_gpu_array, (&dtype).into()).await;
    }

    let dtype = (&new_gpu_array.get_dtype()).into();

    NdArray {
        shape: ndarray.shape.clone(),
        dims: ndarray.dims,
        data: new_gpu_array,
        dtype,
    }
}

// We have to use Pix<Box> here else the code wont compile
pub async fn ufunc_nin2_nout1<'a, F>(
    dyn_function: F,
    ndarray1: &'a NdArray,
    ndarray2: &'a NdArray,
    where_: Option<&'a NdArray>,
    dtype: Option<Dtype>,
) -> NdArray
where
    F: for<'b> FnOnce(
        &'b ArrowArrayGPU,
        &'b ArrowArrayGPU,
    ) -> Pin<Box<dyn Future<Output = ArrowArrayGPU> + Send + 'b>>,
{
    let broadcasted_shape = broadcast_shape(&ndarray1.shape, &ndarray2.shape).unwrap();
    let in1 = broadcast_to(ndarray1, &broadcasted_shape).await;
    let in2 = broadcast_to(ndarray2, &broadcasted_shape).await;

    let mut new_gpu_array = dyn_function(&in1.data, &in2.data).await;

    if let Some(mask) = where_ {
        if let ArrowArrayGPU::BooleanArrayGPU(mask) = &mask.data {
            new_gpu_array = merge_dyn(&new_gpu_array, &in2.data, mask).await;
        }
    }

    if let Some(dtype) = dtype {
        new_gpu_array = cast_dyn(&new_gpu_array, (&dtype).into()).await;
    }

    let dtype = (&new_gpu_array.get_dtype()).into();
    let dims = broadcasted_shape.len() as u16;

    NdArray {
        shape: broadcasted_shape,
        dims,
        data: new_gpu_array,
        dtype,
    }
}
