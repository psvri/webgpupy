use arrow_gpu::{
    array::{broadcast_op_dyn, ArrowArrayGPU},
    gpu_utils::*,
    kernels::{cast_op_dyn, merge_op_dyn},
};

use crate::{broadcast_shape, broadcast_to_op, Dtype, NdArray, ScalarValue};

pub fn ufunc_nin1_nout1<'a, F>(
    dyn_function: F,
    ndarray: &'a NdArray,
    where_: Option<&'a NdArray>,
    dtype: Option<Dtype>,
) -> NdArray
where
    F: FnOnce(&'a ArrowArrayGPU, &mut ArrowComputePipeline) -> ArrowArrayGPU,
{
    let mut pipeline = ArrowComputePipeline::new(ndarray.data.get_gpu_device(), None);
    let mut new_gpu_array = dyn_function(&ndarray.data, &mut pipeline);

    if let Some(mask) = where_ {
        if let ArrowArrayGPU::BooleanArrayGPU(mask) = &mask.data {
            let zero_array = broadcast_op_dyn(
                ScalarValue::zero(&ndarray.dtype).into(),
                ndarray.len() as usize,
                &mut pipeline,
            );
            new_gpu_array = merge_op_dyn(&new_gpu_array, &zero_array, mask, &mut pipeline);
        }
    }

    if let Some(dtype) = dtype {
        new_gpu_array = cast_op_dyn(&new_gpu_array, (&dtype).into(), &mut pipeline);
    }

    let dtype = (&new_gpu_array.get_dtype()).into();
    pipeline.finish();

    NdArray {
        shape: ndarray.shape.clone(),
        dims: ndarray.dims,
        data: new_gpu_array,
        dtype,
    }
}

// We have to use Pix<Box> here else the code wont compile
pub fn ufunc_nin2_nout1<'a, F>(
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
        &mut ArrowComputePipeline,
    ) -> ArrowArrayGPU,
{
    let mut pipeline = ArrowComputePipeline::new(ndarray1.data.get_gpu_device(), None);
    let broadcasted_shape = broadcast_shape(&ndarray1.shape, &ndarray2.shape).unwrap();
    let mut in1 = ndarray1;
    let temp1;
    if ndarray1.shape != broadcasted_shape {
        temp1 = broadcast_to_op(ndarray1, &broadcasted_shape, &mut pipeline);
        in1 = &temp1;
    }
    let mut in2 = ndarray2;
    let temp2;
    if ndarray2.shape != broadcasted_shape {
        temp2 = broadcast_to_op(ndarray2, &broadcasted_shape, &mut pipeline);
        in2 = &temp2;
    }

    let mut new_gpu_array = dyn_function(&in1.data, &in2.data, &mut pipeline);

    if let Some(mask) = where_ {
        if let ArrowArrayGPU::BooleanArrayGPU(mask) = &mask.data {
            let zero_array = broadcast_op_dyn(
                ScalarValue::zero(&new_gpu_array.get_dtype().into()).into(),
                new_gpu_array.len(),
                &mut pipeline,
            );
            new_gpu_array = merge_op_dyn(&new_gpu_array, &zero_array, mask, &mut pipeline);
        }
    }

    if let Some(dtype) = dtype {
        new_gpu_array = cast_op_dyn(&new_gpu_array, (&dtype).into(), &mut pipeline);
    }

    let dtype = (&new_gpu_array.get_dtype()).into();
    let dims = broadcasted_shape.len() as u16;

    pipeline.finish();

    NdArray {
        shape: broadcasted_shape,
        dims,
        data: new_gpu_array,
        dtype,
    }
}

#[macro_export]
macro_rules! ufunc_nin2_nout1_body {
    ($name: ident, $dyn: ident) => {
        pub fn $name(
            input1: &NdArray,
            input2: &NdArray,
            where_: Option<&NdArray>,
            dtype: Option<Dtype>,
        ) -> NdArray {
            ufunc_nin2_nout1($dyn, input1, input2, where_, dtype)
        }
    };
}

#[macro_export]
macro_rules! ufunc_nin1_nout1_body {
    ($name: ident, $dyn: ident) => {
        pub fn $name(ndarray: &NdArray, where_: Option<&NdArray>, dtype: Option<Dtype>) -> NdArray {
            ufunc_nin1_nout1($dyn, ndarray, where_, dtype)
        }
    };
}
