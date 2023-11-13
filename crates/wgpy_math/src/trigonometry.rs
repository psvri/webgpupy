use arrow_gpu::kernels::*;
use webgpupy_core::{ufunc_nin1_nout1, Dtype, NdArray, OperandType, UfuncType};

pub async fn sin(ndarray: &NdArray, where_: Option<&NdArray>, dtype: Option<Dtype>) -> NdArray {
    ufunc_nin1_nout1(sin_dyn, ndarray, where_, dtype).await
}

pub async fn cos(ndarray: &NdArray, where_: Option<&NdArray>, dtype: Option<Dtype>) -> NdArray {
    ufunc_nin1_nout1(cos_dyn, ndarray, where_, dtype).await
}

pub const COS_TYPES: [UfuncType; 5] = [
    UfuncType::UfuncNin1Nout1Type(
        [OperandType::NdArrayType(Dtype::Float32)],
        OperandType::NdArrayType(Dtype::Float32),
    ),
    UfuncType::UfuncNin1Nout1Type(
        [OperandType::NdArrayType(Dtype::UInt8)],
        OperandType::NdArrayType(Dtype::Float32),
    ),
    UfuncType::UfuncNin1Nout1Type(
        [OperandType::NdArrayType(Dtype::UInt16)],
        OperandType::NdArrayType(Dtype::Float32),
    ),
    UfuncType::UfuncNin1Nout1Type(
        [OperandType::NdArrayType(Dtype::Int8)],
        OperandType::NdArrayType(Dtype::Float32),
    ),
    UfuncType::UfuncNin1Nout1Type(
        [OperandType::NdArrayType(Dtype::Int16)],
        OperandType::NdArrayType(Dtype::Float32),
    ),
];

pub const SIN_TYPES: [UfuncType; 5] = COS_TYPES;

#[cfg(test)]
mod test {
    use super::*;
    use arrow_gpu::array::ArrowArrayGPU;
    use test_utils::*;
    use webgpupy_core::GPU_DEVICE;

    test_ufunc_nin1_nout1_f32!(
        test_sin_f32_mask_mixed,
        Dtype::Float32,
        Dtype::Float32,
        [1.0, 1.0],
        [1.0f32.sin(), 0.0],
        [true, false],
        sin
    );

    test_ufunc_nin1_nout1_f32!(
        test_sin_f32_mask_true,
        Dtype::Float32,
        Dtype::Float32,
        [1.0, 2.0],
        [1.0f32.sin(), 2.0f32.sin()],
        [true, true],
        sin
    );
}
