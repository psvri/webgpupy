use arrow_gpu::kernels::*;
use webgpupy_core::{
    ufunc_nin1_nout1, ufunc_nin1_nout1_body, ufunc_nin2_nout1, ufunc_nin2_nout1_body, Dtype,
    NdArray,
};

ufunc_nin2_nout1_body!(bitwise_and, bitwise_and_op_dyn);
ufunc_nin2_nout1_body!(bitwise_or, bitwise_or_op_dyn);
ufunc_nin1_nout1_body!(invert, bitwise_not_op_dyn);

#[cfg(test)]
mod test {
    use super::*;
    use arrow_gpu::array::ArrowArrayGPU;
    use test_utils::*;
    use webgpupy_core::GPU_DEVICE;

    test_ufunc_nin2_nout1!(
        test_u32_bitwise_and_u32,
        [1u32, 2],
        [1u32, 1],
        [1u32, 0],
        UInt32ArrayGPU,
        bitwise_and
    );

    test_ufunc_nin2_nout1!(
        test_u32_bitwise_or_u32,
        [1u32, 2],
        [1u32, 1],
        [1u32, 3],
        UInt32ArrayGPU,
        bitwise_or
    );

    test_ufunc_nin1_nout1!(
        test_u32_bitwise_not_u32,
        [1u32, 2],
        [!1u32, !2],
        UInt32ArrayGPU,
        invert
    );
}
