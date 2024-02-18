use arrow_gpu::kernels::*;
use webgpupy_core::{ufunc_nin2_nout1, Dtype, NdArray};

#[macro_export]
macro_rules! ufunc_compare_nin2_nout1_body {
    ($name: ident, $dyn: ident) => {
        pub fn $name(
            input1: &NdArray,
            input2: &NdArray,
            where_: Option<&NdArray>,
            dtype: Option<Dtype>,
        ) -> NdArray {
            ufunc_nin2_nout1(
                |x, y, z| $dyn(x, y, z).into(),
                input1,
                input2,
                where_,
                dtype,
            )
        }
    };
}

ufunc_compare_nin2_nout1_body!(greater, gt_op_dyn);
ufunc_compare_nin2_nout1_body!(greater_equal, gteq_op_dyn);
ufunc_compare_nin2_nout1_body!(lesser, lt_op_dyn);
ufunc_compare_nin2_nout1_body!(lesser_equal, lteq_op_dyn);
ufunc_compare_nin2_nout1_body!(equal, eq_op_dyn);

#[cfg(test)]
mod test {
    use super::*;
    use arrow_gpu::array::ArrowArrayGPU;
    use test_utils::*;
    use webgpupy_core::GPU_DEVICE;

    test_ufunc_nin2_nout1!(
        test_f32_lt_f32_mask_true,
        [1.0, 2.0],
        [10.0, 20.0],
        [false, false],
        BooleanArrayGPU,
        greater
    );
}
