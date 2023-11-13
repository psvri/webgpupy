use arrow_gpu::kernels::mul_dyn;
use webgpupy_core::{ufunc_nin2_nout1, Dtype, NdArray};

pub async fn multiply(
    input1: &NdArray,
    input2: &NdArray,
    where_: Option<&NdArray>,
    dtype: Option<Dtype>,
) -> NdArray {
    ufunc_nin2_nout1(
        |x, y| Box::pin(mul_dyn(x, y)),
        input1,
        input2,
        where_,
        dtype,
    )
    .await
}
