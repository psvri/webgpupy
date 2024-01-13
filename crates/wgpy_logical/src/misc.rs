use arrow_gpu::{array::ArrowArrayGPU, kernels::LogicalContains};
use webgpupy_core::NdArray;

//TODO make it like numpy api
pub async fn any(x: &NdArray) -> bool {
    if let ArrowArrayGPU::BooleanArrayGPU(y) = &x.data {
        y.any().await
    } else {
        panic!("Cant perform any on dtype {:?}", x.dtype)
    }
}
