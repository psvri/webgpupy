use arrow_gpu::kernels::*;
use webgpupy_core::{
    ufunc_nin1_nout1, ufunc_nin1_nout1_body, Dtype, NdArray,
};

ufunc_nin1_nout1_body!(sqrt, sqrt_dyn);
