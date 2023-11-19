use arrow_gpu::kernels::*;
use webgpupy_core::{ufunc_nin2_nout1, ufunc_nin2_nout1_body, Dtype, NdArray};

ufunc_nin2_nout1_body!(multiply, mul_dyn);
ufunc_nin2_nout1_body!(divide, div_dyn);
ufunc_nin2_nout1_body!(add, add_dyn);
ufunc_nin2_nout1_body!(subtract, sub_dyn);
