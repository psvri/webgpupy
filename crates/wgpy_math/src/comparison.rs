use arrow_gpu::kernels::*;
use webgpupy_core::{ufunc_nin2_nout1, ufunc_nin2_nout1_body, Dtype, NdArray};

ufunc_nin2_nout1_body!(maximum, max_dyn);
ufunc_nin2_nout1_body!(minimum, min_dyn);
