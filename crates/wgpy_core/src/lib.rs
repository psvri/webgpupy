use arrow_gpu::gpu_utils::GpuDevice;
use once_cell::sync::Lazy;
use std::sync::Arc;

pub(crate) mod array_routines;
pub(crate) mod errors;
pub(crate) mod ndarray;
pub(crate) mod operand;
pub(crate) mod types;
pub(crate) mod ufunc;
pub(crate) mod utils;

pub use array_routines::*;
pub use arrow_gpu::utils::ScalarArray;
pub use errors::*;
pub use ndarray::*;
pub use operand::*;
pub use types::*;
pub use ufunc::*;

pub static GPU_DEVICE: Lazy<Arc<GpuDevice>> = Lazy::new(|| Arc::new(GpuDevice::new()));
