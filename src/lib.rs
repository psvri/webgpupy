use std::{cmp::max, sync::Arc};

use arrow_gpu::{
    array::{f32_gpu::Float32ArrayGPU, u32_gpu::UInt32ArrayGPU, ArrowArrayGPU, GpuDevice},
    kernels::arithmetic::*,
    kernels::trigonometry::*,
};
use pollster::FutureExt as _;
use pyo3::prelude::*;

#[pyclass]
struct Ndarray {
    shape: Vec<u32>,
    dims: usize,
    data: Arc<dyn ArrowArrayGPU>,
}

#[pymethods]
impl Ndarray {
    #[new]
    fn new(shape: Vec<u32>) -> Self {
        let dims = shape.len();
        let size = max((shape.iter().product::<u32>()) as usize, 1);
        let gpu_device = Arc::new(GpuDevice::new().block_on());
        Ndarray {
            shape,
            data: Arc::new(Float32ArrayGPU::braodcast(1.0, size, gpu_device).block_on()),
            dims,
        }
    }

    fn display(&self) {
        println!("{:?}", self.data.as_ref());
    }

    fn sin(&mut self) -> Self {
        let data = sin_dyn(self.data.as_ref()).block_on();
        Self {
            shape: self.shape.clone(),
            dims: self.dims.clone(),
            data,
        }
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn webgpupy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Ndarray>()?;
    Ok(())
}
