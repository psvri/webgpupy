use pyo3::prelude::*;
use webgpupy::{Generator, ThreeFry2x32};

use crate::ndarraypy::NdArrayPy;

#[pyclass(name = "threefry")]
#[derive(Debug)]
pub struct ThreeFryPy {
    pub threefry: ThreeFry2x32,
}

#[pymethods]
impl ThreeFryPy {
    #[new]
    fn new(key: u32) -> Self {
        let threefry = ThreeFry2x32::new(key, None);
        Self { threefry }
    }

    fn random(&mut self, shape: Vec<u32>) -> NdArrayPy {
        self.threefry.random(&shape).into()
    }

    fn normal(&mut self, shape: Vec<u32>) -> NdArrayPy {
        self.threefry.normal(&shape).into()
    }
}

#[pyfunction(name = "default_rng")]
pub fn default_rng() -> PyResult<ThreeFryPy> {
    // TODO add array size validation
    Ok(ThreeFryPy::new(0))
}

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_class::<ThreeFryPy>()?;
    Ok(())
}

pub fn random_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "random")?;
    child_module.add_class::<ThreeFryPy>()?;
    child_module.add_function(wrap_pyfunction!(default_rng, child_module)?)?;
    parent_module.add_submodule(child_module)?;
    Ok(())
}
