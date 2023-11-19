pub mod arithmetic;
pub(crate) mod cast;
pub mod misc_math;
pub mod ndarraypy;
pub mod trigonometry;
pub mod types;
pub mod ufunc;

use ndarraypy::NdArrayPy;
use pollster::FutureExt;
use pyo3::{exceptions::PyTypeError, prelude::*, types::*};
use types::OperandPy;
use webgpupy::{NdArray, ScalarValue};

pub(crate) fn convert_pyobj_into_operand(data: &PyAny) -> PyResult<OperandPy> {
    if data.is_instance_of::<NdArrayPy>() {
        let ndarray: &PyCell<NdArrayPy> = data.downcast()?;
        PyResult::Ok((&ndarray.get().ndarray).into())
    } else if data.is_instance_of::<PyFloat>() {
        let value = data.extract::<f32>()?;
        let ndarray = NdArray::from_slice([value].as_slice().into(), vec![1], None).block_on();
        PyResult::Ok(ndarray.into())
    } else if data.is_instance_of::<PyInt>() {
        let value = data.extract::<i32>()?;
        let ndarray = NdArray::from_slice([value].as_slice().into(), vec![1], None).block_on();
        PyResult::Ok(ndarray.into())
    } else {
        PyResult::Err(PyTypeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub(crate) fn convert_pyobj_into_scalar(data: &PyAny) -> PyResult<ScalarValue> {
    if data.is_instance_of::<PyFloat>() {
        PyResult::Ok(data.extract::<f32>()?.into())
    } else if data.is_instance_of::<PyInt>() {
        PyResult::Ok(data.extract::<i32>()?.into())
    } else {
        PyResult::Err(PyTypeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub(crate) fn convert_pyobj_into_array_u32(data: &PyAny) -> PyResult<Vec<u32>> {
    if data.is_instance_of::<PyInt>() {
        PyResult::Ok(vec![data.extract::<u32>()?])
    } else if data.is_instance_of::<PyList>() {
        PyResult::Ok(data.extract::<Vec<u32>>()?)
    } else {
        PyResult::Err(PyTypeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub(crate) fn convert_pyobj_into_vec_ndarray(data: &PyAny) -> PyResult<Vec<&NdArray>> {
    if data.is_instance_of::<PyList>() || data.is_instance_of::<PyTuple>() {
        let len = data.len()?;

        Ok(Python::with_gil(|_py| {
            let mut refs = vec![];
            for i in 0..len {
                let item = data.get_item(i)?;
                if item.is_instance_of::<NdArrayPy>() {
                    refs.push(&item.downcast::<PyCell<NdArrayPy>>()?.get().ndarray)
                } else {
                    return PyResult::Err(PyTypeError::new_err(format!(
                        "Operation not supported for the given type {:?}",
                        item.get_type()
                    )));
                }
            }
            Ok(refs)
        })?)
    } else {
        PyResult::Err(PyTypeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "webgpupy")]
fn webgpupy_module(_py: Python, m: &PyModule) -> PyResult<()> {
    ufunc::create_py_items(m)?;
    trigonometry::create_py_items(m)?;
    misc_math::create_py_items(m)?;
    arithmetic::create_py_items(m)?;
    ndarraypy::create_py_items(m)?;
    Ok(())
}
