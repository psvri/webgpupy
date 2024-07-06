pub mod arithmetic;
pub mod binary;
pub(crate) mod cast;
pub mod gpu_device;
pub mod logical;
pub mod misc;
pub mod misc_math;
pub mod ndarraypy;
pub mod random;
pub mod trigonometry;
pub mod types;
pub mod ufunc;

use ndarraypy::NdArrayPy;
use pyo3::{exceptions::PyTypeError, prelude::*, types::*};
use types::OperandPy;
use webgpupy::{NdArray, ScalarValue};

pub(crate) fn convert_pyobj_into_operand<'a>(
    data: &'a Bound<'a, PyAny>,
) -> PyResult<OperandPy<'a>> {
    if data.is_instance_of::<NdArrayPy>() {
        let ndarray = data.downcast::<NdArrayPy>()?;
        PyResult::Ok((&ndarray.get().ndarray).into())
    } else if data.is_instance_of::<PyFloat>() {
        let value = data.extract::<f32>()?;
        let ndarray = NdArray::from_slice([value].as_slice().into(), vec![1], None);
        PyResult::Ok(ndarray.into())
    } else if data.is_instance_of::<PyInt>() {
        let value = data.extract::<i32>()?;
        let ndarray = NdArray::from_slice([value].as_slice().into(), vec![1], None);
        PyResult::Ok(ndarray.into())
    } else {
        //TODO implment support for pylist and pytuple
        PyResult::Err(PyTypeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub(crate) fn convert_pyobj_into_option_operand<'a>(
    data: &'a Bound<'a, PyAny>,
) -> PyResult<Option<OperandPy<'a>>> {
    if data.is_none() {
        Ok(None)
    } else {
        Ok(Some(convert_pyobj_into_operand(data)?))
    }
}

pub(crate) fn convert_pyobj_into_scalar(data: &Bound<PyAny>) -> PyResult<ScalarValue> {
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

pub(crate) fn convert_pyobj_into_array_u32(data: &Bound<PyAny>) -> PyResult<Vec<u32>> {
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

pub(crate) fn convert_pyobj_into_vec_ndarray<'a>(
    data: &'a Bound<'a, PyAny>,
) -> PyResult<Vec<Bound<NdArrayPy>>> {
    if data.is_instance_of::<PyList>() || data.is_instance_of::<PyTuple>() {
        let len = data.len()?;
        Ok(Python::with_gil(|_py| {
            let mut refs = vec![];
            for i in 0..len {
                let item = data.get_item(i)?;
                if item.is_instance_of::<NdArrayPy>() {
                    refs.push(item.downcast_into::<NdArrayPy>()?)
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

#[pymodule]
#[pyo3(name = "webgpupy")]
fn webgpupy_module(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();
    ufunc::create_py_items(m)?;
    logical::create_py_items(m)?;
    binary::create_py_items(m)?;
    trigonometry::create_py_items(m)?;
    misc::create_py_items(m)?;
    misc_math::create_py_items(m)?;
    arithmetic::create_py_items(m)?;
    ndarraypy::create_py_items(m)?;
    random::random_module(py, m)?;
    Ok(())
}
