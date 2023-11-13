use std::borrow::Cow;

use pollster::FutureExt as _;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::*};
use webgpupy::{full, *};

use crate::{
    arithmetic::_multiply,
    cast::PyObectToRustPrimitive,
    convert_pyobj_into_array_u32, convert_pyobj_into_operand, convert_pyobj_into_scalar,
    convert_pyobj_into_vec_ndarray,
    types::{into_dtypepy, into_optional_dtypepy, DtypePy},
};

#[pyclass(frozen)]
#[derive(Debug)]
pub struct NdArrayPy {
    pub ndarray: NdArray,
}

impl From<NdArray> for NdArrayPy {
    fn from(value: NdArray) -> Self {
        NdArrayPy { ndarray: value }
    }
}

impl<'a> From<&'a NdArrayPy> for &'a NdArray {
    fn from(value: &'a NdArrayPy) -> &'a NdArray {
        &value.ndarray
    }
}

#[pymethods]
impl NdArrayPy {
    pub fn display(&self) -> String {
        format!("Shape {:?} \n {:?}", self.ndarray.shape, self.ndarray.data)
    }

    pub fn tolist(&self, py: Python<'_>) -> Py<PyAny> {
        let values = self.ndarray.data.get_raw_values().block_on();

        match values {
            ScalarArray::U32Vec(x) => x.into_py(py),
            ScalarArray::F32Vec(x) => x.into_py(py),
            ScalarArray::U16Vec(x) => x.into_py(py),
            ScalarArray::U8Vec(x) => x.into_py(py),
            ScalarArray::I32Vec(x) => x.into_py(py),
            ScalarArray::I16Vec(x) => x.into_py(py),
            ScalarArray::I8Vec(x) => x.into_py(py),
            ScalarArray::BOOLVec(x) => x.into_py(py),
        }
    }

    pub fn reshape(&self, py: Python<'_>, shape: Vec<u32>) -> Py<PyAny> {
        let mut new_array = self.ndarray.clone_array().block_on();
        //TODO add validation
        new_array.shape = shape;
        NdArrayPy { ndarray: new_array }.into_py(py)
    }

    #[getter]
    pub fn shape(&self) -> PyResult<Vec<u32>> {
        Ok(self.ndarray.shape.clone())
    }

    pub fn __mul__(slf: &PyCell<Self>, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        Ok(_multiply(py, slf.as_ref(), other, None, None))
    }

    pub fn astype(&self, #[pyo3(from_py_with = "into_dtypepy")] dtype: Dtype) -> PyResult<Self> {
        Ok(Self {
            ndarray: self.ndarray.astype(dtype).block_on(),
        })
    }

    pub fn flatten(&self, py: Python<'_>) -> PyResult<Self> {
        py.allow_threads(|| {
            let mut new_array = self.ndarray.clone_array().block_on();
            new_array.shape = vec![(&new_array.shape).iter().map(|x| *x).product()];
            Ok(NdArrayPy { ndarray: new_array })
        })
    }
}

/// Creates a new array
#[pyfunction(name = "ones")]
pub fn array_ones(
    py: Python<'_>,
    shape: Vec<u32>,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> NdArrayPy {
    py.allow_threads(|| {
        let array_type: &Dtype = match dtype.as_ref() {
            Some(x) => &x.as_ref().dtype,
            None => &Dtype::Float32,
        };
        ones(shape, Some((*array_type).into()), None)
            .block_on()
            .into()
    })
}

/// Creates a new array
#[pyfunction(name = "zeros")]
pub fn array_zeros(
    py: Python<'_>,
    shape: Vec<u32>,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> NdArrayPy {
    py.allow_threads(|| {
        let array_type: &Dtype = match dtype.as_ref() {
            Some(x) => &x.as_ref().dtype,
            None => &Dtype::Float32,
        };
        zeros(shape, Some((*array_type).into()), None)
            .block_on()
            .into()
    })
}

/// Fills array with value
#[pyfunction(name = "full")]
pub fn array_full(
    py: Python<'_>,
    shape: Vec<u32>,
    data: &PyAny,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> PyResult<NdArrayPy> {
    let operand = convert_pyobj_into_scalar(data)?;
    py.allow_threads(|| {
        let dtype: &Dtype = match dtype.as_ref() {
            Some(x) => &x.as_ref().dtype,
            None => &Dtype::Float32,
        };
        // TODO remove unwrap
        PyResult::Ok(
            full(shape, operand.into(), Some(*dtype), None)
                .block_on()
                .into(),
        )
    })
}

pub fn get_shape<'a>(data: &'a PyAny) -> PyResult<Vec<u32>> {
    if data.is_instance_of::<PyList>() {
        let mut shape = vec![];
        shape.push(data.len()? as u32);
        shape.extend(get_shape(data.get_item(0)?)?);
        Ok(shape)
    } else if data.is_instance_of::<PyFloat>()
        || data.is_instance_of::<PyInt>()
        || data.is_instance_of::<PyString>()
    {
        Ok(vec![])
    } else {
        Err(PyRuntimeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub fn get_type<'a>(data: &'a PyAny) -> PyResult<Dtype> {
    if data.is_instance_of::<PyList>() {
        get_type(data.get_item(0)?)
    } else if data.is_instance_of::<PyBool>() {
        Ok(Dtype::Bool)
    } else if data.is_instance_of::<PyFloat>() {
        Ok(Dtype::Float32)
    } else if data.is_instance_of::<PyInt>() {
        Ok(Dtype::Int32)
    } else {
        Err(PyRuntimeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub fn flatten<'a, T: PyObectToRustPrimitive>(
    data: &'a PyAny,
    shape: &[u32],
    depth: usize,
) -> PyResult<Vec<T>> {
    if data.is_instance_of::<PyList>() {
        if data.len()? as u32 != shape[depth] {
            Err(PyRuntimeError::new_err(
                "Operation not supported for the given values",
            ))
        } else {
            let mut values = vec![];
            for i in 0..shape[depth] {
                values.extend(flatten(data.get_item(i)?, shape, depth + 1)?);
            }
            Ok(values)
        }
    } else if data.is_instance_of::<PyFloat>()
        || data.is_instance_of::<PyInt>()
        || data.is_instance_of::<PyString>()
    {
        Ok(vec![T::into_rust(data)?])
    } else {
        Err(PyRuntimeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub fn into_scalar_array<'a>(data: &'a PyAny, dtype: Dtype) -> PyResult<NdArrayPy> {
    let shape = get_shape(data)?;

    match dtype {
        Dtype::Int8 => {
            let values_array = flatten::<i8>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::Int16 => {
            let values_array = flatten::<i16>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::Int32 => {
            let values_array = flatten::<i32>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::UInt8 => {
            let values_array = flatten::<u8>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::UInt16 => {
            let values_array = flatten::<u16>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::UInt32 => {
            let values_array = flatten::<u32>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::Float32 => {
            let values_array = flatten::<f32>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::Bool => {
            let values_array = flatten::<bool>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
    }
}

/// Creates a new array
#[pyfunction(name = "array")]
pub fn array(
    data: &PyAny,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> PyResult<NdArrayPy> {
    match dtype.as_ref() {
        Some(x) => into_scalar_array(data, x.as_ref().dtype),
        None => {
            let dtype = get_type(data)?;
            into_scalar_array(data, dtype)
        }
    }
}

/// Creates a new array
#[pyfunction(name = "broadcast_to")]
pub fn broadcast_to(data: &PyAny, shape: Vec<u32>) -> PyResult<NdArrayPy> {
    let array = convert_pyobj_into_operand(data)?;
    Ok(NdArrayPy {
        ndarray: webgpupy::broadcast_to(array.as_ref(), &shape).block_on(),
    })
}

/// Repeats elements in an array
#[pyfunction(name = "repeat")]
pub fn repeat(
    data: &PyAny,
    #[pyo3(from_py_with = "convert_pyobj_into_array_u32")] repeats: Vec<u32>,
    axis: Option<u32>,
) -> PyResult<NdArrayPy> {
    let array = convert_pyobj_into_operand(data)?;
    // TODO remove unwrap
    Ok(NdArrayPy {
        ndarray: webgpupy::repeat(array.as_ref(), &repeats, axis)
            .block_on()
            .unwrap(),
    })
}

#[pyfunction(name = "dstack")]
pub fn dstack(
    #[pyo3(from_py_with = "convert_pyobj_into_vec_ndarray")] tup: Vec<&NdArray>,
) -> PyResult<NdArrayPy> {
    // TODO add array size validation
    Ok(NdArrayPy {
        ndarray: webgpupy::dstack(tup.as_ref()).block_on(),
    })
}

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(array_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(array_ones, m)?)?;
    m.add_function(wrap_pyfunction!(array_full, m)?)?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(self::broadcast_to, m)?)?;
    m.add_function(wrap_pyfunction!(self::repeat, m)?)?;
    m.add_function(wrap_pyfunction!(self::dstack, m)?)?;
    m.add_class::<NdArrayPy>()?;
    m.add_class::<DtypePy>()?;
    Ok(())
}
