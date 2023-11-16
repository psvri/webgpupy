use std::borrow::Cow;

use pyo3::{
    exceptions::PyTypeError,
    pyclass, pymethods,
    types::{PyBool, PyString},
    FromPyObject, PyAny, PyCell, PyErr, PyResult,
};
use webgpupy::{Dtype, NdArray};

#[derive(Debug)]
pub enum ScalarValuePy {
    F32(f32),
    U32(u32),
    U16(u16),
    U8(u8),
    I32(i32),
    I16(i16),
    I8(i8),
    BOOL(bool),
}

impl<'s> FromPyObject<'s> for ScalarValuePy {
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        if ob.is_instance_of::<PyBool>() {
            PyResult::Ok(Self::BOOL(ob.extract::<bool>().unwrap()))
        } else if let Ok(value) = ob.extract::<i32>() {
            PyResult::Ok(Self::I32(value))
        } else if let Ok(value) = ob.extract::<f32>() {
            PyResult::Ok(Self::F32(value))
        } else {
            PyResult::Err(PyTypeError::new_err("Invalid scalar value type"))
        }
    }
}

#[derive(Debug)]
pub enum ScalarArrayPy {
    F32ARRAY(Vec<f32>),
    U32ARRAY(Vec<u32>),
    U16ARRAY(Vec<u16>),
    U8ARRAY(Vec<u8>),
    I32ARRAY(Vec<i32>),
    I16ARRAY(Vec<i16>),
    I8ARRAY(Vec<i8>),
    BOOLARRAY(Vec<bool>),
}

#[derive(Debug)]
pub enum OperandPy<'a> {
    NdArrayRef(&'a NdArray),
    NdArrayOwned(NdArray),
}

impl AsRef<NdArray> for OperandPy<'_> {
    fn as_ref(&self) -> &NdArray {
        match self {
            OperandPy::NdArrayRef(x) => x,
            OperandPy::NdArrayOwned(x) => x,
        }
    }
}

impl<'a> From<&'a NdArray> for OperandPy<'a> {
    fn from(value: &'a NdArray) -> Self {
        OperandPy::NdArrayRef(value)
    }
}

impl From<NdArray> for OperandPy<'_> {
    fn from(value: NdArray) -> Self {
        OperandPy::NdArrayOwned(value)
    }
}

#[pyclass(frozen)]
#[derive(Debug, Copy, Clone)]
pub struct DtypePy {
    pub dtype: Dtype,
}

#[pymethods]
impl DtypePy {
    #[new]
    fn new(#[pyo3(from_py_with = "into_dtypepy")] value: Dtype) -> Self {
        Self { dtype: value }
    }
}

pub(crate) fn into_optional_dtypepy(obj: &PyAny) -> Result<Option<Cow<DtypePy>>, PyErr> {
    if obj.is_none() {
        PyResult::Ok(None)
    } else if let Ok(c) = obj.downcast::<PyCell<DtypePy>>() {
        PyResult::Ok(Some(Cow::Borrowed(c.get())))
    } else if obj.is_instance_of::<PyString>() {
        PyResult::Ok(Some(Cow::Owned(DtypePy {
            dtype: Dtype::from(obj.extract::<&str>()?),
        })))
    } else {
        PyResult::Err(PyTypeError::new_err(format!(
            "Cannot convert type {} into DtypePy ",
            obj.get_type()
        )))
    }
}

pub(crate) fn into_dtypepy(obj: &PyAny) -> Result<Dtype, PyErr> {
    if obj.is_instance_of::<PyString>() {
        Ok(Dtype::from(obj.extract::<&str>()?))
    } else {
        PyResult::Err(PyTypeError::new_err(format!(
            "Cannot convert type {} into DtypePy ",
            obj.get_type()
        )))
    }
}

impl From<DtypePy> for Dtype {
    fn from(value: DtypePy) -> Self {
        value.dtype
    }
}

impl<'a> From<&'a DtypePy> for &'a Dtype {
    fn from(value: &'a DtypePy) -> &'a Dtype {
        &value.dtype
    }
}

impl<'a> TryFrom<&'a str> for DtypePy {
    type Error = PyErr;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        match value {
            "bool" => Ok(DtypePy { dtype: Dtype::Bool }),
            "float" => Ok(DtypePy {
                dtype: Dtype::Float32,
            }),
            "int" => Ok(DtypePy {
                dtype: Dtype::Int32,
            }),
            _ => Err(PyTypeError::new_err(format!(
                "Cannot convert type {} into DtypePy ",
                value
            ))),
        }
    }
}
