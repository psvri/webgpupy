use std::borrow::Cow;

use pollster::FutureExt;
use pyo3::{pyfunction, Python};

use crate::{
    convert_pyobj_into_operand,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy},
};
use pyo3::prelude::*;

/// Performs cos of ndarray
#[pyfunction]
#[pyo3(signature = (x, /, *, r#where = None, dtype=None))]
pub fn _cos(
    py: Python<'_>,
    x: &PyAny,
    r#where: Option<&NdArrayPy>,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> NdArrayPy {
    let data = convert_pyobj_into_operand(x).unwrap();
    let where_ = r#where.map(|x| x.into());
    let dtype = dtype.map(|x| x.as_ref().dtype);
    py.allow_threads(|| {
        webgpupy::cos(data.as_ref(), where_, dtype)
            .block_on()
            .into()
    })
}

/// Performs sin of ndarray
#[pyfunction]
#[pyo3(signature = (x, /, *, r#where = None, dtype=None))]
pub fn _sin(
    py: Python<'_>,
    x: &PyAny,
    r#where: Option<&NdArrayPy>,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> NdArrayPy {
    let data = convert_pyobj_into_operand(x).unwrap();
    let where_ = r#where.map(|x| x.into());
    let dtype = dtype.map(|x| x.as_ref().dtype);
    py.allow_threads(|| {
        webgpupy::sin(data.as_ref(), where_, dtype)
            .block_on()
            .into()
    })
}

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_sin, m)?)?;
    m.add_function(wrap_pyfunction!(_cos, m)?)?;
    Ok(())
}
