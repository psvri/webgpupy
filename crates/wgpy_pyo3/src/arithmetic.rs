use std::borrow::Cow;

use pollster::FutureExt;
use pyo3::{pyfunction, Python};

use crate::{
    convert_pyobj_into_operand,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy},
};
use pyo3::prelude::*;

/// Performs multiplication of ndarray
#[pyfunction]
#[pyo3(signature = (x, y, /, *, r#where = None, dtype=None))]
pub fn _multiply<'a>(
    py: Python<'_>,
    x: &'a PyAny,
    y: &'a PyAny,
    r#where: Option<&NdArrayPy>,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> NdArrayPy {
    let x = convert_pyobj_into_operand(x).unwrap();
    let y = convert_pyobj_into_operand(y).unwrap();
    let where_ = r#where.map(|x| x.into());
    let dtype = dtype.map(|x| x.as_ref().dtype);
    py.allow_threads(|| {
        webgpupy::multiply(x.as_ref(), y.as_ref(), where_, dtype)
            .block_on()
            .into()
    })
}

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_multiply, m)?)?;
    Ok(())
}
