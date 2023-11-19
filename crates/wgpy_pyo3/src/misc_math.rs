use pollster::FutureExt;
use pyo3::{pyfunction, Python};
use std::borrow::Cow;

use crate::{
    convert_pyobj_into_operand, impl_ufunc_nin1_nout1,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy},
};
use pyo3::prelude::*;

impl_ufunc_nin1_nout1!(sqrt, webgpupy::sqrt);

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    Ok(())
}
