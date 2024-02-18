use pyo3::{pyfunction, Python};
use std::borrow::Cow;

use crate::{
    add_ufunc_nin1_nout1, convert_pyobj_into_operand, impl_ufunc_nin1_nout1,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy},
    ufunc::Ufunc,
};
use pyo3::prelude::*;

impl_ufunc_nin1_nout1!(_cos, webgpupy::cos);
impl_ufunc_nin1_nout1!(_sin, webgpupy::sin);
impl_ufunc_nin1_nout1!(_arccos, webgpupy::arccos);

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_sin, m)?)?;
    m.add_function(wrap_pyfunction!(_cos, m)?)?;
    m.add_function(wrap_pyfunction!(_arccos, m)?)?;
    add_ufunc_nin1_nout1!(m, "sin");
    add_ufunc_nin1_nout1!(m, "cos");
    add_ufunc_nin1_nout1!(m, "arccos");
    Ok(())
}
