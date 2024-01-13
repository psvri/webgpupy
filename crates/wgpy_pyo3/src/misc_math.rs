use crate::{
    add_ufunc_nin1_nout1, add_ufunc_nin2_nout1, convert_pyobj_into_operand, impl_ufunc_nin1_nout1,
    impl_ufunc_nin2_nout1,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy},
    ufunc::Ufunc,
};
use pollster::FutureExt;
use pyo3::prelude::*;
use pyo3::{pyfunction, Python};
use std::borrow::Cow;

impl_ufunc_nin1_nout1!(_sqrt, webgpupy::sqrt);
impl_ufunc_nin2_nout1!(_maximum, webgpupy::maximum);
impl_ufunc_nin2_nout1!(_minimum, webgpupy::minimum);

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(_maximum, m)?)?;
    m.add_function(wrap_pyfunction!(_minimum, m)?)?;

    add_ufunc_nin1_nout1!(m, "sqrt");
    add_ufunc_nin2_nout1!(m, "maximum");
    add_ufunc_nin2_nout1!(m, "minimum");
    Ok(())
}
