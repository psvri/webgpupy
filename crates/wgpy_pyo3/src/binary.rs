use crate::ufunc::Ufunc;
use std::borrow::Cow;

use pollster::FutureExt;
use pyo3::{pyfunction, Python};

use crate::{
    add_ufunc_nin1_nout1, add_ufunc_nin2_nout1, convert_pyobj_into_operand, impl_ufunc_nin1_nout1,
    impl_ufunc_nin2_nout1,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy},
};
use pyo3::prelude::*;

impl_ufunc_nin2_nout1!(_bitwise_and, webgpupy::bitwise_and);
impl_ufunc_nin2_nout1!(_bitwise_or, webgpupy::bitwise_or);
impl_ufunc_nin1_nout1!(_invert, webgpupy::invert);

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(_bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(_invert, m)?)?;

    add_ufunc_nin2_nout1!(m, "bitwise_and");
    add_ufunc_nin2_nout1!(m, "bitwise_or");
    add_ufunc_nin1_nout1!(m, "invert");
    Ok(())
}
