use crate::ufunc::Ufunc;
use std::borrow::Cow;

use pollster::FutureExt;
use pyo3::{pyfunction, Python};

use crate::{
    add_ufunc_nin2_nout1, convert_pyobj_into_operand, impl_ufunc_nin2_nout1,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy},
};
use pyo3::prelude::*;

impl_ufunc_nin2_nout1!(_greater, webgpupy::greater);
impl_ufunc_nin2_nout1!(_lesser, webgpupy::lesser);

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_greater, m)?)?;
    m.add_function(wrap_pyfunction!(_lesser, m)?)?;

    add_ufunc_nin2_nout1!(m, "greater");
    add_ufunc_nin2_nout1!(m, "lesser");
    Ok(())
}
