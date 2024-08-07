use crate::ufunc::Ufunc;
use std::borrow::Cow;

use pyo3::{pyfunction, Python};

use crate::{
    add_ufunc_nin2_nout1, convert_pyobj_into_operand, impl_ufunc_nin2_nout1,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy},
};
use pyo3::prelude::*;

impl_ufunc_nin2_nout1!(_greater, webgpupy::greater);
impl_ufunc_nin2_nout1!(_lesser, webgpupy::lesser);
impl_ufunc_nin2_nout1!(_equal, webgpupy::equal);

pub fn create_py_items(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_greater, m)?)?;
    m.add_function(wrap_pyfunction!(_lesser, m)?)?;
    m.add_function(wrap_pyfunction!(_equal, m)?)?;

    add_ufunc_nin2_nout1!(m, "greater");
    add_ufunc_nin2_nout1!(m, "lesser");
    Ok(())
}
