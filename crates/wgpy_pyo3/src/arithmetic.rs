use crate::ufunc::Ufunc;
use std::borrow::Cow;

use pyo3::{pyfunction, Python};

use crate::{
    add_ufunc_nin2_nout1, convert_pyobj_into_operand, impl_ufunc_nin2_nout1,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy},
};
use pyo3::prelude::*;

impl_ufunc_nin2_nout1!(_multiply, webgpupy::multiply);
impl_ufunc_nin2_nout1!(_divide, webgpupy::divide);
impl_ufunc_nin2_nout1!(_add, webgpupy::add);
impl_ufunc_nin2_nout1!(_subtract, webgpupy::subtract);

pub fn create_py_items(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(_divide, m)?)?;
    m.add_function(wrap_pyfunction!(_add, m)?)?;
    m.add_function(wrap_pyfunction!(_subtract, m)?)?;

    add_ufunc_nin2_nout1!(m, "multiply");
    add_ufunc_nin2_nout1!(m, "divide");
    add_ufunc_nin2_nout1!(m, "add");
    add_ufunc_nin2_nout1!(m, "subtract");
    Ok(())
}
