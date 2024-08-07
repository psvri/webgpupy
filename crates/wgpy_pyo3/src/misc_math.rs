use crate::{
    add_ufunc_nin1_nout1, add_ufunc_nin2_nout1, convert_pyobj_into_operand,
    convert_pyobj_into_option_operand, impl_ufunc_nin1_nout1, impl_ufunc_nin2_nout1,
    ndarraypy::*,
    types::{into_optional_dtypepy, DtypePy, OperandPy},
    ufunc::Ufunc,
};
use pyo3::prelude::*;
use pyo3::{pyfunction, Python};
use std::borrow::Cow;
use webgpupy::clip;
use webgpupy::cross;

impl_ufunc_nin1_nout1!(_sqrt, webgpupy::sqrt);
impl_ufunc_nin1_nout1!(_cbrt, webgpupy::cbrt);
impl_ufunc_nin1_nout1!(_absolute, webgpupy::absolute);
impl_ufunc_nin2_nout1!(_maximum, webgpupy::maximum);
impl_ufunc_nin2_nout1!(_minimum, webgpupy::minimum);
impl_ufunc_nin2_nout1!(_power, webgpupy::power);

// TODO add ufunc kwargs support
#[pyfunction(name = "clip")]
#[pyo3(signature = (a, a_min, a_max))]
pub fn clip_(
    py: Python<'_>,
    a: &NdArrayPy,
    #[pyo3(from_py_with = "convert_pyobj_into_option_operand")] a_min: Option<OperandPy>,
    #[pyo3(from_py_with = "convert_pyobj_into_option_operand")] a_max: Option<OperandPy>,
) -> NdArrayPy {
    let min_arr = a_min.as_ref().map(|a_min| a_min.as_ref());
    let max_arr = a_max.as_ref().map(|a_max| a_max.as_ref());
    py.allow_threads(|| clip(&a.ndarray, min_arr, max_arr).into())
}

#[pyfunction(name = "cross")]
#[pyo3(signature = (a, b))]
pub fn cross_<'a>(py: Python<'_>, a: &'a Bound<'a, PyAny>, b: &'a Bound<'a, PyAny>) -> NdArrayPy {
    let x = convert_pyobj_into_operand(a).unwrap();
    let y = convert_pyobj_into_operand(b).unwrap();
    py.allow_threads(|| cross(x.as_ref(), y.as_ref()).into())
}

pub fn create_py_items(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(_cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(_maximum, m)?)?;
    m.add_function(wrap_pyfunction!(_minimum, m)?)?;
    m.add_function(wrap_pyfunction!(_absolute, m)?)?;
    m.add_function(wrap_pyfunction!(_power, m)?)?;
    m.add_function(wrap_pyfunction!(clip_, m)?)?;

    m.add_function(wrap_pyfunction!(cross_, m)?)?;

    add_ufunc_nin1_nout1!(m, "sqrt");
    add_ufunc_nin1_nout1!(m, "cbrt");
    add_ufunc_nin2_nout1!(m, "maximum");
    add_ufunc_nin2_nout1!(m, "minimum");
    add_ufunc_nin2_nout1!(m, "absolute");
    add_ufunc_nin2_nout1!(m, "power");
    Ok(())
}
