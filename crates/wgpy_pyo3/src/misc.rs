use pyo3::{
    prelude::PyModuleMethods, pyfunction, types::PyModule, wrap_pyfunction, Bound, PyAny, PyResult,
    Python,
};
use webgpupy::where_;

use crate::{convert_pyobj_into_operand, ndarraypy::NdArrayPy};

/// Fix broadcasting
#[pyfunction]
pub fn r#where(mask: &Bound<PyAny>, x: &Bound<PyAny>, y: &Bound<PyAny>) -> NdArrayPy {
    let x = convert_pyobj_into_operand(x).unwrap();
    let y = convert_pyobj_into_operand(y).unwrap();
    let mask = convert_pyobj_into_operand(mask).unwrap();
    NdArrayPy {
        ndarray: where_(mask.as_ref(), x.as_ref(), y.as_ref()),
    }
}

#[pyfunction]
pub fn any(_py: Python<'_>, x: &Bound<PyAny>) -> bool {
    let x = convert_pyobj_into_operand(x).unwrap();
    webgpupy::any(x.as_ref())
}

#[pyfunction]
pub fn all(_py: Python<'_>, x: &Bound<PyAny>) -> bool {
    let x = convert_pyobj_into_operand(x).unwrap();
    webgpupy::all(x.as_ref())
}

pub fn create_py_items(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(r#where, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    Ok(())
}
