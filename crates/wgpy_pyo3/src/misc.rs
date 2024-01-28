use pyo3::{pyfunction, types::PyModule, wrap_pyfunction, PyAny, PyResult, Python};
use webgpupy::where_;

use crate::{convert_pyobj_into_operand, ndarraypy::NdArrayPy};

/// Fix broadcasting
#[pyfunction]
pub fn r#where(py: Python<'_>, mask: &PyAny, x: &PyAny, y: &PyAny) -> NdArrayPy {
    let x = convert_pyobj_into_operand(x).unwrap();
    let y = convert_pyobj_into_operand(y).unwrap();
    let mask = convert_pyobj_into_operand(mask).unwrap();
    NdArrayPy {
        ndarray: where_(mask.as_ref(), x.as_ref(), y.as_ref()),
    }
}

#[pyfunction]
pub fn any(py: Python<'_>, x: &PyAny) -> bool {
    let x = convert_pyobj_into_operand(x).unwrap();
    webgpupy::any(x.as_ref())
}

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(r#where, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    Ok(())
}
