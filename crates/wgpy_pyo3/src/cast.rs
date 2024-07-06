use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyBool, PyFloat, PyInt},
    Bound, PyAny, PyResult,
};
pub trait PyObectToRustPrimitive {
    fn into_rust(object: &Bound<PyAny>) -> PyResult<Self>
    where
        Self: Sized;
}

macro_rules! ImplPyObectToRustPrimitive {
    ($ty: ident) => {
        impl PyObectToRustPrimitive for $ty {
            fn into_rust(object: &Bound<PyAny>) -> PyResult<$ty> {
                if object.is_instance_of::<PyInt>() {
                    Ok(object.extract::<i64>()? as $ty)
                } else if object.is_instance_of::<PyFloat>() {
                    Ok(object.extract::<f64>()? as $ty)
                } else {
                    Err(PyRuntimeError::new_err(format!(
                        stringify!("cannot convert {:?} into ", $ty),
                        object.get_type()
                    )))
                }
            }
        }
    };
}

ImplPyObectToRustPrimitive!(i8);
ImplPyObectToRustPrimitive!(i16);
ImplPyObectToRustPrimitive!(i32);
ImplPyObectToRustPrimitive!(u8);
ImplPyObectToRustPrimitive!(u16);
ImplPyObectToRustPrimitive!(u32);
ImplPyObectToRustPrimitive!(f32);

impl PyObectToRustPrimitive for bool {
    fn into_rust(object: &Bound<PyAny>) -> PyResult<Self>
    where
        Self: Sized,
    {
        if object.is_instance_of::<PyBool>() {
            Ok(object.extract::<bool>()?)
        } else {
            Err(PyRuntimeError::new_err(format!(
                stringify!("cannot convert {:?} into ", $ty),
                object.get_type()
            )))
        }
    }
}
