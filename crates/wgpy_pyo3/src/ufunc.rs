use pyo3::{
    intern,
    prelude::*,
    pyclass, pymethods,
    types::{PyAny, PyDict, PyModule, PyString, PyTuple},
    Py, PyObject, PyResult, Python,
};

#[pyclass(name = "ufunc", subclass)]
pub struct Ufunc {
    pub function_name: &'static str,
    pub doc_string_path: &'static str,
    pub py_func: Py<PyAny>,
    pub nin: u8,
    pub nout: u8,
    pub ntypes: u8,
}

#[pymethods]
impl Ufunc {
    #[getter(__name__)]
    fn __name__(&self) -> String {
        self.function_name.into()
    }

    #[getter(__doc__)]
    fn __doc__(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let resources = py.import_bound("pkgutil")?;
        let function = resources.getattr(intern!(py, "get_data"))?;
        let args = ("webgpupy", self.doc_string_path);
        let result = function.call(args, None)?.getattr("decode")?;
        let result_string = result
            .call1(("utf-8",))?
            .downcast_into::<PyString>()?
            .unbind();
        PyResult::Ok(result_string)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<PyTuple>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> PyResult<PyObject> {
        self.py_func.call_bound(py, args, kwargs)
    }

    fn __repr__(&self) -> String {
        format!("<ufunc webgpupy_{}>", self.function_name)
    }
}

#[macro_export]
macro_rules! impl_ufunc_nin2_nout1 {
    ($name: ident, $fn_name: expr) => {
        #[pyfunction]
        #[pyo3(signature = (x, y, /, *, r#where = None, dtype=None))]
        pub fn $name<'a>(
            py: Python<'_>,
            x: &'a Bound<'a, PyAny>,
            y: &'a Bound<'a, PyAny>,
            r#where: Option<&NdArrayPy>,
            #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
        ) -> NdArrayPy {
            let x = convert_pyobj_into_operand(x).unwrap();
            let y = convert_pyobj_into_operand(y).unwrap();
            let where_ = r#where.map(|x| x.into());
            let dtype = dtype.map(|x| x.as_ref().dtype);
            py.allow_threads(|| $fn_name(x.as_ref(), y.as_ref(), where_, dtype).into())
        }
    };
}

#[macro_export]
macro_rules! impl_ufunc_nin1_nout1 {
    ($name: ident, $fn_name: expr) => {
        #[pyfunction]
        #[pyo3(signature = (x, /, *, r#where = None, dtype=None))]
        pub fn $name(
            py: Python<'_>,
            x: &Bound<PyAny>,
            r#where: Option<&NdArrayPy>,
            #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
        ) -> NdArrayPy {
            let data = convert_pyobj_into_operand(x).unwrap();
            let where_ = r#where.map(|x| x.into());
            let dtype = dtype.map(|x| x.as_ref().dtype);
            py.allow_threads(|| $fn_name(data.as_ref(), where_, dtype).into())
        }
    };
}

#[macro_export]
macro_rules! add_ufunc_nin1_nout1 {
    ($m: ident, $name: literal) => {
        let py_fn = $m.getattr(concat!("_", $name))?.into();
        let ufunc_function = Ufunc {
            function_name: $name,
            doc_string_path: concat!("python_doc/", $name, ".rst"),
            nin: 1,
            nout: 1,
            ntypes: 1,
            py_func: py_fn,
        };
        let ufunc_py = Python::with_gil(|py| {
            let ufunc_py_function = Py::new(py, ufunc_function)?;
            PyResult::Ok(ufunc_py_function)
        })?;
        $m.add($name, ufunc_py)?;
    };
}

#[macro_export]
macro_rules! add_ufunc_nin2_nout1 {
    ($m: ident, $name: literal) => {
        let py_fn = $m.getattr(concat!("_", $name))?.into();
        let ufunc_function = Ufunc {
            function_name: $name,
            doc_string_path: "",
            nin: 2,
            nout: 1,
            ntypes: 1,
            py_func: py_fn,
        };
        let ufunc_py = Python::with_gil(|py| {
            let ufunc_py_function = Py::new(py, ufunc_function)?;
            PyResult::Ok(ufunc_py_function)
        })?;
        $m.add($name, ufunc_py)?;
    };
}

pub fn create_py_items(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Ufunc>()?;
    Ok(())
}
