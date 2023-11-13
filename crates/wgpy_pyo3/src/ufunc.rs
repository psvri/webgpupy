use pyo3::{
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
    fn __doc__<'a>(&self, py: Python<'a>) -> PyResult<&'a PyString> {
        let resources = py.import("pkgutil")?;
        let function = resources.getattr("get_data")?;
        let args = ("webgpupy", self.doc_string_path);
        let result = function.call(args, None)?.getattr("decode")?;
        let result_string = result.call1(("utf-8",))?.downcast()?;
        PyResult::Ok(result_string)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &PyTuple,
        kwargs: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        self.py_func.call(py, args, kwargs)
    }
}

macro_rules! add_ufunc_nin1_nout1 {
    ($m: ident, $name: literal) => {
        let py_fn = $m.getattr(concat!("_", $name))?.into();
        let ufunc_function = Ufunc {
            function_name: $name,
            doc_string_path: concat!($name, ".md"),
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

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_class::<Ufunc>()?;
    add_ufunc_nin1_nout1!(m, "sin");
    add_ufunc_nin1_nout1!(m, "cos");
    add_ufunc_nin2_nout1!(m, "multiply");
    Ok(())
}
