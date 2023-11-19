use std::borrow::Cow;

use pollster::FutureExt as _;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::*};
use webgpupy::{full, *};

use crate::{
    arithmetic::*,
    cast::PyObectToRustPrimitive,
    convert_pyobj_into_array_u32, convert_pyobj_into_operand, convert_pyobj_into_scalar,
    convert_pyobj_into_vec_ndarray,
    types::{into_dtypepy, into_optional_dtypepy, DtypePy},
};

/// N-dimentional array
#[pyclass(name = "ndarray", frozen)]
#[derive(Debug)]
pub struct NdArrayPy {
    pub ndarray: NdArray,
}

impl From<NdArray> for NdArrayPy {
    fn from(value: NdArray) -> Self {
        NdArrayPy { ndarray: value }
    }
}

impl<'a> From<&'a NdArrayPy> for &'a NdArray {
    fn from(value: &'a NdArrayPy) -> &'a NdArray {
        &value.ndarray
    }
}

#[pymethods]
impl NdArrayPy {
    pub fn display(&self) -> String {
        format!("Shape {:?} \n {:?}", self.ndarray.shape, self.ndarray.data)
    }

    #[doc = include_str!("../python/webgpupy/python_doc/ndarray.tolist.rst")]
    pub fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let values = self.ndarray.data.get_raw_values().block_on();
        to_list(py, &values, 0, &self.ndarray.shape, &mut 0)
    }

    pub fn reshape(&self, py: Python<'_>, shape: Vec<u32>) -> Py<PyAny> {
        let mut new_array = self.ndarray.clone_array().block_on();
        //TODO add validation
        new_array.shape = shape;
        NdArrayPy { ndarray: new_array }.into_py(py)
    }

    /// Shape of ndarry
    #[getter]
    pub fn shape(&self) -> PyResult<Vec<u32>> {
        Ok(self.ndarray.shape.clone())
    }

    pub fn __mul__(slf: &PyCell<Self>, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        Ok(_multiply(py, slf.as_ref(), other, None, None))
    }

    pub fn __rmul__(slf: &PyCell<Self>, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        Ok(_multiply(py, slf.as_ref(), other, None, None))
    }

    pub fn __truediv__(slf: &PyCell<Self>, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        Ok(_divide(py, slf.as_ref(), other, None, None))
    }

    pub fn __add__(slf: &PyCell<Self>, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        Ok(_add(py, slf.as_ref(), other, None, None))
    }

    pub fn __radd__(slf: &PyCell<Self>, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        Ok(_add(py, slf.as_ref(), other, None, None))
    }

    pub fn __sub__(slf: &PyCell<Self>, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        Ok(_subtract(py, slf.as_ref(), other, None, None))
    }

    pub fn __rsub__(slf: &PyCell<Self>, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        Ok(_subtract(py, other, slf.as_ref(), None, None))
    }

    pub fn astype(&self, #[pyo3(from_py_with = "into_dtypepy")] dtype: Dtype) -> PyResult<Self> {
        Ok(Self {
            ndarray: self.ndarray.astype(dtype).block_on(),
        })
    }

    pub fn flatten(&self, py: Python<'_>) -> PyResult<Self> {
        py.allow_threads(|| {
            let mut new_array = self.ndarray.clone_array().block_on();
            new_array.shape = vec![(new_array.shape).iter().copied().product()];
            Ok(NdArrayPy { ndarray: new_array })
        })
    }

    pub fn __getitem__(&self, py: Python<'_>, other: &PyAny) -> PyResult<Self> {
        let index_slices = subscripts_to_index_slices(other, &self.ndarray.shape)?;
        let mut indexes = vec![];
        slice_to_index(&self.ndarray.shape, 0, &index_slices, &mut indexes, 0);
        let indexes = NdArray::from_slice(
            indexes.as_slice().into(),
            vec![indexes.len() as u32],
            Some(self.ndarray.data.get_gpu_device()),
        )
        .block_on();
        let ndarray = py.allow_threads(|| self.ndarray.take(&indexes, None).block_on());

        Ok(NdArrayPy { ndarray })
    }
}

/// Creates a new array
#[pyfunction(name = "ones")]
pub fn array_ones(
    py: Python<'_>,
    shape: Vec<u32>,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> NdArrayPy {
    py.allow_threads(|| {
        let array_type: &Dtype = match dtype.as_ref() {
            Some(x) => &x.as_ref().dtype,
            None => &Dtype::Float32,
        };
        ones(shape, Some(*array_type), None).block_on().into()
    })
}

/// Creates a new array
#[pyfunction(name = "zeros")]
pub fn array_zeros(
    py: Python<'_>,
    shape: Vec<u32>,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> NdArrayPy {
    py.allow_threads(|| {
        let array_type: &Dtype = match dtype.as_ref() {
            Some(x) => &x.as_ref().dtype,
            None => &Dtype::Float32,
        };
        zeros(shape, Some(*array_type), None).block_on().into()
    })
}

/// Fills array with value
#[pyfunction(name = "full")]
pub fn array_full(
    py: Python<'_>,
    shape: Vec<u32>,
    data: &PyAny,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> PyResult<NdArrayPy> {
    let operand = convert_pyobj_into_scalar(data)?;
    py.allow_threads(|| {
        let dtype: &Dtype = match dtype.as_ref() {
            Some(x) => &x.as_ref().dtype,
            None => &Dtype::Float32,
        };
        // TODO remove unwrap
        PyResult::Ok(
            full(shape, operand.into(), Some(*dtype), None)
                .block_on()
                .into(),
        )
    })
}

fn to_list(
    py: Python<'_>,
    values: &ScalarArray,
    depth: u32,
    shape: &[u32],
    pos: &mut usize,
) -> PyResult<Py<PyList>> {
    if depth as usize == shape.len() - 1 {
        let len = shape[shape.len() - 1] as usize;
        let list = match values {
            ScalarArray::F32Vec(x) => PyList::new(py, &x[*pos..*pos + len]),
            ScalarArray::U32Vec(x) => PyList::new(py, &x[*pos..*pos + len]),
            ScalarArray::U16Vec(x) => PyList::new(py, &x[*pos..*pos + len]),
            ScalarArray::U8Vec(x) => PyList::new(py, &x[*pos..*pos + len]),
            ScalarArray::I32Vec(x) => PyList::new(py, &x[*pos..*pos + len]),
            ScalarArray::I16Vec(x) => PyList::new(py, &x[*pos..*pos + len]),
            ScalarArray::I8Vec(x) => PyList::new(py, &x[*pos..*pos + len]),
            ScalarArray::BOOLVec(x) => PyList::new(py, &x[*pos..*pos + len]),
        };
        *pos += len;
        PyResult::Ok(list.into_py(py))
    } else {
        let list = PyList::empty(py);
        for _ in 0..shape[depth as usize] {
            list.append(to_list(py, values, depth + 1, shape, pos)?)?;
        }
        PyResult::Ok(list.into_py(py))
    }
}

fn slice_to_index(
    shape: &[u32],
    depth: u32,
    subscripts: &[IndexSlice],
    indexes: &mut Vec<u32>,
    base_index: u32,
) {
    if depth as usize == shape.len() - 1 {
        let slice = &subscripts[depth as usize];
        for i in slice.iterate() {
            indexes.push(base_index + i);
        }
    } else {
        let slice = &subscripts[depth as usize];
        let count = shape[(depth + 1) as usize..].iter().product::<u32>();
        for i in slice.iterate() {
            slice_to_index(
                shape,
                depth + 1,
                subscripts,
                indexes,
                base_index + (i * count),
            );
        }
    }
}

fn slice_to_index_slice(subscripts: &PyAny, length: i32) -> PyResult<IndexSlice> {
    let slice = subscripts.downcast::<PySlice>()?;
    let indices = slice.indices(length)?;
    Ok(IndexSlice::new(
        indices.start as i64,
        indices.stop as i64,
        indices.step as i64,
        length as i64,
    )
    .unwrap())
}

fn subscripts_to_index_slices(subscripts: &PyAny, shape: &[u32]) -> PyResult<Vec<IndexSlice>> {
    if subscripts.is_instance_of::<PyTuple>() {
        let subscripts_tuple = subscripts.downcast::<PyTuple>()?;
        let length = subscripts_tuple.len();
        let mut index_slices = Vec::with_capacity(shape.len());

        for i in 0..length {
            let slice = subscripts_tuple.get_item(i)?;
            if slice.is_instance_of::<PySlice>() {
                index_slices.push(slice_to_index_slice(slice, shape[i] as i32)?);
            } else if slice.is_instance_of::<PyInt>() {
                let index = slice.extract().unwrap();
                index_slices.push(IndexSlice::new(index, index + 1, 1, shape[i] as i64).unwrap());
            } else {
                return Err(PyRuntimeError::new_err(format!(
                    "Operation not supported for the given subscripts {:?}",
                    subscripts.get_type(),
                )));
            }
        }

        for i in length..shape.len() {
            index_slices.push(IndexSlice::new(0, shape[i] as i64, 1, shape[i] as i64).unwrap());
        }

        Ok(index_slices)
    } else if subscripts.is_instance_of::<PyInt>() {
        let index = subscripts.extract().unwrap();
        let mut index_slices = Vec::with_capacity(shape.len());
        index_slices.push(IndexSlice::new(index, index + 1, 1, shape[0] as i64).unwrap());
        for i in 1..shape.len() {
            index_slices.push(IndexSlice::new(0, shape[i] as i64, 1, shape[i] as i64).unwrap());
        }

        Ok(index_slices)
    } else if subscripts.is_instance_of::<PySlice>() {
        let mut index_slices = Vec::with_capacity(shape.len());
        index_slices.push(slice_to_index_slice(subscripts, shape[0] as i32)?);

        for i in 1..shape.len() {
            index_slices.push(IndexSlice::new(0, shape[i] as i64, 1, shape[i] as i64).unwrap());
        }
        Ok(index_slices)
    } else {
        Err(PyRuntimeError::new_err(format!(
            "Operation not supported for the given subscripts {:?}",
            subscripts.get_type(),
        )))
    }
}

pub fn get_shape(data: &PyAny) -> PyResult<Vec<u32>> {
    if data.is_instance_of::<PyList>() {
        let mut shape = vec![];
        shape.push(data.len()? as u32);
        shape.extend(get_shape(data.get_item(0)?)?);
        Ok(shape)
    } else if data.is_instance_of::<PyFloat>()
        || data.is_instance_of::<PyInt>()
        || data.is_instance_of::<PyString>()
    {
        Ok(vec![])
    } else {
        Err(PyRuntimeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub fn get_type(data: &PyAny) -> PyResult<Dtype> {
    if data.is_instance_of::<PyList>() {
        get_type(data.get_item(0)?)
    } else if data.is_instance_of::<PyBool>() {
        Ok(Dtype::Bool)
    } else if data.is_instance_of::<PyFloat>() {
        Ok(Dtype::Float32)
    } else if data.is_instance_of::<PyInt>() {
        Ok(Dtype::Int32)
    } else {
        Err(PyRuntimeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub fn flatten<T: PyObectToRustPrimitive>(
    data: &PyAny,
    shape: &[u32],
    depth: usize,
) -> PyResult<Vec<T>> {
    if data.is_instance_of::<PyList>() {
        if data.len()? as u32 != shape[depth] {
            Err(PyRuntimeError::new_err(
                "Operation not supported for the given values",
            ))
        } else {
            let mut values = vec![];
            for i in 0..shape[depth] {
                values.extend(flatten(data.get_item(i)?, shape, depth + 1)?);
            }
            Ok(values)
        }
    } else if data.is_instance_of::<PyFloat>()
        || data.is_instance_of::<PyInt>()
        || data.is_instance_of::<PyString>()
    {
        Ok(vec![T::into_rust(data)?])
    } else {
        Err(PyRuntimeError::new_err(
            "Operation not supported for the given values",
        ))
    }
}

pub fn into_scalar_array(data: &PyAny, dtype: Dtype) -> PyResult<NdArrayPy> {
    let shape = get_shape(data)?;

    match dtype {
        Dtype::Int8 => {
            let values_array = flatten::<i8>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::Int16 => {
            let values_array = flatten::<i16>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::Int32 => {
            let values_array = flatten::<i32>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::UInt8 => {
            let values_array = flatten::<u8>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::UInt16 => {
            let values_array = flatten::<u16>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::UInt32 => {
            let values_array = flatten::<u32>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::Float32 => {
            let values_array = flatten::<f32>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
        Dtype::Bool => {
            let values_array = flatten::<bool>(data, &shape, 0)?;
            Ok(
                NdArray::from_slice(values_array.as_slice().into(), shape, None)
                    .block_on()
                    .into(),
            )
        }
    }
}

/// Creates a new array
#[pyfunction(name = "array")]
pub fn array(
    data: &PyAny,
    #[pyo3(from_py_with = "into_optional_dtypepy")] dtype: Option<Cow<DtypePy>>,
) -> PyResult<NdArrayPy> {
    match dtype.as_ref() {
        Some(x) => into_scalar_array(data, x.as_ref().dtype),
        None => {
            let dtype = get_type(data)?;
            into_scalar_array(data, dtype)
        }
    }
}

/// Creates a new array
#[pyfunction(name = "broadcast_to")]
pub fn broadcast_to(data: &PyAny, shape: Vec<u32>) -> PyResult<NdArrayPy> {
    let array = convert_pyobj_into_operand(data)?;
    Ok(NdArrayPy {
        ndarray: webgpupy::broadcast_to(array.as_ref(), &shape).block_on(),
    })
}

/// Repeats elements in an array
#[pyfunction(name = "repeat")]
pub fn repeat(
    data: &PyAny,
    #[pyo3(from_py_with = "convert_pyobj_into_array_u32")] repeats: Vec<u32>,
    axis: Option<u32>,
) -> PyResult<NdArrayPy> {
    let array = convert_pyobj_into_operand(data)?;
    // TODO remove unwrap
    Ok(NdArrayPy {
        ndarray: webgpupy::repeat(array.as_ref(), &repeats, axis)
            .block_on()
            .unwrap(),
    })
}

#[pyfunction(name = "dstack")]
pub fn dstack(
    #[pyo3(from_py_with = "convert_pyobj_into_vec_ndarray")] tup: Vec<&NdArray>,
) -> PyResult<NdArrayPy> {
    // TODO add array size validation
    Ok(NdArrayPy {
        ndarray: webgpupy::dstack(tup.as_ref()).block_on(),
    })
}

pub fn create_py_items(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(array_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(array_ones, m)?)?;
    m.add_function(wrap_pyfunction!(array_full, m)?)?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(self::broadcast_to, m)?)?;
    m.add_function(wrap_pyfunction!(self::repeat, m)?)?;
    m.add_function(wrap_pyfunction!(self::dstack, m)?)?;
    m.add_class::<NdArrayPy>()?;
    m.add_class::<DtypePy>()?;
    Ok(())
}
