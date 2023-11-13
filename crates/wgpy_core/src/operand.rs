use crate::{Dtype, NdArray};

use arrow_gpu::kernels::ScalarValue as ArrowScalarValue;

#[derive(Debug)]
pub enum ScalarValue {
    F32(f32),
    U32(u32),
    U16(u16),
    U8(u8),
    I32(i32),
    I16(i16),
    I8(i8),
    BOOL(bool),
}

impl ScalarValue {
    pub fn zero(dtype: &Dtype) -> Self {
        match *dtype {
            Dtype::Int8 => ScalarValue::I8(0),
            Dtype::Int16 => ScalarValue::I16(0),
            Dtype::Int32 => ScalarValue::I32(0),
            Dtype::UInt8 => ScalarValue::U8(0),
            Dtype::UInt16 => ScalarValue::U16(0),
            Dtype::UInt32 => ScalarValue::U32(0),
            Dtype::Float32 => ScalarValue::F32(0.0),
            Dtype::Bool => ScalarValue::BOOL(false),
        }
    }

    pub fn one(dtype: &Dtype) -> Self {
        match dtype {
            Dtype::Int8 => ScalarValue::I8(1),
            Dtype::Int16 => ScalarValue::I16(1),
            Dtype::Int32 => ScalarValue::I32(1),
            Dtype::UInt8 => ScalarValue::U8(1),
            Dtype::UInt16 => ScalarValue::U16(1),
            Dtype::UInt32 => ScalarValue::U32(1),
            Dtype::Float32 => ScalarValue::F32(1.0),
            Dtype::Bool => ScalarValue::BOOL(true),
        }
    }
}

macro_rules! impl_into_scalarvalue {
    ($ty: ident, $svty: ident) => {
        impl From<$ty> for ScalarValue {
            fn from(value: $ty) -> Self {
                Self::$svty(value)
            }
        }
    };
}

impl_into_scalarvalue!(f32, F32);
impl_into_scalarvalue!(u32, U32);
impl_into_scalarvalue!(u16, U16);
impl_into_scalarvalue!(u8, U8);
impl_into_scalarvalue!(i32, I32);
impl_into_scalarvalue!(i16, I16);
impl_into_scalarvalue!(i8, I8);
impl_into_scalarvalue!(bool, BOOL);

#[derive(Debug)]
pub enum ScalarArrayRef<'a> {
    F32ARRAY(&'a [f32]),
    U32ARRAY(&'a [u32]),
    U16ARRAY(&'a [u16]),
    U8ARRAY(&'a [u8]),
    I32ARRAY(&'a [i32]),
    I16ARRAY(&'a [i16]),
    I8ARRAY(&'a [i8]),
    BOOLARRAY(&'a [bool]),
}

macro_rules! impl_into_scalararrayref {
    ($ty: ident, $saty: ident) => {
        impl<'a> From<&'a [$ty]> for ScalarArrayRef<'a> {
            fn from(value: &'a [$ty]) -> Self {
                ScalarArrayRef::$saty(value)
            }
        }
    };
}

impl_into_scalararrayref!(f32, F32ARRAY);
impl_into_scalararrayref!(i32, I32ARRAY);
impl_into_scalararrayref!(i16, I16ARRAY);
impl_into_scalararrayref!(i8, I8ARRAY);
impl_into_scalararrayref!(u32, U32ARRAY);
impl_into_scalararrayref!(u16, U16ARRAY);
impl_into_scalararrayref!(u8, U8ARRAY);
impl_into_scalararrayref!(bool, BOOLARRAY);

impl From<ScalarValue> for ArrowScalarValue {
    fn from(value: ScalarValue) -> Self {
        match value {
            ScalarValue::F32(x) => ArrowScalarValue::F32(x),
            ScalarValue::U32(x) => ArrowScalarValue::U32(x),
            ScalarValue::U16(x) => ArrowScalarValue::U16(x),
            ScalarValue::U8(x) => ArrowScalarValue::U8(x),
            ScalarValue::I32(x) => ArrowScalarValue::I32(x),
            ScalarValue::I16(x) => ArrowScalarValue::I16(x),
            ScalarValue::I8(x) => ArrowScalarValue::I8(x),
            ScalarValue::BOOL(x) => ArrowScalarValue::BOOL(x),
        }
    }
}

// TODO add scalar ndarray support ore remove ScalarArrayRef support
#[derive(Debug)]
pub enum Operand<'a> {
    Scalar(ScalarValue),
    ScalarArrayRef(ScalarArrayRef<'a>),
    NdArrayRef(&'a NdArray),
}

impl<'a> Operand<'a> {
    pub async fn into_nd_array(&self) -> &NdArray {
        match self {
            Operand::Scalar(_) => todo!(),
            Operand::ScalarArrayRef(_) => todo!(),
            Operand::NdArrayRef(x) => *x,
        }
    }
}

impl<'a> From<ScalarArrayRef<'a>> for Operand<'a> {
    fn from(value: ScalarArrayRef<'a>) -> Self {
        Operand::ScalarArrayRef(value)
    }
}

impl<'a> From<ScalarValue> for Operand<'a> {
    fn from(value: ScalarValue) -> Self {
        Operand::Scalar(value)
    }
}

impl<'a> From<&'a NdArray> for Operand<'a> {
    fn from(value: &'a NdArray) -> Self {
        Operand::NdArrayRef(value)
    }
}
