use arrow_gpu::array::ArrowType;

use crate::ScalarValue;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Dtype {
    Int8,
    Int16,
    Int32,
    UInt8,
    UInt16,
    UInt32,
    Float32,
    Bool,
}

impl From<ArrowType> for Dtype {
    fn from(value: ArrowType) -> Self {
        match value {
            ArrowType::Float32Type => Dtype::Float32,
            ArrowType::UInt32Type => Dtype::UInt32,
            ArrowType::UInt16Type => Dtype::UInt16,
            ArrowType::UInt8Type => Dtype::UInt8,
            ArrowType::Int32Type => Dtype::Int32,
            ArrowType::Int16Type => Dtype::Int16,
            ArrowType::Int8Type => Dtype::Int8,
            _ => panic!("Unsupported type"),
        }
    }
}

impl From<&ArrowType> for Dtype {
    fn from(value: &ArrowType) -> Self {
        match *value {
            ArrowType::Float32Type => Dtype::Float32,
            ArrowType::UInt32Type => Dtype::UInt32,
            ArrowType::UInt16Type => Dtype::UInt16,
            ArrowType::UInt8Type => Dtype::UInt8,
            ArrowType::Int32Type => Dtype::Int32,
            ArrowType::Int16Type => Dtype::Int16,
            ArrowType::Int8Type => Dtype::Int8,
            _ => panic!("Unsupported type"),
        }
    }
}

impl<'a> From<&'a str> for Dtype {
    fn from(value: &'a str) -> Self {
        match value {
            "float" | "float32" => Dtype::Float32,
            "uint32" => Dtype::UInt32,
            "uint16" => Dtype::UInt16,
            "uint8" => Dtype::UInt8,
            "int32" => Dtype::Int32,
            "int16" => Dtype::Int16,
            "int8" => Dtype::Int8,
            "bool" => Dtype::Bool,
            _ => panic!("Unsupported type"),
        }
    }
}

impl Into<ArrowType> for Dtype {
    fn into(self) -> ArrowType {
        match self {
            Dtype::Int8 => ArrowType::Int8Type,
            Dtype::Int16 => ArrowType::Int16Type,
            Dtype::Int32 => ArrowType::Int32Type,
            Dtype::UInt8 => ArrowType::UInt8Type,
            Dtype::UInt16 => ArrowType::UInt16Type,
            Dtype::UInt32 => ArrowType::UInt32Type,
            Dtype::Float32 => ArrowType::Float32Type,
            Dtype::Bool => ArrowType::BooleanType,
        }
    }
}

impl<'a> From<&'a Dtype> for &'a ArrowType {
    fn from(value: &'a Dtype) -> Self {
        match value {
            Dtype::Float32 => &ArrowType::Float32Type,
            Dtype::UInt32 => &ArrowType::UInt32Type,
            Dtype::UInt16 => &ArrowType::UInt16Type,
            Dtype::UInt8 => &ArrowType::UInt8Type,
            Dtype::Int32 => &ArrowType::Int32Type,
            Dtype::Int16 => &ArrowType::Int16Type,
            Dtype::Int8 => &ArrowType::Int8Type,
            _ => panic!("Unsupported type"),
        }
    }
}

impl<'a> From<&'a ScalarValue> for Dtype {
    fn from(value: &'a ScalarValue) -> Self {
        match value {
            ScalarValue::F32(_) => Dtype::Float32,
            ScalarValue::U32(_) => Dtype::UInt32,
            ScalarValue::U16(_) => Dtype::UInt16,
            ScalarValue::U8(_) => Dtype::UInt8,
            ScalarValue::I32(_) => Dtype::Int32,
            ScalarValue::I16(_) => Dtype::Int16,
            ScalarValue::I8(_) => Dtype::Int8,
            ScalarValue::BOOL(_) => Dtype::Bool,
        }
    }
}

#[derive(Debug)]
pub enum OperandType {
    NdArrayType(Dtype),
    ScalarType(Dtype),
}

#[derive(Debug)]
pub enum UfuncType {
    UfuncNin1Nout1Type([OperandType; 1], OperandType),
    UfuncNin2Nout1Type([OperandType; 2], OperandType),
}
