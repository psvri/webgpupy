use std::ops::{Range, RangeFrom, RangeTo};

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
            ArrowType::BooleanType => Dtype::Bool,
            _ => panic!("Unsupported dtype"),
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
            ArrowType::BooleanType => Dtype::Bool,
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

impl From<Dtype> for ArrowType {
    fn from(val: Dtype) -> Self {
        match val {
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

// We are using i32 here to handle cases like [10:2:-1]
#[derive(Debug)]
pub struct IndexSlice {
    pub start: u32,
    pub stop: u32,
    pub step: i32,
}

impl IndexSlice {
    pub fn iterate(&self) -> IndexSliceIter {
        IndexSliceIter::new(self)
    }

    pub fn new(mut start: i64, mut stop: i64, step: i64, length: u32) -> Result<Self, String> {
        let length = length as i64;
        if start > length || stop > length {
            return Err("Index greater than length".to_string());
        }
        if start < 0 {
            start += length;
        }
        if stop < 0 {
            stop += length;
        }
        Ok(Self {
            start: start.try_into().unwrap(),
            stop: stop.try_into().unwrap(),
            step: step.try_into().unwrap(),
        })
    }

    pub fn element_count(&self) -> u32 {
        let step = (self.step).abs() as u32;
        let diff = (self.start).abs_diff(self.stop);
        let div_result = diff / step;
        if diff % step == 0 {
            div_result
        } else {
            div_result + 1
        }
    }
}

#[derive(Debug)]
pub struct IndexSliceIter<'a> {
    index_slice: &'a IndexSlice,
    current_pos: u32,
}

impl<'a> IndexSliceIter<'a> {
    pub fn new(index_slice: &'a IndexSlice) -> Self {
        Self {
            index_slice,
            current_pos: index_slice.start,
        }
    }
}

impl Iterator for IndexSliceIter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos != self.index_slice.stop {
            let pos = self.current_pos;
            if self.index_slice.step.is_negative() {
                self.current_pos -= self.index_slice.step.abs() as u32;
            } else {
                self.current_pos += self.index_slice.step as u32;
            }

            Some(pos)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub enum IndexSliceOp {
    Index(i64),
    Range(Range<i64>),
    RangeFrom(RangeFrom<i64>),
    RangeTo(RangeTo<i64>),
    RangeWithStep(Range<i64>, i32),
    RangeFromWithStep(RangeFrom<i64>, i32),
    RangeToWithStep(RangeTo<i64>, i32),
}

pub trait IntoIndexSliceOp {
    fn into_slice_op(self) -> IndexSliceOp;
}

impl IntoIndexSliceOp for i64 {
    fn into_slice_op(self) -> IndexSliceOp {
        IndexSliceOp::Index(self)
    }
}

impl IntoIndexSliceOp for Range<i64> {
    fn into_slice_op(self) -> IndexSliceOp {
        IndexSliceOp::Range(self)
    }
}

impl IntoIndexSliceOp for RangeFrom<i64> {
    fn into_slice_op(self) -> IndexSliceOp {
        IndexSliceOp::RangeFrom(self)
    }
}

impl From<i64> for IndexSliceOp {
    fn from(value: i64) -> Self {
        Self::Index(value)
    }
}

impl From<Range<i64>> for IndexSliceOp {
    fn from(value: Range<i64>) -> Self {
        Self::Range(value)
    }
}

impl From<RangeFrom<i64>> for IndexSliceOp {
    fn from(value: RangeFrom<i64>) -> Self {
        Self::RangeFrom(value)
    }
}

impl From<(Range<i64>, i32)> for IndexSliceOp {
    fn from(value: (Range<i64>, i32)) -> Self {
        Self::RangeWithStep(value.0, value.1)
    }
}

impl IndexSliceOp {
    pub fn into_index_slice(&self, length: u32) -> IndexSlice {
        match self {
            IndexSliceOp::Index(x) => IndexSlice::new(*x, *x + 1, 1, length).unwrap(),
            IndexSliceOp::Range(x) => IndexSlice::new(x.start, x.end, 1, length).unwrap(),
            IndexSliceOp::RangeWithStep(x, step) => {
                IndexSlice::new(x.start, x.end, (*step).into(), length).unwrap()
            }
            IndexSliceOp::RangeFrom(x) => {
                IndexSlice::new(x.start, length.into(), 1, length).unwrap()
            }
            IndexSliceOp::RangeTo(_) => todo!(),
            IndexSliceOp::RangeFromWithStep(_, _) => todo!(),
            IndexSliceOp::RangeToWithStep(_, _) => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::IndexSlice;

    #[test]
    fn test_index_slice_iter() {
        let index_slice = IndexSlice::new(0, 4, 1, 10).unwrap();
        assert_eq!(
            [0, 1, 2, 3].as_ref(),
            &index_slice.iterate().collect::<Vec<u32>>()
        );

        let index_slice = IndexSlice::new(0, 4, 2, 10).unwrap();
        assert_eq!(
            [0, 2].as_ref(),
            &index_slice.iterate().collect::<Vec<u32>>()
        );
    }

    #[test]
    fn test_index_slice_iter_rev() {
        let index_slice = IndexSlice::new(-3, -1, 1, 10).unwrap();
        assert_eq!(
            [7, 8].as_ref(),
            &index_slice.iterate().collect::<Vec<u32>>()
        );
    }
}
