use arrow_gpu::kernels::*;
use webgpupy_core::{ufunc_nin2_nout1, ufunc_nin2_nout1_body, Dtype, NdArray};

ufunc_nin2_nout1_body!(multiply, mul_op_dyn);
ufunc_nin2_nout1_body!(divide, div_op_dyn);
ufunc_nin2_nout1_body!(add, add_op_dyn);
ufunc_nin2_nout1_body!(subtract, sub_op_dyn);

#[cfg(test)]
mod test {
    use webgpupy_core::NdArray;

    use crate::multiply;

    #[test]
    fn test_multiply() {
        let arr1 = NdArray::from_slice([1.0f32, 2.0, 3.0].as_ref().into(), vec![3], None);
        let arr2 = NdArray::from_slice([1.0, 2.0].as_ref().into(), vec![1, 2, 1], None);

        let result = multiply(&arr1, &arr2, None, None);

        assert_eq!(result.shape, [1, 2, 3]);

        assert_eq!(
            result.data.get_raw_values(),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0].into()
        );
    }
}
