use arrow_gpu::{gpu_utils::ArrowComputePipeline, kernels::*};
use webgpupy_core::{
    broadcast_shape, broadcast_shapes, broadcast_to_op, ufunc_nin1_nout1, ufunc_nin1_nout1_body,
    ufunc_nin2_nout1, ufunc_nin2_nout1_body, Dtype, NdArray,
};

ufunc_nin1_nout1_body!(sqrt, sqrt_op_dyn);
ufunc_nin1_nout1_body!(cbrt, cbrt_op_dyn);
ufunc_nin1_nout1_body!(absolute, abs_op_dyn);
ufunc_nin2_nout1_body!(power, power_op_dyn);

pub fn clip(a: &NdArray, a_min: Option<&NdArray>, a_max: Option<&NdArray>) -> NdArray {
    match (a_min, a_max) {
        (None, None) => panic!("Both a_min and a_max cannot be null"),
        (None, Some(max_values)) => {
            let mut pipeline = ArrowComputePipeline::new(a.get_gpu_device(), None);
            let broadcasted_shape = broadcast_shape(&a.shape, &max_values.shape).unwrap();

            let temp1;
            let mut max_arr = max_values;
            if max_values.shape != broadcasted_shape {
                temp1 = broadcast_to_op(max_arr, &broadcasted_shape, &mut pipeline);
                max_arr = &temp1;
            };

            let temp2;
            let mut arr = a;
            if arr.shape != broadcasted_shape {
                temp2 = broadcast_to_op(arr, &broadcasted_shape, &mut pipeline);
                arr = &temp2;
            };

            let data = min_op_dyn(&arr.data, &max_arr.data, &mut pipeline);
            pipeline.finish();
            NdArray {
                shape: broadcasted_shape,
                dims: a.dims,
                data,
                dtype: a.dtype,
            }
        }
        (Some(min_values), None) => {
            let mut pipeline = ArrowComputePipeline::new(a.get_gpu_device(), None);
            let broadcasted_shape = broadcast_shape(&a.shape, &min_values.shape).unwrap();

            let temp1;
            let mut min_arr = min_values;
            if min_values.shape != broadcasted_shape {
                temp1 = broadcast_to_op(min_arr, &broadcasted_shape, &mut pipeline);
                min_arr = &temp1;
            };

            let temp2;
            let mut arr = a;
            if arr.shape != broadcasted_shape {
                temp2 = broadcast_to_op(arr, &broadcasted_shape, &mut pipeline);
                arr = &temp2;
            };

            let data = max_op_dyn(&arr.data, &min_arr.data, &mut pipeline);
            pipeline.finish();
            NdArray {
                shape: broadcasted_shape,
                dims: a.dims,
                data,
                dtype: a.dtype,
            }
        }
        (Some(min_values), Some(max_values)) => {
            let mut pipeline = ArrowComputePipeline::new(a.get_gpu_device(), None);
            let broadcasted_shape =
                broadcast_shapes(&[&a.shape, &min_values.shape, &max_values.shape]).unwrap();

            let temp1;
            let mut min_arr = min_values;
            if min_values.shape != broadcasted_shape {
                temp1 = broadcast_to_op(min_arr, &broadcasted_shape, &mut pipeline);
                min_arr = &temp1;
            };

            let temp2;
            let mut arr = a;
            if arr.shape != broadcasted_shape {
                temp2 = broadcast_to_op(arr, &broadcasted_shape, &mut pipeline);
                arr = &temp2;
            };

            let temp3;
            let mut max_arr = max_values;
            if max_values.shape != broadcasted_shape {
                temp3 = broadcast_to_op(max_arr, &broadcasted_shape, &mut pipeline);
                max_arr = &temp3;
            };

            let mut data = max_op_dyn(&arr.data, &min_arr.data, &mut pipeline);
            data = min_op_dyn(&data, &max_arr.data, &mut pipeline);
            pipeline.finish();
            NdArray {
                shape: broadcasted_shape,
                dims: a.dims,
                data,
                dtype: a.dtype,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::vec;

    #[test]
    fn test_clip() {
        let arr1 = NdArray::from_slice([1.0, 2.0, 3.0, 100.0].as_ref().into(), vec![4], None);
        let min_values = NdArray::from_slice([2.0].as_ref().into(), vec![1], None);
        let max_values = NdArray::from_slice([50.0].as_ref().into(), vec![1], None);

        assert_eq!(
            clip(&arr1, Some(&min_values), None).data.get_raw_values(),
            vec![2.0, 2.0, 3.0, 100.0].into()
        );

        assert_eq!(
            clip(&arr1, None, Some(&max_values)).data.get_raw_values(),
            vec![1.0, 2.0, 3.0, 50.0].into()
        );

        assert_eq!(
            clip(&arr1, Some(&min_values), Some(&max_values))
                .data
                .get_raw_values(),
            vec![2.0, 2.0, 3.0, 50.0].into()
        );
    }
}
