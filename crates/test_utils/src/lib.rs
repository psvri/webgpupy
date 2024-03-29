use arrow_gpu::utils::ScalarArray;

pub fn float_eq_in_error(left: f32, right: f32) -> bool {
    if (left.is_nan() && !right.is_nan()) || (right.is_nan() && !left.is_nan()) {
        return false;
    }
    if left.is_nan() && right.is_nan() {
        return true;
    }
    if (right == f32::NEG_INFINITY && left != f32::NEG_INFINITY)
        || (left == f32::NEG_INFINITY && right != f32::NEG_INFINITY)
    {
        return false;
    }
    if (left == f32::INFINITY && right != f32::INFINITY)
        || (right == f32::INFINITY && left != f32::INFINITY)
    {
        return false;
    }
    if (left.abs() - right.abs()).abs() > 0.01 {
        return false;
    }
    true
}

pub fn float_slice_eq_in_error(v1: ScalarArray, v2: ScalarArray) {
    match (v1, v2) {
        (ScalarArray::F32Vec(x), ScalarArray::F32Vec(y)) => {
            for i in 0..x.len() {
                if !float_eq_in_error(x[i], x[i]) {
                    panic!(
                        "assertion failed: `(left {} == right {}) \n left: `{:?}` \n right: `{:?}`",
                        x[i], y[i], x, y
                    );
                }
            }
        }
        _ => todo!(),
    }
}

#[macro_export]
macro_rules! test_ufunc_nin1_nout1_f32 {
    ($fn_name: ident, $input: expr, $output: expr, $mask: expr, $ufunc: ident) => {
        #[test]
        fn $fn_name() {
            let data = $input;
            let input_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let mask_array = NdArray::from_slice(
                $mask.as_slice().into(),
                vec![$mask.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let new_gpu_array = $ufunc(&input_ndarray, Some(&mask_array), None);
            if let ArrowArrayGPU::Float32ArrayGPU(x) = new_gpu_array.data {
                let new_values = x.raw_values().unwrap();
                for (index, new_value) in (&new_values).iter().enumerate() {
                    if !float_eq_in_error($output[index], *new_value) {
                        panic!(
                            "assertion failed: `(left {} == right {}) \n left: `{:?}` \n right: `{:?}`",
                            $output[index], *new_value, $output, new_values
                        );
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! test_ufunc_nin1_nout1 {
    ($fn_name: ident, $input: expr, $mask: expr, $output: expr, $output_ty: ident, $ufunc: ident) => {
        #[test]
        fn $fn_name() {
            let data = $input;
            let input_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let mask_array = NdArray::from_slice(
                $mask.as_slice().into(),
                vec![$mask.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let new_gpu_array = $ufunc(&input_ndarray, Some(&mask_array), None);
            if let ArrowArrayGPU::$output_ty(x) = new_gpu_array.data {
                let new_values = x.raw_values().unwrap();
                assert_eq!(&new_values, &$output);
            }
        }
    };
    ($fn_name: ident, $input: expr, $output: expr, $output_ty: ident, $ufunc: ident) => {
        #[test]
        fn $fn_name() {
            let data = $input;
            let input_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let new_gpu_array = $ufunc(&input_ndarray, None, None);
            if let ArrowArrayGPU::$output_ty(x) = new_gpu_array.data {
                let new_values = x.raw_values().unwrap();
                assert_eq!(&new_values, &$output);
            }
        }
    };
}

#[macro_export]
macro_rules! test_ufunc_nin2_nout1_f32 {
    ($fn_name: ident, $input_1: expr, $input_2: expr,$output: expr, $mask: expr, $ufunc: ident) => {
        #[test]
        async fn $fn_name() {
            let data = $input_1;
            let input1_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let data = $input_2;
            let input1_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let mask_array = NdArray::from_slice(
                $mask.as_slice().into(),
                vec![$mask.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let new_gpu_array = $ufunc(&input1_ndarray,&input1_ndarray, Some(&mask_array), None);
            if let ArrowArrayGPU::Float32ArrayGPU(x) = new_gpu_array.data {
                let new_values = x.raw_values().unwrap();
                for (index, new_value) in (&new_values).iter().enumerate() {
                    if !float_eq_in_error($output[index], *new_value) {
                        panic!(
                            "assertion failed: `(left {} == right {}) \n left: `{:?}` \n right: `{:?}`",
                            $output[index], *new_value, $output, new_values
                        );
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! test_ufunc_nin2_nout1 {
    ($fn_name: ident, $input_1: expr, $input_2: expr,  $mask: expr, $output: expr, $output_ty: ident, $ufunc: ident) => {
        #[test]
        fn $fn_name() {
            let data = $input_1;
            let input1_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let data = $input_2;
            let input2_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let mask_array = NdArray::from_slice(
                $mask.as_slice().into(),
                vec![$mask.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let new_gpu_array = $ufunc(&input1_ndarray, &input2_ndarray, Some(&mask_array), None);
            if let ArrowArrayGPU::$output_ty(x) = new_gpu_array.data {
                let new_values = x.raw_values().unwrap();
                assert_eq!(&new_values, &$output);
            }
        }
    };
    ($fn_name: ident, $input_1: expr, $input_2: expr, $output: expr, $output_ty: ident, $ufunc: ident) => {
        #[test]
        fn $fn_name() {
            let data = $input_1;
            let input1_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let data = $input_2;
            let input2_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            );
            let new_gpu_array = $ufunc(&input1_ndarray, &input2_ndarray, None, None);
            if let ArrowArrayGPU::$output_ty(x) = new_gpu_array.data {
                let new_values = x.raw_values().unwrap();
                assert_eq!(&new_values, &$output);
            }
        }
    };
}
