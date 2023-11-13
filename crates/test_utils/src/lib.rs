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

#[macro_export]
macro_rules! test_ufunc_nin1_nout1_f32 {
    ($fn_name: ident, $input_ty: expr, $output_ty: expr, $input: expr, $output: expr, $mask: expr, $ufunc: ident) => {
        #[tokio::test]
        async fn $fn_name() {
            let data = $input;
            let input_ndarray = NdArray::from_slice(
                data.as_slice().into(),
                vec![data.len() as u32],
                Some(GPU_DEVICE.clone()),
            )
            .await;
            let mask_array = NdArray::from_slice(
                $mask.as_slice().into(),
                vec![$mask.len() as u32],
                Some(GPU_DEVICE.clone()),
            )
            .await;
            let new_gpu_array = $ufunc(&input_ndarray, Some(&mask_array), None).await;
            if let ArrowArrayGPU::Float32ArrayGPU(x) = new_gpu_array.data {
                let new_values = x.raw_values().await.unwrap();
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
