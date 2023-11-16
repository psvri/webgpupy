use arrow_gpu::{array::ArrowArrayGPU, kernels::merge_dyn};

use crate::{
    broadcast::{broadcast_shape, broadcast_to},
    NdArray,
};

pub async fn where_(mask: &NdArray, x: &NdArray, y: &NdArray) -> NdArray {
    //TODO broadcast mask as well
    if let ArrowArrayGPU::BooleanArrayGPU(bool_mask) = &mask.data {
        let broadcast_shape =
            broadcast_shape(&mask.shape, &broadcast_shape(&x.shape, &y.shape).unwrap()).unwrap();
        let broadcasted_x = broadcast_to(x, &broadcast_shape).await;
        let broadcasted_y = broadcast_to(y, &broadcast_shape).await;
        let merged_array = merge_dyn(&broadcasted_x.data, &broadcasted_y.data, bool_mask).await;
        NdArray {
            shape: x.shape.clone(),
            dims: x.dims,
            data: merged_array,
            dtype: x.dtype,
        }
    } else {
        panic!("Mask is not of boolean type")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::NdArray;

    #[tokio::test]
    async fn test_where() {
        let input_1 =
            NdArray::from_slice([1.0f32, 2.0, 3.0].as_ref().into(), vec![1, 1, 3], None).await;
        let input_2 = NdArray::from_slice([10.0f32].as_ref().into(), vec![1], None).await;
        let mask =
            NdArray::from_slice([true, false, false].as_ref().into(), vec![1, 1, 3], None).await;
        let new_gpu_array = where_(&mask, &input_1, &input_2).await;
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 10.0, 10.0].into()
        );

        let input_1 =
            NdArray::from_slice([1.0f32, 2.0, 3.0].as_ref().into(), vec![1, 1, 3], None).await;
        let input_2 = NdArray::from_slice([10.0f32].as_ref().into(), vec![1], None).await;
        let mask =
            NdArray::from_slice([false, false, false].as_ref().into(), vec![1, 1, 3], None).await;
        let new_gpu_array = where_(&mask, &input_1, &input_2).await;
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![10.0f32, 10.0, 10.0].into()
        );
    }
}
