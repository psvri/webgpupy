use arrow_gpu::{array::UInt32ArrayGPU, kernels::put_dyn};

use crate::{zeros, NdArray};

fn recast_shape(shape: &Vec<u32>) -> Vec<u32> {
    if shape.len() == 2 {
        let mut new_shape = shape.clone();
        new_shape.push(1);
        new_shape
    } else {
        shape.clone()
    }
}

fn validate_and_genereate_new_shapes(tup: &[&NdArray]) -> Vec<Vec<u32>> {
    let mut new_shapes = vec![];
    new_shapes.push(recast_shape(&tup[0].shape));

    for i in 1..tup.len() {
        if tup[0].shape != tup[i].shape {
            panic!(
                "Cant dstack shapes {:?} and {:?}",
                tup[0].shape, tup[i].shape
            )
        }
        if tup[0].dtype != tup[i].dtype {
            panic!(
                "Cant dstack dtypes {:?} and {:?}",
                tup[0].dtype, tup[i].dtype
            )
        }
        new_shapes.push(recast_shape(&tup[i].shape));
    }


    new_shapes
}

// TODO broadcasting of shapes and validating
pub async fn dstack(tup: &[&NdArray]) -> NdArray {
    if tup.len() == 1 {
        tup[1].clone_array().await
    } else {
        // for i in 1..tup.len() {
        //     if tup[0].shape != tup[i].shape {
        //         panic!(
        //             "Cant dstack shapes {:?} and {:?}",
        //             tup[0].shape, tup[i].shape
        //         )
        //     }
        //     if tup[0].dtype != tup[i].dtype {
        //         panic!(
        //             "Cant dstack dtypes {:?} and {:?}",
        //             tup[0].dtype, tup[i].dtype
        //         )
        //     }
        // }

        //let new_array_count: u32 = tup.iter().map(|x| x.shape.iter().product::<u32>()).sum();
        let new_shapes = validate_and_genereate_new_shapes(tup);
        let mut new_shape = new_shapes[0].clone();
        let shape_len = new_shape.len();

        let last_dimension_size = new_shapes.iter().map(|x| x[x.len() - 1]).sum();
        new_shape[shape_len - 1] = last_dimension_size;
        let device = tup[0].data.get_gpu_device().clone();
        let mut new_array = zeros(new_shape, Some(tup[0].dtype), Some(device.clone())).await;

        let mut last_dimension = 0;
        for array in tup {
            let count = array.shape.iter().product();
            let src_indexes = (0..count).into_iter().collect::<Vec<u32>>();
            let dst_indexes = (0..count)
                .into_iter()
                .map(|x| last_dimension + (x * last_dimension_size))
                .collect::<Vec<u32>>();
            let src_indexes_gpu = UInt32ArrayGPU::from_vec(&src_indexes, device.clone());
            let dst_indexes_gpu = UInt32ArrayGPU::from_vec(&dst_indexes, device.clone());
            put_dyn(
                &array.data,
                &src_indexes_gpu,
                &mut new_array.data,
                &dst_indexes_gpu,
            )
            .await;
            last_dimension += array.shape[array.shape.len() - 1];
        }

        new_array
    }
}

#[cfg(test)]
mod tests {
    use crate::{dstack, NdArray};

    #[tokio::test]
    async fn test_dstack_2() {
        let input_1 =
            NdArray::from_slice([1.0f32, 2.0, 3.0, 4.0].as_ref().into(), vec![2, 2, 1], None).await;
        let input_2 =
            NdArray::from_slice([5.0f32, 6.0, 7.0, 8.0].as_ref().into(), vec![2, 2, 1], None).await;
        let new_gpu_array = dstack(&[&input_1, &input_2]).await;
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0].into()
        );
    }

    #[tokio::test]
    async fn test_dstack_3() {
        let input_1 =
            NdArray::from_slice([1.0f32, 2.0, 3.0, 4.0].as_ref().into(), vec![2, 2, 1], None).await;
        let input_2 =
            NdArray::from_slice([5.0f32, 6.0, 7.0, 8.0].as_ref().into(), vec![2, 2, 1], None).await;
        let input_3 = NdArray::from_slice(
            [9.0f32, 10.0, 11.0, 12.0].as_ref().into(),
            vec![2, 2, 1],
            None,
        )
        .await;
        let new_gpu_array = dstack(&[&input_1, &input_2, &input_3]).await;
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0].into()
        );
    }
}
