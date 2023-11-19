use crate::{NdArray, NdArrayError, NdArrayResult};
use arrow_gpu::{array::UInt32ArrayGPU, kernels::take_dyn};

fn generate_repeat_shape(shape: &[u32], repeats: &[u32], axis: u32) -> Vec<u32> {
    let mut new_shape = vec![];

    for i in 0..shape.len() {
        if axis as usize == i {
            if repeats.len() == 1 {
                new_shape.push(shape[i] * repeats[0])
            } else {
                new_shape.push(repeats.iter().sum())
            }
        } else {
            new_shape.push(shape[i])
        }
    }

    new_shape
}

fn generate_repeat_indexes(
    shape: &[u32],
    repeats: &[u32],
    axis: u32,
    base_index: &mut u32,
    depth: u32,
    indexes: &mut Vec<u32>,
) {
    if depth as usize == shape.len() - 1 {
        for i in 0..shape[shape.len() - 1] {
            indexes.push(*base_index + i);
            if axis == depth {
                (0..(repeats[i as usize % repeats.len()] - 1))
                    .for_each(|_| indexes.push(*base_index + i));
            }
        }
        *base_index += shape[shape.len() - 1];
    } else {
        for i in 0..shape[depth as usize] {
            let old_count = indexes.len();
            generate_repeat_indexes(shape, repeats, axis, base_index, depth + 1, indexes);
            let new_count = indexes.len();
            if axis == depth {
                (0..(repeats[i as usize % repeats.len()] - 1))
                    .for_each(|_| indexes.extend_from_within(old_count..new_count));
            }
        }
    }
}

/// Broadcast an array to a new shape
pub async fn repeat(arr: &NdArray, repeats: &[u32], axis: Option<u32>) -> NdArrayResult<NdArray> {
    let array_count = arr.shape.iter().product();
    let dtype = arr.data.get_dtype().into();
    match (repeats.len() as u32, axis) {
        (x, None) if (x != 1) && (x != array_count) => Err(NdArrayError::RepeatError(format!(
            "repeat count {} is not equal to array of count {}",
            repeats.len(),
            array_count
        ))),
        (x, Some(y)) if (x != 1) && arr.shape[y as usize] != x => {
            Err(NdArrayError::RepeatError(format!(
                "repeat count {} is not equal to array count {} at axis {}",
                repeats.len(),
                arr.shape[y as usize],
                y
            )))
        }
        (1, None) => {
            let final_length = array_count * repeats[0];
            let mut indexes = Vec::with_capacity(final_length as usize);

            for i in 0..array_count {
                for _ in 0..repeats[0] {
                    indexes.push(i);
                }
            }
            let indexes = UInt32ArrayGPU::from_slice(&indexes, arr.data.get_gpu_device());
            let dims = 1u16;
            let shape = vec![final_length];
            let data = take_dyn(&arr.data, &indexes).await;

            Ok(NdArray {
                shape,
                dims,
                data,
                dtype,
            })
        }
        (_, None) => {
            let final_length = repeats.iter().sum();
            let mut indexes = Vec::with_capacity(final_length as usize);

            repeats.iter().enumerate().for_each(|(idx, count)| {
                for _ in 0..*count {
                    indexes.push(idx as u32)
                }
            });
            let indexes = UInt32ArrayGPU::from_slice(&indexes, arr.data.get_gpu_device());
            let dims = 1u16;
            let shape = vec![final_length];
            let data = take_dyn(&arr.data, &indexes).await;

            Ok(NdArray {
                shape,
                dims,
                data,
                dtype,
            })
        }
        (_, Some(y)) => {
            let mut indexes = vec![];
            generate_repeat_indexes(&arr.shape, repeats, y, &mut 0, 0, &mut indexes);
            let indexes = UInt32ArrayGPU::from_slice(&indexes, arr.data.get_gpu_device());
            let dims = 1u16;
            let shape = generate_repeat_shape(&arr.shape, repeats, y);
            let data = take_dyn(&arr.data, &indexes).await;

            Ok(NdArray {
                shape,
                dims,
                data,
                dtype,
            })
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::NdArray;

    #[tokio::test]
    async fn test_repeat() {
        let input =
            NdArray::from_slice([1.0f32, 2.0, 3.0].as_ref().into(), vec![1, 1, 3], None).await;
        let new_gpu_array = repeat(&input, &vec![2], None).await.unwrap();
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0].into()
        );
        assert_eq!(&new_gpu_array.shape, &[6]);

        let input =
            NdArray::from_slice([1.0f32, 2.0, 3.0].as_ref().into(), vec![1, 1, 3], None).await;
        let new_gpu_array = repeat(&input, &vec![2, 1, 3], None).await.unwrap();
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 1.0, 2.0, 3.0, 3.0, 3.0].into()
        );
        assert_eq!(&new_gpu_array.shape, &[6]);
    }

    #[test]
    fn test_generate_repeat_indexes() {
        let mut result_indexes = vec![];
        generate_repeat_indexes(&[1, 1, 3], &[2], 1, &mut 0, 0, &mut result_indexes);
        assert_eq!(&result_indexes, &[0, 1, 2, 0, 1, 2]);

        result_indexes = vec![];
        generate_repeat_indexes(&[1, 2, 3], &[2, 1], 1, &mut 0, 0, &mut result_indexes);
        assert_eq!(&result_indexes, &[0, 1, 2, 0, 1, 2, 3, 4, 5]);

        result_indexes = vec![];
        generate_repeat_indexes(&[3, 1], &[2], 1, &mut 0, 0, &mut result_indexes);
        assert_eq!(&result_indexes, &[0, 0, 1, 1, 2, 2]);

        result_indexes = vec![];
        generate_repeat_indexes(&[3, 1], &[2, 3, 1], 0, &mut 0, 0, &mut result_indexes);
        assert_eq!(&result_indexes, &[0, 0, 1, 1, 1, 2]);
    }

    #[tokio::test]
    async fn test_repeat_with_axis() {
        let input =
            NdArray::from_slice([1.0f32, 2.0, 3.0].as_ref().into(), vec![1, 1, 3], None).await;
        let new_gpu_array = repeat(&input, &vec![2], Some(2)).await.unwrap();
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0].into()
        );
        assert_eq!(&new_gpu_array.shape, &[1, 1, 6]);

        let input =
            NdArray::from_slice([1.0f32, 2.0, 3.0].as_ref().into(), vec![1, 1, 3], None).await;
        let new_gpu_array = repeat(&input, &vec![1, 2, 3], Some(2)).await.unwrap();
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 2.0, 2.0, 3.0, 3.0, 3.0].into()
        );
        assert_eq!(&new_gpu_array.shape, &[1, 1, 6]);

        let input =
            NdArray::from_slice([1.0f32, 2.0, 3.0].as_ref().into(), vec![1, 1, 3], None).await;
        let new_gpu_array = repeat(&input, &vec![2], Some(1)).await.unwrap();
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0].into()
        );
        assert_eq!(&new_gpu_array.shape, &[1, 2, 3]);

        let input = NdArray::from_slice(
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].as_ref().into(),
            vec![1, 2, 3],
            None,
        )
        .await;
        let new_gpu_array = repeat(&input, &vec![2, 1], Some(1)).await.unwrap();
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into()
        );
        assert_eq!(&new_gpu_array.shape, &[1, 3, 3]);
    }
}
