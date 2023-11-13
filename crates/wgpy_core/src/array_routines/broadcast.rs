use std::cmp::max;

use arrow_gpu::{array::UInt32ArrayGPU, kernels::take_dyn};

use crate::{NdArray, NdArrayError, NdArrayResult};

pub fn broadcast_shape(x: &[u32], y: &[u32]) -> NdArrayResult<Vec<u32>> {
    let max_shape_length = max(x.len(), y.len());
    let mut new_shape = vec![1; max_shape_length];

    for i in 0..max_shape_length {
        let v1 = if i < x.len() { x[x.len() - 1 - i] } else { 1 };
        let v2 = if i < y.len() { y[y.len() - 1 - i] } else { 1 };
        if v1 == v2 {
            new_shape[max_shape_length - 1 - i] = v1;
        } else if v1 == 1 {
            new_shape[max_shape_length - 1 - i] = v2;
        } else if v2 == 1 {
            new_shape[max_shape_length - 1 - i] = v1;
        } else {
            return Err(NdArrayError::BroadcastError(format!(
                "Cannot broadcast shapes {:?}, {:?}",
                x, y
            )));
        }
    }

    Ok(new_shape)
}

// TODO probably can simpify this
fn shape_to_indexes(to_shape: &[u32], from_shape: &[u32]) -> Vec<u32> {
    let mut indexes = vec![];

    let length = to_shape.len();

    let v1 = to_shape[length - 1];
    let v2 = from_shape[from_shape.len() - 1];

    if v1 == v2 {
        (0..v1).for_each(|x| indexes.push(x));
    } else {
        (0..v2).for_each(|x| indexes.push(x));
        for _ in 0..v1 - 1 {
            indexes.extend_from_within(0..v2 as usize);
        }
    }

    let mut continous_product = v1 as usize;
    let mut previous = v2;

    for i in 1..length {
        let v1 = to_shape[length - 1 - i];
        let v2 = if i < from_shape.len() {
            from_shape[from_shape.len() - 1 - i]
        } else {
            0
        };
        if v1 == v2 {
            for i in 1..v2 as usize {
                indexes.extend_from_within(0..continous_product);
                let new_length = indexes.len();
                for j in 0..continous_product {
                    indexes[new_length - 1 - j] += i as u32 * previous;
                }
            }
        } else {
            for _ in 0..(v1 - 1) as usize {
                indexes.extend_from_within(0..continous_product);
            }
        }
        continous_product *= v1 as usize;
        previous *= v2;
    }

    indexes
}

/// Broadcast an array to a new shape
pub async fn broadcast_to(x: &NdArray, shape: &[u32]) -> NdArray {
    let braodcasted_shape = broadcast_shape(&x.shape, shape).unwrap();

    let indexes = UInt32ArrayGPU::from_vec(
        &shape_to_indexes(&braodcasted_shape, &x.shape),
        x.data.get_gpu_device(),
    );
    let dims = braodcasted_shape.len() as u16;

    let new_data = take_dyn(&x.data, &indexes).await;
    let dtype = new_data.get_dtype().into();

    NdArray {
        shape: braodcasted_shape,
        dims,
        data: new_data,
        dtype,
    }
}

#[cfg(test)]
mod test {
    use super::{broadcast_shape, broadcast_to, shape_to_indexes};
    use crate::NdArray;

    #[test]
    fn test_broadcast() {
        assert_eq!(
            broadcast_shape(&vec![1, 1, 1], &vec![3]).unwrap(),
            vec![1, 1, 3]
        );
        assert_eq!(
            broadcast_shape(&vec![1, 2, 1], &vec![3]).unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            broadcast_shape(&vec![4, 1, 3], &vec![2, 3]).unwrap(),
            vec![4, 2, 3]
        );
        assert_eq!(
            broadcast_shape(&vec![1, 3], &vec![2, 1]).unwrap(),
            vec![2, 3]
        );
        assert!(broadcast_shape(&vec![2], &vec![3]).is_err())
    }

    #[test]
    fn test_shape_to_indexes() {
        assert_eq!(
            shape_to_indexes(&vec![1, 1, 3], &vec![1, 1, 1]),
            vec![0, 0, 0]
        );
        assert_eq!(
            shape_to_indexes(&vec![2, 3], &vec![1, 3]),
            vec![0, 1, 2, 0, 1, 2]
        );
        assert_eq!(
            shape_to_indexes(&vec![2, 3], &vec![2, 1]),
            vec![0, 0, 0, 1, 1, 1]
        );
        assert_eq!(
            shape_to_indexes(&vec![4, 2, 3], &vec![2, 3]),
            vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            shape_to_indexes(&vec![4, 2, 3], &vec![4, 1, 3]),
            vec![0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 9, 10, 11]
        );
    }

    #[tokio::test]
    async fn test_broadcast_to() {
        let input =
            NdArray::from_slice([1.0f32, 2.0, 3.0].as_ref().into(), vec![1, 1, 3], None).await;
        let new_gpu_array = broadcast_to(&input, &vec![1, 2, 3]).await;
        assert_eq!(
            new_gpu_array.data.get_raw_values().await,
            vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0].into()
        );
    }
}
