use arrow_gpu::{
    array::{ArrayUtils, ArrowArrayGPU, ArrowType, Float32ArrayGPU, UInt32ArrayGPU},
    gpu_utils::*,
    kernels::{
        add_op_dyn, bitcast_op_dyn, broadcast::Broadcast, max_op_dyn, mul_op_dyn, sub_op_dyn,
        ArrowScalarMul, Logical,
    },
};
use std::{f32::consts::SQRT_2, fmt::Debug, sync::Arc};
use webgpupy_core::{full, ones, zeros, Dtype, NdArray, ScalarValue, GPU_DEVICE};
use wgpu::Buffer;

use crate::{iota::*, threefry::*};

pub trait Generator {
    fn get_gpu_device(&self) -> Arc<GpuDevice>;

    fn random_bit_op(&mut self, shape: &[u32], pipeline: &mut ArrowComputePipeline) -> Buffer;

    fn random_bit(&mut self, shape: &[u32]) -> Buffer;

    fn random(&mut self, shape: &[u32]) -> NdArray {
        let min_values = zeros(
            shape.to_vec(),
            Some(Dtype::Float32),
            Some(self.get_gpu_device()),
        );
        let max_values = ones(
            shape.to_vec(),
            Some(Dtype::Float32),
            Some(self.get_gpu_device()),
        );
        self.uniform(shape, &min_values, &max_values)
    }

    fn uniform(&mut self, shape: &[u32], min_value: &NdArray, max_value: &NdArray) -> NdArray {
        let mut pipeline = ArrowComputePipeline::new(self.get_gpu_device(), Some("uniform"));
        let result = self.uniform_op(shape, min_value, max_value, &mut pipeline);
        pipeline.finish();
        result
    }

    fn uniform_op(
        &mut self,
        shape: &[u32],
        min_value: &NdArray,
        max_value: &NdArray,
        pipeline: &mut ArrowComputePipeline,
    ) -> NdArray;

    fn normal(&mut self, shape: &[u32]) -> NdArray;
}

pub struct ThreeFry2x32 {
    new_key: Buffer,
    input_key: Buffer,
    gpu_device: Arc<GpuDevice>,
}

impl Debug for ThreeFry2x32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreeFry2x32")
            .field(
                "new_key",
                &Self::read_buffer(&self.gpu_device, &self.input_key),
            )
            .field(
                "input_key",
                &Self::read_buffer(&self.gpu_device, &self.input_key),
            )
            .field("gpu_device", &self.gpu_device)
            .finish()
    }
}

impl ThreeFry2x32 {
    pub fn new(key: u32, gpu_device: Option<Arc<GpuDevice>>) -> Self {
        let gpu_device = gpu_device.unwrap_or(GPU_DEVICE.clone());
        let new_key = gpu_device.create_gpu_buffer_with_data(&[0, key]);
        let input_key = gpu_device.create_gpu_buffer_with_data(&[0, key]);
        Self {
            new_key,
            input_key,
            gpu_device,
        }
    }

    fn read_buffer(device: &GpuDevice, buffer: &Buffer) -> Vec<u32> {
        bytemuck::cast_slice(&device.retrive_data(buffer)).to_vec()
    }

    pub fn get_keys(&self) -> [[u32; 2]; 2] {
        [
            Self::read_buffer(&self.gpu_device, &self.new_key)
                .try_into()
                .unwrap(),
            Self::read_buffer(&self.gpu_device, &self.input_key)
                .try_into()
                .unwrap(),
        ]
    }
}

impl Generator for ThreeFry2x32 {
    fn random_bit_op(&mut self, shape: &[u32], pipeline: &mut ArrowComputePipeline) -> Buffer {
        let rand_input = iota_op(shape, pipeline);
        let key_input = iota_op(&[4], pipeline);
        let rand_buffer = threefry2x32_op(&self.input_key, &rand_input, pipeline);
        let new_keys_buffer = threefry2x32_op(&self.input_key, &key_input, pipeline);

        let new_key = self.gpu_device.create_empty_buffer(2 * 4);
        let input_key = self.gpu_device.create_empty_buffer(2 * 4);

        pipeline.copy_buffer_to_buffer(&new_keys_buffer, 0, &new_key, 0, new_key.size());

        pipeline.copy_buffer_to_buffer(&new_keys_buffer, 2 * 4, &input_key, 0, new_key.size());

        self.new_key = new_key;
        self.input_key = input_key;

        rand_buffer
    }

    fn random_bit(&mut self, shape: &[u32]) -> Buffer {
        let mut pipeline = ArrowComputePipeline::new(self.gpu_device.clone(), Some("random_bits"));
        let result = self.random_bit_op(shape, &mut pipeline);
        pipeline.finish();
        result
    }

    //TODO support other dtypes, other min and max values of array type, odd shaped like &[5]
    fn uniform_op(
        &mut self,
        shape: &[u32],
        min_value: &NdArray,
        max_value: &NdArray,
        pipeline: &mut ArrowComputePipeline,
    ) -> NdArray {
        let rand_buffer = self.random_bit_op(shape, pipeline);
        let count: u32 = shape.iter().product();
        let data = Arc::new(rand_buffer);
        let len = (&data.size() / 4) as usize;
        let data = UInt32ArrayGPU {
            data,
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len,
            null_buffer: None,
        };

        let ones = UInt32ArrayGPU::broadcast_op(1.0f32.to_bits(), count as usize, pipeline);
        let shift_right = UInt32ArrayGPU::broadcast_op(32 - 23, count as usize, pipeline);
        let shr = data.bitwise_shr_op(&shift_right, pipeline);
        let float_bits = shr.bitwise_or_op(&ones, pipeline).into();

        let mut floats = bitcast_op_dyn(&float_bits, &ArrowType::Float32Type, pipeline);
        let one = Float32ArrayGPU::from_slice(&[1.0f32], self.get_gpu_device()).into();
        floats = sub_op_dyn(&floats, &one, pipeline);

        let min_values = &min_value.data;
        let max_values = &max_value.data;

        let diff = sub_op_dyn(max_values, min_values, pipeline);

        floats = mul_op_dyn(&floats, &diff, pipeline);
        floats = add_op_dyn(&floats, min_values, pipeline);

        let data = max_op_dyn(min_values, &floats, pipeline);

        let shape = shape.to_vec();
        let dims = shape.len() as u16;
        NdArray {
            shape,
            dims,
            data,
            dtype: Dtype::UInt32,
        }
    }

    fn get_gpu_device(&self) -> Arc<GpuDevice> {
        self.gpu_device.clone()
    }

    fn normal(&mut self, shape: &[u32]) -> NdArray {
        let mut pipeline = ArrowComputePipeline::new(self.get_gpu_device(), Some("normal"));
        let lo = full(
            shape.to_vec(),
            ScalarValue::F32(-1.0f32 + f32::EPSILON).into(),
            Some(Dtype::Float32),
            Some(self.get_gpu_device()),
        );
        let hi = ones(
            shape.to_vec(),
            Some(Dtype::Float32),
            Some(self.get_gpu_device()),
        );
        let u = self.uniform_op(shape, &lo, &hi, &mut pipeline);
        let sqrt_ans = Float32ArrayGPU::broadcast_op(SQRT_2, 1, &mut pipeline);

        if let ArrowArrayGPU::Float32ArrayGPU(x) = u.data {
            let data = pipeline.apply_unary_function(
                &x.data,
                x.data.size(),
                include_str!("../compute_shader/erfinv.wgsl"),
                "erfinv",
                x.len.div_ceil(256) as u32,
            );
            let result_arr = Float32ArrayGPU {
                data: Arc::new(data),
                gpu_device: x.get_gpu_device(),
                phantom: std::marker::PhantomData,
                len: x.len,
                null_buffer: None,
            };

            let data = result_arr.mul_scalar_op(&sqrt_ans, &mut pipeline).into();

            pipeline.finish();

            NdArray {
                shape: shape.to_vec(),
                dims: shape.len() as u16,
                data,
                dtype: Dtype::Float32,
            }
        } else {
            panic!()
        }
    }
}

#[cfg(test)]
mod test {

    use test_utils::float_slice_eq_in_error;
    use webgpupy_core::GPU_DEVICE;

    use super::*;

    #[test]
    fn test_threefry_generator() {
        let gpu_device = GPU_DEVICE.clone();
        let mut rng = ThreeFry2x32::new(1701, None);
        let rands = rng.random_bit(&[4]);

        assert_eq!(
            bytemuck::cast_slice::<u8, u32>(&gpu_device.retrive_data(&rands)),
            &vec![56197195u32, 1801093307, 961309823, 1704866707]
        );

        assert_eq!(
            rng.get_keys(),
            [[56197195u32, 1801093307], [961309823, 1704866707]]
        )
    }

    #[test]
    fn test_threefry_random() {
        let mut rng = ThreeFry2x32::new(1701, None);

        float_slice_eq_in_error(
            rng.random(&[4]).data.get_raw_values(),
            vec![0.013084412, 0.41934967, 0.22382236, 0.39694512].into(),
        );

        rng = ThreeFry2x32::new(1701, None);
        float_slice_eq_in_error(
            rng.random(&[6]).data.get_raw_values(),
            vec![
                0.21588242, 0.911929, 0.42731953, 0.15760279, 0.7366872, 0.9338119,
            ]
            .into(),
        );
    }

    #[test]
    fn test_threefry_normal() {
        let mut rng = ThreeFry2x32::new(1701, None);

        float_slice_eq_in_error(
            rng.normal(&[4]).data.get_raw_values(),
            vec![-2.2236953, -0.20355737, -0.7593474, -0.26126227].into(),
        );
    }
}
