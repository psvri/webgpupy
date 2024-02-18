use criterion::{black_box, criterion_group, criterion_main, Criterion};
use webgpupy_core::{NdArray, GPU_DEVICE};
use webgpupy_random::{Generator, ThreeFry2x32};

pub fn threefry_clean(fry: &mut ThreeFry2x32, shape: &[u32]) -> NdArray {
    fry.random(shape)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let device = GPU_DEVICE.clone();
    let mut fry = ThreeFry2x32::new(1701, Some(device.clone()));
    let shape = vec![1_000_000];
    let mut group = c.benchmark_group("trheefry");

    group.bench_function("threefry_clean", |b| {
        b.iter(|| threefry_clean(black_box(&mut fry), black_box(&shape)))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
