const FFT_SIZE: u32 = 256u;
const FFT_LOG2: u32 = 8u;
const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

struct FftOceanSettings {
    elapsed_time: f32,
    patch_length: f32,
    _padding_b: f32,
    _padding_c: f32,
}

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

fn complex_conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}
