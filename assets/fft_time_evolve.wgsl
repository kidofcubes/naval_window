#import "fft_common.wgsl"::{FFT_SIZE, TAU, FftOceanSettings, complex_mul, complex_conj}

@group(0) @binding(0) var initial_spectrum: texture_2d<f32>;
@group(0) @binding(1) var initial_spectrum_conj: texture_2d<f32>;
@group(0) @binding(2) var angular_frequency: texture_2d<f32>;
@group(0) @binding(3) var evolved_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(4) var<uniform> settings: FftOceanSettings;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;
    if (row >= FFT_SIZE || col >= FFT_SIZE) {
        return;
    }

    let coord = vec2<i32>(i32(col), i32(row));

    // H0(k) and H0*(-k)
    let h0 = textureLoad(initial_spectrum, coord, 0).xy;
    let h0_conj = textureLoad(initial_spectrum_conj, coord, 0).xy;

    // omega(k)
    let omega = textureLoad(angular_frequency, coord, 0).x;

    // exp(i * omega * t)
    let phase = omega * settings.elapsed_time;
    let exp_pos = vec2<f32>(cos(phase), sin(phase));
    let exp_neg = vec2<f32>(cos(phase), -sin(phase)); // exp(-i * omega * t)

    // H(k, t) = H0(k) * exp(i*w*t) + H0*(-k) * exp(-i*w*t)
    let evolved = complex_mul(h0, exp_pos) + complex_mul(h0_conj, exp_neg);

    textureStore(evolved_out, coord, vec4<f32>(evolved, 0.0, 0.0));
}
