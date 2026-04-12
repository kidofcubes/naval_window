#import "fft_common.wgsl"::{FFT_SIZE, TAU, FftOceanSettings, complex_mul, complex_conj}

@group(0) @binding(0) var initial_spectrum: texture_2d<f32>;
@group(0) @binding(1) var initial_spectrum_conj: texture_2d<f32>;
@group(0) @binding(2) var angular_frequency: texture_2d<f32>;
@group(0) @binding(3) var evolved_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(4) var<uniform> settings: FftOceanSettings;
@group(0) @binding(5) var slope_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(6) var disp_out:  texture_storage_2d<rg32float, write>;

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

    // Slope spectrum C(k) = Sx + i*Sy, packed so IFFT(C).real = slope_x, IFFT(C).imag = slope_z
    // k_row indexes world-x, k_col indexes world-z (matching pre_compute.rs convention)
    let k_row = TAU * (f32(row) - 128.0) / settings.patch_length;
    let k_col = TAU * (f32(col) - 128.0) / settings.patch_length;
    // Sx = i*k_row*H => Sx_r = -k_row*H_i, Sx_i = k_row*H_r
    // Sy = i*k_col*H => Sy_r = -k_col*H_i, Sy_i = k_col*H_r
    // C = Sx + i*Sy => C_r = Sx_r - Sy_i, C_i = Sx_i + Sy_r
    let c_r = -k_row * evolved.y - k_col * evolved.x;
    let c_i =  k_row * evolved.x - k_col * evolved.y;
    textureStore(slope_out, coord, vec4<f32>(c_r, c_i, 0.0, 0.0));

    // Horizontal displacement spectrum D(k) = C(k) / |k|
    // D = Dx + i*Dz, IFFT(D).rg = (dx, dz) spatial displacement
    let k_len = sqrt(k_row * k_row + k_col * k_col);
    if k_len > 1e-6 {
        textureStore(disp_out, coord, vec4<f32>(c_r / k_len, c_i / k_len, 0.0, 0.0));
    } else {
        textureStore(disp_out, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
