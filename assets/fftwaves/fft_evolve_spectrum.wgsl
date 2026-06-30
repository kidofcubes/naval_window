// Time-evolve the wave spectrum each frame; pack all 5 spectra
// (H, Dx, Dz, Sx, Sz) per cascade into one 2D-array texture.

#import "fftwaves/fft_common.wgsl"::{
    FFT_GRID_SIZE, FFT_CASCADE_COUNT, FFT_SPEC_H, FFT_SPEC_DX, FFT_SPEC_DZ,
    FFT_SPEC_SX, FFT_SPEC_SZ, spec_layer,
    FftSettings, FftCascadeBuffer,
    cmul, cmul_i, cmul_neg_i, cexp_i, k_for_cell,
}

@group(0) @binding(0) var h0_in:      texture_storage_2d_array<rg32float, read>;
@group(0) @binding(1) var h0_conj_in: texture_storage_2d_array<rg32float, read>;
@group(0) @binding(2) var omega_in:   texture_storage_2d_array<r32float,  read>;
@group(0) @binding(3) var spectra_out: texture_storage_2d_array<rg32float, write>;
@group(0) @binding(4) var<uniform> settings: FftSettings;
@group(0) @binding(5) var<uniform> cascades: FftCascadeBuffer;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let cascade = gid.z;
    if (row >= FFT_GRID_SIZE || col >= FFT_GRID_SIZE || cascade >= FFT_CASCADE_COUNT) {
        return;
    }
    let coord = vec2<i32>(i32(col), i32(row));
    let cascade_layer = i32(cascade);

    let h0      = textureLoad(h0_in,      coord, cascade_layer).xy;
    let h0_conj = textureLoad(h0_conj_in, coord, cascade_layer).xy;
    let omega   = textureLoad(omega_in,   coord, cascade_layer).x;

    let phase = omega * settings.elapsed_time;
    let exp_pos = cexp_i( phase);
    let exp_neg = cexp_i(-phase);

    let h = cmul(h0, exp_pos) + cmul(h0_conj, exp_neg);

    let patch_unreserved_keyword = cascades.cascades[cascade].x;
    let k = k_for_cell(row, col, patch_unreserved_keyword);
    let k_len = length(k);

    var dx = vec2<f32>(0.0, 0.0);
    var dz = vec2<f32>(0.0, 0.0);
    var sx = vec2<f32>(0.0, 0.0);
    var sz = vec2<f32>(0.0, 0.0);
    if (k_len > 1e-6) {
        let kx_hat = k.x / k_len;
        let kz_hat = k.y / k_len;
        let neg_i_h = cmul_neg_i(h);
        dx = neg_i_h * kx_hat;
        dz = neg_i_h * kz_hat;
        let i_h = cmul_i(h);
        sx = i_h * k.x;
        sz = i_h * k.y;
    }

    let zero_pad = vec2<f32>(0.0, 0.0);
    textureStore(spectra_out, coord, spec_layer(FFT_SPEC_H,  cascade), vec4<f32>(h,  zero_pad));
    textureStore(spectra_out, coord, spec_layer(FFT_SPEC_DX, cascade), vec4<f32>(dx, zero_pad));
    textureStore(spectra_out, coord, spec_layer(FFT_SPEC_DZ, cascade), vec4<f32>(dz, zero_pad));
    textureStore(spectra_out, coord, spec_layer(FFT_SPEC_SX, cascade), vec4<f32>(sx, zero_pad));
    textureStore(spectra_out, coord, spec_layer(FFT_SPEC_SZ, cascade), vec4<f32>(sz, zero_pad));
}
