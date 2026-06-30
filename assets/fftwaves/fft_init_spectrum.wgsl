// Initialize h0(k) and h0_conj(-k) and ω(k) for every cascade.
// Runs ONCE (the Rust side gates this via a "done" flag).

#import "fftwaves/fft_common.wgsl"::{
    FFT_GRID_SIZE, FFT_CASCADE_COUNT, GRAVITY, TAU,
    FftCascadeBuffer,
    jonswap_spectrum, gaussian2, k_for_cell, hash3, cconj,
}

@group(0) @binding(0) var h0_out:      texture_storage_2d_array<rg32float, write>;
@group(0) @binding(1) var h0_conj_out: texture_storage_2d_array<rg32float, write>;
@group(0) @binding(2) var omega_out:   texture_storage_2d_array<r32float,  write>;
@group(0) @binding(3) var<uniform> cascades: FftCascadeBuffer;

// PRNG seed mixing prime; arbitrary fixed value picks a particular ocean realization.
const FFT_PRNG_SALT: u32 = 0xC0FFEE17u;

fn sample_h0(row: u32, col: u32, cascade: u32) -> vec2<f32> {
    let params = cascades.cascades[cascade];
    let patch_unreserved_keyword = params.x;
    let k_min = params.y;
    let k_max = params.z;

    let k = k_for_cell(row, col, patch_unreserved_keyword);
    let k_len = length(k);
    if (k_len < k_min || k_len > k_max) {
        return vec2<f32>(0.0, 0.0);
    }
    let s = jonswap_spectrum(k.x, k.y);
    let scale = sqrt(s) * 0.7071067811865475; // 1/sqrt(2)

    let noise = gaussian2(
        hash3(row, col,  cascade * 2u      + FFT_PRNG_SALT),
        hash3(row, col,  cascade * 2u + 1u + FFT_PRNG_SALT),
    );
    return noise * scale;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let cascade = gid.z;
    if (row >= FFT_GRID_SIZE || col >= FFT_GRID_SIZE || cascade >= FFT_CASCADE_COUNT) {
        return;
    }

    let h0 = sample_h0(row, col, cascade);

    // h0_conj(k) = h0(-k)*, mirrored about (N/2, N/2) to match the CPU init.
    let mr = (FFT_GRID_SIZE - row) % FFT_GRID_SIZE;
    let mc = (FFT_GRID_SIZE - col) % FFT_GRID_SIZE;
    let h0_conj = cconj(sample_h0(mr, mc, cascade));

    let coord = vec2<i32>(i32(col), i32(row));
    let layer = i32(cascade);

    textureStore(h0_out,      coord, layer, vec4<f32>(h0.x,      h0.y,      0.0, 0.0));
    textureStore(h0_conj_out, coord, layer, vec4<f32>(h0_conj.x, h0_conj.y, 0.0, 0.0));

    let patch_unreserved_keyword = cascades.cascades[cascade].x;
    let k = k_for_cell(row, col, patch_unreserved_keyword);
    let k_len = length(k);
    let omega = sqrt(GRAVITY * k_len);
    textureStore(omega_out, coord, layer, vec4<f32>(omega, 0.0, 0.0, 0.0));
}
