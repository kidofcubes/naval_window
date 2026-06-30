// Pack IFFT outputs (real parts) into the textures sampled by the material.
// Displacement: vec4(dx, height, dz, 0) per cascade.
// Slope:        vec4(dh_dx, dh_dz, 0, 0) per cascade.

#import "fftwaves/fft_common.wgsl"::{
    FFT_GRID_SIZE, FFT_CASCADE_COUNT,
    FFT_SPEC_H, FFT_SPEC_DX, FFT_SPEC_DZ, FFT_SPEC_SX, FFT_SPEC_SZ,
    spec_layer,
}

@group(0) @binding(0) var spectra_in:      texture_storage_2d_array<rg32float,  read>;
@group(0) @binding(1) var displacement_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(2) var slope_out:        texture_storage_2d_array<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let cascade = gid.z;
    if (row >= FFT_GRID_SIZE || col >= FFT_GRID_SIZE || cascade >= FFT_CASCADE_COUNT) {
        return;
    }
    let coord = vec2<i32>(i32(col), i32(row));
    let layer = i32(cascade);

    let h  = textureLoad(spectra_in, coord, spec_layer(FFT_SPEC_H,  cascade)).x;
    let dx = textureLoad(spectra_in, coord, spec_layer(FFT_SPEC_DX, cascade)).x;
    let dz = textureLoad(spectra_in, coord, spec_layer(FFT_SPEC_DZ, cascade)).x;
    let sx = textureLoad(spectra_in, coord, spec_layer(FFT_SPEC_SX, cascade)).x;
    let sz = textureLoad(spectra_in, coord, spec_layer(FFT_SPEC_SZ, cascade)).x;

    textureStore(displacement_out, coord, layer, vec4<f32>(dx, h,  dz, 0.0));
    textureStore(slope_out,        coord, layer, vec4<f32>(sx, sz, 0.0, 0.0));
}
