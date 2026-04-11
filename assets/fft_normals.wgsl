#import "fft_common.wgsl"::{FFT_SIZE, FftOceanSettings}

@group(0) @binding(0) var height_field: texture_2d<f32>;
@group(0) @binding(1) var normal_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> settings: FftOceanSettings;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;
    if (row >= FFT_SIZE || col >= FFT_SIZE) {
        return;
    }

//    let cell_size = settings.patch_length / f32(FFT_SIZE);
    let cell_size = 26.0 / f32(FFT_SIZE);

    // Central differences with wrapping
    let left  = textureLoad(height_field, vec2<i32>(i32((col + FFT_SIZE - 1u) % FFT_SIZE), i32(row)), 0).x;
    let right = textureLoad(height_field, vec2<i32>(i32((col + 1u) % FFT_SIZE), i32(row)), 0).x;
    let down  = textureLoad(height_field, vec2<i32>(i32(col), i32((row + FFT_SIZE - 1u) % FFT_SIZE)), 0).x;
    let up    = textureLoad(height_field, vec2<i32>(i32(col), i32((row + 1u) % FFT_SIZE)), 0).x;

    let dh_dx = (right - left) / (2.0 * cell_size);
    let dh_dz = (up - down) / (2.0 * cell_size);

    let normal = normalize(vec3<f32>(-dh_dx, 1.0, -dh_dz));

    textureStore(normal_out, vec2<i32>(i32(col), i32(row)), vec4<f32>(normal, 0.0));
}
