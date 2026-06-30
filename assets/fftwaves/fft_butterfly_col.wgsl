// 1D radix-2 IFFT down each column, with Tessendorf fftshift sign correction.
// Dispatch: (1, FFT_GRID_SIZE, FFT_SPECTRUM_COUNT * FFT_CASCADE_COUNT)
// Workgroup processes one (col, packed_layer) of the spectra texture.

#import "fftwaves/fft_common.wgsl"::{
    FFT_GRID_SIZE, FFT_GRID_SIZE_LOG2,
    cmul, cexp_i,
}

@group(0) @binding(0) var spectra_in:  texture_storage_2d_array<rg32float, read>;
@group(0) @binding(1) var spectra_out: texture_storage_2d_array<rg32float, write>;

var<workgroup> sm: array<vec2<f32>, FFT_GRID_SIZE>;

fn bit_reverse_8(x: u32) -> u32 {
    var v = x;
    v = ((v & 0xAAu) >> 1u) | ((v & 0x55u) << 1u);
    v = ((v & 0xCCu) >> 2u) | ((v & 0x33u) << 2u);
    v = ((v & 0xF0u) >> 4u) | ((v & 0x0Fu) << 4u);
    return v;
}

const TAU: f32 = 6.283185307179586;

@compute @workgroup_size(FFT_GRID_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let col = gid.y;
    let layer = i32(gid.z);
    let tid = lid.x;

    // Load: column index = col, row index = tid (the FFT axis).
    let br = bit_reverse_8(tid);
    let coord_in = vec2<i32>(i32(col), i32(tid));
    sm[br] = textureLoad(spectra_in, coord_in, layer).xy;
    workgroupBarrier();

    for (var s: u32 = 0u; s < FFT_GRID_SIZE_LOG2; s = s + 1u) {
        let m = 1u << (s + 1u);
        let half_m = m >> 1u;
        if ((tid & half_m) == 0u) {
            let k = tid & (half_m - 1u);
            let i1 = tid;
            let i2 = tid | half_m;
            let twiddle = cexp_i(TAU * f32(k) / f32(m));
            let a = sm[i1];
            let b = cmul(sm[i2], twiddle);
            sm[i1] = a + b;
            sm[i2] = a - b;
        }
        workgroupBarrier();
    }

    // Tessendorf fftshift sign: (-1)^(row+col), applied during the column pass.
    let row = tid;
    let sign = select(-1.0, 1.0, ((row + col) & 1u) == 0u);
    let val = sm[tid] * sign;
    let coord_out = vec2<i32>(i32(col), i32(tid));
    textureStore(spectra_out, coord_out, layer, vec4<f32>(val.x, val.y, 0.0, 0.0));
}
