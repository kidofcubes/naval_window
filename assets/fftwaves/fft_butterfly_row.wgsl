// 1D radix-2 IFFT across each row.
// Dispatch: (1, FFT_GRID_SIZE, FFT_SPECTRUM_COUNT * FFT_CASCADE_COUNT)
// Workgroup processes one (row, packed_layer) of the spectra texture.

#import "fftwaves/fft_common.wgsl"::{
    FFT_GRID_SIZE, FFT_GRID_SIZE_LOG2,
    cmul, cexp_i,
}

@group(0) @binding(0) var spectra_in:  texture_storage_2d_array<rg32float, read>;
@group(0) @binding(1) var spectra_out: texture_storage_2d_array<rg32float, write>;

var<workgroup> sm: array<vec2<f32>, FFT_GRID_SIZE>;

fn bit_reverse_n(x: u32) -> u32 {
    var v: u32 = 0u;
    var t: u32 = x;
    for (var i: u32 = 0u; i < FFT_GRID_SIZE_LOG2; i = i + 1u) {
        v = (v << 1u) | (t & 1u);
        t = t >> 1u;
    }
    return v;
}

const TAU: f32 = 6.283185307179586;

@compute @workgroup_size(FFT_GRID_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let row = gid.y;
    let layer = i32(gid.z);
    let tid = lid.x;

    // Load with bit-reverse permutation into shared memory.
    let br = bit_reverse_n(tid);
    let coord_in = vec2<i32>(i32(tid), i32(row));
    sm[br] = textureLoad(spectra_in, coord_in, layer).xy;
    workgroupBarrier();

    // Cooley-Tukey radix-2 DIT, inverse twiddle (positive exponent).
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

    let coord_out = vec2<i32>(i32(tid), i32(row));
    let val = sm[tid];
    textureStore(spectra_out, coord_out, layer, vec4<f32>(val.x, val.y, 0.0, 0.0));
}
