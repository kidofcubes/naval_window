#import "fft_common.wgsl"::{FFT_SIZE, FFT_LOG2, TAU, complex_mul}

// Stockham auto-sort IFFT over rows.
// Each workgroup processes one row (256 threads = 256 elements).
// global_id.y selects the row.

@group(0) @binding(0) var spectrum_in: texture_2d<f32>;
@group(0) @binding(1) var spectrum_out: texture_storage_2d<rg32float, write>;

var<workgroup> ping: array<vec2<f32>, 256>;
var<workgroup> pong: array<vec2<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    let t = local_id.x;

    if (row >= FFT_SIZE) {
        return;
    }

    // Load row from texture
    ping[t] = textureLoad(spectrum_in, vec2<i32>(i32(t), i32(row)), 0).xy;
    workgroupBarrier();

    // Stockham butterfly: 8 stages for N=256
    // At each stage s, butterflies have stride 2^s
    for (var s: u32 = 0u; s < FFT_LOG2; s++) {
        let m = 1u << (s + 1u);       // butterfly group size: 2, 4, 8, ..., 256
        let half_m = 1u << s;          // half of group size: 1, 2, 4, ..., 128

        let group = t / m;             // which butterfly group
        let j = t % m;                 // position within group

        let even_idx = group * m + (j % half_m);
        let odd_idx = even_idx + half_m;

        // Twiddle factor: exp(+2*pi*i * j / m) for IFFT (positive sign)
        let twiddle_exp = (j % half_m);
        let angle = TAU * f32(twiddle_exp) / f32(m);
        let w = vec2<f32>(cos(angle), sin(angle));

        var a: vec2<f32>;
        var b: vec2<f32>;
        if (s % 2u == 0u) {
            a = ping[even_idx];
            b = ping[odd_idx];
        } else {
            a = pong[even_idx];
            b = pong[odd_idx];
        }

        let wb = complex_mul(b, w);
        var result: vec2<f32>;
        if (j < half_m) {
            result = a + wb;
        } else {
            result = a - wb;
        }

        if (s % 2u == 0u) {
            pong[t] = result;
        } else {
            ping[t] = result;
        }

        workgroupBarrier();
    }

    // After 8 stages (even count), result is in pong
    let final_val = pong[t];

    textureStore(spectrum_out, vec2<i32>(i32(t), i32(row)), vec4<f32>(final_val, 0.0, 0.0));
}
