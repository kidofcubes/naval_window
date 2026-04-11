#import "fft_common.wgsl"::{FFT_SIZE, FFT_LOG2, TAU, complex_mul}

// Stockham auto-sort IFFT over columns + sign correction, keeping both real and imaginary parts.
// Used for the slope spectrum: IFFT(C).real = slope_x, IFFT(C).imag = slope_z.
// Each workgroup processes one column (256 threads = 256 elements).
// global_id.y selects the column.

@group(0) @binding(0) var spectrum_in: texture_2d<f32>;
@group(0) @binding(1) var slope_out: texture_storage_2d<rg32float, write>;

var<workgroup> ping: array<vec2<f32>, 256>;
var<workgroup> pong: array<vec2<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let col = global_id.y;
    let t = local_id.x; // row index within the column

    if (col >= FFT_SIZE) {
        return;
    }

    // Load column from texture (transposed access: x=col, y=row)
    ping[t] = textureLoad(spectrum_in, vec2<i32>(i32(col), i32(t)), 0).xy;
    workgroupBarrier();

    // Stockham butterfly: 8 stages for N=256
    for (var s: u32 = 0u; s < FFT_LOG2; s++) {
        let m = 1u << (s + 1u);
        let half_m = 1u << s;

        let group = t / m;
        let j = t % m;

        let even_idx = group * m + (j % half_m);
        let odd_idx = even_idx + half_m;

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
    let val = pong[t];

    // FFT shift sign correction: (-1)^(row+col)
    let row = t;
    let sign = select(-1.0, 1.0, (row + col) % 2u == 0u);

    // Keep both real (slope_x) and imaginary (slope_z) parts with sign correction
    textureStore(slope_out, vec2<i32>(i32(col), i32(row)), vec4<f32>(val.x * sign, val.y * sign, 0.0, 0.0));
}
