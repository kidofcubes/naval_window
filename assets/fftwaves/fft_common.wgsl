// Shared constants & helpers for the FFT-based ocean wave cascade.

const FFT_GRID_SIZE: u32 = 512u;
const FFT_GRID_SIZE_LOG2: u32 = 9u;
const FFT_CASCADE_COUNT: u32 = 4u;

// Spectra packed into a single 2D-array texture: layer = spectrum * C + cascade.
const FFT_SPECTRUM_COUNT: u32 = 5u;
const FFT_SPEC_H:  u32 = 0u;
const FFT_SPEC_DX: u32 = 1u;
const FFT_SPEC_DZ: u32 = 2u;
const FFT_SPEC_SX: u32 = 3u;
const FFT_SPEC_SZ: u32 = 4u;

fn spec_layer(spectrum: u32, cascade: u32) -> i32 {
    return i32(spectrum * FFT_CASCADE_COUNT + cascade);
}

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const GRAVITY: f32 = 9.81;

// JONSWAP parameters (mirror big_watering::OceanConfig defaults)
const WIND_SPEED: f32 = 5.0;
const WIND_DIR_X: f32 = 0.707;
const WIND_DIR_Z: f32 = 0.707;
const FETCH: f32 = 100000.0;
const GAMMA: f32 = 3.3;
const AMPLITUDE: f32 = 0.05;

struct FftSettings {
    elapsed_time: f32,
    _padding_a: f32,
    _padding_b: f32,
    _padding_c: f32,
}

// One per cascade. xyzw = (patch_length, k_min, k_max, _pad)
struct FftCascadeBuffer {
    cascades: array<vec4<f32>, FFT_CASCADE_COUNT>,
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cconj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

// e^{i*theta}
fn cexp_i(theta: f32) -> vec2<f32> {
    return vec2<f32>(cos(theta), sin(theta));
}

// i*z = (-z.y, z.x)
fn cmul_i(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-z.y, z.x);
}

// -i*z = (z.y, -z.x)
fn cmul_neg_i(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z.y, -z.x);
}

// Deterministic per-cell PRNG (PCG-hash style), yields f32 in (0,1).
fn hash_u32(x: u32) -> u32 {
    var v = x;
    v = v ^ (v >> 16u);
    v = v * 0x7feb352du;
    v = v ^ (v >> 15u);
    v = v * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

fn hash3(a: u32, b: u32, c: u32) -> u32 {
    return hash_u32(a + hash_u32(b + hash_u32(c)));
}

fn u32_to_unit_f32(u: u32) -> f32 {
    // 24-bit mantissa precision, strictly > 0 to avoid log(0).
    let m = (u >> 8u) & 0x00ffffffu;
    return (f32(m) + 1.0) / f32(0x01000000u);
}

// Two independent N(0,1) samples via Box-Muller.
fn gaussian2(seed_a: u32, seed_b: u32) -> vec2<f32> {
    let u1 = u32_to_unit_f32(hash_u32(seed_a));
    let u2 = u32_to_unit_f32(hash_u32(seed_b));
    let radius = sqrt(-2.0 * log(u1));
    let angle = TAU * u2;
    return vec2<f32>(radius * cos(angle), radius * sin(angle));
}

// JONSWAP spectrum (port of big_watering::jonswap_spectrum)
fn jonswap_spectrum(kx: f32, kz: f32) -> f32 {
    let wave_number = sqrt(kx * kx + kz * kz);
    if (wave_number < 1e-8) {
        return 0.0;
    }
    let omega = sqrt(GRAVITY * wave_number);
    if (omega < 1e-8) {
        return 0.0;
    }
    let omega_peak = 22.0 * pow(GRAVITY * GRAVITY / (WIND_SPEED * FETCH), 1.0 / 3.0);

    let alpha = 0.0081;
    let omega_ratio = omega_peak / omega;
    let pm = (alpha * GRAVITY * GRAVITY / pow(omega, 5.0))
           * exp(-1.25 * pow(omega_ratio, 4.0));

    let sigma = select(0.09, 0.07, omega <= omega_peak);
    let exponent_arg = pow(omega - omega_peak, 2.0) / (2.0 * sigma * sigma * omega_peak * omega_peak);
    let gamma_factor = pow(GAMMA, exp(-exponent_arg));
    let jonswap = pm * gamma_factor;

    let d_omega_dk = GRAVITY / (2.0 * omega);
    let spectrum_k = jonswap * d_omega_dk / TAU;

    let wave_dir = vec2<f32>(kx, kz) / wave_number;
    let cos_theta = wave_dir.x * WIND_DIR_X + wave_dir.y * WIND_DIR_Z;
    let directional = select(0.0, cos_theta * cos_theta, cos_theta > 0.0);

    return max(0.0, AMPLITUDE * spectrum_k * directional);
}

// Map (row, col) integer indices to (kx, kz) for a given patch length,
// matching the CPU init convention: kx, kz = 2π * (i - N/2) / patch_length.
fn k_for_cell(row: u32, col: u32, patch_length: f32) -> vec2<f32> {
    let half = f32(FFT_GRID_SIZE) * 0.5;
    let kx = TAU * (f32(row) - half) / patch_length;
    let kz = TAU * (f32(col) - half) / patch_length;
    return vec2<f32>(kx, kz);
}
