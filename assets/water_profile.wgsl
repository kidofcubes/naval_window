// ─────────────────────────────────────────────────────────────────────────────
// water_profile.wgsl
//
// Pass 1 of the Water Surface Wavelets pipeline.
//
// Builds the 1-D profile buffer Ψ(p) by integrating the wave spectrum S(k)
// over a discrete set of wavenumbers.
//
// Ψ(p) = ∫ √S(k) · cos(k·p) dk  ≈  Σ_i √S(k_i) · cos(k_i · p) · Δk
//
// The profile encodes the wave shape along a single wave's travel direction.
// At height reconstruction time, we evaluate:
//   η(x) = Σ_d  A(x, d) · Ψ( k̂_d · x_world + φ_d )
//
// This texture is 1 × PROFILE_SIZE (so indexing is: texel (p, 0)).
// ─────────────────────────────────────────────────────────────────────────────

const TAU: f32 = 6.283185307;

// Number of wavenumber samples used to build Ψ.
// More = smoother profile; 16 is usually sufficient.
const NUM_K_SAMPLES: u32 = 16u;

struct WaterSettings {
    domain_size:          f32,
    time:                 f32,
    dt:                   f32,
    k0:                   f32,
    gravity:              f32,
    wind_direction:       f32,
    wind_speed:           f32,
    diffusion_spatial:    f32,
    diffusion_angular:    f32,
//    ship_pos:             vec2<f32>,
    ship_pos_x:           f32,
    ship_pos_z:           f32,
    ship_source_strength: f32,
    grid_size:            u32,
    num_directions:       u32,
    profile_size:         u32,
}

@group(0) @binding(0) var profile_out: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var<uniform> s: WaterSettings;

// ─── Phillips spectrum ────────────────────────────────────────────────────────

/// Full directional Phillips spectrum  √S(k), scalar version.
/// The directional spreading is handled in the advection layer;
/// here we just need the frequency-domain energy envelope.
fn phillips_sqrt(k: f32) -> f32 {
    if k < 0.001 { return 0.0; }
    let v  = s.wind_speed;
    let g  = s.gravity;
    let L  = v * v / g;          // dominant wavelength
    let kL = k * L;
    let k2 = k * k;
    // Phillips: S(k) = A * exp(-1/(kL)²) / k⁴
    // We absorb constant A into ship_source_strength elsewhere; here A=1.
    return sqrt(exp(-1.0 / (kL * kL)) / (k2 * k2));
}

// ─── Wave profile ─────────────────────────────────────────────────────────────

/// Deep-water phase speed: cp = √(g/k)
fn phase_speed(k: f32) -> f32 { return sqrt(s.gravity / k); }

/// Trochoidal/Gerstner-style profile contribution for wavenumber k
/// at profile coordinate p (in metres), at current time t.
///
/// This gives the classic peaked-crest, flat-trough wave shape.
fn profile_sample(k: f32, p: f32) -> f32 {
    let omega = sqrt(s.gravity * k);          // angular frequency
    let phase = k * p - omega * s.time;      // travelling wave phase
    // cos gives standard sinusoidal; using a Stokes correction
    // (cos + 0.5·k·a·cos(2φ)) sharpens crests slightly.
    // Here we approximate: just cos for clarity, easy to extend.
    return cos(phase);
}

// ─── Main Kernel ─────────────────────────────────────────────────────────────
//
// One thread per profile sample p_i ∈ [0, domain_size].
// Each thread accumulates contributions from NUM_K_SAMPLES wavenumbers.

@compute @workgroup_size(64, 1, 1)
fn build_profile(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p_idx = gid.x;
    if p_idx >= s.profile_size { return; }

    // Map index to physical distance in [0, domain_size].
    let p = (f32(p_idx) / f32(s.profile_size)) * s.domain_size;

    // Wavenumber range: sample logarithmically from k_min to k_max.
    // k_min ≈ 2π/domain_size (longest wave that fits in the domain)
    // k_max ≈ 16 * k0        (shortest relevant wave)
    let k_min = TAU / s.domain_size;
    let k_max = s.k0 * 16.0;
    let log_kmin = log(k_min);
    let log_kmax = log(k_max);
    let log_step = (log_kmax - log_kmin) / f32(NUM_K_SAMPLES);

    var psi = 0.0;
    for (var i = 0u; i < NUM_K_SAMPLES; i++) {
        let log_k = log_kmin + (f32(i) + 0.5) * log_step;
        let k     = exp(log_k);
        let dk    = k * (exp(log_step) - 1.0);  // Δk in log-space

        let weight = phillips_sqrt(k) * dk;
        psi += weight * profile_sample(k, p);
    }

    textureStore(profile_out, vec2<u32>(p_idx, 0u), vec4<f32>(psi, 0.0, 0.0, 0.0));
}
