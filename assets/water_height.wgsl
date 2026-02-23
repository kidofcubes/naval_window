// ─────────────────────────────────────────────────────────────────────────────
// water_height.wgsl
//
// Pass 3 of the Water Surface Wavelets pipeline.
//
// Reconstructs the visible height field η(x, y) from the amplitude field:
//
//   η(x, y)  =  Σ_{d=0}^{N-1}  A(x, y, d) · Ψ( k̂_d · x_world + φ_d )
//
// where:
//   A(x, y, d)  = amplitude of waves at cell (x,y) propagating in direction d
//   Ψ(p)        = 1-D profile buffer (wave shape function, built by Pass 1)
//   k̂_d         = unit wavevector for direction d  (cos θ_d, sin θ_d)
//   x_world     = world-space position of cell (x, y) in metres
//   φ_d         = per-direction phase offset (breaks up lattice artifacts)
//
// The result is written to the height_out storage texture, which is then
// sampled by the water surface vertex shader.
// ─────────────────────────────────────────────────────────────────────────────

const NUM_DIRECTIONS: u32 = 8u;
const TAU:            f32 = 6.283185307;
// Golden-ratio-based angle to decorrelate directional phase offsets.
const GOLDEN_ANGLE:   f32 = 2.399963229;

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

@group(0) @binding(0) var amplitude:   texture_storage_2d_array<r32float, read>;
@group(0) @binding(1) var profile_buf: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var height_out:  texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> s:  WaterSettings;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn dir_vec(d: u32) -> vec2<f32> {
    let theta = (f32(d) / f32(NUM_DIRECTIONS)) * TAU;
    return vec2<f32>(cos(theta), sin(theta));
}

/// Sample Ψ(p) from the profile buffer.
/// p is in world-space metres; we map it into [0, profile_size) by wrapping.
fn sample_profile(p: f32) -> f32 {
    // Wrap p into [0, domain_size) then map to texture column index.
    let p_wrapped = ((p % s.domain_size) + s.domain_size) % s.domain_size;
    let p_idx     = u32(p_wrapped / s.domain_size * f32(s.profile_size)) % s.profile_size;
    return textureLoad(profile_buf, vec2<u32>(p_idx, 0u)).r;
}

// ─── Main Kernel ─────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn reconstruct_height(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= s.grid_size || y >= s.grid_size { return; }

    // World-space position of this cell (in metres).
    let half    = s.domain_size * 0.5;
    let cell_m  = s.domain_size / f32(s.grid_size);
    let wx      = f32(x) * cell_m - half;
    let wz      = f32(y) * cell_m - half;
    let x_world = vec2<f32>(wx, wz);

    var eta = 0.0;

    for (var d = 0u; d < NUM_DIRECTIONS; d++) {
        let A = textureLoad(amplitude, vec2<u32>(x, y), d).r;

        // k̂_d · x_world gives the scalar projection of world position
        // onto the wave travel direction.
        let kd  = dir_vec(d);
        let proj = s.k0 * dot(kd, x_world);   // units: radians (k [rad/m] × x [m])

        // Per-direction phase offset using golden angle to decorrelate directions
        // and avoid visible lattice artifacts at low NUM_DIRECTIONS counts.
        let phi = f32(d) * GOLDEN_ANGLE;

        // Convert radians → metres for the profile lookup.
        // The profile was built in metres, so: p = (proj + phi) / k0
        let p = (proj + phi) / s.k0;

        eta += A * sample_profile(p);
    }

    textureStore(height_out, vec2<u32>(x, y), vec4<f32>(eta, 0.0, 0.0, 0.0));
}
