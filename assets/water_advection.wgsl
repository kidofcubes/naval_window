// ─────────────────────────────────────────────────────────────────────────────
// water_advection.wgsl
//
// Pass 2 of the Water Surface Wavelets pipeline.
//
// Implements semi-Lagrangian advection of the amplitude field A(x, y, θ_d).
// For each cell (x, y) and each wave direction d:
//
//   1. Compute group velocity vector  cg * (cos θ_d, sin θ_d).
//   2. Trace backward one timestep:   prev = (x,y) - cg * dt * dir.
//   3. If the ray crossed an obstacle, reflect it off the obstacle boundary.
//   4. Bilinear-sample A_in at prev to get the new amplitude.
//   5. Apply spatial + angular diffusion.
//   6. Add a ship source term (Kelvin wake emission).
// ─────────────────────────────────────────────────────────────────────────────

// Must match Rust constants.
const NUM_DIRECTIONS: u32  = 8u;
const TAU:            f32  = 6.283185307;
const GOLDEN_ANGLE:   f32  = 2.399963229; // 2π(1 - 1/φ), for phase decorrelation

// ─── Uniforms ────────────────────────────────────────────────────────────────

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

@group(0) @binding(0) var amplitude_in:   texture_storage_2d_array<r32float, read>;
@group(0) @binding(1) var amplitude_out:  texture_storage_2d_array<r32float, write>;
@group(0) @binding(2) var obstacle_mask:  texture_storage_2d<r8unorm, read>;
@group(0) @binding(3) var<uniform> s: WaterSettings;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Deep-water group speed: cg = √(g/k) / 2
fn group_speed(k: f32, g: f32) -> f32 {
    return sqrt(g / k) * 0.5;
}

/// Direction vector for discrete direction index d.
fn dir_vec(d: u32) -> vec2<f32> {
    let theta = (f32(d) / f32(NUM_DIRECTIONS)) * TAU;
    return vec2<f32>(cos(theta), sin(theta));
}

/// Bilinear sample from amplitude_in at continuous position p (in grid cells),
/// clamping to [0, grid_size-1].  Layer = direction index d.
fn sample_amplitude(p: vec2<f32>, d: u32) -> f32 {
    let gs = f32(s.grid_size);
    // Wrap (toroidal domain boundary for the open-ocean edges).
    let pw = (p % gs + gs) % gs;

    let x0 = u32(pw.x) % s.grid_size;
    let y0 = u32(pw.y) % s.grid_size;
    let x1 = (x0 + 1u) % s.grid_size;
    let y1 = (y0 + 1u) % s.grid_size;

    let fx = fract(pw.x);
    let fy = fract(pw.y);

    let a00 = textureLoad(amplitude_in, vec2<u32>(x0, y0), d).r;
    let a10 = textureLoad(amplitude_in, vec2<u32>(x1, y0), d).r;
    let a01 = textureLoad(amplitude_in, vec2<u32>(x0, y1), d).r;
    let a11 = textureLoad(amplitude_in, vec2<u32>(x1, y1), d).r;

    return mix(mix(a00, a10, fx), mix(a01, a11, fx), fy);
}

/// Returns true if grid cell (x, y) is solid (obstacle mask > 0.5).
fn is_solid(x: u32, y: u32) -> bool {
    return textureLoad(obstacle_mask, vec2<u32>(x, y)).r > 0.5;
}

/// Reflect a 2-D direction vector off a grid boundary normal.
/// We approximate the normal as the gradient of the obstacle mask.
fn reflect_direction(dir: vec2<f32>, cell: vec2<u32>) -> vec2<f32> {
    // Sample neighbours to estimate obstacle gradient (approx. outward normal).
    let cx = cell.x;
    let cy = cell.y;
    let gs = s.grid_size;

    let ox = select(0.0, 1.0, cx > 0u && is_solid(cx - 1u, cy))
           - select(0.0, 1.0, cx + 1u < gs && is_solid(cx + 1u, cy));
    let oy = select(0.0, 1.0, cy > 0u && is_solid(cx, cy - 1u))
           - select(0.0, 1.0, cy + 1u < gs && is_solid(cx, cy + 1u));

    let n = normalize(vec2<f32>(ox, oy) + vec2<f32>(0.001)); // avoid zero normal
    return dir - 2.0 * dot(dir, n) * n;
}

/// Phillips spectrum amplitude for wavenumber k under given wind.
/// Returns √S(k) so it can be used as an amplitude directly.
fn phillips_amplitude(k: f32) -> f32 {
    let v  = s.wind_speed;
    let g  = s.gravity;
    // Limiting wavelength driven by wind: L = v²/g
    let L  = v * v / g;
    let k2 = k * k;
    let kL = k * L;
    // Phillips: S(k) ∝ exp(−1/(kL)²) / k⁴
    // Cap at k_min to avoid blowup at k→0
    if k < 0.001 { return 0.0; }
    let S = exp(-1.0 / (kL * kL)) / (k2 * k2);
    return sqrt(S);
}

// ─── Main Kernel ─────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x  = gid.x;
    let y  = gid.y;
    let d  = gid.z;  // direction layer [0, NUM_DIRECTIONS)

    if x >= s.grid_size || y >= s.grid_size || d >= NUM_DIRECTIONS { return; }

    // ── 1. Group velocity vector ───────────────────────────────────────────
    let dv  = dir_vec(d);                        // unit vector in θ_d
    let cg  = group_speed(s.k0, s.gravity);      // m/s
    let cell_m = s.domain_size / f32(s.grid_size); // metres per cell

    // Advection in grid-cell units per timestep.
    let advect_cells = cg * s.dt / cell_m;

    // ── 2. Trace backward ──────────────────────────────────────────────────
    var prev = vec2<f32>(f32(x), f32(y)) - dv * advect_cells;

    // ── 3. Obstacle reflection ─────────────────────────────────────────────
    // Check if the straight-back path crosses an obstacle cell.
    // Simple: test the grid cell at the halfway point.
    let mid  = vec2<f32>(f32(x), f32(y)) - dv * (advect_cells * 0.5);
    let mx   = u32(clamp(mid.x, 0.0, f32(s.grid_size - 1u)));
    let my   = u32(clamp(mid.y, 0.0, f32(s.grid_size - 1u)));

    var reflected_dir = dv;
    if is_solid(mx, my) {
        reflected_dir = reflect_direction(dv, vec2<u32>(mx, my));
        // Re-trace from current cell in reflected direction.
        prev = vec2<f32>(f32(x), f32(y)) - reflected_dir * advect_cells;
    }

    // ── 4. Sample amplitude at back-traced position ────────────────────────
    var amp = sample_amplitude(prev, d);

    // ── 5. Diffusion ───────────────────────────────────────────────────────
    // Spatial diffusion (smooths along travel direction, mimics group spread).
    let sp = s.diffusion_spatial;
    if sp > 0.0 {
        let ahead = sample_amplitude(vec2<f32>(f32(x), f32(y)) + dv, d);
        let behind = sample_amplitude(vec2<f32>(f32(x), f32(y)) - dv, d);
        amp = mix(amp, (amp + ahead + behind) / 3.0, sp);
    }

    // Angular diffusion: blend with neighbouring directional layers.
    // This spreads a point source into a circle rather than discrete beams.
    let ang = s.diffusion_angular;
    if ang > 0.0 {
        let d_prev = (d + NUM_DIRECTIONS - 1u) % NUM_DIRECTIONS;
        let d_next = (d + 1u) % NUM_DIRECTIONS;
        let a_prev = sample_amplitude(prev, d_prev);
        let a_next = sample_amplitude(prev, d_next);
        amp = mix(amp, (amp + a_prev + a_next) / 3.0, ang);
    }

    // ── 6. Ship source term ────────────────────────────────────────────────
    // Emit a small Kelvin-wake source term at the ship's grid position.
    // The ship injects energy into all directions, but preferentially
    // in the direction opposite to travel (stern → wake).
    let cell_pos = vec2<f32>(f32(x), f32(y));
    let half_gs  = f32(s.grid_size) * 0.5;
    let ship_grid = vec2<f32>(s.ship_pos_x,s.ship_pos_z) / s.domain_size * f32(s.grid_size) + half_gs;
    let dist      = length(cell_pos - ship_grid);

    // Gaussian footprint of ship source, radius ≈ 3 cells.
    let source_radius = 3.0;
    let source_weight = exp(-dist * dist / (2.0 * source_radius * source_radius));

    // Directional weighting: bias emission backwards from the wind/ship heading
    let ship_dir_bias = max(0.0, dot(-dv, vec2<f32>(cos(s.wind_direction), sin(s.wind_direction))));
    let source_amp    = phillips_amplitude(s.k0) * s.ship_source_strength * source_weight * (0.5 + 0.5 * ship_dir_bias);

    amp = amp + source_amp;

    // ── 7. Clamp and write ─────────────────────────────────────────────────
    // Clamp to avoid numerical explosions.
    amp = clamp(amp, 0.0, 10.0);
    textureStore(amplitude_out, vec2<u32>(x, y), d, vec4<f32>(amp, 0.0, 0.0, 0.0));
}
