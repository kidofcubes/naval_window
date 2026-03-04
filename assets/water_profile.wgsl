// Build the 1-D profile buffer Psi(p)
//
// Each entry stores a vec4:
//   .x = horizontal Gerstner displacement (along wave direction)
//   .y = vertical height
//   .z = d(horizontal) / dp   (spatial derivative, for normals)
//   .w = d(vertical)   / dp   (spatial derivative, for normals)
//
// The output texture is 2D:
//   x-axis = profile index [0, PROFILE_BUFFER_LENGTH)
//   y-axis = zeta band     [0, TOTAL_WAVE_NUMBERS)
//
// Dispatched with workgroups (ceil(PROFILE_BUFFER_LENGTH / 128), 1, 1)
// Each invocation loops over all TOTAL_WAVE_NUMBERS bands for its profile index.
//
// IMPORTANT: WaterSettings must include a `wind_speed: f32` field.
// This controls which wavelengths dominate in the Pierson-Moskowitz spectrum.
// A value around 10.0 (m/s) is a reasonable default for open ocean.

#import "water_common.wgsl"::{
    GRID_X_SIZE, GRID_Z_SIZE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_WAVE_NUMBERS, PROFILE_BUFFER_LENGTH,
    ZETA_MIN, ZETA_MAXWAVE_NUMBERS, DIRECTIONAL_OFFSETS,
    GRAVITY, TIME_MULT, TAU, WaterSettings, WaterDebugSettings, VoxelSettings,
    index_to_radians, layer_index
}

@group(0) @binding(0) var profile_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var<uniform> settings: WaterSettings;

// How many sub-steps to use for the midpoint-rule integration over zeta.
// The C++ reference uses 100. 64 is accurate enough on GPU and keeps
// register pressure reasonable.
const INTEGRATION_STEPS: i32 = 64;

// The profile period is set to (PERIODICITY * max_wavelength_in_band).
// Using 2 means the buffer covers two full cycles of the longest wave,
// which is enough to avoid aliasing at the wrap point.
const PERIODICITY: f32 = 2.0;

// ---------------------------------------------------------------------------
// Pierson-Moskowitz spectrum density at zeta = log2(wavelength).
// Returns amplitude density (i.e. sqrt of energy density).
//
// The formula is a reparameterisation of the standard Pierson-Moskowitz model
// into zeta space. See Spectrum.cpp in the reference implementation.
//
// `wind_speed` is in m/s. Higher wind speed shifts the peak toward shorter
// wavelengths and lets higher frequencies survive the decay term B.
// ---------------------------------------------------------------------------
fn spectrum(zeta: f32, wind_speed: f32) -> f32 {
    // A: energy growth with increasing wavenumber (shorter waves).
    // Base 1.1 is a discretisation constant tuned for the zeta spacing used here.
    // (The original reference used pow(2, 1.5*zeta) but 1.1 was found to give
    // better calibrated results for this zeta spacing.)
    let A = pow(1.1, 1.5 * zeta);

    // B: high-frequency cutoff. Kills wavelengths too short for the wind to sustain.
    // The magic constant 1.8038897788076411 comes from the Pierson-Moskowitz
    // derivation. Dividing by wind_speed^4 means stronger wind = less decay.
    let B = exp(-1.8038897788076411 * pow(4.0, zeta) / pow(wind_speed, 4.0));

    // 0.139098 is a normalisation constant. sqrt converts energy -> amplitude.
    return 0.139098 * sqrt(A * B);
}

// ---------------------------------------------------------------------------
// Deep-water dispersion relation: omega = sqrt(g * k)
// Gives the angular frequency of a wave with wavenumber k.
// ---------------------------------------------------------------------------
fn dispersion_relation(k: f32) -> f32 {
    return sqrt(GRAVITY * k);
}

// ---------------------------------------------------------------------------
// Cubic bump — used to blend two phase evaluations together so the profile
// buffer wraps with C1 continuity (no visible seam).
//
// This is the p0 cubic Hermite basis function:
//   equals 1 at x=0, smoothly falls to 0 at |x|=1, zero outside.
// See ProfileBuffer.cpp::cubic_bump in the reference implementation.
// ---------------------------------------------------------------------------
fn cubic_bump(x: f32) -> f32 {
    let ax = abs(x);
    if ax >= 1.0 {
        return 0.0;
    }
    return ax * ax * (2.0 * ax - 3.0) + 1.0;
}

// ---------------------------------------------------------------------------
// Gerstner (trochoidal) wave evaluated at a given phase and wavenumber.
//
// Returns vec4(horizontal_disp, vertical_height, d_horiz/dp, d_vert/dp).
//
// The horizontal displacement is what distinguishes Gerstner from plain cosine:
// it shifts water particles horizontally toward crests, creating the sharp
// peaked crests and broad flat troughs seen on real ocean waves.
//
// Matches ProfileBuffer::gerstner_wave exactly:
//   return {-sin, cos, -k*cos, -k*sin}
// ---------------------------------------------------------------------------
fn gerstner_wave(phase: f32, knum: f32) -> vec4<f32> {
    let s = sin(phase);
    let c = cos(phase);
    return vec4<f32>(
        -s,         // horizontal displacement
         c,         // vertical height
        -knum * c,  // d(horizontal)/dp
        -knum * s   // d(vertical)/dp
    );
}

// ---------------------------------------------------------------------------
// Main compute shader entry point.
//
// For each profile index p_idx (dispatched in x), we loop over all
// TOTAL_WAVE_NUMBERS zeta bands and write one vec4 per (p_idx, band) pair.
// ---------------------------------------------------------------------------
@compute @workgroup_size(128, 1, 1)
fn build_profile(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p_idx = global_id.x;
    if p_idx >= PROFILE_BUFFER_LENGTH {
        return;
    }

    let zeta_full_range = ZETA_MAX - ZETA_MIN;

    for (var wave_index: u32 = 0u; wave_index < TOTAL_WAVE_NUMBERS; wave_index++) {

        // --- Determine this band's zeta range ---
        // The full [ZETA_MIN, ZETA_MAX] range is divided evenly among bands,
        // matching how WaveGrid sets up the zeta axis:
        //   zeta_center = ZETA_MIN + (wave_index + 0.5) * band_width
        //   zeta_band_min = center - 0.5 * band_width
        //   zeta_band_max = center + 0.5 * band_width
        let band_width    = zeta_full_range / f32(TOTAL_WAVE_NUMBERS);
        let zeta_band_min = ZETA_MIN + f32(wave_index) * band_width;
        let zeta_band_max = zeta_band_min + band_width;

        // --- Compute the period for this band ---
        // Period = PERIODICITY * (largest wavelength in this band).
        // This ensures the buffer is wide enough for at least 2 full cycles
        // of the slowest (longest) wave without aliasing at the wrap point.
        // Matches: m_period = periodicity * pow(2, zeta_max) in ProfileBuffer.h
        let max_wavelength_in_band = pow(2.0, zeta_band_max);
        let period = PERIODICITY * max_wavelength_in_band;

        // --- Convert profile index to physical position p (in metres) ---
        // p ranges from 0 to period across the buffer.
        // This is the coordinate along the wave's direction of travel.
        // Matches: p = (i * m_period) / resolution in ProfileBuffer::precompute
        let p = f32(p_idx) / f32(PROFILE_BUFFER_LENGTH) * period;

        // --- Periodicity blend weights ---
        // Instead of evaluating one wave at p, we blend between the wave
        // evaluated at p (phase1) and at p-period (phase2).
        //
        // Why: summing many wavelengths gives a profile that is not naturally
        // periodic over `period`. Naively wrapping the buffer would create a
        // visible seam. The blend enforces:
        //   - at p=0:      weight1=0, weight2=1 → only phase2 contributes
        //   - at p=period: weight1=1, weight2=0 → only phase1 contributes
        //     but phase1 at p=period == phase2 at p=0 (both = -omega*t)
        //     so the buffer value is the same at both ends → C1 continuity.
        //
        // cubic_bump provides smooth (C1) falloff so the seam has no kink.
        let weight1 = p / period;        // 0 → 1 across the buffer
        let weight2 = 1.0 - weight1;    // 1 → 0 across the buffer

        // --- Numerically integrate over zeta (midpoint rule) ---
        // This computes equation (21) from the paper:
        //
        //   Psi(p) = integral from zeta_min to zeta_max of:
        //              wavelength * spectrum(zeta) * gerstner(k*p - omega*t, k) dzeta
        //
        // The `wavelength` factor is the Jacobian from changing integration
        // variable: since zeta = log2(wavelength), dk/dzeta = k * ln2, and
        // wavelength ≈ k / ln2 absorbs that factor to give correctly scaled output.
        //
        // We use `dz` as the integration weight (midpoint rectangle rule),
        // matching Integration.h::integrate() from the reference.
        var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        let dz = (zeta_band_max - zeta_band_min) / f32(INTEGRATION_STEPS);

        for (var i: i32 = 0; i < INTEGRATION_STEPS; i++) {

            // Midpoint of this sub-interval in zeta space.
            // Matches: x = x_min + 0.5 * dx  (first node) in Integration.h
            let zeta = zeta_band_min + (f32(i) + 0.5) * dz;

            let wave_length = pow(2.0, zeta);
            let wave_number = TAU / wave_length;
            let omega       = dispersion_relation(wave_number);

            // phase1: the wave evaluated at position p.
            // Matches: phase1 = waveNumber * p - dispersionRelation(waveNumber) * time
            let phase1 = wave_number * p
                       - omega * settings.elapsed_time * TIME_MULT;

            // phase2: the wave evaluated as if p were wrapped back by one period.
            // This is what gets blended in near the start of the buffer to
            // ensure the seam is seamless.
            // Matches: phase2 = waveNumber * (p - m_period) - dispersionRelation(...) * time
            let phase2 = wave_number * (p - period)
                       - omega * settings.elapsed_time * TIME_MULT;

            let spec_weight = spectrum(zeta, settings.wind_speed);

            // Blend the two phase evaluations with cubic_bump weights,
            // then scale by wavelength * spectrum density.
            // This is the full integrand from ProfileBuffer::precompute.
            let contribution = wave_length * spec_weight * (
                cubic_bump(weight1) * gerstner_wave(phase1, wave_number) +
                cubic_bump(weight2) * gerstner_wave(phase2, wave_number)
            );

            // Accumulate using the midpoint rectangle rule weight dz.
            result += dz * contribution;
        }

        // --- Store the result ---
        // The vec4 layout matches what waterSurface() in WaveGrid.cpp reads:
        //   .x (r) = horizontal Gerstner displacement  → used for surface position offset
        //   .y (g) = vertical height                   → the actual water height
        //   .z (b) = d(horizontal) / dp                → used to build surface tangent tx
        //   .w (a) = d(vertical)   / dp                → used to build surface tangent ty
        //
        // In your reconstruction shader, sample this texture and use all 4 channels.
        // The normal is then cross(tx, ty) where tx and ty are built from .z and .w.
        textureStore(
            profile_out,
            vec2<i32>(i32(p_idx), i32(wave_index)),
            vec4<f32>(result.x, result.y, result.z, result.w)
        );
    }
}