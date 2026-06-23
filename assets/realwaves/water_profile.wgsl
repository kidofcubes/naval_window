#import "water_common.wgsl"::{
    GRID_X_SIZE, GRID_Z_SIZE, RENDER_X_SIZE, RENDER_Z_SIZE,
    DIRECTIONS, TOTAL_ZETA_NUMBERS, PROFILE_BUFFER_LENGTH,
    PERIODICITY, ZETA_MIN, ZETA_MAX, DIRECTIONAL_OFFSETS,
    GRAVITY, TIME_MULT, TAU, WaterSettings, WaterDebugSettings, VoxelSettings, WaterWaveGroupSpeeds,
    index_to_radians, layer_index
}

@group(0) @binding(0) var profile_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var<uniform> settings: WaterSettings;

const INTEGRATION_STEPS: u32 = 64;
fn spectrum(zeta: f32, wind_speed: f32) -> f32 {
    let A = pow(1.1, 1.5 * zeta);
    let B = exp(-1.8038897788076411 * pow(4.0, zeta) / pow(wind_speed, 4.0));
    return 0.139098 * sqrt(A * B);
}

fn dispersionRelation(k: f32) -> f32 {
    return sqrt(GRAVITY * k);
}

fn cubic_bump(x: f32) -> f32 {
    if abs(x) >= 1.0 {
        return 0.0;
    }
    return x * x * ((2.0 * abs(x)) - 3.0) + 1.0;
}

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

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let profile_index = global_id.x;
    if profile_index >= PROFILE_BUFFER_LENGTH {
        return;
    }

    let zeta_full_range = ZETA_MAX - ZETA_MIN;

    for (var zeta_index: u32 = 0u; zeta_index < TOTAL_ZETA_NUMBERS; zeta_index++) {

        let min_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) - 0.0) / f32(TOTAL_ZETA_NUMBERS)));
        let max_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 1.0) / f32(TOTAL_ZETA_NUMBERS)));
        let mid_zeta = ZETA_MIN + (zeta_full_range * ((f32(zeta_index) + 0.5) / f32(TOTAL_ZETA_NUMBERS)));
        let period: f32 = f32(PERIODICITY) * pow(2,mid_zeta);

        let p = (f32(profile_index) * period) / f32(PROFILE_BUFFER_LENGTH);

        let dx = (max_zeta - min_zeta) / f32(INTEGRATION_STEPS);

        var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        for(var i: u32 = 0u; i < INTEGRATION_STEPS; i++){
            let zeta = min_zeta + ((max_zeta - min_zeta) * ((f32(i) + 0.5) / f32(INTEGRATION_STEPS)));

//            let zeta = max_zeta;

            let waveLength = pow(2, zeta);
            let waveNumber = TAU / waveLength;
//            let phase1 = waveNumber * (p         ) - (dispersionRelation(waveNumber) * (settings.elapsed_time+100.0));
//            let phase2 = waveNumber * (p - period) - (dispersionRelation(waveNumber) * (settings.elapsed_time+100.0));
            let phase1 = (waveNumber * (p         )) - (dispersionRelation(waveNumber) * ((settings.elapsed_time)));
            let phase2 = (waveNumber * (p - period)) - (dispersionRelation(waveNumber) * ((settings.elapsed_time)));

            let weight1 = p / period;
            let weight2 = 1 - weight1;
            result += dx * waveLength * spectrum(zeta, 3.0) *
                   (cubic_bump(weight1) * gerstner_wave(phase1, waveNumber) +
                    cubic_bump(weight2) * gerstner_wave(phase2, waveNumber));

//            result += dx * waveLength * spectrum(zeta, 3.0) *
//                   (gerstner_wave(phase1, waveNumber));
        }

        textureStore(
            profile_out,
            vec2<i32>(i32(profile_index), i32(zeta_index)),
            vec4<f32>(result.x, result.y, result.z, result.w)
        );
    }
}