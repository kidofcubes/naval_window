//! water_lod.rs — Quadtree LOD mesh for ocean rendering in Bevy.
//!
//! Generates a dynamic mesh with high vertex density near the camera and
//! lower density further away. T-junctions between LOD levels are hidden
//! using skirt geometry (thin curtains hanging below the water surface).
//!
//! # Usage
//! ```rust
//! app.add_plugins(WaterLodPlugin::default());
//! ```
//! Then replace the placeholder `StandardMaterial` in `setup_water_mesh`
//! with your custom water shader material.
//!
//! # Cargo.toml
//! ```toml
//! [dependencies]
//! bevy = "0.15"
//! ```

use bevy::{
    prelude::*,
};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
// ─── Configuration ─────────────────────────────────────────────────────────────

/// Vertices per edge of each patch. 17 → 16×16 quads per patch.
/// Increasing this raises per-patch detail but also per-patch triangle count.
// const PATCH_VERTS: usize = 17;
// const PATCH_VERTS: usize = 9;
const PATCH_VERTS: usize = 3;

/// Total world-space width/height of the water plane in meters.
pub const WATER_SIZE: f32 = 16384.0;

/// Maximum quadtree depth. At level N, patch world size = WATER_SIZE / 2^N.
/// With WATER_SIZE = 4096 and MAX_LEVELS = 7, finest patches are 32 m wide.
const MAX_LEVELS: u32 = 13;

/// Subdivide a node when the camera is closer than node.size × SPLIT_FACTOR.
/// Increase to subdivide more aggressively (more triangles near camera).
const SPLIT_FACTOR: f32 = 32.0;

/// Depth the skirt curtain extends below y = 0 to hide LOD cracks.
/// Should be deeper than your maximum wave trough displacement.
const SKIRT_DEPTH: f32 = 20.0;

/// Minimum camera movement (meters) before the LOD mesh is rebuilt.
/// Prevents rebuilding every frame when the camera is nearly stationary.
const REBUILD_THRESHOLD: f32 = 1.0;

// ─── Plugin ────────────────────────────────────────────────────────────────────

pub struct WaterLodPlugin {}

impl Plugin for WaterLodPlugin {
    fn build(&self, app: &mut App) {
        app
            // .insert_resource(WaterLodConfig { center: self.center })
            // .add_systems(Startup, setup_water_mesh)
            .add_systems(Update, update_water_lod);
    }
}

// ─── Resources & Components ────────────────────────────────────────────────────


/// Marker component attached to the water mesh entity.
#[derive(Component)]
pub struct WaterLodMesh {
    /// XZ camera position the last time the mesh was rebuilt.
    pub last_camera_pos: Vec3,
    pub center: Vec2,
}

// ─── Quadtree ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct QuadNode {
    /// World-space XZ center of this node.
    center: Vec2,
    /// Full side length in world space.
    size: f32,
    /// Depth from root (root = level 0).
    level: u32,
    /// [SW, SE, NW, NE] children, or None if this is a leaf node.
    children: Option<Box<[QuadNode; 4]>>,
}

impl QuadNode {
    fn new(center: Vec2, size: f32, level: u32) -> Self {
        Self { center, size, level, children: None }
    }

    /// Recursively subdivide the tree based on camera XZ distance.
    /// Nodes closer than `size × SPLIT_FACTOR` are split into four children.
    fn subdivide(&mut self, camera: Vec3) {
        if self.level >= MAX_LEVELS {
            return;
        }

        let dist = camera.distance(vec3(self.center.x, 0.0, self.center.y));
        if dist > self.size * SPLIT_FACTOR {
            return; // Far enough away — remain a leaf.
        }

        let half = self.size * 0.5;
        let quarter = self.size * 0.25;

        // Offsets for SW / SE / NW / NE children in XZ space.
        let mut children = [
            QuadNode::new(self.center + Vec2::new(-quarter, -quarter), half, self.level + 1),
            QuadNode::new(self.center + Vec2::new( quarter, -quarter), half, self.level + 1),
            QuadNode::new(self.center + Vec2::new(-quarter,  quarter), half, self.level + 1),
            QuadNode::new(self.center + Vec2::new( quarter,  quarter), half, self.level + 1),
        ];

        for child in &mut children {
            child.subdivide(camera);
        }

        self.children = Some(Box::new(children));
    }

    /// Append all leaf nodes (nodes without children) to `out`.
    fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a QuadNode>) {
        match &self.children {
            Some(children) => children.iter().for_each(|c| c.collect_leaves(out)),
            None => out.push(self),
        }
    }
}

// ─── Mesh Generation ───────────────────────────────────────────────────────────

/// Build a complete LOD `Mesh` for the given camera XZ position.
/// Called on startup and whenever the camera moves past `REBUILD_THRESHOLD`.
pub(crate) fn build_lod_mesh(camera_xz: Vec3, config: &WaterLodMesh) -> Mesh {
    // Subdivide the quadtree based on camera proximity.
    let mut root = QuadNode::new(config.center, WATER_SIZE, 0);
    root.subdivide(camera_xz);

    let mut leaves = Vec::new();
    root.collect_leaves(&mut leaves);

    // Accumulate geometry from every leaf patch.
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals:   Vec<[f32; 3]> = Vec::new();
    let mut uvs:       Vec<[f32; 2]> = Vec::new();
    let mut indices:   Vec<u32>       = Vec::new();

    for node in &leaves {
        emit_patch(
            node.center,
            node.size,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut indices,
        );
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL,   normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0,     uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Emit vertex and index data for one PATCH_VERTS×PATCH_VERTS surface patch,
/// plus four skirt curtains on its edges to hide cracks at LOD boundaries.
fn emit_patch(
    center:    Vec2,
    size:      f32,
    positions: &mut Vec<[f32; 3]>,
    normals:   &mut Vec<[f32; 3]>,
    uvs:       &mut Vec<[f32; 2]>,
    indices:   &mut Vec<u32>,
) {
    let base = positions.len() as u32;
    let half = size * 0.5;
    let n    = PATCH_VERTS;

    // ── Surface grid ──────────────────────────────────────────────────────────
    //
    // Vertices are laid out row-major. Row 0 is the –Z edge, row n–1 is +Z.
    // Within each row, column 0 is –X and column n–1 is +X.
    // Y is always 0 here — your vertex shader displaces it using the wave field.
    //
    // World-space UV (uv_0) covers [0,1]² across the full WATER_SIZE domain
    // so you can sample the amplitude grid at the correct position in a shader.

    for row in 0..n {
        for col in 0..n {
            let tx = col as f32 / (n - 1) as f32; // [0, 1] across patch width
            let tz = row as f32 / (n - 1) as f32; // [0, 1] across patch depth

            let x = center.x - half + tx * size;
            let z = center.y - half + tz * size;

            positions.push([x, 0.0, z]);
            // Normals start pointing straight up; the fragment shader overwrites
            // these analytically using the wave field gradient (see water shader).
            normals.push([0.0, 1.0, 0.0]);
            // Store world-space UV so the shader can locate this vertex in the
            // amplitude grid regardless of which patch it belongs to.
            uvs.push([
                (x - (-WATER_SIZE * 0.5)) / WATER_SIZE,
                (z - (-WATER_SIZE * 0.5)) / WATER_SIZE,
            ]);
        }
    }

    // ── Surface triangles ─────────────────────────────────────────────────────
    //
    // Each quad → two CCW triangles (winding from above, y-up):
    //
    //   tl ── tr
    //   │  ╲  │
    //   bl ── br
    //
    //   tri 1: tl → bl → tr
    //   tri 2: tr → bl → br

    for row in 0..(n - 1) {
        for col in 0..(n - 1) {
            let tl = base + (row * n + col) as u32;
            let tr = tl + 1;
            let bl = base + ((row + 1) * n + col) as u32;
            let br = bl + 1;

            indices.extend_from_slice(&[tl, bl, tr]);
            indices.extend_from_slice(&[tr, bl, br]);
        }
    }

    // ── Skirt curtains ────────────────────────────────────────────────────────
    //
    // Without skirts, adjacent patches at different LOD levels have mismatched
    // edges — the finer patch has more vertices along the shared edge than the
    // coarser patch, leaving visible cracks. Rather than doing complex neighbour
    // stitching, we hang a thin curtain of geometry straight down from each
    // patch edge to SKIRT_DEPTH below y=0. The curtain covers any crack from
    // any viewing angle above the waterline.

    for edge in [SkirtEdge::South, SkirtEdge::North, SkirtEdge::West, SkirtEdge::East] {
        emit_skirt(base, edge, positions, normals, uvs, indices);
    }
}

/// Which edge of the patch this skirt hangs from.
#[derive(Clone, Copy)]
enum SkirtEdge { North, South, East, West }

/// Emit one skirt curtain — a strip of quads hanging from a patch edge to
/// y = -SKIRT_DEPTH.
fn emit_skirt(
    patch_base: u32,
    edge:       SkirtEdge,
    positions:  &mut Vec<[f32; 3]>,
    normals:    &mut Vec<[f32; 3]>,
    uvs:        &mut Vec<[f32; 2]>,
    indices:    &mut Vec<u32>,
) {
    let n = PATCH_VERTS;

    // Surface vertex indices along the requested edge, in order left→right
    // or bottom→top as viewed from outside the patch.
    let surface_indices: Vec<u32> = match edge {
        SkirtEdge::South => (0..n).map(|c| patch_base + c as u32).collect(),
        SkirtEdge::North => (0..n).map(|c| patch_base + ((n - 1) * n + c) as u32).collect(),
        SkirtEdge::West  => (0..n).map(|r| patch_base + (r * n) as u32).collect(),
        SkirtEdge::East  => (0..n).map(|r| patch_base + (r * n + n - 1) as u32).collect(),
    };

    // Emit one "bottom" skirt vertex directly below each surface edge vertex.
    let skirt_base = positions.len() as u32;
    for &surf_idx in &surface_indices {
        let [x, _, z] = positions[surf_idx as usize];
        positions.push([x, -SKIRT_DEPTH, z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([
            (x + WATER_SIZE * 0.5) / WATER_SIZE,
            (z + WATER_SIZE * 0.5) / WATER_SIZE,
        ]);
    }

    // Connect surface edge verts to their skirt copies as a quad strip.
    //
    //   s0 ── s1      (y = 0, surface)
    //   │  ╲  │
    //   b0 ── b1      (y = -SKIRT_DEPTH)
    //
    // Winding doesn't need to be perfect here — skirts are only visible
    // from the side and will be occluded by the main surface patch above.
    for i in 0..(n as u32 - 1) {
        let s0 = surface_indices[i as usize];
        let s1 = surface_indices[i as usize + 1];
        let b0 = skirt_base + i;
        let b1 = skirt_base + i + 1;

        indices.extend_from_slice(&[s0, s1, b0]);
        indices.extend_from_slice(&[s1, b1, b0]);
    }
}

// ─── Bevy Systems ──────────────────────────────────────────────────────────────

// fn setup_water_mesh(
//     mut commands:  Commands,
//     mut meshes:    ResMut<Assets<Mesh>>,
//     mut materials: ResMut<Assets<StandardMaterial>>,
// ) {
//     let mesh     = build_lod_mesh(Vec3::ZERO, &config);
//     let mesh_h   = meshes.add(mesh);
//
//     // ── Replace this material with your custom water shader material ──────────
//     let material = materials.add(StandardMaterial {
//         base_color:           Color::srgb(0.04, 0.28, 0.46),
//         perceptual_roughness: 0.05,
//         reflectance:          0.9,
//         ..default()
//     });
//
//     commands.spawn((
//         Mesh3d(mesh_h),
//         MeshMaterial3d(material),
//         Transform::default(),
//         WaterLodMesh {
//             last_camera_pos: Vec3::ZERO,
//             center:           Vec2::ZERO,
//         },
//     ));
// }

fn update_water_lod(
    mut water_q:  Query<(&mut WaterLodMesh, &Mesh3d)>,
    camera_q:     Query<&Transform, With<Camera3d>>,
    mut meshes:   ResMut<Assets<Mesh>>,
) {
    let Ok(camera_tf) = camera_q.single() else { return };

    let camera_pos = camera_tf.translation;
    for (mut water, mesh_handle) in water_q.iter_mut() {
        // Skip rebuild if the camera hasn't moved far enough.
        if camera_pos.distance(water.last_camera_pos) < REBUILD_THRESHOLD {
            return;
        }

        water.last_camera_pos = camera_pos;

        // Hot-swap the existing mesh asset so the entity handle stays valid.
        if let Some(mesh) = meshes.get_mut(&mesh_handle.0) {
            *mesh = build_lod_mesh(camera_pos, water.as_ref());
        }
        water.last_camera_pos = camera_pos;
    }

    // let Ok((mut water, mesh_handle)) = water_q.single_mut() else { return };
}
