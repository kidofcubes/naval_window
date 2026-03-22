//! Mesh optimization module for combining and simplifying ship meshes
//!
//! This module provides functionality to combine multiple mesh parts into a single mesh
//! and simplify them using the meshopt library for reduced triangle counts.

use anyhow::anyhow;
use bevy::prelude::*;
use bevy::mesh::{Mesh, Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::asset::RenderAssetUsages;
use meshopt::{simplify_sloppy, typed_to_bytes, VertexDataAdapter};
use parry3d::shape::{SharedShape, ConvexPolyhedron};
use parry3d::transformation::vhacd::VHACDParameters;
use parry3d::math::Pose3;
use crate::voxelize::{triangle_box_intersection, voxel_to_world, world_to_voxel, VoxelGrid, VOXEL_BOUNDARY};

/// Vertex struct for meshopt processing
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Vertex {
    position: [f32; 3],
}

/// Combine multiple meshes with their transforms into a single mesh
///
/// Each mesh is transformed to world space using its associated GlobalTransform,
/// and all vertices and indices are accumulated into a single combined mesh.
pub fn combine_meshes(
    mesh_transform_pairs: &[(&Mesh, &GlobalTransform)],
) -> Option<Mesh> {
    if mesh_transform_pairs.is_empty() {
        return None;
    }

    let mut all_vertices: Vec<[f32; 3]> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    for (mesh, transform) in mesh_transform_pairs {
        // Extract positions
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(p)) => p,
            _ => continue,
        };

        // Get current vertex offset for index remapping
        let vertex_offset = all_vertices.len() as u32;

        // Transform vertices to world space and add
        for pos in positions {
            let world_pos = transform.transform_point(Vec3::from(*pos));
            all_vertices.push([world_pos.x, world_pos.y, world_pos.z]);
        }

        // Extract and offset indices
        let indices: Vec<u32> = match mesh.indices() {
            Some(Indices::U32(idx)) => idx.clone(),
            Some(Indices::U16(idx)) => idx.iter().map(|&i| i as u32).collect(),
            None => {
                // No indices means sequential vertices for triangle list
                if mesh.primitive_topology() == PrimitiveTopology::TriangleList {
                    (0..positions.len() as u32).collect()
                } else {
                    continue;
                }
            }
        };

        for idx in indices {
            all_indices.push(idx + vertex_offset);
        }
    }

    if all_vertices.is_empty() || all_indices.is_empty() {
        return None;
    }

    // Build combined mesh
    let mut combined = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::all(),
    );
    combined.insert_attribute(Mesh::ATTRIBUTE_POSITION, all_vertices);
    combined.insert_indices(Indices::U32(all_indices));
    println!("are there indices? {:?}", combined.indices().is_some());
    // combined.compute_flat_normals();
    // combined.duplicate_vertices();
    // combined.compute_flat_normals();
    combined.compute_smooth_normals();
    println!("are there indices? {:?}", combined.indices().is_some());

    Some(combined)
}

/// Simplify a mesh using meshopt
///
/// The target_error parameter controls the maximum allowed error during simplification.
/// This is relative to mesh extents:
/// - 0.01 = 1% error (high quality, less simplification)
/// - 0.1 = 10% error (moderate quality)
/// - 1.0 = 100% error (aggressive simplification)
///
/// The mesh will be simplified as much as possible while staying within
/// the specified error threshold.
pub fn simplify_mesh(
    mesh: &Mesh,
    target_error: f32,
) -> anyhow::Result<Mesh> {
    // Extract vertices
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(p)) => p,
        _ => return Err(anyhow!("no attribute")),
    };

    // Extract indices
    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(idx)) => idx.clone(),
        Some(Indices::U16(idx)) => idx.iter().map(|&i| i as u32).collect(),
        None => return Err(anyhow!("no indices")) ,
    };

    if indices.is_empty() || positions.is_empty() {
        return Err(anyhow!("no indices or vertices")) ;
    }

    // Convert to Vertex structs for meshopt
    let vertex_data: Vec<Vertex> = positions
        .iter()
        .map(|p| Vertex { position: *p })
        .collect();

    // Create byte slice for meshopt's VertexDataAdapter using typed_to_bytes
    let vertex_bytes = typed_to_bytes(&vertex_data);

    let adapter = match VertexDataAdapter::new(
        vertex_bytes,
        std::mem::size_of::<Vertex>(),
        0, // position offset in Vertex struct
    ) {
        Ok(a) => a,
        Err(_) => return Err(anyhow!("meshopt adapter error")),
    };

    // Use minimum target count (3 indices = 1 triangle) to let error be the limiting factor
    let target_indices = 3;

    // Simplify the mesh using sloppy simplification
    // This sacrifices some mesh appearance for better performance
    // and doesn't require SimplifyOptions which isn't exported in v0.6.2
    let simplified_indices = simplify_sloppy(
        &indices,
        &adapter,
        target_indices,
        target_error,
        None,
    );

    if simplified_indices.is_empty() {
        return Err(anyhow!("no simplified indices"));
    }

    // Build simplified mesh (reuse original vertices, only indices change)
    let mut simplified = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::all(),
    );
    simplified.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
    simplified.insert_indices(Indices::U32(simplified_indices));
    simplified.duplicate_vertices();
    simplified.compute_flat_normals();

    Ok(simplified)
}
/// Remove unused vertices from a mesh after simplification
///
/// The simplify function only reduces indices - unused vertices remain.
/// This function creates a new mesh with only the vertices that are
/// actually referenced by the indices.
pub fn compact_mesh(mesh: &Mesh) -> Option<Mesh> {
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(p)) => p,
        _ => return None,
    };

    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(idx)) => idx.clone(),
        Some(Indices::U16(idx)) => idx.iter().map(|&i| i as u32).collect(),
        None => return None,
    };

    // Find which vertices are actually used
    let mut used_vertices = vec![false; positions.len()];
    for &idx in &indices {
        if (idx as usize) < used_vertices.len() {
            used_vertices[idx as usize] = true;
        }
    }

    // Create mapping from old indices to new indices
    let mut old_to_new: Vec<u32> = vec![0; positions.len()];
    let mut new_positions: Vec<[f32; 3]> = Vec::new();

    for (old_idx, &is_used) in used_vertices.iter().enumerate() {
        if is_used {
            old_to_new[old_idx] = new_positions.len() as u32;
            new_positions.push(positions[old_idx]);
        }
    }

    // Remap indices
    let new_indices: Vec<u32> = indices
        .iter()
        .map(|&idx| old_to_new[idx as usize])
        .collect();

    // Build compacted mesh
    let mut compacted = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::all(),
    );
    compacted.insert_attribute(Mesh::ATTRIBUTE_POSITION, new_positions);
    compacted.insert_indices(Indices::U32(new_indices));
    compacted.compute_flat_normals();

    Some(compacted)
}

/// Decompose a mesh into convex hulls using V-HACD algorithm
///
/// Returns a vector of convex hull meshes that approximate the original mesh.
/// Useful for physics collision shapes where convex geometry is required.
///
/// Parameters:
/// - `mesh`: Input mesh to decompose
/// - `params`: Optional VHACDParameters for fine-tuning (uses defaults if None)
pub fn convex_decomposition(
    mesh: &Mesh,
    params: Option<VHACDParameters>,
) -> anyhow::Result<Vec<Mesh>> {
    // 1. Extract vertices as Vec<Vec3>
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(p)) => p,
        _ => return Err(anyhow!("no position attribute")),
    };

    let vertices: Vec<Vec3> = positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();

    // 2. Extract indices as Vec<[u32; 3]>
    let indices: Vec<[u32; 3]> = match mesh.indices() {
        Some(Indices::U32(idx)) => idx.chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect(),
        Some(Indices::U16(idx)) => idx.chunks(3)
            .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
            .collect(),
        None => return Err(anyhow!("no indices")),
    };

    // 3. Run convex decomposition
    let params = params.unwrap_or_default();
    let decomposed = SharedShape::convex_decomposition_with_params(
        &vertices,
        &indices,
        &params,
    );

    // 4. Convert each convex hull back to Bevy Mesh
    let mut result_meshes = Vec::new();

    // decomposed is a Compound shape containing multiple convex parts
    for (pose, shape) in decomposed.as_compound().unwrap().shapes() {
        if let Some(convex) = shape.as_convex_polyhedron() {
            let hull_mesh = convex_polyhedron_to_mesh(convex, pose)?;
            result_meshes.push(hull_mesh);
        }
    }

    Ok(result_meshes)
}

/// Convert a parry ConvexPolyhedron to a Bevy Mesh
fn convex_polyhedron_to_mesh(
    convex: &ConvexPolyhedron,
    pose: &Pose3,
) -> anyhow::Result<Mesh> {
    let points = convex.points();

    // Transform points by pose and convert to [f32; 3]
    let vertices: Vec<[f32; 3]> = points
        .iter()
        .map(|p| {
            let transformed = pose.rotation * *p + pose.translation;
            [transformed.x, transformed.y, transformed.z]
        })
        .collect();

    // Get triangle indices by triangulating convex hull faces
    let mut flat_indices: Vec<u32> = Vec::new();
    for face in convex.faces() {
        let first_idx = face.first_vertex_or_edge;
        let num_vertices = face.num_vertices_or_edges;
        // Fan triangulation of the convex face
        for i in 1..num_vertices - 1 {
            let v0 = convex.vertices_adj_to_face()[first_idx as usize] as u32;
            let v1 = convex.vertices_adj_to_face()[(first_idx + i) as usize] as u32;
            let v2 = convex.vertices_adj_to_face()[(first_idx + i + 1) as usize] as u32;
            flat_indices.push(v0);
            flat_indices.push(v1);
            flat_indices.push(v2);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::all(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_indices(Indices::U32(flat_indices));
    mesh.duplicate_vertices();
    mesh.compute_flat_normals();

    Ok(mesh)
}


pub fn get_boundary_mesh_conservative(grid: &VoxelGrid, mesh: Mesh, transform: &GlobalTransform) -> Option<Mesh>{
    // Extract vertex positions
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(positions)) => positions,
        _ => return None,
    };

    // Extract indices
    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(indices)) => indices.clone(),
        Some(Indices::U16(indices)) => indices.iter().map(|&i| i as u32).collect(),
        None => {
            if mesh.primitive_topology() == PrimitiveTopology::TriangleList {
                println!("Mesh has no indices!!!!!!!!!!!!!");
                (0..positions.len() as u32).collect()
            } else {
                return None;
            }
        }
    };
    let mut to_keep_indices: Vec<u32> = Vec::with_capacity(indices.len());

    // Process each triangle
    for (i,triangle_indices) in indices.chunks(3).enumerate() {
        if triangle_indices.len() != 3 {
            continue;
        }

        let v0 = Vec3A::from(transform.transform_point(Vec3::from(
            positions[triangle_indices[0] as usize],
        )));
        let v1 = Vec3A::from(transform.transform_point(Vec3::from(
            positions[triangle_indices[1] as usize],
        )));
        let v2 = Vec3A::from(transform.transform_point(Vec3::from(
            positions[triangle_indices[2] as usize],
        )));

        // if triangle_intersects_boundary(grid, v0, v1, v2) {
        to_keep_indices.push(i as u32);
        // }
    }
    // return Some(filter_mesh_triangles(&mesh, &to_keep_indices));
    return Some(filter_mesh_triangles(&mesh, |a,b,c| {
        let mut contains = 0;
        for pos in [a,b,c] {
            let pos = round_to_voxel(Vec3A::from(transform.transform_point(pos.into())) - grid.offset, grid.voxel_size);
            if(near_voxel(grid, pos.0, pos.1, pos.2, 3, |voxel| { (voxel & VOXEL_BOUNDARY) != 0})){
                contains+=1;
            }
        }
        return contains >= 1;

        // return triangle_intersects_boundary(grid, a.into(), b.into(), c.into());
    }));
}
fn round_to_voxel(vec: Vec3A, voxel_size: f32) -> (i32, i32, i32) {
    return ((vec.x / voxel_size).round() as i32, (vec.y / voxel_size).round() as i32, (vec.z / voxel_size).round() as i32)
}

fn near_voxel(grid: &VoxelGrid, x: i32, y: i32, z: i32, radius: i32, predicate: impl Fn(u8) -> bool) -> bool{
    for x_offset in -radius as i32..radius as i32 {
        for y_offset in -radius as i32..radius as i32 {
            for z_offset in -radius as i32..radius as i32 {
                if let Some(voxel) = grid.get_voxel_bounds_check(x + x_offset, y + y_offset, z + z_offset) {
                    if predicate(voxel) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

fn triangle_intersects_boundary(grid: &VoxelGrid, v0: Vec3A, v1: Vec3A, v2: Vec3A) -> bool {
    // Calculate triangle bounding box in voxel coordinates
    let min_world = v0.min(v1).min(v2);
    let max_world = v0.max(v1).max(v2);

    let min_voxel = world_to_voxel(grid, min_world);
    let max_voxel = world_to_voxel(grid, max_world);

    // Clamp to grid bounds and add 1-voxel padding for safety
    let start_x = (min_voxel.x - 1).max(0);
    let start_y = (min_voxel.y - 1).max(0);
    let start_z = (min_voxel.z - 1).max(0);
    let end_x = (max_voxel.x + 2).min(grid.dimensions.0 as i32);
    let end_y = (max_voxel.y + 2).min(grid.dimensions.1 as i32);
    let end_z = (max_voxel.z + 2).min(grid.dimensions.2 as i32);

    let half_size = grid.voxel_size * 0.5;

    // Test each voxel in the bounding box
    for z in start_z..end_z {
        for y in start_y..end_y {
            for x in start_x..end_x {
                if let Some(boundary) = grid.is_boundary(x as u32, y as u32, z as u32){
                    if(boundary){
                        let voxel_center = voxel_to_world(grid, x, y, z);
                        // Test if triangle intersects this voxel
                        if triangle_box_intersection(v0, v1, v2, voxel_center, half_size) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

/// Rebuild `source` keeping only triangles for which `predicate` returns `true`.
///
/// # Arguments
/// * `source`    — The original `Mesh`. Must use `TriangleList` topology.
/// * `predicate` — Called once per triangle with its three vertex positions
///                 (in the mesh's local space). Return `true` to keep the
///                 triangle, `false` to discard it.
///
/// # Returns
/// A new `Mesh` with the same vertex attributes as the source, but containing
/// only the kept triangles. Unreferenced vertices are compacted out so the
/// output vertex buffer contains no dead entries.
///
/// # Panics
/// Panics if the mesh topology is not `TriangleList`, or if the mesh has no
/// `ATTRIBUTE_POSITION` data.
pub fn filter_mesh_triangles(
    source:    &Mesh,
    predicate: impl Fn(Vec3, Vec3, Vec3) -> bool,
) -> Mesh {
    assert_eq!(
        source.primitive_topology(),
        PrimitiveTopology::TriangleList,
        "filter_mesh_triangles only supports TriangleList topology"
    );

    // ── Extract source positions ───────────────────────────────────────────

    let positions = extract_positions(source);

    // ── Build index list ───────────────────────────────────────────────────

    // If the mesh has explicit indices, use them. If not (rare for real meshes
    // but valid), synthesize sequential indices 0,1,2,3,… so the rest of the
    // code is uniform.
    let src_indices: Vec<u32> = match source.indices() {
        Some(Indices::U32(v)) => v.clone(),
        Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
        None                  => (0..positions.len() as u32).collect(),
    };

    assert_eq!(
        src_indices.len() % 3,
        0,
        "Index count is not a multiple of 3 — mesh is malformed"
    );

    // ── Filter triangles ───────────────────────────────────────────────────

    // Collect the index triples for triangles that pass the predicate.
    let kept_triples: Vec<[u32; 3]> = src_indices
        .chunks_exact(3)
        .filter_map(|tri| {
            let [i0, i1, i2] = [tri[0], tri[1], tri[2]];
            let a = positions[i0 as usize];
            let b = positions[i1 as usize];
            let c = positions[i2 as usize];
            if predicate(a, b, c) { Some([i0, i1, i2]) } else { None }
        })
        .collect();

    // ── Compact vertex buffer ──────────────────────────────────────────────

    // Build a mapping from old vertex index → new compact index, only for
    // vertices that are actually referenced by a kept triangle.
    // Vertices referenced by no kept triangle are simply dropped.
    let compact = build_compact_map(&kept_triples, positions.len());

    // Remap triangle indices to the new compact space.
    let new_indices: Vec<u32> = kept_triples
        .iter()
        .flat_map(|&[i0, i1, i2]| {
            [
                compact[i0 as usize],
                compact[i1 as usize],
                compact[i2 as usize],
            ]
        })
        .collect();

    // ── Assemble output mesh ───────────────────────────────────────────────

    let mut out = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );

    // Copy every vertex attribute from the source, compacting each one using
    // the same old→new index map. This preserves normals, UVs, tangents, and
    // any custom attributes without needing to know about them explicitly.
    copy_compacted_attributes(source, &mut out, &compact);

    out.insert_indices(Indices::U32(new_indices));

    out
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/// Extract positions from the mesh as `Vec<Vec3>`.
fn extract_positions(mesh: &Mesh) -> Vec<Vec3> {
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(v)) => {
            v.iter().map(|&[x, y, z]| Vec3::new(x, y, z)).collect()
        }
        _ => panic!("Mesh has no Float32x3 ATTRIBUTE_POSITION data"),
    }
}

/// For each old vertex index, compute the new compact index it maps to.
/// Vertices not referenced by any kept triangle get `u32::MAX` as a sentinel.
fn build_compact_map(kept: &[[u32; 3]], vertex_count: usize) -> Vec<u32> {
    // Mark which old indices are referenced.
    let mut used = vec![false; vertex_count];
    for &[i0, i1, i2] in kept {
        used[i0 as usize] = true;
        used[i1 as usize] = true;
        used[i2 as usize] = true;
    }

    // Assign sequential new indices to referenced vertices.
    let mut map     = vec![u32::MAX; vertex_count];
    let mut next_id = 0u32;
    for (old_idx, &is_used) in used.iter().enumerate() {
        if is_used {
            map[old_idx] = next_id;
            next_id      += 1;
        }
    }
    map
}

/// Copy all vertex attributes from `src` into `dst`, reordering/compacting
/// each attribute array according to `compact_map` (old index → new index).
///
/// Vertices whose compact_map entry is `u32::MAX` are not emitted.
fn copy_compacted_attributes(src: &Mesh, dst: &mut Mesh, compact_map: &[u32]) {
    let new_count = compact_map.iter().filter(|&&v| v != u32::MAX).count();

    for (attr_id, values) in src.attributes() {
        match values {
            VertexAttributeValues::Float32(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    compact_vec(v, compact_map, new_count, 0.0f32),
                );
            }
            VertexAttributeValues::Float32x2(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    compact_vec(v, compact_map, new_count, [0.0f32; 2]),
                );
            }
            VertexAttributeValues::Float32x3(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    compact_vec(v, compact_map, new_count, [0.0f32; 3]),
                );
            }
            VertexAttributeValues::Float32x4(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    compact_vec(v, compact_map, new_count, [0.0f32; 4]),
                );
            }
            VertexAttributeValues::Uint32(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    compact_vec(v, compact_map, new_count, 0u32),
                );
            }
            VertexAttributeValues::Sint32(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    compact_vec(v, compact_map, new_count, 0i32),
                );
            }
            VertexAttributeValues::Uint16x2(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    VertexAttributeValues::Uint16x2(compact_vec(v, compact_map, new_count, [0u16; 2])),
                );
            }
            VertexAttributeValues::Uint16x4(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    VertexAttributeValues::Uint16x4(compact_vec(v, compact_map, new_count, [0u16; 4])),
                );
            }
            VertexAttributeValues::Sint16x2(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    VertexAttributeValues::Sint16x2(compact_vec(v, compact_map, new_count, [0i16; 2])),
                );
            }
            VertexAttributeValues::Sint16x4(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    VertexAttributeValues::Sint16x4(compact_vec(v, compact_map, new_count, [0i16; 4])),
                );
            }
            VertexAttributeValues::Uint8x2(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    VertexAttributeValues::Uint8x2(compact_vec(v, compact_map, new_count, [0u8; 2])),
                );
            }
            VertexAttributeValues::Uint8x4(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    VertexAttributeValues::Uint8x4(compact_vec(v, compact_map, new_count, [0u8; 4])),
                );
            }
            VertexAttributeValues::Sint8x2(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    VertexAttributeValues::Sint8x2(compact_vec(v, compact_map, new_count, [0i8; 2])),
                );
            }
            VertexAttributeValues::Sint8x4(v) => {
                dst.insert_attribute(
                    attr_id.clone(),
                    VertexAttributeValues::Sint8x4(compact_vec(v, compact_map, new_count, [0i8; 4])),
                );
            }
            other => {
                warn!(
                    "filter_mesh_triangles: skipping unhandled attribute {:?}",
                    other
                );
            }
        }
    }
}

/// Generic compact helper — works for any `Copy + Default` element type.
/// Scatters `src[old_idx]` into `out[new_idx]` for every mapped vertex.
fn compact_vec<T: Copy>(
    src:       &[T],
    map:       &[u32],
    new_count: usize,
    default:   T,
) -> Vec<T> {
    let mut out = vec![default; new_count];
    for (old_idx, &new_idx) in map.iter().enumerate() {
        if new_idx != u32::MAX {
            out[new_idx as usize] = src[old_idx];
        }
    }
    out
}
