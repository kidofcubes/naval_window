//! Mesh optimization module for combining and simplifying ship meshes
//!
//! This module provides functionality to combine multiple mesh parts into a single mesh
//! and simplify them using the meshopt library for reduced triangle counts.

use anyhow::anyhow;
use bevy::prelude::*;
use bevy::mesh::{Mesh, Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::asset::RenderAssetUsages;
use meshopt::{simplify_sloppy, typed_to_bytes, VertexDataAdapter};

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
