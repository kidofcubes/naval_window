//! Voxelization module for converting meshes to 3D voxel grids
//!
//! This module provides functionality to convert iterators of meshes into
//! solid voxel representations with customizable resolution and voxel size constraints.

use bevy::prelude::*;
use bevy::mesh::{Mesh, Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::math::{Vec3A, Vec3Swizzles};

/// Configuration for voxelization process
#[derive(Debug, Clone)]
pub struct VoxelizationConfig {
    /// Total resolution of the voxel grid (cubic grid)
    pub resolution: u32,
    /// Optional minimum voxel size constraint
    pub min_voxel_size: Option<f32>,
    /// Optional maximum voxel size constraint  
    pub max_voxel_size: Option<f32>,
}

impl Default for VoxelizationConfig {
    fn default() -> Self {
        Self {
            resolution: 64,
            min_voxel_size: None,
            max_voxel_size: None,
        }
    }
}

/// 3D voxel grid representation
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    /// 3D grid of voxels stored as flat array
    pub voxels: Vec<bool>,
    /// Grid dimensions (width, height, depth)
    pub dimensions: (u32, u32, u32),
    /// minimum position of the voxel grid
    pub offset: Vec3A,
    /// Size of each voxel in world units
    pub voxel_size: f32,
}

impl VoxelGrid {
    /// Create a new empty voxel grid
    pub fn new(dimensions: (u32, u32, u32), offset: Vec3A) -> Self {
        let voxel_count = (dimensions.0 * dimensions.1 * dimensions.2) as usize;
        let voxel_size = calculate_voxel_size(dimensions);

        Self {
            voxels: vec![false; voxel_count],
            dimensions,
            offset,
            voxel_size,
        }
    }
    
    /// Get the linear index for a 3D coordinate
    pub fn get_index(&self, x: u32, y: u32, z: u32) -> Option<usize> {
        if x < self.dimensions.0 && y < self.dimensions.1 && z < self.dimensions.2 {
            Some((z * self.dimensions.0 * self.dimensions.1 + y * self.dimensions.0 + x) as usize)
        } else {
            None
        }
    }
    
    /// Set a voxel at the given coordinates
    pub fn set_voxel(&mut self, x: u32, y: u32, z: u32, filled: bool) -> bool {
        if let Some(index) = self.get_index(x, y, z) {
            self.voxels[index] = filled;
            true
        } else {
            false
        }
    }
    
    /// Check if a voxel is filled at the given coordinates
    pub fn is_filled(&self, x: u32, y: u32, z: u32) -> Option<bool> {
        self.get_index(x, y, z).map(|index| self.voxels[index])
    }
}

/// Calculate appropriate voxel size based on grid dimensions
fn calculate_voxel_size(dimensions: (u32, u32, u32)) -> f32 {
    // Use a default world size assumption or calculate based on dimensions
    // This assumes a reasonable world space size for the voxel grid
    let world_size = 100.0f32; // Default world space size
    world_size / (dimensions.0.max(dimensions.1).max(dimensions.2)) as f32
}

pub fn render_voxel_grid(grid: &VoxelGrid, gizmo: &mut Gizmos) {
    let voxel_size = grid.voxel_size;
    let offset = grid.offset;

    for z in 0..grid.dimensions.2 {
        for y in 0..grid.dimensions.1 {
            for x in 0..grid.dimensions.0 {
                if let Some(true) = grid.is_filled(x, y, z) {
                    let world_pos = offset + Vec3A::new(
                        (x as f32 + 0.5) * voxel_size,
                        (y as f32 + 0.5) * voxel_size,
                        (z as f32 + 0.5) * voxel_size,
                    );

                    // Draw wireframe cube for each filled voxel
                    gizmo.cube(
                        Transform::from_translation(world_pos.into())
                            .with_scale(Vec3::splat(voxel_size * 0.95)), // Slightly smaller for visual clarity
                        Color::srgba(0.0, 1.0, 0.0, 0.3), // Semi-transparent green
                    );
                }
            }
        }
    }
}


pub fn voxelize_meshes(
    mesh_transform_pairs: Vec<(&Mesh, &GlobalTransform)>,
    config: VoxelizationConfig,
) -> VoxelGrid {
    // ... [same bounds calculation as before] ...

    let mut min_bound = Vec3A::splat(f32::MAX);
    let mut max_bound = Vec3A::splat(f32::MIN);

    for (mesh, transform) in &mesh_transform_pairs {
        if let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            if let VertexAttributeValues::Float32x3(vertices) = positions {
                for vertex in vertices {
                    let world_pos = transform.transform_point(Vec3::from(*vertex));
                    let world_pos = Vec3A::from(world_pos);
                    min_bound = min_bound.min(world_pos);
                    max_bound = max_bound.max(world_pos);
                }
            }
        }
    }

    let padding = (max_bound - min_bound).max_element() * 0.01;
    min_bound -= Vec3A::splat(padding);
    max_bound += Vec3A::splat(padding);

    let bounds_size = max_bound - min_bound;
    let max_dimension = bounds_size.max_element();

    let mut voxel_size = max_dimension / config.resolution as f32;

    if let Some(min_size) = config.min_voxel_size {
        voxel_size = voxel_size.max(min_size);
    }
    if let Some(max_size) = config.max_voxel_size {
        voxel_size = voxel_size.min(max_size);
    }

    let dimensions = (
        (bounds_size.x / voxel_size).ceil() as u32,
        (bounds_size.y / voxel_size).ceil() as u32,
        (bounds_size.z / voxel_size).ceil() as u32,
    );

    let mut grid = VoxelGrid {
        voxels: vec![false; (dimensions.0 * dimensions.1 * dimensions.2) as usize],
        dimensions,
        offset: min_bound,
        voxel_size,
    };

    // Step 1: Surface voxelization
    for (mesh, transform) in &mesh_transform_pairs {
        voxelize_single_mesh(&mut grid, mesh, transform);
    }

    // Step 2: Fill interior using scanline algorithm
    fill_interior_floodfill(&mut grid);

    grid
}

/// Voxelize a single mesh into the grid
fn voxelize_single_mesh(grid: &mut VoxelGrid, mesh: &Mesh, transform: &GlobalTransform) {
    // Get vertex positions
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(positions)) => positions,
        _ => return, // Skip meshes without proper position data
    };

    // Get indices - handle both indexed and non-indexed meshes
    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(indices)) => indices.clone(),
        Some(Indices::U16(indices)) => indices.iter().map(|&i| i as u32).collect(),
        None => {
            // Non-indexed mesh - generate sequential indices
            if mesh.primitive_topology() == PrimitiveTopology::TriangleList {
                (0..positions.len() as u32).collect()
            } else {
                return; // Skip non-triangle meshes
            }
        }
    };

    // Process triangles
    for triangle_indices in indices.chunks(3) {
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

        voxelize_triangle(grid, v0, v1, v2);
    }
}

/// Conservative triangle voxelization using 3D DDA-style traversal
fn voxelize_triangle(grid: &mut VoxelGrid, v0: Vec3A, v1: Vec3A, v2: Vec3A) {
    // Calculate triangle bounding box in voxel coordinates
    let min_world = v0.min(v1).min(v2);
    let max_world = v0.max(v1).max(v2);

    let min_voxel = world_to_voxel(grid, min_world);
    let max_voxel = world_to_voxel(grid, max_world);

    // Clamp to grid bounds
    let start_x = min_voxel.x.max(0);
    let start_y = min_voxel.y.max(0);
    let start_z = min_voxel.z.max(0);
    let end_x = (max_voxel.x + 1).min(grid.dimensions.0 as i32);
    let end_y = (max_voxel.y + 1).min(grid.dimensions.1 as i32);
    let end_z = (max_voxel.z + 1).min(grid.dimensions.2 as i32);

    // Test each voxel in the bounding box
    for z in start_z..end_z {
        for y in start_y..end_y {
            for x in start_x..end_x {
                let voxel_center = voxel_to_world(grid, x, y, z);
                let half_size = grid.voxel_size * 0.5;

                // Test if triangle intersects this voxel
                if triangle_box_intersection(v0, v1, v2, voxel_center, half_size) {
                    grid.set_voxel(x as u32, y as u32, z as u32, true);
                }
            }
        }
    }
}

/// Convert world position to voxel grid coordinates
fn world_to_voxel(grid: &VoxelGrid, world_pos: Vec3A) -> IVec3 {
    let local_pos = world_pos - grid.offset;
    IVec3::new(
        (local_pos.x / grid.voxel_size).floor() as i32,
        (local_pos.y / grid.voxel_size).floor() as i32,
        (local_pos.z / grid.voxel_size).floor() as i32,
    )
}

/// Convert voxel coordinates to world position (center of voxel)
fn voxel_to_world(grid: &VoxelGrid, x: i32, y: i32, z: i32) -> Vec3A {
    grid.offset
        + Vec3A::new(
        (x as f32 + 0.5) * grid.voxel_size,
        (y as f32 + 0.5) * grid.voxel_size,
        (z as f32 + 0.5) * grid.voxel_size,
    )
}

/// Triangle-box intersection test using separating axis theorem
/// Based on Tomas Akenine-Möller's algorithm
fn triangle_box_intersection(v0: Vec3A, v1: Vec3A, v2: Vec3A, box_center: Vec3A, box_half_size: f32) -> bool {
    // Translate triangle to box space (box center at origin)
    let v0 = v0 - box_center;
    let v1 = v1 - box_center;
    let v2 = v2 - box_center;

    // Compute triangle edges
    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;

    // Test triangle plane
    let normal = e0.cross(e1);
    let d = -normal.dot(v0);
    let r = box_half_size * (normal.x.abs() + normal.y.abs() + normal.z.abs());
    if d.abs() > r {
        return false;
    }

    // Test the 9 edge cross products (separating axes)
    let box_half = Vec3A::splat(box_half_size);

    // Edge 0
    if !test_axis(e0.yz(), v0.yz(), v2.yz(), box_half.yz()) {
        return false;
    }
    if !test_axis(e0.zx(), v0.zx(), v2.zx(), box_half.zx()) {
        return false;
    }
    if !test_axis(e0.xy(), v0.xy(), v2.xy(), box_half.xy()) {
        return false;
    }

    // Edge 1
    if !test_axis(e1.yz(), v0.yz(), v2.yz(), box_half.yz()) {
        return false;
    }
    if !test_axis(e1.zx(), v0.zx(), v2.zx(), box_half.zx()) {
        return false;
    }
    if !test_axis(e1.xy(), v0.xy(), v2.xy(), box_half.xy()) {
        return false;
    }

    // Edge 2
    if !test_axis(e2.yz(), v0.yz(), v2.yz(), box_half.yz()) {
        return false;
    }
    if !test_axis(e2.zx(), v0.zx(), v2.zx(), box_half.zx()) {
        return false;
    }
    if !test_axis(e2.xy(), v0.xy(), v2.xy(), box_half.xy()) {
        return false;
    }

    // Test overlap in all 3 axes
    let min_x = v0.x.min(v1.x).min(v2.x);
    let max_x = v0.x.max(v1.x).max(v2.x);
    if min_x > box_half_size || max_x < -box_half_size {
        return false;
    }

    let min_y = v0.y.min(v1.y).min(v2.y);
    let max_y = v0.y.max(v1.y).max(v2.y);
    if min_y > box_half_size || max_y < -box_half_size {
        return false;
    }

    let min_z = v0.z.min(v1.z).min(v2.z);
    let max_z = v0.z.max(v1.z).max(v2.z);
    if min_z > box_half_size || max_z < -box_half_size {
        return false;
    }

    true
}

/// Helper for 2D separating axis test
fn test_axis(edge: Vec2, v0: Vec2, v2: Vec2, box_half: Vec2) -> bool {
    let p0 = edge.perp_dot(v0);
    let p2 = edge.perp_dot(v2);
    let (min_p, max_p) = if p0 < p2 { (p0, p2) } else { (p2, p0) };
    let r = box_half.x * edge.y.abs() + box_half.y * edge.x.abs();
    !(max_p < -r || min_p > r)
}

/// Fill interior by flood-filling exterior space, then inverting
fn fill_interior_floodfill(grid: &mut VoxelGrid) {
    use std::collections::VecDeque;

    let (width, height, depth) = grid.dimensions;
    let mut exterior = vec![false; grid.voxels.len()];
    let mut queue = VecDeque::new();

    // Seed from all 6 faces of the bounding box
    for x in 0..width {
        for y in 0..height {
            queue.push_back((x, y, 0));
            queue.push_back((x, y, depth - 1));
        }
    }
    for x in 0..width {
        for z in 0..depth {
            queue.push_back((x, 0, z));
            queue.push_back((x, height - 1, z));
        }
    }
    for y in 0..height {
        for z in 0..depth {
            queue.push_back((0, y, z));
            queue.push_back((width - 1, y, z));
        }
    }

    // Flood fill exterior space
    while let Some((x, y, z)) = queue.pop_front() {
        if let Some(idx) = grid.get_index(x, y, z) {
            // Skip if already marked as exterior or if it's a surface voxel
            if exterior[idx] || grid.voxels[idx] {
                continue;
            }

            exterior[idx] = true;

            // Add neighbors
            if x > 0 { queue.push_back((x - 1, y, z)); }
            if x + 1 < width { queue.push_back((x + 1, y, z)); }
            if y > 0 { queue.push_back((x, y - 1, z)); }
            if y + 1 < height { queue.push_back((x, y + 1, z)); }
            if z > 0 { queue.push_back((x, y, z - 1)); }
            if z + 1 < depth { queue.push_back((x, y, z + 1)); }
        }
    }

    // Everything not exterior and not surface is interior - fill it
    for i in 0..grid.voxels.len() {
        if !exterior[i] {
            grid.voxels[i] = true;
        }
    }
}