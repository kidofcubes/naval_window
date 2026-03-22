//! Voxelization module for converting meshes to 3D voxel grids
//!
//! This module provides functionality to convert iterators of meshes into
//! solid voxel representations with customizable resolution and voxel size constraints.

use std::collections::{HashMap, HashSet};
use bevy::app::ctrlc::Error::System;
use bevy::asset::RenderAssetUsages;
use bevy::audio::CpalSample;
use bevy::prelude::*;
use bevy::mesh::{Mesh, Indices, PrimitiveTopology, VertexAttributeValues, MeshVertexAttributeId, MeshVertexAttribute};
use bevy::math::{Vec3A, Vec3Swizzles};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use isosurface::source::Source;
use isosurface::marching_cubes::MarchingCubes;
use rand::random_bool;

/// Voxel bit flags
pub const VOXEL_FILLED: u8 =    0b00000001;    // bit 0: filled flag (0 = empty, 1 = filled)
pub const VOXEL_BOUNDARY: u8 =  0b00000010;  // bit 1: boundary flag (0 = not boundary, 1 = boundary)

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
    /// 3D grid of voxels stored as flat array (u8 with bit flags)
    pub voxels: Vec<u8>,
    /// Grid dimensions (width, height, depth)
    pub dimensions: (u32, u32, u32),
    /// minimum position of the voxel grid
    pub offset: Vec3A,
    /// Size of each voxel in world units
    pub voxel_size: f32,
    /// Signed distance field: negative inside, positive outside, zero at boundary
    pub sdf: Vec<f32>,
}

impl VoxelGrid {
    /// Create a new empty voxel grid
    pub fn new(dimensions: (u32, u32, u32), offset: Vec3A) -> Self {
        let voxel_count = (dimensions.0 * dimensions.1 * dimensions.2) as usize;
        let voxel_size = calculate_voxel_size(dimensions);

        Self {
            voxels: vec![0u8; voxel_count],
            dimensions,
            offset,
            voxel_size,
            sdf: vec![0.0; voxel_count],
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

    /// Get the linear index for a 3D coordinate
    pub fn get_index_bounds_check(&self, x: i32, y: i32, z: i32) -> Option<usize> {
        if x >= 0 && y >= 0 && z >= 0 && x < self.dimensions.0 as i32 && y < self.dimensions.1 as i32 && z < self.dimensions.2 as i32 {
            Some(((z as u32 * self.dimensions.0 * self.dimensions.1) + (y as u32 * self.dimensions.0) + x as u32) as usize)
        } else {
            None
        }
    }

    /// Get the raw u8 value at the given coordinates
    pub fn get_voxel(&self, x: u32, y: u32, z: u32) -> Option<u8> {
        self.get_index(x, y, z).map(|index| self.voxels[index])
    }

    /// Get the raw u8 value at the given coordinates
    pub fn get_voxel_bounds_check(&self, x: i32, y: i32, z: i32) -> Option<u8> {
        self.get_index_bounds_check(x, y, z).map(|index| self.voxels[index])
    }

    /// Set a voxel filled flag at the given coordinates
    pub fn set_voxel(&mut self, x: u32, y: u32, z: u32, filled: bool) -> bool {
        if let Some(index) = self.get_index(x, y, z) {
            if filled {
                self.voxels[index] |= VOXEL_FILLED;
            } else {
                self.voxels[index] &= !VOXEL_FILLED;
            }
            true
        } else {
            false
        }
    }

    /// Check if a voxel is filled at the given coordinates
    pub fn is_filled(&self, x: u32, y: u32, z: u32) -> Option<bool> {
        self.get_index(x, y, z).map(|index| (self.voxels[index] & VOXEL_FILLED) != 0)
    }

    /// Set the boundary flag at the given coordinates
    pub fn set_boundary(&mut self, x: u32, y: u32, z: u32) -> bool {
        if let Some(index) = self.get_index(x, y, z) {
            self.voxels[index] |= VOXEL_BOUNDARY;
            true
        } else {
            false
        }
    }

    /// Check if a voxel is a boundary voxel
    pub fn is_boundary(&self, x: u32, y: u32, z: u32) -> Option<bool> {
        self.get_index(x, y, z).map(|index| (self.voxels[index] & VOXEL_BOUNDARY) != 0)
    }

    /// Flood fill from a starting position, visiting all connected voxels matching `predicate`.
    /// Returns a list of flat indices that were flooded.
    /// Uses 6-connectivity (face neighbors only).
    pub fn flood_fill(&self, start_x: u32, start_y: u32, start_z: u32, predicate: impl Fn(u8) -> bool) -> Vec<usize> {
        let start_index = match self.get_index(start_x, start_y, start_z) {
            Some(i) => i,
            None => return Vec::new(),
        };

        if !predicate(self.voxels[start_index]) {
            return Vec::new();
        }

        let mut visited = vec![false; self.voxels.len()];
        let mut result = Vec::new();
        let mut stack = vec![(start_x, start_y, start_z)];
        visited[start_index] = true;

        while let Some((x, y, z)) = stack.pop() {
            result.push(self.get_index(x, y, z).unwrap());

            let neighbors: [(i64, i64, i64); 6] = [
                (x as i64 - 1, y as i64, z as i64),
                (x as i64 + 1, y as i64, z as i64),
                (x as i64, y as i64 - 1, z as i64),
                (x as i64, y as i64 + 1, z as i64),
                (x as i64, y as i64, z as i64 - 1),
                (x as i64, y as i64, z as i64 + 1),
            ];

            for (nx, ny, nz) in neighbors {
                if nx < 0 || ny < 0 || nz < 0 {
                    continue;
                }
                let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
                if let Some(idx) = self.get_index(nx, ny, nz) {
                    if !visited[idx] && predicate(self.voxels[idx]) {
                        visited[idx] = true;
                        stack.push((nx, ny, nz));
                    }
                }
            }
        }

        result
    }
}

/// Implement Source trait for isosurface extraction
impl Source for VoxelGrid {
    fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        // Convert normalized [0,1] coords to voxel coords
        let vx = (x * self.dimensions.0 as f32) as i32;
        let vy = (y * self.dimensions.1 as f32) as i32;
        let vz = (z * self.dimensions.2 as f32) as i32;

        if (vx < 0 || vy < 0 || vz < 0 || vx >= self.dimensions.0 as i32 || vy >= self.dimensions.1 as i32 || vz >= self.dimensions.2 as i32) {
            return 10.0;
        }

    // Clamp to valid range
        let vx = (vx as u32).min(self.dimensions.0.saturating_sub(1));
        let vy = (vy as u32).min(self.dimensions.1.saturating_sub(1));
        let vz = (vz as u32).min(self.dimensions.2.saturating_sub(1));

        // Return negative if filled (inside), positive if empty (outside)
        if self.is_filled(vx, vy, vz).unwrap_or(false) {
            // -1.0
            // -1.5
            -0.25 / 2.0
        } else {
            // 1.0
            // 0.5
            1.75 * 2.0
        }
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
                let voxel = grid.get_voxel(x, y, z).unwrap_or(0);
                let is_filled = (voxel & VOXEL_FILLED) != 0;
                let is_boundary = (voxel & VOXEL_BOUNDARY) != 0;

                // Skip empty non-boundary voxels
                if !is_filled && !is_boundary {
                    continue;
                }
                if !is_boundary {
                    continue;
                }

                let world_pos = offset + Vec3A::new(
                    (x as f32 + 0.5) * voxel_size,
                    (y as f32 + 0.5) * voxel_size,
                    (z as f32 + 0.5) * voxel_size,
                );

                // Choose color based on voxel state
                let color = if is_filled && is_boundary {
                    // Boundary filled voxels: red
                    Color::srgba(1.0, 0.0, 0.0, 0.3)
                } else if is_filled {
                    // Interior filled voxels: green
                    Color::srgba(0.0, 1.0, 0.0, 0.3)
                } else {
                    // Boundary empty voxels: yellow (for debugging)
                    Color::srgba(1.0, 1.0, 0.0, 0.2)
                };

                // Draw wireframe cube
                gizmo.cube(
                    Transform::from_translation(world_pos.into())
                        .with_scale(Vec3::splat(voxel_size * 0.95)),
                    color,
                );
            }
        }
    }
}

pub fn render_sdf(grid: &VoxelGrid, gizmo: &mut Gizmos) {
    let voxel_size = grid.voxel_size;
    let offset = grid.offset;

    // Find max absolute SDF value for normalization
    let max_dist = grid.sdf.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1.0);

    for z in 0..grid.dimensions.2 {
        for y in 0..grid.dimensions.1 {
            for x in 0..grid.dimensions.0 {
                let idx = grid.get_index(x, y, z).unwrap();
                let sdf_val = grid.sdf[idx];

                // Only show voxels near the surface (within a few voxels)
                if sdf_val.abs() > 2.0 {
                    continue;
                }

                let world_pos = offset + Vec3A::new(
                    (x as f32 + 0.5) * voxel_size,
                    (y as f32 + 0.5) * voxel_size,
                    (z as f32 + 0.5) * voxel_size,
                );

                let t = (sdf_val.abs() / 5.0).min(1.0);
                let alpha = 0.4 * (1.0 - t); // fade out further from surface

                let color = if sdf_val <= 0.0 {
                    // Inside (negative): blue, brighter near surface
                    Color::srgba(0.0, 0.2 * t, 1.0 - 0.5 * t, alpha)
                } else {
                    // Outside (positive): red, brighter near surface
                    Color::srgba(1.0 - 0.5 * t, 0.2 * t, 0.0, alpha)
                };

                gizmo.cube(
                    Transform::from_translation(world_pos.into())
                        .with_scale(Vec3::splat(voxel_size * 0.95)),
                    color,
                );
            }
        }
    }
}

pub fn mark_boundary_voxels_floodfill(grid: &mut VoxelGrid) {
    let outside_voxels: HashSet<usize> = HashSet::from_iter(grid.flood_fill(0,0,0, |voxel| {
        return (voxel & VOXEL_FILLED) == 0;
    }));
    println!("outside voxel length {:?}", outside_voxels.len());
    println!("voxel length {:?}", grid.voxels.len());
    let mut count = 0;
    for x in 0..grid.dimensions.0 {
        for y in 0..grid.dimensions.1 {
            for z in 0..grid.dimensions.2 {
                if !grid.is_filled(x, y, z).unwrap() {
                    continue;
                }
                let mut is_boundary: bool = false;
                for x_offset in -1i32..=1 {
                    for y_offset in -1i32..=1 {
                        for z_offset in -1i32..=1 {
                            if x_offset == 0 && y_offset == 0 && z_offset == 0 {
                                continue;
                            }
                            if grid.get_index_bounds_check(x as i32 + x_offset, y as i32 + y_offset, z as i32 + z_offset).map(|i| outside_voxels.contains(&i)).unwrap_or(true) {
                                grid.set_boundary(x, y, z);
                                is_boundary = true;
                                break;
                            }
                        }
                        if is_boundary {
                            break;
                        }
                    }
                    if is_boundary {
                        break;
                    }
                }
                // is_boundary |= grid.get_index(x+1, y+0, z+0).map(|i| outside_voxels.contains(&i)).unwrap_or(true);
                // if(is_boundary){
                //     grid.set_boundary(x, y, z);
                //     count+=1;
                //     continue;
                // }
                // is_boundary |= grid.get_index(x.wrapping_sub(1), y+0, z+0).map(|i| outside_voxels.contains(&i)).unwrap_or(true);
                // if(is_boundary){
                //     grid.set_boundary(x, y, z);
                //     count+=1;
                //     continue;
                // }
                // is_boundary |= grid.get_index(x+0, y+1, z+0).map(|i| outside_voxels.contains(&i)).unwrap_or(true);
                // if(is_boundary){
                //     grid.set_boundary(x, y, z);
                //     count+=1;
                //     continue;
                // }
                // is_boundary |= grid.get_index(x+0, y.wrapping_sub(1), z+0).map(|i| outside_voxels.contains(&i)).unwrap_or(true);
                // if(is_boundary){
                //     grid.set_boundary(x, y, z);
                //     count+=1;
                //     continue;
                // }
                // is_boundary |= grid.get_index(x+0, y+0, z+1).map(|i| outside_voxels.contains(&i)).unwrap_or(true);
                // if(is_boundary){
                //     grid.set_boundary(x, y, z);
                //     count+=1;
                //     continue;
                // }
                // is_boundary |= grid.get_index(x+0, y+0, z.wrapping_sub(1)).map(|i| outside_voxels.contains(&i)).unwrap_or(true);
                // if(is_boundary){
                //     grid.set_boundary(x, y, z);
                //     count+=1;
                //     continue;
                // }
            }
        }
    }
    println!("boundary voxels length {:?}", count)
}

/// Mark boundary voxels using isosurface extraction
pub fn mark_boundary_voxels(grid: &mut VoxelGrid) {
    let size = grid.dimensions.0.max(grid.dimensions.1).max(grid.dimensions.2) as usize;
    let mut mc = MarchingCubes::new(size);

    let mut vertices: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    mc.extract(grid, &mut vertices, &mut indices);

    // Convert all vertices from normalized [0,1] space to world space and store
    let num_vertices = vertices.len() / 3;
    let mut world_vertices: Vec<Vec3A> = Vec::with_capacity(num_vertices);
    for i in 0..num_vertices {
        let x = vertices[i * 3];
        let y = vertices[i * 3 + 1];
        let z = vertices[i * 3 + 2];
        world_vertices.push(Vec3A::new(
            grid.offset.x + x * (grid.dimensions.0 as f32) * grid.voxel_size,
            grid.offset.y + y * (grid.dimensions.1 as f32) * grid.voxel_size,
            grid.offset.z + z * (grid.dimensions.2 as f32) * grid.voxel_size,
        ));
    }

    // For each triangle in the boundary mesh
    for tri in indices.chunks(3) {
        if tri.len() != 3 {
            continue;
        }
        let v0 = world_vertices[tri[0] as usize];
        let v1 = world_vertices[tri[1] as usize];
        let v2 = world_vertices[tri[2] as usize];

        // Mark voxels intersecting this triangle with VOXEL_BOUNDARY bit
        mark_triangle_voxels_boundary(grid, v0, v1, v2);
    }

    // Store the boundary mesh in the grid
    // grid.boundary_vertices = Some(world_vertices);
    // grid.boundary_indices = Some(indices);
}

/// Mark voxels that intersect a boundary triangle
fn mark_triangle_voxels_boundary(grid: &mut VoxelGrid, v0: Vec3A, v1: Vec3A, v2: Vec3A) {
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

    // Test each voxel in the bounding box
    for z in start_z..end_z {
        for y in start_y..end_y {
            for x in start_x..end_x {
                let voxel_center = voxel_to_world(grid, x, y, z);
                let half_size = grid.voxel_size * 0.5;

                // Test if triangle intersects this voxel
                if triangle_box_intersection(v0, v1, v2, voxel_center, half_size) {
                    grid.set_boundary(x as u32, y as u32, z as u32);
                }
            }
        }
    }
}

/// Marks all voxels that border empty voxels as boundary voxels
pub fn mark_bordering_voxels_as_boundary(grid: &mut VoxelGrid) {
    let (width, height, depth) = grid.dimensions;

    // Iterate through all voxels in the grid
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                // Check if current voxel is filled
                if let Some(is_filled) = grid.is_filled(x, y, z) {
                    if is_filled {
                        // Check if any neighboring voxel is empty
                        let has_empty_neighbor =
                            // Check 6 direct neighbors (up, down, left, right, front, back)
                            (!grid.is_filled(x.wrapping_sub(1), y, z).unwrap_or(true)) ||
                            (!grid.is_filled(x.wrapping_add(1), y, z).unwrap_or(true)) ||
                            (!grid.is_filled(x, y.wrapping_sub(1), z).unwrap_or(true)) ||
                            (!grid.is_filled(x, y.wrapping_add(1), z).unwrap_or(true)) ||
                            (!grid.is_filled(x, y, z.wrapping_sub(1)).unwrap_or(true)) ||
                            (!grid.is_filled(x, y, z.wrapping_add(1)).unwrap_or(true));

                        // If voxel has an empty neighbor, mark it as boundary
                        if has_empty_neighbor {
                            grid.set_boundary(x, y, z);
                        }
                        //temp
                        grid.set_boundary(x, y, z);
                    }
                }
                // //temp
                // grid.set_boundary(x, y, z);
            }
        }
    }
}

pub fn voxelize_meshes(
    mesh_transform_pairs: Vec<(&Mesh, &GlobalTransform)>,
    config: VoxelizationConfig,
) -> VoxelGrid {
    // Step 1: Calculate combined bounding box
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

    // Step 2: Calculate voxel grid dimensions
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

    let voxel_count = (dimensions.0 * dimensions.1 * dimensions.2) as usize;
    let mut grid = VoxelGrid {
        voxels: vec![0u8; voxel_count],
        dimensions,
        offset: min_bound,
        voxel_size,
        // boundary_vertices: None,
        // boundary_indices: None,
        sdf: vec![0.0; voxel_count],
    };

    // Step 3: Conservative voxelization - mark any voxel a triangle touches
    for (mesh, transform) in &mesh_transform_pairs {
        voxelize_mesh_conservative(&mut grid, mesh, transform);
    }

    // Step 4: Mark boundary voxels using isosurface extraction
    // mark_boundary_voxels(&mut grid);
    // mark_bordering_voxels_as_boundary(&mut grid);
    mark_boundary_voxels_floodfill(&mut grid);
    compute_sdf(&mut grid);

    grid
}

/// Compute the signed distance field using Felzenszwalb-Huttenlocher 3D EDT.
/// Boundary voxels are the zero-crossing; filled voxels get negative distances,
/// empty voxels get positive distances. Distances are in voxel units.
fn compute_sdf(grid: &mut VoxelGrid) {
    let (w, h, d) = grid.dimensions;
    let n = (w * h * d) as usize;
    let max_dim = w.max(h).max(d) as usize;

    // Initialize squared distances: boundary = 0, others = large sentinel
    let sentinel = (w * w + h * h + d * d) as f32 + 1.0;
    let mut sq_dist = vec![sentinel; n];

    for i in 0..n {
        if (grid.voxels[i] & VOXEL_BOUNDARY) != 0 {
            sq_dist[i] = 0.0;
        }
    }

    // Scratch buffers for 1D EDT (reused across all rows)
    let mut f = vec![0.0f32; max_dim];
    let mut z = vec![0.0f32; max_dim + 1];
    let mut v = vec![0usize; max_dim];
    let mut output = vec![0.0f32; max_dim];

    // Pass 1: EDT along X axis
    for zi in 0..d {
        for yi in 0..h {
            let row_start = (zi * w * h + yi * w) as usize;
            for xi in 0..w as usize {
                f[xi] = sq_dist[row_start + xi];
            }
            edt_1d(&f, w as usize, &mut v, &mut z, &mut output);
            for xi in 0..w as usize {
                sq_dist[row_start + xi] = output[xi];
            }
        }
    }

    // Pass 2: EDT along Y axis
    for zi in 0..d {
        for xi in 0..w {
            // Gather column
            for yi in 0..h as usize {
                f[yi] = sq_dist[(zi * w * h + yi as u32 * w + xi) as usize];
            }
            edt_1d(&f, h as usize, &mut v, &mut z, &mut output);
            // Scatter back
            for yi in 0..h as usize {
                sq_dist[(zi * w * h + yi as u32 * w + xi) as usize] = output[yi];
            }
        }
    }

    // Pass 3: EDT along Z axis
    for yi in 0..h {
        for xi in 0..w {
            // Gather depth-line
            for zi in 0..d as usize {
                f[zi] = sq_dist[(zi as u32 * w * h + yi * w + xi) as usize];
            }
            edt_1d(&f, d as usize, &mut v, &mut z, &mut output);
            // Scatter back
            for zi in 0..d as usize {
                sq_dist[(zi as u32 * w * h + yi * w + xi) as usize] = output[zi];
            }
        }
    }

    // Convert squared distances to signed distances
    for i in 0..n {
        let dist = sq_dist[i].sqrt();
        grid.sdf[i] = if (grid.voxels[i] & VOXEL_FILLED) != 0 {
            -dist
        } else {
            dist
        };
    }
}

/// 1D squared Euclidean distance transform using the Felzenszwalb-Huttenlocher algorithm.
/// `f` contains the input values, `n` is the length.
/// Result is written to `output`.
fn edt_1d(f: &[f32], n: usize, v: &mut [usize], z: &mut [f32], output: &mut [f32]) {
    if n == 0 {
        return;
    }
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    let mut k = 0usize;

    for q in 1..n {
        loop {
            let vk = v[k];
            // Intersection of parabolas at v[k] and q
            let s = ((f[q] + (q * q) as f32) - (f[vk] + (vk * vk) as f32))
                / (2.0 * (q as f32 - vk as f32));
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
            if k == 0 {
                v[0] = q;
                z[1] = f32::INFINITY;
                break;
            }
            k -= 1;
        }
    }

    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let diff = q as f32 - v[k] as f32;
        output[q] = diff * diff + f[v[k]];
    }
}

/// Conservative voxelization - marks voxel if triangle intersects it at all
fn voxelize_mesh_conservative(grid: &mut VoxelGrid, mesh: &Mesh, transform: &GlobalTransform) {
    // Extract vertex positions
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(positions)) => positions,
        _ => return,
    };

    // Extract indices
    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(indices)) => indices.clone(),
        Some(Indices::U16(indices)) => indices.iter().map(|&i| i as u32).collect(),
        None => {
            if mesh.primitive_topology() == PrimitiveTopology::TriangleList {
                (0..positions.len() as u32).collect()
            } else {
                return;
            }
        }
    };

    // Process each triangle
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

        voxelize_triangle_conservative(grid, v0, v1, v2);
    }
}

/// Conservatively voxelize a single triangle
fn voxelize_triangle_conservative(grid: &mut VoxelGrid, v0: Vec3A, v1: Vec3A, v2: Vec3A) {
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

/// Triangle-box intersection test using separating axis theorem
pub(crate) fn triangle_box_intersection(v0: Vec3A, v1: Vec3A, v2: Vec3A, box_center: Vec3A, box_half_size: f32) -> bool {
    // Translate triangle to box space (box center at origin)
    let v0 = v0 - box_center;
    let v1 = v1 - box_center;
    let v2 = v2 - box_center;

    // Compute triangle edges
    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;

    // Test the 3 face normals from the AABB
    let box_half = Vec3A::splat(box_half_size);

    // Test X axis
    let min_x = v0.x.min(v1.x).min(v2.x);
    let max_x = v0.x.max(v1.x).max(v2.x);
    if min_x > box_half_size || max_x < -box_half_size {
        return false;
    }

    // Test Y axis
    let min_y = v0.y.min(v1.y).min(v2.y);
    let max_y = v0.y.max(v1.y).max(v2.y);
    if min_y > box_half_size || max_y < -box_half_size {
        return false;
    }

    // Test Z axis
    let min_z = v0.z.min(v1.z).min(v2.z);
    let max_z = v0.z.max(v1.z).max(v2.z);
    if min_z > box_half_size || max_z < -box_half_size {
        return false;
    }

    // Test triangle plane
    let normal = e0.cross(e1);
    let d = -normal.dot(v0);
    let r = box_half_size * (normal.x.abs() + normal.y.abs() + normal.z.abs());
    if d.abs() > r {
        return false;
    }

    // Test the 9 edge cross-product axes
    // Axis: edge0 x box axis
    if !test_axis_cross(e0, v0, v1, v2, box_half, Vec3A::X) {
        return false;
    }
    if !test_axis_cross(e0, v0, v1, v2, box_half, Vec3A::Y) {
        return false;
    }
    if !test_axis_cross(e0, v0, v1, v2, box_half, Vec3A::Z) {
        return false;
    }

    // Axis: edge1 x box axis
    if !test_axis_cross(e1, v0, v1, v2, box_half, Vec3A::X) {
        return false;
    }
    if !test_axis_cross(e1, v0, v1, v2, box_half, Vec3A::Y) {
        return false;
    }
    if !test_axis_cross(e1, v0, v1, v2, box_half, Vec3A::Z) {
        return false;
    }

    // Axis: edge2 x box axis
    if !test_axis_cross(e2, v0, v1, v2, box_half, Vec3A::X) {
        return false;
    }
    if !test_axis_cross(e2, v0, v1, v2, box_half, Vec3A::Y) {
        return false;
    }
    if !test_axis_cross(e2, v0, v1, v2, box_half, Vec3A::Z) {
        return false;
    }

    true
}

/// Test separating axis formed by edge cross box axis
fn test_axis_cross(edge: Vec3A, v0: Vec3A, v1: Vec3A, v2: Vec3A, box_half: Vec3A, axis: Vec3A) -> bool {
    let a = edge.cross(axis);

    // Project triangle vertices onto axis
    let p0 = v0.dot(a);
    let p1 = v1.dot(a);
    let p2 = v2.dot(a);

    let min_p = p0.min(p1).min(p2);
    let max_p = p0.max(p1).max(p2);

    // Project box onto axis
    let r = box_half.x * (a.x.abs()) + box_half.y * (a.y.abs()) + box_half.z * (a.z.abs());

    // Test for separation
    !(max_p < -r || min_p > r)
}

// Helper functions
pub fn world_to_voxel(grid: &VoxelGrid, world_pos: Vec3A) -> IVec3 {
    let local_pos = world_pos - grid.offset;
    IVec3::new(
        (local_pos.x / grid.voxel_size).floor() as i32,
        (local_pos.y / grid.voxel_size).floor() as i32,
        (local_pos.z / grid.voxel_size).floor() as i32,
    )
}

pub fn voxel_to_world(grid: &VoxelGrid, x: i32, y: i32, z: i32) -> Vec3A {
    grid.offset
        + Vec3A::new(
        (x as f32 + 0.5) * grid.voxel_size,
        (y as f32 + 0.5) * grid.voxel_size,
        (z as f32 + 0.5) * grid.voxel_size,
    )
}

pub fn create_voxel_texture(
    grid: &VoxelGrid,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    let (width, height, depth) = grid.dimensions;

    // Use raw u8 voxel data directly (contains bit flags for filled/boundary)
    let data: Vec<u8> = grid.voxels.clone();

    let image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
        TextureDimension::D3,
        data,
        TextureFormat::R8Uint,
        // TextureFormat::R32Uint,
        Default::default(),
    );
    // for x in 0..width {
    //     for y in 0..height{
    //         for z in 0..depth{
    //             image.get_color_at_3d(x,y,z).and_then(|color| {
    //                 println!("color at {},{},{} is {:?}",x,y,z,color);
    //                 Ok("awesome")
    //             });
    //         }
    //     }
    // }

    return images.add(image)
}
pub fn create_sdf_texture(
    grid: &VoxelGrid,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    let (width, height, depth) = grid.dimensions;

    // Convert f32 SDF values to bytes (little-endian)
    let data: Vec<u8> = grid.sdf.iter().flat_map(|v| v.to_le_bytes()).collect();

    let image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
        TextureDimension::D3,
        data,
        TextureFormat::R32Float,
        Default::default(),
    );
    images.add(image)
}

pub fn null_texture(images: &mut Assets<Image>, dimension: TextureDimension, format: TextureFormat) -> Handle<Image> {
    let data: Vec<u8>;
    if let TextureFormat::R8Uint = format {
        data = vec![0];
    } else {
        data = vec![0, 0, 0, 0];
    }

    let mut image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        dimension,
        data,
        format,
        Default::default(),
    );
    return images.add(image)
}