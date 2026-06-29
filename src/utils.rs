use std::f32::consts::TAU;
use bevy::math::{Vec2, Vec3, Vec4};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::default;
use crate::watering::DIRECTIONS;

/// Build a flat XZ grid mesh with `cols` × `rows` vertices.
/// Vertex (col, row) sits at `(origin.x + col * cell_size_x, 0, origin.y + row * cell_size_z)`.
pub fn build_grid_mesh(cols: usize, rows: usize, cell_size_x: f32, cell_size_z: f32, origin: Vec2) -> Mesh {
    let vertex_count = cols * rows;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);

    for row in 0..rows {
        for col in 0..cols {
            let world_x = origin.x + col as f32 * cell_size_x;
            let world_z = origin.y + row as f32 * cell_size_z;
            positions.push([world_x, 0.0, world_z]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([col as f32 / cols as f32, row as f32 / rows as f32]);
        }
    }

    let mut indices = Vec::with_capacity((cols - 1) * (rows - 1) * 6);
    for row in 0..(rows - 1) {
        for col in 0..(cols - 1) {
            let top_left = (row * cols + col) as u32;
            let top_right = top_left + 1;
            let bottom_left = top_left + cols as u32;
            let bottom_right = bottom_left + 1;
            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);
            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Bilinear interpolation into an N×N grid of f32.
pub fn sample_bilinear(grid: &Vec<Vec<f32>>, fractional_col: f32, fractional_row: f32) -> f32 {
    let grid_size = grid.len();
    let col0 = (fractional_col as usize).min(grid_size - 1);
    let row0 = (fractional_row as usize).min(grid_size - 1);
    let col1 = (col0 + 1).min(grid_size - 1);
    let row1 = (row0 + 1).min(grid_size - 1);
    let t_col = fractional_col - col0 as f32;
    let t_row = fractional_row - row0 as f32;

    let v00 = grid[row0][col0];
    let v10 = grid[row0][col1];
    let v01 = grid[row1][col0];
    let v11 = grid[row1][col1];

    let top = v00 + (v10 - v00) * t_col;
    let bot = v01 + (v11 - v01) * t_col;
    top + (bot - top) * t_row
}

/// Bilinear interpolation into a grid of Vec4 indexed as `grid[row][col]`.
pub fn sample_bilinear_vec4(grid: &Vec<Vec<Vec4>>, fractional_col: f32, fractional_row: f32) -> Vec4 {
    let rows = grid.len();
    let cols = grid[0].len();
    let col0 = (fractional_col as usize).min(cols - 1);
    let row0 = (fractional_row as usize).min(rows - 1);
    let col1 = (col0 + 1).min(cols - 1);
    let row1 = (row0 + 1).min(rows - 1);
    let t_col = fractional_col - col0 as f32;
    let t_row = fractional_row - row0 as f32;

    let v00 = grid[row0][col0];
    let v10 = grid[row0][col1];
    let v01 = grid[row1][col0];
    let v11 = grid[row1][col1];
    let top = v00 + (v10 - v00) * t_col;
    let bot = v01 + (v11 - v01) * t_col;
    top + (bot - top) * t_row
}

/// Bilinear interpolation into an N×N grid of Vec3, returns normalized result.
pub fn sample_bilinear_normal(grid: &Vec<Vec<Vec3>>, fractional_col: f32, fractional_row: f32) -> Vec3 {
    let grid_size = grid.len();
    let col0 = (fractional_col as usize).min(grid_size - 1);
    let row0 = (fractional_row as usize).min(grid_size - 1);
    let col1 = (col0 + 1).min(grid_size - 1);
    let row1 = (row0 + 1).min(grid_size - 1);
    let t_col = fractional_col - col0 as f32;
    let t_row = fractional_row - row0 as f32;

    let v00 = grid[row0][col0];
    let v10 = grid[row0][col1];
    let v01 = grid[row1][col0];
    let v11 = grid[row1][col1];
    let top = v00 + (v10 - v00) * t_col;
    let bot = v01 + (v11 - v01) * t_col;
    let result = top + (bot - top) * t_row;
    result.normalize_or_zero()
}

/// Convert direction index to radians
#[inline]
pub fn index_to_radians(i: usize) -> f32 {
    (i as f32 + 0.0) * (TAU / DIRECTIONS as f32)
}

/// Compute layer index for amplitude texture array
#[inline]
pub fn layer_index(theta_index: usize, zeta_index: usize) -> usize {
    zeta_index * DIRECTIONS + theta_index
}

/// Modulo that handles negative numbers correctly
#[inline]
pub fn modulo(x: f32, n: f32) -> f32 {
    x - ((x / n).floor() * n)
}