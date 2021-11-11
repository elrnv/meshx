/**
 * This module provides convenience functions for building common meshes.
 */
use super::{PolyMesh, TetMesh, TriMesh};

/// Axis plane orientation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AxisPlaneOrientation {
    XY,
    YZ,
    ZX,
}

/// Parameters that define a grid that lies in one of the 3 axis planes in 3D space.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GridBuilder {
    /// Number of grid cells in each column.
    pub rows: usize,
    /// Number of grid cells in each row.
    pub cols: usize,
    /// Axis orientation of the grid.
    pub orientation: AxisPlaneOrientation,
}

impl GridBuilder {
    /// Generate a [-1,1]x[-1,1] mesh grid with the given cell resolution and grid orientation. The
    /// grid nodes are spcified in row major order.
    pub fn build(self) -> PolyMesh<f64> {
        let GridBuilder {
            rows,
            cols,
            orientation,
        } = self;

        let mut positions = Vec::new();

        // iterate over vertices
        for j in 0..=cols {
            for i in 0..=rows {
                let r = -1.0 + 2.0 * (i as f64) / rows as f64;
                let c = -1.0 + 2.0 * (j as f64) / cols as f64;
                let node_pos = match orientation {
                    AxisPlaneOrientation::XY => [r, c, 0.0],
                    AxisPlaneOrientation::YZ => [0.0, r, c],
                    AxisPlaneOrientation::ZX => [c, 0.0, r],
                };
                positions.push(node_pos);
            }
        }

        let mut indices = Vec::new();

        // iterate over faces
        for i in 0..rows {
            for j in 0..cols {
                indices.push(4);
                indices.push((rows + 1) * j + i);
                indices.push((rows + 1) * j + i + 1);
                indices.push((rows + 1) * (j + 1) + i + 1);
                indices.push((rows + 1) * (j + 1) + i);
            }
        }

        PolyMesh::new(positions, &indices)
    }
}

/// Builder for a [-1,1]x[-1,1]x[-1,1] tetmesh box with the given cell resolution per axis.
/// The tetrahedralization is a simple 6 tets per cube with a regular pattern.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SolidBoxBuilder {
    pub res: [usize; 3],
}

impl Default for SolidBoxBuilder {
    fn default() -> Self {
        SolidBoxBuilder { res: [1, 1, 1] }
    }
}

impl SolidBoxBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn build(self) -> TetMesh<f64> {
        let mut positions = Vec::new();
        let [nx, ny, nz] = self.res;

        // iterate over vertices
        for ix in 0..=nx {
            for iy in 0..=ny {
                for iz in 0..=nz {
                    let x = -1.0 + 2.0 * (ix as f64) / nx as f64;
                    let y = -1.0 + 2.0 * (iy as f64) / ny as f64;
                    let z = -1.0 + 2.0 * (iz as f64) / nz as f64;
                    positions.push([x, y, z]);
                }
            }
        }

        let mut indices = Vec::new();

        // iterate over faces
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let index = |x, y, z| ((ix + x) * (ny + 1) + (iy + y)) * (nz + 1) + (iz + z);
                    // Populate tets in a star pattern
                    let first = index(0, 0, 0);
                    let second = index(1, 1, 1);
                    // Tet 1
                    indices.push([first, second, index(0, 1, 1), index(0, 1, 0)]);
                    // Tet 2
                    indices.push([first, second, index(0, 1, 0), index(1, 1, 0)]);
                    // Tet 3
                    indices.push([first, second, index(1, 1, 0), index(1, 0, 0)]);
                    // Tet 4
                    indices.push([first, second, index(1, 0, 0), index(1, 0, 1)]);
                    // Tet 5
                    indices.push([first, second, index(1, 0, 1), index(0, 0, 1)]);
                    // Tet 6
                    indices.push([first, second, index(0, 0, 1), index(0, 1, 1)]);
                }
            }
        }

        TetMesh::new(positions, indices)
    }
}

// For now this builder builds only regular shapes. This can be extended with a variety of options.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PlatonicSolidBuilder {}

impl PlatonicSolidBuilder {
    pub fn build_octahedron() -> TriMesh<f64> {
        let vertices = vec![
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
        ];

        #[rustfmt::skip]
        let indices = vec![
            [0, 5, 3],
            [4, 0, 3],
            [1, 4, 3],
            [5, 1, 3],
            [5, 0, 2],
            [0, 4, 2],
            [4, 1, 2],
            [1, 5, 2],
        ];

        TriMesh::new(vertices, indices)
    }

    pub fn build_tetrahedron() -> TetMesh<f64> {
        let sqrt_8_by_9 = f64::sqrt(8.0 / 9.0);
        let sqrt_2_by_9 = f64::sqrt(2.0 / 9.0);
        let sqrt_2_by_3 = f64::sqrt(2.0 / 3.0);
        let vertices = vec![
            [0.0, 1.0, 0.0],
            [-sqrt_8_by_9, -1.0 / 3.0, 0.0],
            [sqrt_2_by_9, -1.0 / 3.0, sqrt_2_by_3],
            [sqrt_2_by_9, -1.0 / 3.0, -sqrt_2_by_3],
        ];

        let indices = vec![[3, 1, 0, 2]];

        TetMesh::new(vertices, indices)
    }

    pub fn build_icosahedron() -> TriMesh<f64> {
        let sqrt5 = 5.0_f64.sqrt();
        let a = 1.0 / sqrt5;
        let w1 = 0.25 * (sqrt5 - 1.0);
        let h1 = (0.125 * (5.0 + sqrt5)).sqrt();
        let w2 = 0.25 * (sqrt5 + 1.0);
        let h2 = (0.125 * (5.0 - sqrt5)).sqrt();
        let vertices = vec![
            // North pole
            [0.0, 0.0, 1.0],
            // Alternating ring
            [0.0, 2.0 * a, a],
            [2.0 * a * h2, 2.0 * a * w2, -a],
            [2.0 * a * h1, 2.0 * a * w1, a],
            [2.0 * a * h1, -2.0 * a * w1, -a],
            [2.0 * a * h2, -2.0 * a * w2, a],
            [0.0, -2.0 * a, -a],
            [-2.0 * a * h2, -2.0 * a * w2, a],
            [-2.0 * a * h1, -2.0 * a * w1, -a],
            [-2.0 * a * h1, 2.0 * a * w1, a],
            [-2.0 * a * h2, 2.0 * a * w2, -a],
            // South pole
            [0.0, 0.0, -1.0],
        ];

        #[rustfmt::skip]
        let indices = vec![
            // North triangles
            [0, 1, 3],
            [0, 3, 5],
            [0, 5, 7],
            [0, 7, 9],
            [0, 9, 1],
            // Equatorial triangles
            [1, 2, 3],
            [2, 4, 3],
            [3, 4, 5],
            [4, 6, 5],
            [5, 6, 7],
            [6, 8, 7],
            [7, 8, 9],
            [8, 10, 9],
            [9, 10, 1],
            [10, 2, 1],
            // South triangles
            [11, 2, 10],
            [11, 4, 2],
            [11, 6, 4],
            [11, 8, 6],
            [11, 10, 8],
        ];

        TriMesh::new(vertices, indices)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TorusBuilder {
    pub outer_radius: f32,
    pub inner_radius: f32,
    pub outer_divs: usize,
    pub inner_divs: usize,
}

impl Default for TorusBuilder {
    fn default() -> Self {
        TorusBuilder {
            outer_radius: 0.5,
            inner_radius: 0.25,
            outer_divs: 24,
            inner_divs: 12,
        }
    }
}

impl TorusBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn build(self) -> PolyMesh<f64> {
        let TorusBuilder {
            outer_radius,
            inner_radius,
            outer_divs,
            inner_divs,
        } = self;

        let mut vertices = Vec::with_capacity(outer_divs * inner_divs);
        let mut indices = Vec::with_capacity(5 * outer_divs * inner_divs);

        let outer_step = 2.0 * std::f64::consts::PI / outer_divs as f64;
        let inner_step = 2.0 * std::f64::consts::PI / inner_divs as f64;

        for i in 0..outer_divs {
            let theta = outer_step * i as f64;
            for j in 0..inner_divs {
                let phi = inner_step * j as f64;
                // Add vertex
                let idx = vertices.len();
                vertices.push([
                    theta.cos() * (outer_radius as f64 + phi.cos() * inner_radius as f64),
                    phi.sin() * inner_radius as f64,
                    theta.sin() * (outer_radius as f64 + phi.cos() * inner_radius as f64),
                ]);

                // Add polygon
                indices.extend_from_slice(&[
                    4, // Number of vertices in the polygon
                    idx,
                    (((idx + 1) % inner_divs) + inner_divs * (idx / inner_divs))
                        % (inner_divs * outer_divs),
                    ((1 + idx) % inner_divs + (1 + idx / inner_divs) * inner_divs)
                        % (inner_divs * outer_divs),
                    (idx % inner_divs + (1 + idx / inner_divs) * inner_divs)
                        % (inner_divs * outer_divs),
                ]);
            }
        }

        PolyMesh::new(vertices, &indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the icosahedron has unit radius.
    #[test]
    fn icosahedron_unity_test() {
        use crate::mesh::VertexPositions;
        use approx::assert_relative_eq;
        use math::Vector3;

        let icosa = PlatonicSolidBuilder::build_icosahedron();
        for &v in icosa.vertex_positions() {
            assert_relative_eq!(Vector3::from(v).norm(), 1.0);
        }
    }

    #[test]
    fn grid_test() {
        use crate::ops::*;
        let grid = GridBuilder {
            rows: 1,
            cols: 1,
            orientation: AxisPlaneOrientation::ZX,
        }
        .build();
        let bbox = grid.bounding_box();
        assert_eq!(bbox.min_corner(), [-1.0, 0.0, -1.0]);
        assert_eq!(bbox.max_corner(), [1.0, 0.0, 1.0]);
    }
}
