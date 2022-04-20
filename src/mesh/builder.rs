/**
 * This module provides convenience functions for building common meshes.
 */
use super::{PolyMesh, TetMesh, TriMesh};
use crate::Real;

/// A trait for building meshes representing various objects.
///
/// This interface is optional. All builders have standalone public build functions for specific
/// mesh types.
///
/// To use this interface one must specify the desired output mesh type beside the output binding as follows:
/// ```
/// use meshx::{builder::MeshBuilder, mesh::PolyMesh, builder::BoxBuilder};
/// let mesh: PolyMesh<f64> = BoxBuilder { divisions: [0; 3] }.build();
/// ```
/// in contrast to
/// ```
/// use meshx::{builder::MeshBuilder, mesh::PolyMesh, builder::BoxBuilder};
/// let mesh = BoxBuilder { divisions: [0; 3] }.build_polymesh::<f64>();
/// ```
///
/// NOTE: `PlatonicSolidBuilder` does not implement this interface since it has specialized
/// output types based on the type of polyhedron being built. This may change in the future.
pub trait MeshBuilder<M> {
    /// Builds a mesh of the given type `M`.
    fn build(self) -> M;
}

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
    pub fn build<T: Real>(self) -> PolyMesh<T> {
        let GridBuilder {
            rows,
            cols,
            orientation,
        } = self;

        let mut positions = Vec::new();

        // iterate over vertices
        for j in 0..=cols {
            for i in 0..=rows {
                let r = T::from(-1.0 + 2.0 * (i as f64) / rows as f64).unwrap();
                let c = T::from(-1.0 + 2.0 * (j as f64) / cols as f64).unwrap();
                let node_pos = match orientation {
                    AxisPlaneOrientation::XY => [r, c, T::zero()],
                    AxisPlaneOrientation::YZ => [T::zero(), r, c],
                    AxisPlaneOrientation::ZX => [c, T::zero(), r],
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

impl<T: Real> MeshBuilder<PolyMesh<T>> for GridBuilder {
    /// Generate a [-1,1]x[-1,1] mesh grid with the given cell resolution and grid orientation. The
    /// grid nodes are spcified in row major order.
    fn build(self) -> PolyMesh<T> {
        self.build::<T>()
    }
}

/// Builder for a [-1,1]x[-1,1]x[-1,1] mesh box with the given number of divisions per axis.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoxBuilder {
    /// Number of divisions in each axis direction.
    ///
    /// `0` indicates no divisions, meaning that `[0,0,0]` creates a regular cube with 6 faces
    /// (assuming polygon mesh output).
    pub divisions: [u32; 3],
}

impl Default for BoxBuilder {
    /// Creates a default builder of a regular box with the specified number of divisions along
    /// each axis.
    fn default() -> Self {
        BoxBuilder {
            divisions: [0, 0, 0],
        }
    }
}

impl BoxBuilder {
    /// Creates a default builder of a 1x1x1 resolution box.
    pub fn new() -> Self {
        Default::default()
    }

    /// Sets the number of divisions on the box.
    pub fn with_divisions(self, divisions: [u32; 3]) -> Self {
        Self { divisions, ..self }
    }

    /// Builds a tetmesh box.
    ///
    /// The tetrahedralization is a simple 6 tets per cube with a regular pattern.
    fn build_tetmesh<T: Real>(self) -> TetMesh<T> {
        let mut positions = Vec::new();
        let [dx, dy, dz] = self.divisions;
        let nx = dx as usize + 1;
        let ny = dy as usize + 1;
        let nz = dz as usize + 1;

        // iterate over vertices
        for ix in 0..=nx {
            for iy in 0..=ny {
                for iz in 0..=nz {
                    let x = T::from(-1.0 + 2.0 * (ix as f64) / (nx as f64)).unwrap();
                    let y = T::from(-1.0 + 2.0 * (iy as f64) / (ny as f64)).unwrap();
                    let z = T::from(-1.0 + 2.0 * (iz as f64) / (nz as f64)).unwrap();
                    positions.push([x, y, z]);
                }
            }
        }

        let mut indices = Vec::new();

        // Iterate over cells.
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

    /// Builds a polymesh box.
    pub fn build_polymesh<T: Real>(self) -> PolyMesh<T> {
        // TODO: It may be worthwhile to refactor this function.

        let mut positions = Vec::new();
        let [dx, dy, dz] = self.divisions;
        let nx = dx as usize + 1;
        let ny = dy as usize + 1;
        let nz = dz as usize + 1;

        for ix in [0, nx] {
            for iy in 0..=ny {
                for iz in 0..=nz {
                    let x = T::from(-1.0 + 2.0 * (ix as f64) / (nx as f64)).unwrap();
                    let y = T::from(-1.0 + 2.0 * (iy as f64) / (ny as f64)).unwrap();
                    let z = T::from(-1.0 + 2.0 * (iz as f64) / (nz as f64)).unwrap();
                    positions.push([x, y, z]);
                }
            }
        }

        let offset1 = positions.len();

        for ix in 1..nx {
            for iy in [0, ny] {
                for iz in 0..=nz {
                    let x = T::from(-1.0 + 2.0 * (ix as f64) / (nx as f64)).unwrap();
                    let y = T::from(-1.0 + 2.0 * (iy as f64) / (ny as f64)).unwrap();
                    let z = T::from(-1.0 + 2.0 * (iz as f64) / (nz as f64)).unwrap();
                    positions.push([x, y, z]);
                }
            }
        }

        let offset2 = positions.len();

        for ix in 1..nx {
            for iy in 1..ny {
                for iz in [0, nz] {
                    let x = T::from(-1.0 + 2.0 * (ix as f64) / (nx as f64)).unwrap();
                    let y = T::from(-1.0 + 2.0 * (iy as f64) / (ny as f64)).unwrap();
                    let z = T::from(-1.0 + 2.0 * (iz as f64) / (nz as f64)).unwrap();
                    positions.push([x, y, z]);
                }
            }
        }

        let mut indices = Vec::new();

        // -x and +x sides
        for ix in 0..2 {
            for iy in 0..ny {
                for iz in 0..nz {
                    let index = |x: usize, y: usize, z: usize| {
                        ((ix + x) * (ny + 1) + (iy + y)) * (nz + 1) + (iz + z)
                    };
                    if ix > 0 {
                        // reversed
                        indices.extend_from_slice(&[
                            4,
                            index(0, 0, 1),
                            index(0, 1, 1),
                            index(0, 1, 0),
                            index(0, 0, 0),
                        ]);
                    } else {
                        indices.extend_from_slice(&[
                            4,
                            index(0, 0, 0),
                            index(0, 1, 0),
                            index(0, 1, 1),
                            index(0, 0, 1),
                        ]);
                    }
                }
            }
        }

        // -y and +y sides
        let x_index = |ix: usize, iy: usize, iz: usize| (ix * (ny + 1) + iy) * (nz + 1) + iz;
        let y_index = |ix: usize, iy: usize, iz: usize| offset1 + (ix * 2 + iy) * (nz + 1) + iz;
        let create_side_polys = |ix: usize, iy: usize, iz: usize, indices: &mut Vec<usize>| {
            if (iy > 0) ^ (ix > 0) {
                // reversed
                indices.extend_from_slice(&[
                    4,
                    y_index(ix * (nx - 2), iy, iz + 1),
                    x_index(ix, iy * ny, iz + 1),
                    x_index(ix, iy * ny, iz),
                    y_index(ix * (nx - 2), iy, iz),
                ]);
            } else {
                indices.extend_from_slice(&[
                    4,
                    y_index(ix * (nx - 2), iy, iz),
                    x_index(ix, iy * ny, iz),
                    x_index(ix, iy * ny, iz + 1),
                    y_index(ix * (nx - 2), iy, iz + 1),
                ]);
            }
        };
        for iy in 0..2 {
            if nx > 1 {
                for iz in 0..nz {
                    create_side_polys(0, iy, iz, &mut indices);
                }
                for ix in 0..nx - 2 {
                    for iz in 0..nz {
                        let index = |x: usize, y: usize, z: usize| {
                            offset1 + ((ix + x) * 2 + (iy + y)) * (nz + 1) + (iz + z)
                        };
                        if iy > 0 {
                            // reversed
                            indices.extend_from_slice(&[
                                4,
                                index(0, 0, 0),
                                index(1, 0, 0),
                                index(1, 0, 1),
                                index(0, 0, 1),
                            ]);
                        } else {
                            indices.extend_from_slice(&[
                                4,
                                index(0, 0, 1),
                                index(1, 0, 1),
                                index(1, 0, 0),
                                index(0, 0, 0),
                            ]);
                        }
                    }
                }
                for iz in 0..nz {
                    create_side_polys(1, iy, iz, &mut indices);
                }
            } else {
                for iz in 0..nz {
                    if iy > 0 {
                        // reversed
                        indices.extend_from_slice(&[
                            4,
                            x_index(1, iy * ny, iz + 1),
                            x_index(0, iy * ny, iz + 1),
                            x_index(0, iy * ny, iz),
                            x_index(1, iy * ny, iz),
                        ]);
                    } else {
                        indices.extend_from_slice(&[
                            4,
                            x_index(1, iy, iz),
                            x_index(0, iy * ny, iz),
                            x_index(0, iy * ny, iz + 1),
                            x_index(1, iy, iz + 1),
                        ]);
                    }
                }
            }
        }

        // -z and +z sides
        let z_index = |ix: usize, iy: usize, iz: usize| offset2 + (ix * (ny - 1) + iy) * 2 + iz;
        let create_corner_poly = |ix: usize, iy: usize, iz: usize, indices: &mut Vec<usize>| {
            if ny > 1 {
                if (iz > 0) ^ (iy > 0) ^ (ix > 0) {
                    indices.extend_from_slice(&[
                        4,
                        x_index(ix, iy * ny, iz * nz),
                        x_index(ix, iy * (ny - 2) + 1, iz * nz),
                        z_index(ix * (nx - 2), iy * (ny - 2), iz),
                        y_index(ix * (nx - 2), iy, iz * nz),
                    ]);
                } else {
                    indices.extend_from_slice(&[
                        4,
                        y_index(ix * (nx - 2), iy, iz * nz),
                        z_index(ix * (nx - 2), iy * (ny - 2), iz),
                        x_index(ix, (ny - 2) * iy + 1, iz * nz),
                        x_index(ix, ny * iy, iz * nz),
                    ]);
                }
            } else if iy == 0 {
                if (iz > 0) ^ (ix > 0) {
                    indices.extend_from_slice(&[
                        4,
                        x_index(ix, iy * ny, iz * nz),
                        x_index(ix, iy + 1, iz * nz),
                        y_index(ix * (nx - 2), iy + 1, iz * nz),
                        y_index(ix * (nx - 2), iy, iz * nz),
                    ]);
                } else {
                    indices.extend_from_slice(&[
                        4,
                        y_index(ix * (nx - 2), iy, iz * nz),
                        y_index(ix * (nx - 2), iy + 1, iz * nz),
                        x_index(ix, iy + 1, iz * nz),
                        x_index(ix, iy, iz * nz),
                    ]);
                }
            }
        };
        let create_x_side_polys = |ix: usize, iy: usize, iz: usize, indices: &mut Vec<usize>| {
            if (iz > 0) ^ (ix > 0) {
                indices.extend_from_slice(&[
                    4,
                    x_index(ix, iy + 1, iz * nz),
                    x_index(ix, iy + 2, iz * nz),
                    z_index(ix * (nx - 2), iy + 1, iz),
                    z_index(ix * (nx - 2), iy, iz),
                ]);
            } else {
                indices.extend_from_slice(&[
                    4,
                    z_index(ix * (nx - 2), iy, iz),
                    z_index(ix * (nx - 2), iy + 1, iz),
                    x_index(ix, iy + 2, iz * nz),
                    x_index(ix, iy + 1, iz * nz),
                ]);
            }
        };

        for iz in 0..2 {
            if nx > 1 {
                if ny > 1 {
                    create_corner_poly(0, 0, iz, &mut indices);
                    for iy in 0..ny - 2 {
                        create_x_side_polys(0, iy, iz, &mut indices);
                    }
                    create_corner_poly(0, 1, iz, &mut indices);
                    let create_y_side_polys =
                        |ix: usize, iy: usize, iz: usize, indices: &mut Vec<usize>| {
                            if (iz > 0) ^ (iy > 0) {
                                indices.extend_from_slice(&[
                                    4,
                                    y_index(ix, iy, iz * nz),
                                    z_index(ix, iy * (ny - 2), iz),
                                    z_index(ix + 1, iy * (ny - 2), iz),
                                    y_index(ix + 1, iy, iz * nz),
                                ]);
                            } else {
                                indices.extend_from_slice(&[
                                    4,
                                    y_index(ix + 1, iy, iz * nz),
                                    z_index(ix + 1, iy * (ny - 2), iz),
                                    z_index(ix, iy * (ny - 2), iz),
                                    y_index(ix, iy, iz * nz),
                                ]);
                            }
                        };
                    for ix in 0..nx - 2 {
                        create_y_side_polys(ix, 0, iz, &mut indices);
                        for iy in 0..ny - 2 {
                            let index = |x: usize, y: usize, z: usize| {
                                offset2 + ((ix + x) * (ny - 1) + (iy + y)) * 2 + (iz + z)
                            };
                            if iz > 0 {
                                // reversed
                                indices.extend_from_slice(&[
                                    4,
                                    index(0, 0, 0),
                                    index(0, 1, 0),
                                    index(1, 1, 0),
                                    index(1, 0, 0),
                                ]);
                            } else {
                                indices.extend_from_slice(&[
                                    4,
                                    index(1, 0, 0),
                                    index(1, 1, 0),
                                    index(0, 1, 0),
                                    index(0, 0, 0),
                                ]);
                            }
                        }
                        create_y_side_polys(ix, 1, iz, &mut indices);
                    }
                    create_corner_poly(1, 0, iz, &mut indices);
                    for iy in 0..ny - 2 {
                        create_x_side_polys(1, iy, iz, &mut indices);
                    }
                    create_corner_poly(1, 1, iz, &mut indices);
                } else {
                    create_corner_poly(0, 0, iz, &mut indices);
                    for ix in 0..nx - 2 {
                        if iz > 0 {
                            indices.extend_from_slice(&[
                                4,
                                y_index(ix, 0, iz * nz),
                                y_index(ix, 1, iz * nz),
                                y_index(ix + 1, 1, iz * nz),
                                y_index(ix + 1, 0, iz * nz),
                            ]);
                        } else {
                            indices.extend_from_slice(&[
                                4,
                                y_index(ix + 1, 0, iz * nz),
                                y_index(ix + 1, 1, iz * nz),
                                y_index(ix, 1, iz * nz),
                                y_index(ix, 0, iz * nz),
                            ]);
                        }
                    }
                    create_corner_poly(1, 0, iz, &mut indices);
                }
            } else {
                for iy in 0..ny {
                    if iz > 0 {
                        indices.extend_from_slice(&[
                            4,
                            x_index(0, iy, iz * nz),
                            x_index(0, iy + 1, iz * nz),
                            x_index(1, iy + 1, iz * nz),
                            x_index(1, iy, iz * nz),
                        ]);
                    } else {
                        indices.extend_from_slice(&[
                            4,
                            x_index(1, iy, iz * nz),
                            x_index(1, iy + 1, iz * nz),
                            x_index(0, iy + 1, iz * nz),
                            x_index(0, iy, iz * nz),
                        ]);
                    }
                }
            }
        }

        PolyMesh::new(positions, &indices)
    }
    /// Builds a trimesh box.
    pub fn build_trimesh<T: Real>(self) -> TriMesh<T> {
        TriMesh::from(self.build_polymesh::<T>())
    }
}

impl<T: Real> MeshBuilder<PolyMesh<T>> for BoxBuilder {
    fn build(self) -> PolyMesh<T> {
        self.build_polymesh::<T>()
    }
}

impl<T: Real> MeshBuilder<TetMesh<T>> for BoxBuilder {
    fn build(self) -> TetMesh<T> {
        self.build_tetmesh::<T>()
    }
}

impl<T: Real> MeshBuilder<TriMesh<T>> for BoxBuilder {
    fn build(self) -> TriMesh<T> {
        self.build_trimesh::<T>()
    }
}

/// Convex regular polyhedron builder.
///
/// [Platonic solides](https://en.wikipedia.org/wiki/Platonic_solid) have congruent regular polygon faces
/// with constant vertex valence.
/// There are 5 such shapes:
///  - Tetrahedron,
///  - Cube,
///  - Octahedron,
///  - Dodecahedron, and
///  - Icosahedron
///
/// Each shape built using this builder has a consistent radius, meaning vertices are always `radius`
/// distance away from the origin.
///
/// NOTE: Vertices of cubes created using this builder will not be exact in floating
/// point arithmetic. If this is a requirement, use the `BoxBuilder`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PlatonicSolidBuilder {
    radius: f32,
}

impl Default for PlatonicSolidBuilder {
    fn default() -> Self {
        PlatonicSolidBuilder { radius: 1.0 }
    }
}

impl PlatonicSolidBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the radius of the platonic solid to be built.
    pub fn radius(self, radius: f32) -> Self {
        Self { radius }
    }

    /// Builds a regular octahedron as a `TriMesh`.
    pub fn build_octahedron<T: Real>(self) -> TriMesh<T> {
        let h = T::from(self.radius).unwrap();
        let z = T::zero();
        let vertices = vec![
            [-h, z, z],
            [h, z, z],
            [z, -h, z],
            [z, h, z],
            [z, z, -h],
            [z, z, h],
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

    /// Builds a regular cube as a `PolyMesh`.
    ///
    /// All vertices on the cube have norm approximately equal to `1.0`.
    /// To create a cube with exact vertex positions, use `BoxBuilder`.
    pub fn build_cube<T: Real>(self) -> PolyMesh<T> {
        #[rustfmt::skip]
        let h = T::from(self.radius / num_traits::Float::sqrt(3.0)).unwrap();
        let vertices = vec![
            [h, -h, h],
            [-h, -h, h],
            [h, h, h],
            [-h, h, h],
            [-h, -h, -h],
            [h, -h, -h],
            [-h, h, -h],
            [h, h, -h],
        ];

        #[rustfmt::skip]
        let indices = vec![
            4, 0, 1, 3, 2,
            4, 4, 5, 7, 6,
            4, 6, 7, 2, 3,
            4, 5, 4, 1, 0,
            4, 5, 0, 2, 7,
            4, 1, 4, 6, 3,
        ];

        PolyMesh::new(vertices, &indices)
    }

    /// Builds a regular tetrahedron as a `TetMesh`.
    pub fn build_tetrahedron<T: Real>(self) -> TetMesh<T> {
        let r = T::from(self.radius).unwrap();
        let sqrt_8_by_9 = r * T::from(f64::sqrt(8.0 / 9.0)).unwrap();
        let sqrt_2_by_9 = r * T::from(f64::sqrt(2.0 / 9.0)).unwrap();
        let sqrt_2_by_3 = r * T::from(f64::sqrt(2.0 / 3.0)).unwrap();
        let third = r * T::from(1.0 / 3.0).unwrap();
        let vertices = vec![
            [T::zero(), r, T::zero()],
            [-sqrt_8_by_9, -third, T::zero()],
            [sqrt_2_by_9, -third, sqrt_2_by_3],
            [sqrt_2_by_9, -third, -sqrt_2_by_3],
        ];

        let indices = vec![[3, 1, 0, 2]];

        TetMesh::new(vertices, indices)
    }

    /// Builds a regular icosahedron as a `TriMesh`.
    pub fn build_icosahedron<T: Real>(self) -> TriMesh<T> {
        let sqrt5_f64 = 5.0_f64.sqrt();
        let sqrt5 = T::from(sqrt5_f64).unwrap();
        let r = T::from(self.radius).unwrap();
        let a = r / sqrt5;
        let w1 = T::from(0.25 * (sqrt5_f64 - 1.0)).unwrap();
        let h1 = T::from((0.125 * (5.0 + sqrt5_f64)).sqrt()).unwrap();
        let w2 = T::from(0.25 * (sqrt5_f64 + 1.0)).unwrap();
        let h2 = T::from((0.125 * (5.0 - sqrt5_f64)).sqrt()).unwrap();
        let two = T::from(2.0).unwrap();
        let vertices = vec![
            // North pole
            [T::zero(), T::zero(), r],
            // Alternating ring
            [T::zero(), two * a, a],
            [two * a * h2, two * a * w2, -a],
            [two * a * h1, two * a * w1, a],
            [two * a * h1, -two * a * w1, -a],
            [two * a * h2, -two * a * w2, a],
            [T::zero(), -two * a, -a],
            [-two * a * h2, -two * a * w2, a],
            [-two * a * h1, -two * a * w1, -a],
            [-two * a * h1, two * a * w1, a],
            [-two * a * h2, two * a * w2, -a],
            // South pole
            [T::zero(), T::zero(), -r],
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

    /// Builds a regular dodecahedron as a `PolyMesh`.
    pub fn build_dodecahedron<T: Real>(self) -> PolyMesh<T> {
        use num_traits::Float;
        let r = T::from(self.radius as f64 / 3.0.sqrt()).unwrap();
        let phi = T::from(0.5 * (1.0 + 5.0.sqrt())).unwrap();
        let phi_inv = T::one() / phi;

        let rphi = r * phi;
        let rphi_inv = r * phi_inv;

        let vertices = vec![
            // Orange
            [-r, -r, -r], // 0
            [-r, -r, r],  // 1
            [-r, r, -r],  // 2
            [-r, r, r],   // 3
            [r, -r, -r],  // 4
            [r, -r, r],   // 5
            [r, r, -r],   // 6
            [r, r, r],    // 7
            // Green
            [T::zero(), -rphi, -rphi_inv], // 8
            [T::zero(), -rphi, rphi_inv],  // 9
            [T::zero(), rphi, -rphi_inv],  // 10
            [T::zero(), rphi, rphi_inv],   // 11
            // Blue
            [-rphi_inv, T::zero(), -rphi], // 12
            [-rphi_inv, T::zero(), rphi],  // 13
            [rphi_inv, T::zero(), -rphi],  // 14
            [rphi_inv, T::zero(), rphi],   // 15
            // Pink
            [-rphi, -rphi_inv, T::zero()], // 16
            [-rphi, rphi_inv, T::zero()],  // 17
            [rphi, -rphi_inv, T::zero()],  // 18
            [rphi, rphi_inv, T::zero()],   // 19
        ];

        let indices = vec![
            5, 8, 4, 14, 12, 0, // 0
            5, 14, 4, 18, 19, 6, // 1
            5, 14, 6, 10, 2, 12, // 2
            5, 6, 19, 7, 11, 10, // 3
            5, 10, 11, 3, 17, 2, // 4
            5, 2, 17, 16, 0, 12, // 5
            5, 5, 15, 7, 19, 18, // 6
            5, 9, 1, 13, 15, 5, // 7
            5, 8, 9, 5, 18, 4, // 8
            5, 16, 1, 9, 8, 0, // 9
            5, 1, 16, 17, 3, 13, // 10
            5, 13, 3, 11, 7, 15, // 11
        ];

        PolyMesh::new(vertices, &indices)
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

    pub fn build_polymesh<T: Real>(self) -> PolyMesh<T> {
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
                    T::from(theta.cos() * (outer_radius as f64 + phi.cos() * inner_radius as f64))
                        .unwrap(),
                    T::from(phi.sin() * inner_radius as f64).unwrap(),
                    T::from(theta.sin() * (outer_radius as f64 + phi.cos() * inner_radius as f64))
                        .unwrap(),
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
    use crate::topology::{NumFaces, NumVertices};
    use crate::VertexPositions;
    use math::Vector3;

    /// Verify that the platonic solids have unit radius.
    #[test]
    fn platonic_unity_test() {
        use crate::mesh::VertexPositions;
        use approx::assert_relative_eq;
        use math::Vector3;

        let icosa = PlatonicSolidBuilder::default().build_icosahedron::<f64>();
        for &v in icosa.vertex_positions() {
            assert_relative_eq!(Vector3::from(v).norm(), 1.0);
        }

        let tet = PlatonicSolidBuilder::default().build_tetrahedron::<f64>();
        for &v in tet.vertex_positions() {
            assert_relative_eq!(Vector3::from(v).norm(), 1.0);
        }

        let cube = PlatonicSolidBuilder::default().build_cube::<f64>();
        for &v in cube.vertex_positions() {
            assert_relative_eq!(Vector3::from(v).norm(), 1.0, max_relative = 1e-6);
        }

        let octa = PlatonicSolidBuilder::default().build_octahedron::<f64>();
        for &v in octa.vertex_positions() {
            assert_relative_eq!(Vector3::from(v).norm(), 1.0);
        }

        let dodeca = PlatonicSolidBuilder::default().build_dodecahedron::<f64>();
        for &v in dodeca.vertex_positions() {
            assert_relative_eq!(Vector3::from(v).norm(), 1.0, max_relative = 1e-6);
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
        .build::<f64>();
        let bbox = grid.bounding_box();
        assert_eq!(bbox.min_corner(), [-1.0, 0.0, -1.0]);
        assert_eq!(bbox.max_corner(), [1.0, 0.0, 1.0]);
    }

    #[test]
    fn box_test() {
        let check_regular_box = |mesh: PolyMesh<f64>, n: usize| {
            // Check that all vertices lie on the cube.
            assert!(mesh
                .vertex_positions()
                .iter()
                .all(|&p| { Vector3::from(p).amax() == 1.0 }));

            // Check that the area of each polygon is the same.
            let p = mesh.vertex_positions();
            for f in mesh.face_iter() {
                let p0 = Vector3::from(p[f[0]]);
                let p3 = Vector3::from(p[f[3]]);
                let e01 = Vector3::from(p[f[1]]) - p0;
                let e02 = Vector3::from(p[f[2]]) - p0;
                let e11 = Vector3::from(p[f[1]]) - p3;
                let e12 = Vector3::from(p[f[2]]) - p3;
                let a0 = 0.5 * e02.cross(&e01).norm();
                let a1 = 0.5 * e12.cross(&e11).norm();
                // These are probably exact since vertex coordinates are powers of 2.
                assert!((a0 - a1).abs() <= f64::EPSILON);
                let expected_area = 4.0 / ((n * n) as f64);
                assert!((a0 + a1 - expected_area).abs() <= f64::EPSILON);
            }
        };

        // Regular boxes
        for i in 0..5 {
            let polybox: PolyMesh<f64> = BoxBuilder { divisions: [i; 3] }.build();
            check_regular_box(polybox, i as usize + 1);
        }

        let check_irregular_box = |mesh: PolyMesh<f64>, [i, j, k]: [usize; 3]| {
            assert_eq!(mesh.num_faces(), 2 * (i * j + i * k + j * k));
            assert_eq!(
                mesh.num_vertices(),
                2 * ((i + 1) * (j + 1) + (i + 1) * (k - 1) + (j - 1) * (k - 1))
            );
        };

        // Irregular shaped boxes
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let polybox: PolyMesh<f64> = BoxBuilder {
                        divisions: [i, j, k],
                    }
                    .build();
                    check_irregular_box(polybox, [i as usize + 1, j as usize + 1, k as usize + 1]);
                }
            }
        }
    }
}
