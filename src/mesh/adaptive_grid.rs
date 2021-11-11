//! Warning: This is data structure is a work in progress.

use crate::{Index, FaceVertex, FaceVertexIndex, VertexIndex, FaceIndex};
use std::collections::HashMap;

/// A 3-dimensional adaptive grid designed for computation.
/// If there are n cells at the coarse level in one dimension, there are n+1 grid nodes.
pub struct AdaptiveGrid {
    /// Number of levels of adaptation. Level of 1 produces a regular grid.
    levels: u8,

    /// Size of the coarsest level grid cell.
    scale: f32,

    /// Positions of the first node of the grid. The grid grows in the positive direction in each
    /// dimension.
    origin: [f32; 3],

    /// A signed 1-based index of an outer grid cell. This can reference a true cell (positive `index`)
    /// or a subgrid (negative `index`). A zero value corresponds to an invald index. This logic is
    /// abstracted away trough the `GridIndex` enum, we use `isize` to optimize storage.
    indices: Vec<isize>,

    /// A sparse collection of subgrids in this adaptive grid.
    subgrids: Vec<Grid>,

    /// Number of divisions in each dimension.
    num_divisions: [usize; 3],
}

impl AdaptiveGrid {
    /// Number of subdivisions per level.
    const N: u8 = 2;

    /// Get the grid index for the cell at the given index.
    fn cell(i: usize, j: usize, k: usize) -> GridIndex {
        let [nx, ny, nz] = num_divisions.into();
        GridIndex::new(indices[nz*(ny * i + j) + k])
    }
}

pub enum GridIndex {
    Cell(usize),
    SubGrid(usize),
    Invalid,
}

impl GridIndex {
    /// Given a signed 1-based index of an outer grid cell, this function returns either a true
    /// cell index (for a positive `idx`) or a subgrid index (for a negative `idx`).
    /// A zero value corresponds to an invald index.
    /// This function is not meant to be part of the public interface.
    fn new(idx: isize) -> Self {
        if idx < 0 {
            GridIndex::SubGrid(idx as usize)
        } else if idx > 0 {
            GridIndex::Cell(idx as usize)
        } else {
            GridIndex::Invalid
        }
    }
}

impl MeshData {
    pub fn new(verts: Vec<[f32; 3]>, indices: Vec<usize>) -> MeshData {
        MeshData {
            vertex_positions: verts,
            indices, 
        }
    }
}

/// Implement indexing for the Vec types used in the mesh
/// The macro `impl_index_for_vec` is defined in the `index` module
impl_index_for_vec!(usize);
impl_index_for_vec!([f32; 3]);

pub struct TriMesh {
    /// Vertex positions and indices into that array.
    pub mesh_data: MeshData,
    pub vertex_normals: Option<Vec<[f32; 3]>>,
    //pub face_vertex_uvs: Option<Vec<Vector2<f32>>>,
}

impl TriMesh {
    pub fn new(verts: Vec<[f32; 3]>, indices: Vec<usize>) -> TriMesh {
        TriMesh { 
            mesh_data: MeshData::new(verts,indices),
			vertex_normals: None,
        }
    }
    pub fn vertex_normals(mut self, normals: Vec<[f32; 3]>) -> Self {
        self.vertex_normals = Some(normals);
        self
    }
    pub fn num_faces(&self) -> usize {
        self.mesh_data.indices.len()/3
    }
    /*
    pub fn face_normals(mut self, normals: Vec<Vector3<f32>>) -> Self {
        self.mesh_data.face_normals = Some(normals);
        self
    }
    pub fn face_vertex_uvs(mut self, uvs: Vec<Vector2<f32>>) -> Self {
        self.mesh_data.face_vertex_uvs = Some(uvs);
        self
    }
    */
}

/// Mesh composed of tetrahedra.
#[allow(dead_code)]
pub struct TetMesh {
    /// Vertex positions and indices into that array.
    mesh_data: MeshData,
}

/// Mesh with arbitrarily shaped faces. It could have polygons with any number of sides.
#[allow(dead_code)]
pub struct PolyMesh {
    mesh_data: MeshData,
    /// Offsets into the indices Vec. Two consecutive offsets identify a polygon face
    offsets: Vec<usize>,
}

impl FaceVertex for TriMesh {
	#[inline]
    fn face_vertex<FI,FVI>(&self, fidx: FI, fvidx: FVI) -> VertexIndex
		where FI: Into<FaceIndex> + Into<Index>,
			  FVI: Into<FaceVertexIndex> + Into<Index> {
		let fidx_raw: Index = fidx.into();
		let fvidx_raw: Index = fvidx.into();
        debug_assert!({
		   	if fvidx_raw.is_valid() {
			   fvidx_raw < 3
                        } else {
			   true // ok if invalid index
			}
		});

        self.mesh_data.indices[3*fidx_raw + fvidx_raw].into()
    }
}
