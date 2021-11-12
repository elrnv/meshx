//!
//! Unstructured mesh module.
//!
//! This module defines a mesh data structure cunstructed from unstructured
//! cells of arbitrary shape.
//!

use crate::mesh::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::vertex_positions::VertexPositions;
use crate::Real;

use flatk::*;

/// A marker for the type of cell contained in a Mesh.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CellType {
    Tetrahedron,
    Triangle,
}

/// Mesh with arbitrarily shaped elements or cells.
///
/// NOTE: We stick with the terminology cell but these could very well be called
/// elements. The exact terminology would depend on how this mesh is used.
#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct Mesh<T: Real> {
    /// Vertex positions intrinsic attribute.
    #[intrinsic(VertexPositions)]
    pub vertex_positions: IntrinsicAttribute<[T; 3], VertexIndex>,
    /// Indices into `vertices`. Each chunk represents a cell, and cells
    /// can have an arbitrary number of referenced vertices. Each clump
    /// represents cells of the same kind.
    pub indices: flatk::Clumped<Vec<usize>>,
    /// Types of cells, one for each clump in `indices`.
    pub types: Vec<CellType>,
    /// Vertex attributes.
    pub vertex_attributes: AttribDict<VertexIndex>,
    /// Cell attributes.
    pub cell_attributes: AttribDict<CellIndex>,
    /// Cell vertex attributes.
    pub cell_vertex_attributes: AttribDict<CellVertexIndex>,
    /// Indirect attribute value cache
    pub attribute_value_cache: AttribValueCache,
}

impl<T: Real> Mesh<T> {
    /// Construct a `Mesh` from an array of vertices and cells with associated types given queried with a function.
    ///
    /// The `cells` array contains the indices into the vertex array for each cell preceeded by the
    /// number of vertices in the corresponding cell. I.e. `cells` is expected to be structured as
    /// a contiguous array of a number (corresponding to the number of vertices in the cell)
    /// followed by the vertex indices (in the same cell):
    /// ```verbatim
    ///     n i_1 i_2 ... i_n m j_1 j_2 ... j_m ...
    /// ```
    ///
    /// # Examples
    /// ```
    /// use meshx::mesh::{Mesh, CellType};
    /// let points = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0]];
    /// let cells = vec![3, 0, 1, 2, // first triangle
    ///                  3, 1, 3, 2, // second triangle
    ///                  4, 0, 1, 5, 4]; // tetrahedron
    ///
    /// let mesh = Mesh::from_cells_with_type(points, &cells, |i| if i < 2 { CellType::Triangle } else { CellType::Tetrahedron });
    ///
    /// assert_eq!(mesh.indices.data, vec![0, 1, 2, 1, 3, 2, 0, 1, 5, 4]);
    ///
    /// ```
    pub fn from_cells_with_type(
        verts: Vec<[T; 3]>,
        cells: &[usize],
        type_at: impl Fn(usize) -> CellType,
    ) -> Mesh<T> {
        let mut indices = Vec::new();
        let mut offsets = Vec::new();
        let mut i = 0;
        while i < cells.len() {
            let n = cells[i];
            offsets.push(indices.len());
            i += 1;

            for k in 0..n {
                indices.push(cells[i + k]);
            }

            i += n;
        }

        offsets.push(indices.len());

        let chunked_indices = flatk::Chunked::from_offsets(offsets, indices);

        // TODO: use the From trait to convert to Clumped.
        let clumped_indices = flatk::Clumped {
            chunks: flatk::ClumpedOffsets::from(chunked_indices.chunks),
            data: chunked_indices.data,
        };

        let types = (0..clumped_indices.chunks.num_clumps())
            .map(type_at)
            .collect();

        Mesh {
            vertex_positions: IntrinsicAttribute::from_vec(verts),
            indices: clumped_indices,
            types,
            vertex_attributes: AttribDict::new(),
            cell_attributes: AttribDict::new(),
            cell_vertex_attributes: AttribDict::new(),
            attribute_value_cache: AttribValueCache::with_hasher(Default::default()),
        }
    }

    /// Construct a `Mesh` from an array of vertices and clumped cells with associated types.
    ///
    /// This is the most efficient way to construct a mesh, but also the easiest to get wrong.
    pub fn from_clumped_cells_and_types(
        verts: Vec<[T; 3]>,
        cells: Vec<usize>,
        offsets: Vec<usize>,
        chunk_offsets: Vec<usize>,
        types: Vec<CellType>,
    ) -> Mesh<T> {
        Mesh {
            vertex_positions: IntrinsicAttribute::from_vec(verts),
            indices: flatk::Clumped::from_clumped_offsets(chunk_offsets, offsets, cells),
            types,
            vertex_attributes: AttribDict::new(),
            cell_attributes: AttribDict::new(),
            cell_vertex_attributes: AttribDict::new(),
            attribute_value_cache: AttribValueCache::with_hasher(Default::default()),
        }
    }

    pub fn cell_iter(&self) -> impl ExactSizeIterator<Item = &[usize]> {
        self.indices.iter()
    }

    pub fn cell_iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [usize]> {
        self.indices.iter_mut()
    }

    /// Reverse the order of each cell in this mesh.
    #[inline]
    pub fn reverse(&mut self) {
        for cell in self.cell_iter_mut() {
            cell.reverse();
        }

        let Self {
            cell_vertex_attributes,
            indices,
            ..
        } = self;

        // TODO: Consider doing reversing lazily using a flag field.

        // Since each vertex has an associated cell vertex attribute, we must remap those
        // as well.
        // Reverse cell vertex attributes
        for (_, attrib) in cell_vertex_attributes.iter_mut() {
            let mut data_slice = attrib.data_mut_slice();
            for cell in indices.iter() {
                let mut i = 0usize;
                let num_verts = cell.len();
                while i < num_verts / 2 {
                    data_slice.swap(cell[i], cell[num_verts - i - 1]);
                    i += 1;
                }
            }
        }
    }

    /// Reverse the order of each cell in this mesh.
    ///
    ///  This is the consuming version of the `reverse` method.
    #[inline]
    pub fn reversed(mut self) -> Mesh<T> {
        self.reverse();
        self
    }
}

impl<T: Real> Default for Mesh<T> {
    /// Produce an empty mesh.
    ///
    /// This is not particularly useful on its own, however it can be used as a
    /// null case for various mesh algorithms.
    ///
    /// This function allocates two `Vec`s of size 1.
    fn default() -> Self {
        Mesh::from_clumped_cells_and_types(vec![], vec![], vec![0], vec![0], vec![])
    }
}

impl<T: Real> NumVertices for Mesh<T> {
    #[inline]
    fn num_vertices(&self) -> usize {
        self.vertex_positions.len()
    }
}

impl<T: Real> NumCells for Mesh<T> {
    #[inline]
    fn num_cells(&self) -> usize {
        self.indices.len()
    }
}

impl<T: Real> CellVertex for Mesh<T> {
    #[inline]
    fn vertex<CVI>(&self, cv_idx: CVI) -> VertexIndex
    where
        CVI: Copy + Into<CellVertexIndex>,
    {
        let cv_idx = usize::from(cv_idx.into());
        debug_assert!(cv_idx < self.num_cell_vertices());
        self.indices.data[cv_idx].into()
    }

    #[inline]
    fn cell_vertex<CI>(&self, cidx: CI, which: usize) -> Option<CellVertexIndex>
    where
        CI: Copy + Into<CellIndex>,
    {
        if which >= self.num_vertices_at_cell(cidx) {
            None
        } else {
            let cidx = usize::from(cidx.into());
            debug_assert!(cidx < self.num_cells());

            // SAFETY: cidx is known to be less than num_offsets - 1, we
            // actually only need cidx to be less than num_offsets for safety.
            Some((unsafe { self.indices.chunks.offset_value_unchecked(cidx) } + which).into())
        }
    }

    #[inline]
    fn num_cell_vertices(&self) -> usize {
        self.indices.data.len()
    }

    #[inline]
    fn num_vertices_at_cell<CI>(&self, cidx: CI) -> usize
    where
        CI: Copy + Into<CellIndex>,
    {
        let cidx = usize::from(cidx.into());
        self.indices.view().at(cidx).len()
    }
}

impl<T: Real> From<super::TriMesh<T>> for Mesh<T> {
    /// Convert a triangle mesh into an unstructured mesh.
    fn from(mesh: super::TriMesh<T>) -> Mesh<T> {
        let super::TriMesh {
            vertex_positions,
            indices,
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            .. // Missing face-edge and vertex-face topo attributes
        } = mesh;

        let types = vec![CellType::Triangle; 1];

        // The cell and cell-vertex topology is the same as the original face
        // and face-vertex topology, so we just need to move all the attributes
        // as is.

        let mut cell_attributes = AttribDict::new();
        for (name, attrib) in face_attributes {
            let mut new_attrib = attrib.promote_empty::<CellIndex>();
            new_attrib.data = attrib.data;
            cell_attributes.insert(name, new_attrib);
        }

        let mut cell_vertex_attributes = AttribDict::new();
        for (name, attrib) in face_vertex_attributes {
            let mut new_attrib = attrib.promote_empty::<CellVertexIndex>();
            new_attrib.data = attrib.data;
            cell_vertex_attributes.insert(name, new_attrib);
        }

        Mesh {
            vertex_positions,
            indices: flatk::Clumped::from_clumped_offsets(
                vec![0, indices.len()],
                vec![0, 3 * indices.len()],
                flatk::Chunked3::from_array_vec(indices.into_vec()).into_inner(),
            ),
            types,
            vertex_attributes,
            cell_attributes,
            cell_vertex_attributes,
            attribute_value_cache: AttribValueCache::default(),
        }
    }
}

impl<T: Real> From<super::TetMesh<T>> for Mesh<T> {
    /// Convert a tetmesh into an unstructured mesh.
    fn from(mesh: super::TetMesh<T>) -> Mesh<T> {
        let super::TetMesh {
            vertex_positions,
            indices,
            vertex_attributes,
            cell_attributes,
            cell_vertex_attributes,
            ..
        } = mesh;

        let types = vec![CellType::Tetrahedron; 1];

        Mesh {
            vertex_positions,
            indices: flatk::Clumped::<Vec<usize>>::from_clumped_offsets(
                vec![0, indices.len()],
                vec![0, 4 * indices.len()],
                flatk::Chunked4::from_array_vec(indices.into_vec()).into_inner(),
            ),
            types,
            vertex_attributes,
            cell_attributes,
            cell_vertex_attributes,
            attribute_value_cache: AttribValueCache::default(),
        }
    }
}

/// Convert a point cloud into a mesh.
impl<T: Real> From<super::PointCloud<T>> for Mesh<T> {
    fn from(mesh: super::PointCloud<T>) -> Mesh<T> {
        let super::PointCloud {
            vertex_positions,
            vertex_attributes,
        } = mesh;

        Mesh {
            vertex_attributes,
            ..Mesh::from_clumped_cells_and_types(
                vertex_positions.into(),
                vec![],
                vec![0],
                vec![0],
                vec![],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;

    #[test]
    fn mesh_test() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let cells = vec![
            3, 0, 1, 2, // first triangle
            3, 1, 3, 2, // second triangle
            4, 0, 1, 5, 4, // tetrahedron
        ];

        let mesh = Mesh::from_cells_with_type(points, &cells, |i| {
            if i < 2 {
                CellType::Triangle
            } else {
                CellType::Tetrahedron
            }
        });
        assert_eq!(mesh.num_vertices(), 6);
        assert_eq!(mesh.num_cells(), 3);
        assert_eq!(mesh.num_cell_vertices(), 10);

        assert_eq!(Index::from(mesh.cell_to_vertex(1, 1)), 3);
        assert_eq!(Index::from(mesh.cell_to_vertex(0, 2)), 2);
    }

    fn sample_points() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn from_trimesh_test() {
        use crate::mesh::TriMesh;
        let points = sample_points();
        let tri_faces = vec![
            3, 0, 1, 2, // first triangle
            3, 1, 3, 2, // second triangle
        ];

        let trimesh = TriMesh::new(points.clone(), vec![[0, 1, 2], [1, 3, 2]]);
        let tri_mesh =
            Mesh::from_cells_with_type(points.clone(), &tri_faces, |_| CellType::Triangle);
        assert_eq!(Mesh::from(trimesh), tri_mesh);
    }

    #[test]
    fn from_tetmesh_test() {
        use crate::mesh::TetMesh;
        let points = sample_points();
        let tets = vec![
            4, 0, 1, 3, 2, // just one tet
        ];

        let tetmesh = TetMesh::new(points.clone(), vec![[0, 1, 3, 2]]);
        let tet_mesh = Mesh::from_cells_with_type(points, &tets, |_| CellType::Tetrahedron);
        assert_eq!(Mesh::from(tetmesh), tet_mesh);
    }

    #[test]
    fn from_pointcloud_test() {
        use crate::mesh::PointCloud;
        let points = sample_points();

        let ptcld = PointCloud::new(points.clone());
        let ptcld_mesh = Mesh::from_cells_with_type(
            points.clone(),
            &[],
            |_| CellType::Triangle, /* never called */
        );
        assert_eq!(Mesh::from(ptcld), ptcld_mesh);
    }
}
