//!
//! Extended Tetmesh. Describes a tetrahedron mesh data structure that is accompanied by its dual
//! voronoi topology.
//!

pub use super::surface::*;
use super::TetMesh;

use crate::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::vertex_positions::*;
use crate::prim::Tetrahedron;
use crate::Real;
use std::slice::{Iter, IterMut};

/// Mesh composed of tetrahedra, extended with its dual voronoi topology.
#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct TetMeshExt<T: Real> {
    #[attributes(Vertex, Cell, CellVertex, CellFace)]
    #[intrinsics(VertexPositions::vertex_positions)]
    pub tetmesh: TetMesh<T>,
    /// Lists of cell indices for each vertex. Since each vertex can have a variable number of cell
    /// neighbours, the `cell_offsets` field keeps track of where each subarray of indices begins.
    pub cell_indices: Vec<usize>,
    /// Offsets into the `cell_indices` array, one for each vertex. The last offset is always
    /// equal to the size of `cell_indices` for convenience.
    pub cell_offsets: Vec<usize>,
    /// Vertex cell Attributes.
    pub vertex_cell_attributes: AttribDict<VertexCellIndex>,
}

impl<T: Real> TetMeshExt<T> {
    /// This constant defines the triangle faces of each tet. The rule is that `i`th face of the
    /// tet is the one opposite to the `i`th vertex. The triangle starts with the smallest index.
    pub const TET_FACES: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]];

    pub fn new(verts: Vec<[T; 3]>, indices: Vec<[usize; 4]>) -> TetMeshExt<T> {
        let (cell_indices, cell_offsets) = Self::compute_dual_topology(verts.len(), &indices);

        TetMeshExt {
            tetmesh: TetMesh::new(verts, indices),
            cell_indices,
            cell_offsets,
            vertex_cell_attributes: AttribDict::new(),
        }
    }

    pub(crate) fn compute_dual_topology(
        num_verts: usize,
        indices: &[[usize; 4]],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut cell_indices = Vec::new();
        cell_indices.resize(num_verts, Vec::new());
        for (cidx, cell) in indices.iter().enumerate() {
            for &vidx in cell {
                cell_indices[vidx].push(cidx);
            }
        }

        let mut cell_offsets = Vec::with_capacity(indices.len());
        cell_offsets.push(0);
        for neighbours in cell_indices.iter() {
            let last = *cell_offsets.last().unwrap();
            cell_offsets.push(last + neighbours.len());
        }

        (
            cell_indices
                .iter()
                .flat_map(|x| x.iter().cloned())
                .collect(),
            cell_offsets,
        )
    }

    #[inline]
    pub fn cell_iter(&self) -> Iter<[usize; 4]> {
        self.tetmesh.cell_iter()
    }

    #[cfg(feature = "rayon")]
    #[inline]
    pub fn cell_par_iter(&self) -> rayon::slice::Iter<[usize; 4]> {
        self.tetmesh.cell_par_iter()
    }

    #[inline]
    pub fn cell_iter_mut(&mut self) -> IterMut<[usize; 4]> {
        self.tetmesh.cell_iter_mut()
    }

    #[cfg(feature = "rayon")]
    #[inline]
    pub fn cell_par_iter_mut(&mut self) -> rayon::slice::IterMut<[usize; 4]> {
        self.tetmesh.cell_par_iter_mut()
    }

    /// Cell accessor. These are vertex indices.
    #[inline]
    pub fn cell<CI: Into<CellIndex>>(&self, cidx: CI) -> &[usize; 4] {
        self.tetmesh.cell(cidx.into())
    }

    /// Return a slice of individual cells.
    #[inline]
    pub fn cells(&self) -> &[[usize; 4]] {
        self.tetmesh.cells()
    }

    /// Tetrahedron iterator.
    #[inline]
    pub fn tet_iter(&self) -> impl Iterator<Item = Tetrahedron<T>> + '_ {
        self.tetmesh.tet_iter()
    }

    /// Get a tetrahedron primitive corresponding to the given vertex indices.
    #[inline]
    pub fn tet_from_indices(&self, indices: &[usize; 4]) -> Tetrahedron<T> {
        self.tetmesh.tet_from_indices(indices)
    }

    /// Get a tetrahedron primitive corresponding to the given cell index.
    #[inline]
    pub fn tet<CI: Into<CellIndex>>(&self, cidx: CI) -> Tetrahedron<T> {
        self.tet_from_indices(self.cell(cidx))
    }

    /// Consumes the current mesh to produce a mesh with inverted tetrahedra.
    #[inline]
    pub fn inverted(mut self) -> TetMeshExt<T> {
        self.invert();
        self
    }

    /// Non consuming verion of the `inverted` function which simply modifies the given mesh.
    #[inline]
    pub fn invert(&mut self) {
        self.tetmesh.invert();
    }

    /// Convert this mesh into canonical form. This function inverts any inverted tetrahedron such
    /// that all tetrahedra are in canonical (non-inverted) form. The canonical form is determined
    /// by the shape matrix determinant of each tetrahedron. Canonical tetrahedra have a positive
    /// shape matrix determinant (see the `meshx::ops::ShapeMatrix` trait and the
    /// `meshx::prim::tetrahedron` module).
    #[inline]
    pub fn canonicalized(mut self) -> TetMeshExt<T> {
        self.canonicalize();
        self
    }

    /// Convert this mesh into canonical form. This function inverts any inverted tetrahedron such
    /// that all tetrahedra are in canonical (non-inverted) form. This is a non-consuming version
    /// of `canonicalized`.
    #[inline]
    pub fn canonicalize(&mut self) {
        self.tetmesh.canonicalize();
    }
}

impl<T: Real> Default for TetMeshExt<T> {
    /// Produce an empty `TetMeshExt`. This is not particularly useful on its own, however it can be
    /// used as a null case for various mesh algorithms.
    fn default() -> Self {
        TetMeshExt::new(vec![], vec![])
    }
}

/**
 * Define `TetMeshExt` topology
 */

impl<T: Real> NumVertices for TetMeshExt<T> {
    #[inline]
    fn num_vertices(&self) -> usize {
        self.tetmesh.num_vertices()
    }
}

impl<T: Real> NumCells for TetMeshExt<T> {
    #[inline]
    fn num_cells(&self) -> usize {
        self.tetmesh.num_cells()
    }
}

impl<T: Real> CellVertex for TetMeshExt<T> {
    #[inline]
    fn vertex<CVI>(&self, cv_idx: CVI) -> VertexIndex
    where
        CVI: Copy + Into<CellVertexIndex>,
    {
        self.tetmesh.vertex(cv_idx)
    }

    #[inline]
    fn cell_vertex<CI>(&self, cidx: CI, which: usize) -> Option<CellVertexIndex>
    where
        CI: Copy + Into<CellIndex>,
    {
        self.tetmesh.cell_vertex(cidx, which)
    }

    #[inline]
    fn num_cell_vertices(&self) -> usize {
        self.tetmesh.num_cell_vertices()
    }

    #[inline]
    fn num_vertices_at_cell<CI>(&self, cidx: CI) -> usize
    where
        CI: Copy + Into<CellIndex>,
    {
        self.tetmesh.num_vertices_at_cell(cidx.into())
    }
}

impl<T: Real> CellFace for TetMeshExt<T> {
    #[inline]
    fn face<CFI>(&self, cf_idx: CFI) -> FaceIndex
    where
        CFI: Copy + Into<CellFaceIndex>,
    {
        self.tetmesh.face(cf_idx)
    }

    #[inline]
    fn cell_face<CI>(&self, cidx: CI, which: usize) -> Option<CellFaceIndex>
    where
        CI: Copy + Into<CellIndex>,
    {
        self.tetmesh.cell_face(cidx.into(), which)
    }

    #[inline]
    fn num_cell_faces(&self) -> usize {
        self.tetmesh.num_cell_faces()
    }

    #[inline]
    fn num_faces_at_cell<CI>(&self, cidx: CI) -> usize
    where
        CI: Copy + Into<CellIndex>,
    {
        self.tetmesh.num_faces_at_cell(cidx.into())
    }
}

impl<T: Real> VertexCell for TetMeshExt<T> {
    #[inline]
    fn cell<VCI>(&self, vc_idx: VCI) -> CellIndex
    where
        VCI: Copy + Into<VertexCellIndex>,
    {
        let vc_idx = usize::from(vc_idx.into());
        debug_assert!(vc_idx < self.num_vertex_cells());
        self.cell_indices[vc_idx].into()
    }

    #[inline]
    fn vertex_cell<VI>(&self, vidx: VI, which: usize) -> Option<VertexCellIndex>
    where
        VI: Copy + Into<VertexIndex>,
    {
        if which >= self.num_cells_at_vertex(vidx) {
            return None;
        }

        let vidx = usize::from(vidx.into());

        debug_assert!(vidx < self.num_vertices());

        Some((self.cell_offsets[vidx] + which).into())
    }

    #[inline]
    fn num_vertex_cells(&self) -> usize {
        self.cell_indices.len()
    }

    #[inline]
    fn num_cells_at_vertex<VI>(&self, vidx: VI) -> usize
    where
        VI: Copy + Into<VertexIndex>,
    {
        let vidx = usize::from(vidx.into());
        self.cell_offsets[vidx + 1] - self.cell_offsets[vidx]
    }
}

impl<T: Real> From<TetMesh<T>> for TetMeshExt<T> {
    fn from(tetmesh: TetMesh<T>) -> TetMeshExt<T> {
        let (cell_indices, cell_offsets) =
            Self::compute_dual_topology(tetmesh.vertex_positions.len(), tetmesh.indices.as_slice());

        TetMeshExt {
            tetmesh,
            cell_indices,
            cell_offsets,
            vertex_cell_attributes: AttribDict::new(),
        }
    }
}

impl<T: Real> From<TetMeshExt<T>> for TetMesh<T> {
    fn from(ext: TetMeshExt<T>) -> TetMesh<T> {
        ext.tetmesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;

    fn simple_tetmesh() -> TetMeshExt<f64> {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let indices = vec![[0, 4, 2, 5], [0, 5, 2, 3], [5, 3, 0, 1]];

        TetMeshExt::new(points, indices)
    }

    #[test]
    fn tetmesh_test() {
        let mesh = simple_tetmesh();
        assert_eq!(mesh.num_vertices(), 6);
        assert_eq!(mesh.num_cells(), 3);
        assert_eq!(mesh.num_cell_vertices(), 12);
        assert_eq!(mesh.num_cell_faces(), 12);

        assert_eq!(Index::from(mesh.cell_vertex(1, 1)), 5);
        assert_eq!(Index::from(mesh.cell_vertex(0, 2)), 2);
        assert_eq!(Index::from(mesh.cell_face(2, 3)), 11);

        // Verify dual topology
        let vertex_cells = vec![
            vec![0, 1, 2],
            vec![2],
            vec![0, 1],
            vec![1, 2],
            vec![0],
            vec![0, 1, 2],
        ];
        for i in 0..vertex_cells.len() {
            assert_eq!(mesh.num_cells_at_vertex(i), vertex_cells[i].len());
            let mut local_cells: Vec<usize> = (0..mesh.num_cells_at_vertex(i))
                .map(|j| mesh.vertex_to_cell(i, j).unwrap().into())
                .collect();
            local_cells.sort();
            assert_eq!(local_cells, vertex_cells[i]);
        }
    }

    #[test]
    fn tet_iter_test() {
        use math::Vector3;
        let mesh = simple_tetmesh();
        let points = mesh.vertex_positions().to_vec();

        let pt = |i| Vector3::from(points[i]);

        let tets = vec![
            Tetrahedron(pt(0), pt(4), pt(2), pt(5)),
            Tetrahedron(pt(0), pt(5), pt(2), pt(3)),
            Tetrahedron(pt(5), pt(3), pt(0), pt(1)),
        ];

        for (tet, exptet) in mesh.tet_iter().zip(tets.into_iter()) {
            assert_eq!(tet, exptet);
        }
    }

    /// Verify that inverting canonical tets causes their signed volume to become negative.
    #[test]
    fn invert_test() {
        use crate::ops::Volume;
        let mut mesh = simple_tetmesh();

        // Before inversion, canonical tets should have positive volume.
        let mut vols = Vec::new();
        for ref tet in mesh.tet_iter() {
            vols.push(tet.volume());
            assert!(tet.signed_volume() > 0.0);
        }

        mesh.invert();

        // After inversion, all tets should have negative volume.
        for (tet, vol) in mesh.tet_iter().zip(vols) {
            assert_eq!(tet.signed_volume(), -vol);
        }
    }

    /// Test that canonicalizing tets fixes all inverted tets but doesn't touch tets that are
    /// already in canonical form (not inverted).
    #[test]
    fn canonicalize_test() {
        use crate::ops::Volume;

        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let indices = vec![[0, 4, 2, 5], [0, 5, 3, 2], [5, 3, 1, 0]];

        let mut mesh = TetMeshExt::new(points.clone(), indices);

        // Two tets are inverted
        let vols: Vec<_> = mesh.tet_iter().map(|t| t.volume()).collect();
        assert!(mesh.tet(0).signed_volume() > 0.0);
        assert!(mesh.tet(1).signed_volume() < 0.0);
        assert!(mesh.tet(2).signed_volume() < 0.0);

        mesh.canonicalize();

        // Canonicalization fixes up all inverted tets
        for (tet, vol) in mesh.tet_iter().zip(vols) {
            assert_eq!(tet.signed_volume(), vol);
        }
    }
}
