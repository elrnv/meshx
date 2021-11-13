//!
//! Tetmesh module. Describes tetrahedron mesh data structures and possible
//! operations on them.
//!
//! The root module defines the most basic tetmesh that other tetmeshes can extend.
//!

mod extended;
mod surface;

pub use extended::*;
pub use surface::*;

use crate::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::vertex_positions::*;
use crate::prim::Tetrahedron;
use crate::utils::slice::*;
use crate::Real;
use math::Matrix3;
use std::slice::{Iter, IterMut};

/// A basic mesh composed of tetrahedra. This mesh is based on vertex positions and a list of
/// vertex indices representing tetrahedra.
#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct TetMesh<T: Real> {
    /// Vertex positions.
    #[intrinsic(VertexPositions)]
    pub vertex_positions: IntrinsicAttribute<[T; 3], VertexIndex>,
    /// Quadruples of indices into `vertices` representing tetrahedra.
    /// The canonical non-inverted tetrahedron is indexed as follows:
    /// ```verbatim
    ///       3
    ///      /|\
    ///     / | \
    ///    /  |  \
    ///  2/...|...\0
    ///   \   |   /
    ///    \  |  /
    ///     \ | /
    ///      \|/
    ///       1
    /// ```
    /// (the dotted line is behind the dashed).
    pub indices: IntrinsicAttribute<[usize; 4], CellIndex>,
    /// Vertex Attributes.
    pub vertex_attributes: AttribDict<VertexIndex>,
    /// Cell Attributes.
    pub cell_attributes: AttribDict<CellIndex>,
    /// Cell vertex Attributes.
    pub cell_vertex_attributes: AttribDict<CellVertexIndex>,
    /// Cell face Attributes.
    pub cell_face_attributes: AttribDict<CellFaceIndex>,
    /// Indirect attribute value cache
    pub attribute_value_cache: AttribValueCache,
}

impl<T: Real> TetMesh<T> {
    /// This constant defines the triangle faces of each tet. The rule is that `i`th face of the
    /// tet is the one opposite to the `i`th vertex. The triangle starts with the smallest index.
    pub const TET_FACES: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]];

    pub fn new(verts: Vec<[T; 3]>, indices: Vec<[usize; 4]>) -> TetMesh<T> {
        TetMesh {
            vertex_positions: IntrinsicAttribute::from_vec(verts),
            indices: IntrinsicAttribute::from_vec(indices),
            vertex_attributes: AttribDict::new(),
            cell_attributes: AttribDict::new(),
            cell_vertex_attributes: AttribDict::new(),
            cell_face_attributes: AttribDict::new(),
            attribute_value_cache: AttribValueCache::default(),
        }
    }

    #[inline]
    pub fn cell_iter(&self) -> Iter<[usize; 4]> {
        self.indices.iter()
    }

    #[cfg(feature = "rayon")]
    #[inline]
    pub fn cell_par_iter(&self) -> rayon::slice::Iter<[usize; 4]> {
        self.indices.par_iter()
    }

    #[inline]
    pub fn cell_iter_mut(&mut self) -> IterMut<[usize; 4]> {
        self.indices.iter_mut()
    }

    #[cfg(feature = "rayon")]
    #[inline]
    pub fn cell_par_iter_mut(&mut self) -> rayon::slice::IterMut<[usize; 4]> {
        self.indices.par_iter_mut()
    }

    /// Cell accessor. These are vertex indices.
    #[inline]
    pub fn cell<CI: Into<CellIndex>>(&self, cidx: CI) -> &[usize; 4] {
        &self.indices[cidx.into()]
    }

    /// Return a slice of individual cells.
    #[inline]
    pub fn cells(&self) -> &[[usize; 4]] {
        self.indices.as_slice()
    }

    /// Tetrahedron iterator.
    #[inline]
    pub fn tet_iter(&self) -> impl Iterator<Item = Tetrahedron<T>> + '_ {
        self.cell_iter().map(move |tet| self.tet_from_indices(tet))
    }

    /// Get a tetrahedron primitive corresponding to the given vertex indices.
    #[inline]
    pub fn tet_from_indices(&self, indices: &[usize; 4]) -> Tetrahedron<T> {
        Tetrahedron::from_indexed_slice(indices, self.vertex_positions.as_slice())
    }

    /// Get a tetrahedron primitive corresponding to the given cell index.
    #[inline]
    pub fn tet<CI: Into<CellIndex>>(&self, cidx: CI) -> Tetrahedron<T> {
        self.tet_from_indices(self.cell(cidx))
    }

    /// Consumes the current mesh to produce a mesh with inverted tetrahedra.
    #[inline]
    pub fn inverted(mut self) -> TetMesh<T> {
        self.invert();
        self
    }

    /// A helper function to invert a single tet of the tetmesh. This keeps the inversion
    /// consistent among all methods that use it (e.g. `invert` and `canonicalize`).
    #[inline]
    fn invert_tet_cell(cell: &mut [usize; 4]) {
        cell.swap(2, 3);
    }

    /// Non consuming version of the `inverted` function which simply modifies the given mesh.
    #[inline]
    pub fn invert(&mut self) {
        for cell in self.indices.iter_mut() {
            Self::invert_tet_cell(cell);
        }

        // TODO: Consider doing reversing lazily using a flag field.
        // Since each vertex has an associated cell vertex attribute, we must remap those
        // as well.
        // Reorder cell vertex attributes
        for (_, attrib) in self.cell_vertex_attributes.iter_mut() {
            let mut data_slice = attrib.data_mut_slice();
            for mut vals in data_slice.chunks_exact_mut(4) {
                vals.swap(2, 3);
            }
        }

        // Reorder cell face attributes
        for (_, attrib) in self.cell_face_attributes.iter_mut() {
            for mut vals in attrib.data_mut_slice().chunks_exact_mut(4) {
                vals.swap(2, 3)
            }
        }
    }

    /// Convert this mesh into canonical form. This function inverts any inverted tetrahedron such
    /// that all tetrahedra are in canonical (non-inverted) form. The canonical form is determined
    /// by the shape matrix determinant of each tetrahedron. Canonical tetrahedra have a positive
    /// shape matrix determinant (see the `meshx::ops::ShapeMatrix` trait and the
    /// `meshx::prim::tetrahedron` module).
    #[inline]
    pub fn canonicalized(mut self) -> TetMesh<T> {
        self.canonicalize();
        self
    }

    /// Convert this mesh into canonical form. This function inverts any inverted tetrahedron such
    /// that all tetrahedra are in canonical (non-inverted) form. This is a non-consuming version
    /// of `canonicalized`.
    #[inline]
    pub fn canonicalize(&mut self) {
        use crate::ops::ShapeMatrix;

        let TetMesh {
            ref vertex_positions,
            ref mut cell_vertex_attributes,
            ref mut cell_face_attributes,
            ref mut indices,
            ..
        } = *self;

        // Record what was inverted.
        let mut inverted = vec![false; indices.len()];

        for (cell, inv) in indices.iter_mut().zip(inverted.iter_mut()) {
            let tet = Tetrahedron::from_indexed_slice(cell, vertex_positions.as_slice());
            if Matrix3::from(tet.shape_matrix()).determinant() < T::zero() {
                Self::invert_tet_cell(cell);
                *inv = true;
            }
        }

        // TODO: Consider doing reversing lazily using a flag field.
        // Since each vertex has an associated cell vertex attribute, we must remap those
        // as well.
        // Reorder cell vertex attributes
        for (_, attrib) in cell_vertex_attributes.iter_mut() {
            for (mut vals, _) in attrib
                .data_mut_slice()
                .chunks_exact_mut(4)
                .zip(inverted.iter())
                .filter(|(_, &inv)| inv)
            {
                vals.swap(2, 3);
            }
        }

        // Reorder cell face attributes
        for (_, attrib) in cell_face_attributes.iter_mut() {
            for (mut vals, _) in attrib
                .data_mut_slice()
                .chunks_exact_mut(4)
                .zip(inverted.iter())
                .filter(|(_, &inv)| inv)
            {
                vals.swap(2, 3);
            }
        }
    }

    /// Sort vertices by the given key values, and return the reulting order (permutation).
    ///
    /// This function assumes we have at least one vertex.
    // TODO: This function is identical to the one used in uniform_poly_mesh.
    // We need to figure out how to remove this code duplication whether it is through traits or otherwise.
    pub fn sort_vertices_by_key<K, F>(&mut self, mut f: F) -> Vec<usize>
    where
        F: FnMut(usize) -> K,
        K: Ord,
    {
        // Early exit.
        if self.num_vertices() == 0 {
            return Vec::new();
        }

        let num = self.attrib_size::<VertexIndex>();
        debug_assert!(num > 0);

        // Original vertex indices.
        let mut order: Vec<usize> = (0..num).collect();

        // Sort vertex indices by the given key.
        order.sort_by_key(|k| f(*k));

        // Now sort all mesh data according to the sorting given by order.

        let TetMesh {
            ref mut vertex_positions,
            ref mut indices,
            ref mut vertex_attributes,
            .. // cell and cell_{vertex,face} attributes are unchanged
        } = *self;

        let mut seen = vec![false; vertex_positions.len()];

        // Apply the order permutation to vertex_positions in place
        apply_permutation_with_seen(&order, vertex_positions.as_mut_slice(), &mut seen);

        // Apply permutation to each vertex attribute
        for (_, attrib) in vertex_attributes.iter_mut() {
            let mut data_slice = attrib.data_mut_slice();

            // Clear seen
            seen.iter_mut().for_each(|b| *b = false);

            apply_permutation_with_seen(&order, &mut data_slice, &mut seen);
        }

        // Build a reverse mapping for convenience.
        let mut new_indices = vec![0; order.len()];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            new_indices[old_idx] = new_idx;
        }

        // Remap cell vertices.
        for cell in indices.iter_mut() {
            for vtx_idx in cell.iter_mut() {
                *vtx_idx = new_indices[*vtx_idx];
            }
        }

        order
    }
}

impl<T: Real> Default for TetMesh<T> {
    /// Produce an empty `TetMesh`.
    ///
    /// This is not particularly useful on its own, however it can be
    /// used as a null case for various mesh algorithms.
    fn default() -> Self {
        TetMesh::new(vec![], vec![])
    }
}

/**
 * Define `TetMesh` topology
 */

impl<T: Real> NumVertices for TetMesh<T> {
    #[inline]
    fn num_vertices(&self) -> usize {
        self.vertex_positions.len()
    }
}

impl<T: Real> NumCells for TetMesh<T> {
    #[inline]
    fn num_cells(&self) -> usize {
        self.indices.len()
    }
}

impl<T: Real> CellVertex for TetMesh<T> {
    #[inline]
    fn vertex<CVI>(&self, cv_idx: CVI) -> VertexIndex
    where
        CVI: Copy + Into<CellVertexIndex>,
    {
        let cv_idx = usize::from(cv_idx.into());
        debug_assert!(cv_idx < self.num_cell_vertices());
        self.indices[cv_idx / 4][cv_idx % 4].into()
    }

    #[inline]
    fn cell_vertex<CI>(&self, cidx: CI, which: usize) -> Option<CellVertexIndex>
    where
        CI: Copy + Into<CellIndex>,
    {
        if which >= 4 {
            None
        } else {
            Some((4 * usize::from(cidx.into()) + which).into())
        }
    }

    #[inline]
    fn num_cell_vertices(&self) -> usize {
        self.indices.len() * 4
    }

    #[inline]
    fn num_vertices_at_cell<CI>(&self, _: CI) -> usize
    where
        CI: Copy + Into<CellIndex>,
    {
        4
    }
}

impl<T: Real> CellFace for TetMesh<T> {
    #[inline]
    fn face<CFI>(&self, cf_idx: CFI) -> FaceIndex
    where
        CFI: Copy + Into<CellFaceIndex>,
    {
        // Faces are indexed to be opposite to the corresponding vertices.
        let cf_idx = usize::from(cf_idx.into());
        debug_assert!(cf_idx < self.num_cell_faces());
        self.indices[cf_idx / 4][cf_idx % 4].into()
    }

    #[inline]
    fn cell_face<CI>(&self, cidx: CI, which: usize) -> Option<CellFaceIndex>
    where
        CI: Copy + Into<CellIndex>,
    {
        // Faces are indexed to be opposite to the corresponding vertices.
        if which >= 4 {
            None
        } else {
            Some((4 * usize::from(cidx.into()) + which).into())
        }
    }

    #[inline]
    fn num_cell_faces(&self) -> usize {
        self.indices.len() * 4
    }

    #[inline]
    fn num_faces_at_cell<CI>(&self, _: CI) -> usize
    where
        CI: Copy + Into<CellIndex>,
    {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;

    fn simple_tetmesh() -> TetMesh<f64> {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let indices = vec![[0, 4, 2, 5], [0, 5, 2, 3], [5, 3, 0, 1]];

        TetMesh::new(points, indices)
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

        let mut mesh = TetMesh::new(points.clone(), indices);

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
