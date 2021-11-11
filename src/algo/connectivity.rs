/*!
 * This module defines routines for dealing with meshes composed of multiple connected components.
 */

use crate::index::*;
use crate::mesh::topology::*;
use crate::mesh::{attrib::*, PolyMesh, TetMeshExt, TriMesh};
use crate::Real;

/// A trait defining the primary method for determining connectivity in a mesh.
///
/// `Src` specifies the element index for which to determine connectivity.
/// `Via` specifies a secondary element index which identifies elements through which the
/// connectivity is determined.
pub trait Connectivity<Src: ElementIndex<usize>, Via: ElementIndex<usize>> {
    /// Additional topology that may aid in computing connectivity.
    ///
    /// This is computed with `precompute_topo` and used in `push_neighbours`.
    type Topo: Default;

    /// Precompute additional topology information prior to determining connectivity.
    ///
    /// An optional function that allows implementers to precompute topology information to help
    /// with the implementation of `push_neighbours` when the mesh doesn't already support a
    /// certain type of topology.
    fn precompute_topo(&self) -> Self::Topo {
        Default::default()
    }

    /// Get the number of elements which are considered for connectivity
    ///
    /// E.g. triangles in triangle meshes or tets in a tetmesh.
    fn num_elements(&self) -> usize;

    /// Push all neighbours of the element at the given `index` to the given `stack`.
    ///
    /// Additionally, topology data `topo` computed using `precomute_topo` and an
    /// optional `attribute` on the `Src` topology is provided to help determine connectivity.
    fn push_neighbours<T: Default + PartialEq>(
        &self,
        index: Src,
        stack: &mut Vec<Src>,
        topo: &Self::Topo,
        attribute: Option<&[T]>,
    );

    /// Determine the connectivity of a set of meshes.
    ///
    /// Return a `Vec` with the size of `self.num_elements()` indicating a unique ID of the
    /// connected component each element belongs to. For instance, if two triangles in a triangle
    /// mesh blong to the same connected component, they will have the same ID. Also return the
    /// total number of components generated.
    fn connectivity(&self) -> (Vec<usize>, usize) {
        self.connectivity_via_attrib_fn::<(), _>(|| None)
    }

    /// Determine the connectivity of a set of meshes.
    ///
    /// Return a `Vec` with the size of `self.num_elements()` indicating a unique ID of the
    /// connected component each element belongs to. For instance, if two triangles in a triangle
    /// mesh blong to the same connected component, they will have the same ID. Also return the
    /// total number of components generated.
    ///
    /// This is a more general version of `connectivity` that accepts an optional attribute of type
    /// `T` on the `Src` topology to determine connectivity.
    fn connectivity_via_attrib<T>(&self, attrib: Option<&str>) -> (Vec<usize>, usize)
    where
        Self: Attrib,
        Src: AttribIndex<Self>,
        T: Default + PartialEq + 'static,
    {
        self.connectivity_via_attrib_fn::<T, _>(|| {
            attrib.and_then(|name| self.attrib_as_slice::<T, Src>(name).ok())
        })
    }

    /// Determine the connectivity of a set of meshes.
    ///
    /// Return a `Vec` with the size of `self.num_elements()` indicating a unique ID of the
    /// connected component each element belongs to. For instance, if two triangles in a triangle
    /// mesh blong to the same connected component, they will have the same ID. Also return the
    /// total number of components generated.
    ///
    /// This is the most general version of `connectivity` that accepts a function that providees
    /// attribute data of type `T` on the `Src` topology to determine connectivity.
    /// Note that the provided slice must have the same length as the number of `Src` indices.
    fn connectivity_via_attrib_fn<'a, T, F>(&self, f: F) -> (Vec<usize>, usize)
    where
        T: Default + PartialEq + 'a,
        F: FnOnce() -> Option<&'a [T]>,
    {
        // The ID of the current connected component.
        let mut cur_component_id = 0;

        let mut stack: Vec<Src> = Vec::new();

        let num_element_indices = self.num_elements();

        // The vector of component ids (one for each element).
        let mut component_ids = vec![Index::INVALID; num_element_indices];

        let data = self.precompute_topo();

        let attrib_data = f();

        // Perform a depth first search through the mesh topology to determine connected components.
        for elem in 0..num_element_indices {
            if component_ids[elem].is_valid() {
                continue;
            }

            // elem is the representative element for the current connected component.
            stack.push(elem.into());

            while let Some(elem) = stack.pop() {
                let elem_idx: usize = elem.into();
                if !component_ids[elem_idx].is_valid() {
                    // Process element if it hasn't been seen before.
                    component_ids[elem_idx] = cur_component_id.into();
                    self.push_neighbours(elem, &mut stack, &data, attrib_data);
                }
            }

            // Finished with the current component, no more connected elements.
            cur_component_id += 1;
        }

        // Ensure that all ids are valid before we reinterpret the vector.
        debug_assert!(component_ids.iter().all(|&x| x.is_valid()));
        (bytemuck::cast_vec(component_ids), cur_component_id)
    }
}

// The default connectivity for standard meshes (PolyMesh, TriMesh, QuadMesh and TetMesh) is taken
// to be vertex connectivity. This means that two vertices are in the same connected component iff
// there is a path between them along a set of edges. Other types of connectivity may be
// implemented, but this type allows meshes to be split and rejoined without changing the number of
// total vertices and possibly their order.

/// Implement vertex connectivity for cell based meshes (e.g. TetMesh).
impl<M: VertexCell + CellVertex + NumVertices> Connectivity<VertexIndex, CellIndex> for M {
    type Topo = ();
    fn num_elements(&self) -> usize {
        self.num_vertices()
    }

    fn push_neighbours<T: Default + PartialEq>(
        &self,
        index: VertexIndex,
        stack: &mut Vec<VertexIndex>,
        _: &(),
        _: Option<&[T]>,
    ) {
        for which_cell in 0..self.num_cells_at_vertex(index) {
            let cell = self.vertex_to_cell(index, which_cell).unwrap();
            for which_vtx in 0..self.num_vertices_at_cell(cell) {
                let neigh_vtx = self.cell_to_vertex(cell, which_vtx).unwrap();
                if neigh_vtx != index {
                    stack.push(neigh_vtx);
                }
            }
        }
    }
}

/// Implement vertex connectivity for face based meshes (e.g. PolyMesh, TriMesh and QuadMesh).
impl<M: FaceVertex + NumVertices + NumFaces> Connectivity<VertexIndex, FaceIndex> for M {
    type Topo = (Vec<usize>, Vec<usize>);
    fn precompute_topo(&self) -> Self::Topo {
        self.reverse_topo()
    }
    fn num_elements(&self) -> usize {
        self.num_vertices()
    }

    fn push_neighbours<T: Default + PartialEq>(
        &self,
        index: VertexIndex,
        stack: &mut Vec<VertexIndex>,
        topo: &Self::Topo,
        _: Option<&[T]>,
    ) {
        let (face_indices, face_offsets) = topo;
        let idx = usize::from(index);
        for face in (face_offsets[idx]..face_offsets[idx + 1]).map(|i| face_indices[i]) {
            for which_vtx in 0..self.num_vertices_at_face(face) {
                let neigh_vtx = self.face_to_vertex(face, which_vtx).unwrap();
                if neigh_vtx != index {
                    stack.push(neigh_vtx);
                }
            }
        }
    }
}

/// Implement face vertex connectivity for face based meshes (e.g. PolyMesh, TriMesh and QuadMesh).
///
/// This can be useful for splitting meshes based on texture coordinates, so that they can be
/// exported in formats that don't support additional face-vertex topologies like glTF.
impl<M: FaceVertex + NumFaces + NumVertices> Connectivity<FaceVertexIndex, VertexIndex> for M {
    type Topo = (Vec<usize>, Vec<usize>);
    fn precompute_topo(&self) -> Self::Topo {
        self.reverse_source_topo() // vertex -> (face->vertex) topo
    }

    fn num_elements(&self) -> usize {
        self.num_face_vertices()
    }

    fn push_neighbours<T: Default + PartialEq>(
        &self,
        index: FaceVertexIndex,
        stack: &mut Vec<FaceVertexIndex>,
        topo: &Self::Topo,
        attrib: Option<&[T]>,
    ) {
        // For each vertex, topo contains a set of face-vertex indices.
        let (fv_indices, fv_offsets) = topo;

        let vtx_idx = usize::from(self.vertex(index));

        // Push all neighbours of a face-vertex based on the value of the given attribute.
        let idx = usize::from(index);

        // Attribute value of the primary face-vertex given by `index`.
        let def_val = T::default();
        let primary_attrib_val = attrib.map(|a| &a[idx]).unwrap_or_else(|| &def_val);

        for face_vertex in (fv_offsets[vtx_idx]..fv_offsets[vtx_idx + 1]).map(|i| fv_indices[i]) {
            let neigh_attrib_val = attrib.map(|a| &a[face_vertex]).unwrap_or_else(|| &def_val);
            if primary_attrib_val == neigh_attrib_val {
                stack.push(face_vertex.into());
            }
        }
    }
}

// Implement default connectivity shorthands on meshes to avoid having to specify index types for
// the Connectivity trait.

impl<T: Real> TriMesh<T> {
    pub fn vertex_connectivity(&self) -> (Vec<usize>, usize) {
        Connectivity::<VertexIndex, FaceIndex>::connectivity(self)
    }
}

impl<T: Real> PolyMesh<T> {
    pub fn vertex_connectivity(&self) -> (Vec<usize>, usize) {
        Connectivity::<VertexIndex, FaceIndex>::connectivity(self)
    }
}

impl<T: Real> TetMeshExt<T> {
    pub fn vertex_connectivity(&self) -> (Vec<usize>, usize) {
        Connectivity::<VertexIndex, CellIndex>::connectivity(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{TetMeshExt, TriMesh};

    #[test]
    fn tetmesh_connectivity() {
        // The vertex positions are actually unimportant here.
        let verts = vec![[0.0; 3]; 12];

        // One connected component consisting of two tets connected at a face, and another
        // consisting of two tets connected at a single vertex.
        let indices = vec![[0, 1, 2, 3], [1, 2, 3, 4], [5, 6, 7, 8], [8, 9, 10, 11]];

        let tetmesh = TetMeshExt::new(verts, indices);

        assert_eq!(
            tetmesh.connectivity(),
            (vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 2)
        );
    }

    #[test]
    fn trimesh_connectivity() {
        // The vertex positions are actually unimportant here.
        let verts = vec![[0.0; 3]; 7];

        // One component with two connected triangles at an edge and another with a single triangle
        // that is disconnected
        let indices = vec![[0, 1, 2], [1, 2, 3], [4, 5, 6]];

        let trimesh = TriMesh::new(verts, indices);

        assert_eq!(
            trimesh.vertex_connectivity(),
            (vec![0, 0, 0, 0, 1, 1, 1], 2)
        );
    }
}
