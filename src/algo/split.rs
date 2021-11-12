/*!
 * Traits and algorithms for splitting generic meshes.
 */

use super::connectivity::*;
use crate::index::*;
use crate::mesh::attrib::AttribValueCache;
use crate::mesh::topology::*;
use crate::mesh::{attrib::*, PolyMesh, QuadMesh, TetMesh, TetMeshExt, TriMesh};
use crate::Real;

/// Helper to split attributes based on the given connectivity info.
fn split_attributes<A: Clone, I: Into<Option<usize>>>(
    src_dict: &AttribDict<A>,
    num_components: usize,
    connectivity: impl Iterator<Item = I> + Clone,
    caches: &mut [AttribValueCache],
) -> Vec<AttribDict<A>> {
    split_attributes_with(src_dict, num_components, |attrib, num_components| {
        let mut new_attribs = vec![attrib.duplicate_empty(); num_components];
        // Get an iterator of typeless values for this attribute.
        match &attrib.data {
            AttributeData::Direct(d) => {
                connectivity
                    .clone()
                    .zip(d.data_ref().iter())
                    .filter_map(|(comp_id, val_ref)| {
                        comp_id.into().map(|comp_id| (comp_id, val_ref))
                    })
                    .for_each(|(valid_idx, val_ref)| {
                        new_attribs[valid_idx]
                            .data
                            .direct_data_mut()
                            .unwrap()
                            .push_cloned(val_ref)
                            .unwrap();
                    });
            }
            AttributeData::Indirect(i) => {
                for (valid_comp_id, val_ref) in
                    connectivity.clone().zip(i.data_ref().iter()).filter_map(
                        |(comp_id, val_ref)| comp_id.into().map(|comp_id| (comp_id, val_ref)),
                    )
                {
                    new_attribs[valid_comp_id]
                        .data
                        .indirect_data_mut()
                        .unwrap()
                        .push_cloned(val_ref, &mut caches[valid_comp_id])
                        .unwrap();
                }
            }
        }

        new_attribs
    })
}

/// Helper to split attributes using a given closure to transfer data from each source attribute to
/// the destination collection of individual empty component attributes.
fn split_attributes_with<A: Clone>(
    src_dict: &AttribDict<A>,
    num_components: usize,
    mut split_attribute: impl FnMut(&Attribute<A>, usize) -> Vec<Attribute<A>>,
) -> Vec<AttribDict<A>> {
    let mut comp_attributes = vec![AttribDict::new(); num_components];
    for (name, attrib) in src_dict.iter() {
        // Split the given attribute into one attribute per component.
        let new_attribs = split_attribute(attrib, num_components);
        assert_eq!(new_attribs.len(), num_components);

        // Save the new attributes to their corresponding attribute dictionaries.
        for (attrib_dict, new_attrib) in comp_attributes.iter_mut().zip(new_attribs.into_iter()) {
            attrib_dict.insert(name.to_string(), new_attrib);
        }
    }
    comp_attributes
}

/// Split the object at the `Src` topology (e.g. vertices) into multiple objects of the same type.
pub trait Split<Src>
where
    Self: Sized,
{
    fn split(self, partition: &[usize], num_parts: usize) -> Vec<Self>;
}

// TODO: Refactor the below implementations by extracting common patterns. This can also be
// combined with implementations conversions between meshes.

impl<T: Real> Split<VertexIndex> for TetMesh<T> {
    #[inline]
    fn split(self, partition: &[usize], num_parts: usize) -> Vec<Self> {
        self.split_by_vertex_partition(partition, num_parts).0
    }
}

impl<T: Real> TetMesh<T> {
    /// Split the mesh by the given partition.
    ///
    /// Returns a vector of tetmeshes and the mapping from old cell index to new cell index.
    fn split_by_vertex_partition(
        self,
        vertex_partition: &[usize],
        num_parts: usize,
    ) -> (Vec<Self>, Vec<Index>) {
        // Fast path, when everything is connected.
        if num_parts == 1 {
            return (vec![self], vec![]);
        }

        // Deconstruct the original mesh.
        let TetMesh {
            vertex_positions,
            indices,
            vertex_attributes,
            cell_attributes,
            cell_vertex_attributes,
            cell_face_attributes,
            ..
        } = self;

        // Record where the new vertices end up (just the index within their respective
        // components). The component ids themselves are recorded separately.
        let mut new_vertex_indices = vec![Index::INVALID; vertex_positions.len()];

        // Transfer vertex positions
        let mut comp_vertex_positions = vec![Vec::new(); num_parts];
        for (vidx, &comp_id) in vertex_partition.iter().enumerate() {
            new_vertex_indices[vidx] = comp_vertex_positions[comp_id].len().into();
            comp_vertex_positions[comp_id].push(vertex_positions[vidx]);
        }

        // Validate that all vertices have been properly mapped.
        debug_assert!(new_vertex_indices.iter().all(|&idx| idx.is_valid()));
        let new_vertex_index_slice: &[usize] = bytemuck::cast_slice(new_vertex_indices.as_slice());

        // Record cell connectivity. Note that if cells have vertices on different components,
        // they will be ignored in the output and their connectivity will be "invalid".
        let mut cell_connectivity = vec![Index::INVALID; indices.len()];
        let mut new_cell_indices = vec![Index::INVALID; indices.len()];

        // Transfer cells
        let mut comp_vertex_indices = vec![Vec::new(); num_parts];
        for (cell_idx, &cell) in indices.iter().enumerate() {
            let comp_id = vertex_partition[cell[0]];
            if cell.iter().all(|&i| vertex_partition[i] == comp_id) {
                let new_cell = [
                    new_vertex_index_slice[cell[0]],
                    new_vertex_index_slice[cell[1]],
                    new_vertex_index_slice[cell[2]],
                    new_vertex_index_slice[cell[3]],
                ];
                new_cell_indices[cell_idx] = comp_vertex_indices[comp_id].len().into();
                comp_vertex_indices[comp_id].push(new_cell);
                cell_connectivity[cell_idx] = Index::from(comp_id);
            }
        }

        // Initialize attribute value caches for indirect attributes.
        let mut comp_attribute_value_caches = vec![AttribValueCache::default(); num_parts];

        // Transfer vertex attributes
        let comp_vertex_attributes = split_attributes(
            &vertex_attributes,
            num_parts,
            vertex_partition.iter().cloned(),
            &mut comp_attribute_value_caches,
        );

        // Transfer cell attributes
        let comp_cell_attributes = split_attributes(
            &cell_attributes,
            num_parts,
            cell_connectivity.iter().cloned(),
            &mut comp_attribute_value_caches,
        );

        // Transfer cell vertex attributes
        let comp_cell_vertex_attributes = split_attributes(
            &cell_vertex_attributes,
            num_parts,
            cell_connectivity
                .iter()
                .flat_map(|c| std::iter::repeat(c).take(4).cloned()),
            &mut comp_attribute_value_caches,
        );

        // Transfer cell face attributes
        let comp_cell_face_attributes = split_attributes(
            &cell_face_attributes,
            num_parts,
            cell_connectivity
                .iter()
                .flat_map(|c| std::iter::repeat(c).take(4).cloned()),
            &mut comp_attribute_value_caches,
        );

        // Generate a Vec of meshes.
        (
            comp_vertex_positions
                .into_iter()
                .zip(comp_vertex_indices.into_iter())
                .zip(comp_vertex_attributes.into_iter())
                .zip(comp_cell_attributes.into_iter())
                .zip(comp_cell_vertex_attributes.into_iter())
                .zip(comp_cell_face_attributes.into_iter())
                .zip(comp_attribute_value_caches.into_iter())
                .map(|((((((vp, vi), va), ca), cva), cfa), avc)| TetMesh {
                    vertex_positions: vp.into(),
                    indices: vi.into(),
                    vertex_attributes: va,
                    cell_attributes: ca,
                    cell_vertex_attributes: cva,
                    cell_face_attributes: cfa,
                    attribute_value_cache: avc,
                })
                .collect(),
            new_cell_indices,
        )
    }
}

macro_rules! impl_split_for_uniform_mesh {
    ($mesh_type:ident; $n:expr; $($ns:expr),*) => {
        impl<T: Real> Split<VertexIndex> for $mesh_type<T> {
            #[inline]
            fn split(self, partition: &[usize], num_parts: usize) -> Vec<Self> {
                self.split_by_vertex_partition(partition, num_parts).0
            }
        }

        impl<T: Real> $mesh_type<T> {
            /// Split the mesh by the given partition.
            ///
            /// Returns a vector of meshes and the mapping from old cell index to new cell index.
            fn split_by_vertex_partition(
                self,
                vertex_partition: &[usize],
                num_parts: usize,
            ) -> (Vec<Self>, Vec<Index>) {
                // Fast path, when everything is connected.
                if num_parts == 1 {
                    return (vec![self], vec![]);
                }

                // Deconstruct the original mesh.
                let $mesh_type {
                    vertex_positions,
                    indices,
                    vertex_attributes,
                    face_attributes,
                    face_vertex_attributes,
                    face_edge_attributes,
                    ..
                } = self;

                // Record where the new vertices end up (just the index within their respective
                // components). The component ids themselves are recorded separately.
                let mut new_vertex_indices = vec![Index::INVALID; vertex_positions.len()];

                // Transfer vertex positions
                let mut comp_vertex_positions = vec![Vec::new(); num_parts];
                for (vidx, &comp_id) in vertex_partition.iter().enumerate() {
                    new_vertex_indices[vidx] = comp_vertex_positions[comp_id].len().into();
                    comp_vertex_positions[comp_id].push(vertex_positions[vidx]);
                }

                // Validate that all vertices have been properly mapped.
                debug_assert!(new_vertex_indices.iter().all(|&idx| idx.is_valid()));
                let new_vertex_index_slice: &[usize] = bytemuck::cast_slice(new_vertex_indices.as_slice());

                // Record face connectivity. Note that if faces have vertices on different components,
                // they will be ignored in the output and their connectivity will be "invalid".
                let mut face_connectivity = vec![Index::INVALID; indices.len()];
                let mut new_face_indices = vec![Index::INVALID; indices.len()];

                // Transfer faces
                let mut comp_vertex_indices = vec![Vec::new(); num_parts];
                for (&face, (face_comp_id, new_face_idx)) in indices.iter().zip(face_connectivity.iter_mut().zip(new_face_indices.iter_mut())) {
                    let comp_id = vertex_partition[face[0]];
                    if face.iter().all(|&i| vertex_partition[i] == comp_id) {
                        let new_face = [
                            $(
                                new_vertex_index_slice[face[$ns]],
                            )*
                        ];
                        *new_face_idx = comp_vertex_indices[comp_id].len().into();
                        comp_vertex_indices[comp_id].push(new_face);
                        *face_comp_id = Index::from(comp_id);
                    }
                }

                // Initialize attribute value caches for indirect attributes.
                let mut comp_attribute_value_caches = vec![AttribValueCache::default(); num_parts];

                // Transfer vertex attributes
                let comp_vertex_attributes = split_attributes(
                    &vertex_attributes,
                    num_parts,
                    vertex_partition.iter().cloned(),
                    &mut comp_attribute_value_caches,
                );

                // Transfer face attributes
                let comp_face_attributes = split_attributes(
                    &face_attributes,
                    num_parts,
                    face_connectivity.iter().cloned(),
                    &mut comp_attribute_value_caches,
                );

                // Transfer face vertex attributes
                let comp_face_vertex_attributes = split_attributes(
                    &face_vertex_attributes,
                    num_parts,
                    face_connectivity
                        .iter()
                        .flat_map(|c| std::iter::repeat(c).take($n).cloned()),
                    &mut comp_attribute_value_caches,
                );

                // Transfer face edge attributes
                let comp_face_edge_attributes = split_attributes(
                    &face_edge_attributes,
                    num_parts,
                    face_connectivity
                        .iter()
                        .flat_map(|c| std::iter::repeat(c).take($n).cloned()),
                    &mut comp_attribute_value_caches,
                );

                // Generate a Vec of meshes.
                (
                    comp_vertex_positions
                        .into_iter()
                        .zip(comp_vertex_indices.into_iter())
                        .zip(comp_vertex_attributes.into_iter())
                        .zip(comp_face_attributes.into_iter())
                        .zip(comp_face_vertex_attributes.into_iter())
                        .zip(comp_face_edge_attributes.into_iter())
                        .zip(comp_attribute_value_caches.into_iter())
                        .map(|((((((vp, vi), va), ca), cva), cfa), avc)| Self {
                            vertex_positions: vp.into(),
                            indices: vi.into(),
                            vertex_attributes: va,
                            face_attributes: ca,
                            face_vertex_attributes: cva,
                            face_edge_attributes: cfa,
                            attribute_value_cache: avc,
                        })
                        .collect(),
                    new_face_indices,
                )
            }
        }
    }
}

impl_split_for_uniform_mesh!(TriMesh; 3; 0, 1, 2);
impl_split_for_uniform_mesh!(QuadMesh; 4; 0, 1, 2, 3);

impl<T: Real> Split<VertexIndex> for TetMeshExt<T> {
    fn split(self, vertex_partition: &[usize], num_parts: usize) -> Vec<Self> {
        // Fast path, when everything is connected.
        if num_parts == 1 {
            return vec![self];
        }

        // Deconstruct the original mesh.
        let TetMeshExt {
            tetmesh,
            cell_offsets,
            cell_indices,
            vertex_cell_attributes,
            ..
        } = self;

        let (mut comp_tetmesh, new_cell_indices) =
            tetmesh.split_by_vertex_partition(vertex_partition, num_parts);

        // Transfer vertex to cell topology
        let mut comp_cell_indices = vec![Vec::new(); num_parts];
        let mut comp_cell_offsets = vec![vec![0]; num_parts];
        for (vidx, &comp_id) in vertex_partition.iter().enumerate() {
            let off = cell_offsets[vidx];
            for &cell_idx in &cell_indices[off..cell_offsets[vidx + 1]] {
                new_cell_indices[cell_idx]
                    .if_valid(|new_cidx| comp_cell_indices[comp_id].push(new_cidx));
            }
            comp_cell_offsets[comp_id].push(comp_cell_indices[comp_id].len());
        }

        // Transfer vertex-cell attributes

        // A helper closure to map a given attribute value to the corresponding component id if any
        // `i` is the index of the original attribute value.
        let transfer_comp_id = |vtx_idx: &mut usize, i| -> Option<usize> {
            // Determine the vertex index here using offsets
            let off = cell_offsets[*vtx_idx + 1];
            if i == off {
                *vtx_idx += 1;
            }
            let comp_id = vertex_partition[*vtx_idx];
            let cell_idx = cell_indices[i];

            // Add value for this vertex to the appropriate component data.
            let idx: Index = new_cell_indices[cell_idx];
            idx.map(|_| comp_id).into()
        };

        let comp_vertex_cell_attributes =
            split_attributes_with(&vertex_cell_attributes, num_parts, |attrib, num_parts| {
                let mut new_attribs = vec![attrib.duplicate_empty(); num_parts];

                let mut vtx_idx = 0;

                match &attrib.data {
                    AttributeData::Direct(direct) => {
                        for (i, val_ref) in direct.data_ref().iter().enumerate() {
                            if let Some(comp_id) = transfer_comp_id(&mut vtx_idx, i) {
                                new_attribs[comp_id]
                                    .data
                                    .direct_data_mut()
                                    .unwrap()
                                    .push_cloned(val_ref)
                                    .unwrap();
                            }
                        }
                    }
                    AttributeData::Indirect(indirect) => {
                        for (i, val_ref) in indirect.data_ref().iter().enumerate() {
                            if let Some(comp_id) = transfer_comp_id(&mut vtx_idx, i) {
                                new_attribs[comp_id]
                                    .data
                                    .indirect_data_mut()
                                    .unwrap()
                                    .push_cloned(
                                        val_ref,
                                        &mut comp_tetmesh[comp_id].attribute_value_cache,
                                    )
                                    .unwrap();
                            }
                        }
                    }
                };

                new_attribs
            });

        // Generate a Vec of meshes.
        comp_tetmesh
            .into_iter()
            .zip(comp_cell_indices.into_iter())
            .zip(comp_cell_offsets.into_iter())
            .zip(comp_vertex_cell_attributes.into_iter())
            .map(|(((tm, ci), co), vca)| TetMeshExt {
                tetmesh: tm,
                cell_indices: ci,
                cell_offsets: co,
                vertex_cell_attributes: vca,
            })
            .collect()
    }
}

impl<T: Real> Split<VertexIndex> for PolyMesh<T> {
    fn split(self, vertex_partition: &[usize], num_parts: usize) -> Vec<Self> {
        // Fast path, when everything is connected.
        if num_parts == 1 {
            return vec![self];
        }

        // Record where the new vertices end up (just the index within their respective
        // components). The component ids themselves are recorded separately.
        let mut new_vertex_indices = vec![Index::INVALID; self.vertex_positions.len()];

        // Transfer vertex positions
        let mut comp_vertex_positions = vec![Vec::new(); num_parts];
        for (vidx, &comp_id) in vertex_partition.iter().enumerate() {
            new_vertex_indices[vidx] = comp_vertex_positions[comp_id].len().into();
            comp_vertex_positions[comp_id].push(self.vertex_positions[vidx]);
        }

        // Validate that all vertices have been properly mapped.
        debug_assert!(new_vertex_indices.iter().all(|&idx| idx.is_valid()));
        let new_vertex_index_slice: &[usize] = bytemuck::cast_slice(new_vertex_indices.as_slice());

        // Record face connectivity. Note that if faces have vertices on different components,
        // they will be ignored in the output and their connectivity will be "invalid".
        let mut face_connectivity = vec![Index::INVALID; self.num_faces()];

        // Transfer faces
        let mut comp_indices = vec![Vec::new(); num_parts];
        let mut comp_offsets = vec![vec![0]; num_parts];
        for (face, face_comp_id) in self.face_iter().zip(face_connectivity.iter_mut()) {
            let comp_id = vertex_partition[face[0]];
            if face.iter().all(|&i| vertex_partition[i] == comp_id) {
                let new_face_vtx_iter = face.iter().map(|&vi| new_vertex_index_slice[vi]);
                comp_indices[comp_id].extend(new_face_vtx_iter);
                comp_offsets[comp_id].push(comp_indices[comp_id].len());
                *face_comp_id = Index::from(comp_id);
            }
        }

        // Initialize attribute value caches for indirect attributes.
        let mut comp_attribute_value_caches =
            vec![AttribValueCache::with_hasher(Default::default()); num_parts];

        // Transfer vertex attributes
        let comp_vertex_attributes = split_attributes(
            &self.vertex_attributes,
            num_parts,
            vertex_partition.iter().cloned(),
            &mut comp_attribute_value_caches,
        );

        // Transfer face attributes
        let comp_face_attributes = split_attributes(
            &self.face_attributes,
            num_parts,
            face_connectivity.iter().cloned(),
            &mut comp_attribute_value_caches,
        );

        // Transfer face vertex attributes
        let comp_face_vertex_attributes = split_attributes(
            &self.face_vertex_attributes,
            num_parts,
            face_connectivity.iter().enumerate().flat_map(|(fi, c)| {
                std::iter::repeat(c)
                    .take(self.num_vertices_at_face(fi))
                    .cloned()
            }),
            &mut comp_attribute_value_caches,
        );

        // Transfer face edge attributes
        let comp_face_edge_attributes = split_attributes(
            &self.face_edge_attributes,
            num_parts,
            face_connectivity.iter().enumerate().flat_map(|(fi, c)| {
                std::iter::repeat(c)
                    .take(self.num_edges_at_face(fi))
                    .cloned()
            }),
            &mut comp_attribute_value_caches,
        );

        // Generate a Vec of meshes.
        comp_vertex_positions
            .into_iter()
            .zip(comp_indices.into_iter())
            .zip(comp_offsets.into_iter())
            .zip(comp_vertex_attributes.into_iter())
            .zip(comp_face_attributes.into_iter())
            .zip(comp_face_vertex_attributes.into_iter())
            .zip(comp_face_edge_attributes.into_iter())
            .zip(comp_attribute_value_caches.into_iter())
            .map(|(((((((vp, i), o), va), fa), fva), fea), avc)| PolyMesh {
                vertex_positions: vp.into(),
                indices: i,
                offsets: o,
                vertex_attributes: va,
                face_attributes: fa,
                face_vertex_attributes: fva,
                face_edge_attributes: fea,
                attribute_value_cache: avc,
            })
            .collect()
    }
}

pub trait SplitIntoConnectedComponents<Src, Via>
where
    Src: ElementIndex<usize>,
    Via: ElementIndex<usize>,
    Self: Sized,
{
    fn split_into_connected_components(self) -> Vec<Self>;
}

impl<T: Real> SplitIntoConnectedComponents<VertexIndex, CellIndex> for TetMesh<T> {
    fn split_into_connected_components(self) -> Vec<Self> {
        let tetmesh_ext = TetMeshExt::from(self);
        tetmesh_ext
            .split_into_connected_components()
            .into_iter()
            .map(TetMesh::from)
            .collect()
    }
}

impl<T: Real> SplitIntoConnectedComponents<VertexIndex, CellIndex> for TetMeshExt<T> {
    fn split_into_connected_components(self) -> Vec<Self> {
        let (vertex_connectivity, num_components) = self.connectivity();
        self.split(&vertex_connectivity, num_components)
    }
}

impl<T: Real> SplitIntoConnectedComponents<VertexIndex, FaceIndex> for PolyMesh<T> {
    fn split_into_connected_components(self) -> Vec<Self> {
        // First we partition the vertices.
        let (vertex_connectivity, num_components) =
            Connectivity::<VertexIndex, FaceIndex>::connectivity(&self);
        self.split(&vertex_connectivity, num_components)
    }
}

impl<T: Real> SplitIntoConnectedComponents<VertexIndex, FaceIndex> for TriMesh<T> {
    fn split_into_connected_components(self) -> Vec<Self> {
        // First we partition the vertices.
        let (vertex_connectivity, num_components) =
            Connectivity::<VertexIndex, FaceIndex>::connectivity(&self);

        self.split(&vertex_connectivity, num_components)
    }
}

impl<T: Real> SplitIntoConnectedComponents<VertexIndex, FaceIndex> for QuadMesh<T> {
    fn split_into_connected_components(self) -> Vec<Self> {
        // First we partition the vertices.
        let (vertex_connectivity, num_components) =
            Connectivity::<VertexIndex, FaceIndex>::connectivity(&self);

        self.split(&vertex_connectivity, num_components)
    }
}

// TODO: Generalize split_vertices_by_face_vertex_attrib between the meshes.
//       This will involve converging on how to represent/access indices for rewiring meshes
//       through a trait.

impl<T: Real> TriMesh<T> {
    /// Split vertices by a given face-vertex attribute.
    ///
    /// If a pair of face-vertices have different values for the same vertex, then they will be
    /// split into distinct vertices. New vertex positions are appended at the end of the vertex
    /// position array.
    ///
    /// If the given attribute doesn't exist, then nothing is changed.
    pub fn split_vertices_by_face_vertex_attrib(&mut self, attrib_name: &str) {
        // For each vertex, topo contains a set of face-vertex indices.
        let (fv_indices, fv_offsets) = self.reverse_source_topo();

        // This function doesn't affect the number of faces or face-vertex topology.
        let TriMesh {
            vertex_positions,
            indices,
            vertex_attributes,
            face_vertex_attributes,
            // Other attributes remain unchanged.
            ..
        } = self;

        if let Some(attrib) = face_vertex_attributes.get(attrib_name) {
            let attrib_values = attrib.data_slice();

            // The partitioning of unique values in the neighbourhood of one vertex.
            let mut local_partition = Vec::new();
            let mut unique_values = Vec::new();

            // Remember which vertices were newly created so we can transfer vertex attributes.
            let mut new_vertices = Vec::new();

            for vtx_idx in 0..vertex_positions.len() {
                local_partition.clear();
                unique_values.clear();

                for face_vertex in
                    (fv_offsets[vtx_idx]..fv_offsets[vtx_idx + 1]).map(|i| fv_indices[i])
                {
                    let val = attrib_values.get(face_vertex);
                    if let Some(idx) = unique_values.iter().position(|uv| uv == &val) {
                        local_partition.push((idx, face_vertex));
                    } else {
                        local_partition.push((unique_values.len(), face_vertex));
                        unique_values.push(val);
                    }
                }

                local_partition.sort_by_key(|a| a.0);
                let mut partition_iter = local_partition.iter();
                if let Some(mut prev) = partition_iter.next() {
                    // First element will have a unique vertex by definition.
                    for next in partition_iter {
                        if next.0 != prev.0 {
                            // Found a different face-vertex attribute. Split the vertex.
                            // Rewire appropriate vertex index to the new vertex.
                            let pos = vertex_positions[vtx_idx];
                            indices[next.1 / 3][next.1 % 3] = vertex_positions.len();
                            vertex_positions.as_mut_vec().push(pos);
                            new_vertices.push(vtx_idx);
                            prev = next;
                        } else {
                            // Same bucket but new vertices may have been created, so we must still
                            // rewire to the last newly created vertex.
                            indices[next.1 / 3][next.1 % 3] = indices[prev.1 / 3][prev.1 % 3];
                        }
                    }
                }
            }

            // Duplicate vertex attributes for newly created vertices.
            for (_, attrib) in vertex_attributes.iter_mut() {
                let num = attrib.len();
                attrib.extend_by(new_vertices.len());

                // Split the extended attribute into original byte slice and and newly extended
                // uninitialized slice.
                let mut data_slice = attrib.data_mut_slice();
                let (old, mut new) = data_slice.split_at(num);
                for (&vtx_idx, mut new_val) in new_vertices.iter().zip(new.iter()) {
                    // Initialize the extended part.
                    //let bytes = &old[vtx_idx * element_size..(vtx_idx + 1) * element_size];
                    //new[i * element_size..(i + 1) * element_size].copy_from_slice(bytes);
                    new_val.clone_from_other(old.get(vtx_idx)).unwrap();
                }
            }
        }
    }
}

impl<T: Real> PolyMesh<T> {
    /// Split vertices by a given face-vertex attribute.
    ///
    /// If a pair of face-vertices have different values for the same vertex, then they will be
    /// split into distinct vertices. New vertex positions are appended at the end of the vertex
    /// position array.
    ///
    /// If the given attribute doesn't exist, then nothing is changed.
    pub fn split_vertices_by_face_vertex_attrib<U: PartialOrd + PartialEq + Copy + 'static>(
        &mut self,
        attrib: &str,
    ) {
        // For each vertex, topo contains a set of face-vertex indices.
        let (fv_indices, fv_offsets) = self.reverse_source_topo();

        // This function doesn't affect the number of faces or face-vertex topology.
        let PolyMesh {
            vertex_positions,
            indices,
            vertex_attributes,
            face_vertex_attributes,
            // Other attributes remain unchanged.
            ..
        } = self;

        if let Some(attrib) = face_vertex_attributes
            .get(attrib)
            .and_then(|a| a.as_slice::<U>().ok())
        {
            // The partitioning of unique values in the neighbourhood of one vertex.
            let mut local_partition = Vec::new();

            // Remember which vertices were newly created so we can transfer vertex attributes.
            let mut new_vertices = Vec::new();

            for vtx_idx in 0..vertex_positions.len() {
                local_partition.clear();
                for face_vertex in
                    (fv_offsets[vtx_idx]..fv_offsets[vtx_idx + 1]).map(|i| fv_indices[i])
                {
                    local_partition.push((face_vertex, &attrib[face_vertex]));
                }
                local_partition
                    .sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less));
                let mut partition_iter = local_partition.iter();
                if let Some(mut prev) = partition_iter.next() {
                    // First element will have a unique vertex by definition.
                    for next in partition_iter {
                        if next.1 != prev.1 {
                            // Found a different face-vertex attribute. Split the vertex.
                            // Rewire appropriate vertex index to the new vertex.
                            let pos = vertex_positions[vtx_idx];
                            indices[next.0] = vertex_positions.len();
                            vertex_positions.as_mut_vec().push(pos);
                            new_vertices.push(vtx_idx);
                            prev = next;
                        } else {
                            // Same bucket but new vertices may have been created, so we must still
                            // rewire to the last newly created vertex.
                            indices[next.0] = indices[prev.0];
                        }
                    }
                }
            }

            // Duplicate vertex attributes for newly created vertices.
            for (_, attrib) in vertex_attributes.iter_mut() {
                let num = attrib.len();
                attrib.extend_by(new_vertices.len());

                // Split the extended attribute into original byte slice and newly extended
                // uninitialized slice.
                let mut data_slice = attrib.data_mut_slice();
                let (old, mut new) = data_slice.split_at(num);

                for (&vtx_idx, mut new_val) in new_vertices.iter().zip(new.iter()) {
                    // Initialize the extended part.
                    //let bytes = &old[vtx_idx * element_size..(vtx_idx + 1) * element_size];
                    //new[i * element_size..(i + 1) * element_size].copy_from_slice(bytes);
                    new_val.clone_from_other(old.get(vtx_idx)).unwrap();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::test_utils::*;
    use crate::mesh::{TetMeshExt, TriMesh};

    fn build_tetmesh_sample() -> (TetMeshExt<f64>, TetMeshExt<f64>, TetMeshExt<f64>) {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.0, 0.5],
        ];

        // One connected component consisting of two tets connected at a face, and another
        // consisting of a single tet.
        let indices = vec![[7, 6, 2, 4], [5, 7, 2, 4], [0, 1, 3, 8]];

        let tetmesh = TetMeshExt::new(verts, indices);
        let comp1 = TetMeshExt::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.5, 0.0, 0.5],
            ],
            vec![[0, 1, 2, 3]],
        );
        let comp2 = TetMeshExt::new(
            vec![
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            vec![[4, 3, 0, 1], [2, 4, 0, 1]],
        );
        (tetmesh, comp1, comp2)
    }

    #[test]
    fn tetmesh_split() {
        let (tetmesh, comp1, comp2) = build_tetmesh_sample();

        // First lets verify the vertex partitioning.
        assert_eq!(tetmesh.connectivity(), (vec![0, 0, 1, 0, 1, 1, 1, 1, 0], 2));

        let res = tetmesh.split_into_connected_components();
        assert_eq!(res, vec![comp1, comp2]);
    }

    #[test]
    fn tetmesh_split_with_vertex_attributes() {
        let (mut tetmesh, mut comp1, mut comp2) = build_tetmesh_sample();
        tetmesh
            .add_attrib_data::<usize, VertexIndex>("v", (0..tetmesh.num_vertices()).collect())
            .unwrap();
        comp1
            .add_attrib_data::<usize, VertexIndex>("v", vec![0, 1, 3, 8])
            .unwrap();
        comp2
            .add_attrib_data::<usize, VertexIndex>("v", vec![2, 4, 5, 6, 7])
            .unwrap();
        let res = tetmesh.split_into_connected_components();
        assert_eq!(res, vec![comp1, comp2]);
    }

    #[test]
    fn tetmesh_split_with_cell_attributes() {
        let (mut tetmesh, mut comp1, mut comp2) = build_tetmesh_sample();
        tetmesh
            .add_attrib_data::<usize, CellIndex>("c", (0..tetmesh.num_cells()).collect())
            .unwrap();
        comp1
            .add_attrib_data::<usize, CellIndex>("c", vec![2])
            .unwrap();
        comp2
            .add_attrib_data::<usize, CellIndex>("c", vec![0, 1])
            .unwrap();
        let res = tetmesh.split_into_connected_components();
        assert_eq!(res, vec![comp1, comp2]);
    }

    #[test]
    fn tetmesh_split_with_cell_vertex_attributes() {
        let (mut tetmesh, mut comp1, mut comp2) = build_tetmesh_sample();
        tetmesh
            .add_attrib_data::<usize, CellVertexIndex>("cv", (0..tetmesh.num_cells() * 4).collect())
            .unwrap();

        comp1
            .add_attrib_data::<usize, CellVertexIndex>("cv", vec![8, 9, 10, 11])
            .unwrap();
        comp2
            .add_attrib_data::<usize, CellVertexIndex>("cv", vec![0, 1, 2, 3, 4, 5, 6, 7])
            .unwrap();
        let res = tetmesh.split_into_connected_components();
        assert_eq!(res, vec![comp1, comp2]);
    }

    #[test]
    fn tetmesh_split_with_cell_face_attributes() {
        let (mut tetmesh, mut comp1, mut comp2) = build_tetmesh_sample();
        tetmesh
            .add_attrib_data::<usize, CellFaceIndex>("cf", (0..tetmesh.num_cells() * 4).collect())
            .unwrap();

        comp1
            .add_attrib_data::<usize, CellFaceIndex>("cf", vec![8, 9, 10, 11])
            .unwrap();
        comp2
            .add_attrib_data::<usize, CellFaceIndex>("cf", vec![0, 1, 2, 3, 4, 5, 6, 7])
            .unwrap();
        let res = tetmesh.split_into_connected_components();
        assert_eq!(res, vec![comp1, comp2]);
    }

    #[test]
    fn tetmesh_split_with_vertex_cell_attributes() {
        let (mut tetmesh, mut comp1, mut comp2) = build_tetmesh_sample();
        tetmesh
            .add_attrib_data::<usize, VertexCellIndex>("vc", (0..tetmesh.num_cells() * 4).collect())
            .unwrap();

        comp1
            .add_attrib_data::<usize, VertexCellIndex>("vc", vec![0, 1, 4, 11])
            .unwrap();
        comp2
            .add_attrib_data::<usize, VertexCellIndex>("vc", vec![2, 3, 5, 6, 7, 8, 9, 10])
            .unwrap();
        let res = tetmesh.split_into_connected_components();
        assert_eq!(res, vec![comp1, comp2]);
    }

    #[test]
    fn tetmesh_split_with_all_attributes() {
        let (mut tetmesh, mut comp1, mut comp2) = build_tetmesh_sample();
        tetmesh
            .add_attrib_data::<usize, VertexIndex>("v", (0..tetmesh.num_vertices()).collect())
            .unwrap();
        tetmesh
            .add_attrib_data::<usize, CellIndex>("c", (0..tetmesh.num_cells()).collect())
            .unwrap();
        tetmesh
            .add_attrib_data::<usize, CellVertexIndex>("cv", (0..tetmesh.num_cells() * 4).collect())
            .unwrap();
        tetmesh
            .add_attrib_data::<usize, CellFaceIndex>("cf", (0..tetmesh.num_cells() * 4).collect())
            .unwrap();
        tetmesh
            .add_attrib_data::<usize, VertexCellIndex>("vc", (0..tetmesh.num_cells() * 4).collect())
            .unwrap();
        comp1
            .add_attrib_data::<usize, VertexIndex>("v", vec![0, 1, 3, 8])
            .unwrap();
        comp1
            .add_attrib_data::<usize, CellIndex>("c", vec![2])
            .unwrap();
        comp1
            .add_attrib_data::<usize, CellVertexIndex>("cv", vec![8, 9, 10, 11])
            .unwrap();
        comp1
            .add_attrib_data::<usize, CellFaceIndex>("cf", vec![8, 9, 10, 11])
            .unwrap();
        comp1
            .add_attrib_data::<usize, VertexCellIndex>("vc", vec![0, 1, 4, 11])
            .unwrap();

        comp2
            .add_attrib_data::<usize, VertexIndex>("v", vec![2, 4, 5, 6, 7])
            .unwrap();
        comp2
            .add_attrib_data::<usize, CellIndex>("c", vec![0, 1])
            .unwrap();
        comp2
            .add_attrib_data::<usize, CellVertexIndex>("cv", vec![0, 1, 2, 3, 4, 5, 6, 7])
            .unwrap();
        comp2
            .add_attrib_data::<usize, CellFaceIndex>("cf", vec![0, 1, 2, 3, 4, 5, 6, 7])
            .unwrap();
        comp2
            .add_attrib_data::<usize, VertexCellIndex>("vc", vec![2, 3, 5, 6, 7, 8, 9, 10])
            .unwrap();
        let res = tetmesh.split_into_connected_components();
        assert_eq!(res, vec![comp1, comp2]);
    }

    #[test]
    fn polymesh_split() {
        let (mesh, comp1, comp2) = build_polymesh_sample();

        // First lets verify the vertex partitioning.
        assert_eq!(
            mesh.vertex_connectivity(),
            (vec![0, 0, 0, 0, 1, 1, 1, 1], 2)
        );

        let res = mesh.split_into_connected_components();
        assert_eq!(res, vec![comp1, comp2]);
    }

    #[test]
    fn polymesh_split_with_attributes() {
        let mut sample = build_polymesh_sample();
        add_attribs_to_polymeshes(&mut sample);
        let (mesh, comp1, comp2) = sample;
        let res = mesh.split_into_connected_components();
        assert_eq!(res, vec![comp1, comp2]);
    }

    #[test]
    fn polymesh_split_vertices_by_face_vertex_attrib() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ];

        // Two triangles connected at an edge, a quad, and two triangles connecting these
        // inbetweeen.
        let indices = vec![
            3, 0, 1, 2, 3, 2, 1, 3, 4, 4, 5, 7, 6, 3, 0, 1, 4, 3, 1, 5, 4,
        ];

        let mut polymesh = PolyMesh::new(verts, &indices);

        // Add an arbitrary vertex attribute
        polymesh
            .add_attrib_data::<usize, VertexIndex>("v", (0..polymesh.num_vertices()).collect())
            .unwrap();

        polymesh
            .add_attrib_data::<usize, FaceVertexIndex>(
                "no_split",
                vec![0, 1, 2, 2, 1, 3, 4, 5, 7, 6, 0, 1, 4, 1, 5, 4],
            )
            .unwrap();

        let mut no_split = polymesh.clone();
        no_split.split_vertices_by_face_vertex_attrib::<usize>("no_split");
        assert_eq!(no_split, polymesh);

        polymesh
            .add_attrib_data::<i32, FaceVertexIndex>(
                "vertex1_split",
                vec![0, 10, 2, 2, 11, 3, 4, 5, 7, 6, 0, 12, 4, 13, 5, 4],
            )
            .unwrap();

        let mut vertex1_split = polymesh.clone();
        vertex1_split.split_vertices_by_face_vertex_attrib::<i32>("vertex1_split");
        assert_eq!(vertex1_split.num_vertices(), polymesh.num_vertices() + 3);
        assert_eq!(
            vertex1_split.num_face_vertices(),
            polymesh.num_face_vertices()
        );
        assert_eq!(
            vertex1_split.attrib::<FaceVertexIndex>("vertex1_split"),
            polymesh.attrib::<FaceVertexIndex>("vertex1_split")
        );
        assert_eq!(
            vertex1_split.attrib_as_slice::<usize, VertexIndex>("v"),
            Ok(&[0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1][..])
        );

        polymesh
            .add_attrib_data::<usize, FaceVertexIndex>(
                "full_split",
                (0..polymesh.num_face_vertices()).collect(),
            )
            .unwrap();

        let mut full_split = polymesh.clone();
        full_split.split_vertices_by_face_vertex_attrib::<usize>("full_split");
        assert_eq!(full_split.num_vertices(), polymesh.num_face_vertices());
        assert_eq!(full_split.num_face_vertices(), polymesh.num_face_vertices());
        assert_eq!(
            full_split.attrib_as_slice::<usize, VertexIndex>("v"),
            Ok(&[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 1, 1, 2, 4, 4, 5][..])
        );
    }

    #[test]
    fn trimesh_split_vertices_by_face_vertex_attrib() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ];

        let indices = vec![
            [0, 1, 2],
            [2, 1, 3],
            [4, 5, 6],
            [6, 5, 7],
            [0, 1, 4],
            [1, 5, 4],
        ];

        let mut mesh = TriMesh::new(verts, indices);

        // Add an arbitrary vertex attribute
        mesh.add_attrib_data::<usize, VertexIndex>("v", (0..mesh.num_vertices()).collect())
            .unwrap();

        mesh.add_attrib_data::<usize, FaceVertexIndex>(
            "no_split",
            vec![0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7, 0, 1, 4, 1, 5, 4],
        )
        .unwrap();

        let mut no_split = mesh.clone();
        no_split.split_vertices_by_face_vertex_attrib("no_split");
        assert_eq!(no_split, mesh);

        mesh.add_attrib_data::<f32, FaceVertexIndex>(
            "vertex1_split",
            vec![
                0.0f32,
                10.0 / 3.0,
                2.0,
                2.0,
                11.0,
                3.0,
                4.0,
                5.0,
                6.0 / 4.0,
                6.0 / 4.0,
                5.0,
                7.0,
                0.0,
                12.0,
                4.0,
                13.0,
                5.0,
                4.0,
            ],
        )
        .unwrap();

        let mut vertex1_split = mesh.clone();
        vertex1_split.split_vertices_by_face_vertex_attrib("vertex1_split");
        assert_eq!(vertex1_split.num_vertices(), mesh.num_vertices() + 3);
        assert_eq!(vertex1_split.num_face_vertices(), mesh.num_face_vertices());
        assert_eq!(
            vertex1_split.attrib::<FaceVertexIndex>("vertex1_split"),
            mesh.attrib::<FaceVertexIndex>("vertex1_split")
        );
        assert_eq!(
            vertex1_split.attrib_as_slice::<usize, VertexIndex>("v"),
            Ok(&[0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1][..])
        );

        mesh.add_attrib_data::<usize, FaceVertexIndex>(
            "full_split",
            (0..mesh.num_face_vertices()).collect(),
        )
        .unwrap();

        let mut full_split = mesh.clone();
        full_split.split_vertices_by_face_vertex_attrib("full_split");
        assert_eq!(full_split.num_vertices(), mesh.num_face_vertices());
        assert_eq!(full_split.num_face_vertices(), mesh.num_face_vertices());
        assert_eq!(
            full_split.attrib_as_slice::<usize, VertexIndex>("v"),
            Ok(&[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 1, 1, 2, 4, 4, 5, 5, 6][..])
        );
    }

    /// This is a more complex regression test for splitting vertices.
    #[test]
    fn trimesh_split_vertices_by_face_vertex_attrib_and_promote_complex() {
        let verts = vec![
            [-0.520833, -0.5, 0.5],
            [-0.520833, 0.5, 0.5],
            [-0.520833, -0.5, -0.5],
            [-0.520833, 0.5, -0.5],
            [0.520833, -0.5, 0.5],
            [0.520833, 0.5, 0.5],
            [0.520833, -0.5, -0.5],
            [0.520833, 0.5, -0.5],
        ];

        #[rustfmt::skip]
        let indices = vec![
            [0, 1, 3],
            [4, 5, 7],
            [6, 7, 2],
            [5, 4, 1],
            [5, 0, 2],
            [1, 4, 6],
            [6, 3, 1],
            [2, 7, 5],
            [1, 0, 5],
            [2, 3, 6],
            [7, 6, 4],
            [3, 2, 0],
        ];

        let mut mesh = TriMesh::new(verts, indices);

        // We split the vertices according to the following attribute and then test that
        // there are no more collisions.
        // This tests both functions: split_vertices_by_face_vertex_attrib and attrib_promote.

        mesh.add_attrib_data::<[f32; 2], FaceVertexIndex>(
            "uv",
            vec![
                [0.630043, 0.00107052],
                [0.370129, 0.00107052],
                [0.370129, 0.250588],
                [0.370129, 0.749623],
                [0.630043, 0.749623],
                [0.630043, 0.500105],
                [0.370129, 0.500105],
                [0.630043, 0.500105],
                [0.630043, 0.250588],
                [0.630043, 0.749623],
                [0.370129, 0.749623],
                [0.370129, 0.99914],
                [0.879561, 0.500105],
                [0.879561, 0.250588],
                [0.630043, 0.250588],
                [0.120612, 0.250588],
                [0.120612, 0.500105],
                [0.370129, 0.500105],
                [0.370129, 0.500105],
                [0.370129, 0.250588],
                [0.120612, 0.250588],
                [0.630043, 0.250588],
                [0.630043, 0.500105],
                [0.879561, 0.500105],
                [0.370129, 0.99914],
                [0.630043, 0.99914],
                [0.630043, 0.749623],
                [0.630043, 0.250588],
                [0.370129, 0.250588],
                [0.370129, 0.500105],
                [0.630043, 0.500105],
                [0.370129, 0.500105],
                [0.370129, 0.749623],
                [0.370129, 0.250588],
                [0.630043, 0.250588],
                [0.630043, 0.00107052],
            ],
        )
        .unwrap();

        mesh.split_vertices_by_face_vertex_attrib("uv");

        mesh.attrib_promote::<[f32; 2], _>("uv", |a, b| assert_eq!(a, b))
            .unwrap();
    }

    /// The same test for polymeshes.
    #[test]
    fn polymesh_split_vertices_by_face_vertex_attrib_and_promote_complex() {
        let verts = vec![
            [-0.520833, -0.5, 0.5],
            [-0.520833, 0.5, 0.5],
            [-0.520833, -0.5, -0.5],
            [-0.520833, 0.5, -0.5],
            [0.520833, -0.5, 0.5],
            [0.520833, 0.5, 0.5],
            [0.520833, -0.5, -0.5],
            [0.520833, 0.5, -0.5],
        ];

        #[rustfmt::skip]
        let indices = vec![
            [0, 1, 3],
            [2, 4, 5],
            [7, 6, 6],
            [7, 2, 3],
            [5, 4, 1],
            [0, 5, 0],
            [2, 7, 1],
            [4, 6, 3],
        ];

        let mut mesh = TriMesh::new(verts, indices);

        // We split the vertices according to the following attribute and then test that
        // there are no more collisions.
        // This tests both functions: split_vertices_by_face_vertex_attrib and attrib_promote.

        mesh.add_attrib_data::<[f32; 2], FaceVertexIndex>(
            "uv",
            vec![
                [0.630043, 0.00107052],
                [0.370129, 0.00107052],
                [0.370129, 0.250588],
                [0.630043, 0.250588],
                [0.370129, 0.749623],
                [0.630043, 0.749623],
                [0.630043, 0.500105],
                [0.370129, 0.500105],
                [0.370129, 0.500105],
                [0.630043, 0.500105],
                [0.630043, 0.250588],
                [0.370129, 0.250588],
                [0.630043, 0.749623],
                [0.370129, 0.749623],
                [0.370129, 0.99914],
                [0.630043, 0.99914],
                [0.879561, 0.500105],
                [0.879561, 0.250588],
                [0.630043, 0.250588],
                [0.630043, 0.500105],
                [0.120612, 0.250588],
                [0.120612, 0.500105],
                [0.370129, 0.500105],
                [0.370129, 0.250588],
            ],
        )
        .unwrap();

        mesh.split_vertices_by_face_vertex_attrib("uv");

        mesh.attrib_promote::<[f32; 2], _>("uv", |a, b| assert_eq!(a, b))
            .unwrap();
    }
}
