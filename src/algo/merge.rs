/*!
 * This module defines the `Merge` trait which defines how objects are merged together.
 * Implementations for common mesh types are also included here.
 */

use crate::mesh::attrib::AttribValueCache;
use crate::mesh::attrib::*;
use crate::mesh::pointcloud::PointCloud;
use crate::mesh::polymesh::PolyMesh;
use crate::mesh::tetmesh::{TetMesh, TetMeshExt};
use crate::mesh::topology::*;
use crate::mesh::uniform_poly_mesh::{QuadMesh, QuadMeshExt, TriMesh, TriMeshExt};
use crate::mesh::VertexPositions;
use crate::Real;
use math::{Vector3, Vector4};

/// A trait describing the action of merging mutltiple objects into a single object of the same
/// type. This represents the inverse of splitting a mesh into components of the same type.
pub trait Merge {
    /// Merge another object into self and return the resulting merged object as a mutable
    /// reference.
    fn merge(&mut self, other: Self) -> &mut Self;

    /// Merge an iterator of objects into one of the same type.
    fn merge_iter(iterable: impl IntoIterator<Item = Self>) -> Self
    where
        Self: Default,
    {
        let mut iter = iterable.into_iter();

        let mut obj = iter.next().unwrap_or_else(Default::default);

        for other in iter {
            obj.merge(other);
        }

        obj
    }

    /// Merge a `Vec` of objects into one of the same type.
    fn merge_vec(vec: Vec<Self>) -> Self
    where
        Self: Default,
    {
        Self::merge_iter(vec)
    }

    /// In contrast to `merge_vec`, this function takes an immutable reference to a collection of
    /// meshes, and creates a brand new mesh that is a union of all the given meshes.
    fn merge_slice(slice: &[Self]) -> Self
    where
        Self: Clone + Default,
    {
        Self::merge_iter(slice.into_iter().cloned())
    }
}

/// Helper function to merge two attribute tables together.
///
/// `num_elements` must correspond to the number of elements in each attribute inside `dict`.
/// `num_additional_elements` must correspond to the number of elements in each attribute inside
/// `additional_dict`.
fn merge_attribute_dicts<I>(
    dict: &mut AttribDict<I>,
    num_elements: usize,
    additional_dict: AttribDict<I>,
    num_additional_elements: usize,
) {
    for (name, mut other_attrib) in additional_dict.into_iter() {
        // Check if self already has this attribute. If so, we append, otherwise we create a
        // brand new one.
        match dict.entry(name.to_owned()) {
            Entry::Occupied(entry) => {
                let attrib = entry.into_mut();

                // The copy may be unsuccessful because attributes could have different types.
                // Since we don't actually know the types, we will simply fill our
                // attribute with a default value.
                match &mut attrib.data {
                    AttributeData::Direct(ref mut d) => {
                        if other_attrib
                            .data
                            .direct_data_mut()
                            .ok()
                            .and_then(|other_data| d.data_mut().append(other_data.data_mut()))
                            .is_none()
                        {
                            assert_eq!(other_attrib.len(), num_additional_elements);
                            d.extend_by(other_attrib.len());
                        }
                    }
                    AttributeData::Indirect(ref mut i) => {
                        if other_attrib
                            .data
                            .indirect_data_mut()
                            .ok()
                            .and_then(|other_data| i.data_mut().append(other_data.data_mut()))
                            .is_none()
                        {
                            assert_eq!(other_attrib.len(), num_additional_elements);
                            i.extend_by(other_attrib.len());
                        }
                    }
                }
            }
            Entry::Vacant(entry) => {
                other_attrib.extend_by(num_elements);
                other_attrib.rotate_right(num_elements);
                entry.insert(other_attrib);
            }
        }
    }

    // Extend any attributes in dict that weren't present in `additional_dict` to make sure there
    // are the same number of elements in each attribute in `dict` after this function is
    // completed.

    for (_, attrib) in dict.iter_mut() {
        if attrib.len() < num_elements + num_additional_elements {
            attrib.extend_by(num_additional_elements);
        }

        assert_eq!(attrib.len(), num_elements + num_additional_elements);
    }
}

/// Helper function to merge two attribute tables together using the given source index for
/// determining the merge order.
///
///  - `num_elements` is the number of elements desired in each attribute inside `dict`.
///    Note that the actual number of elements per attribute in `dict` may be smaller, in which
///    case all attributes will be extended with the default value to have length equal to
///    `num_elements`.
///  - `num_additional_elements` must correspond to the number of elements in each attribute inside
/// `additional_dict`.
///  - `source` is a slice of indices mapping attributes in `additional_dict` to entries in `dict`.
///     This means that `source` must size of entries in `additional_dict` and indices in the range
///     `0..num_elements`.
fn merge_attribute_dicts_with_source<I>(
    dict: &mut AttribDict<I>,
    num_elements: usize,
    additional_dict: &AttribDict<I>,
    num_additional_elements: usize,
    source: &[usize],
) {
    let write_data = |mut buf: DataSliceMut, other_buf: DataSlice| {
        for (&i, other_val) in source.iter().zip(other_buf.iter()) {
            // Ignore type mismatch, we don't write anything and just leave the default value intact.
            buf.get_mut(i).clone_from_other(other_val).ok();
        }
    };
    for (name, other_attrib) in additional_dict.iter() {
        // Check if self already has this attribute. If so, we append, otherwise we create a
        // brand new one.
        match dict.entry(name.to_owned()) {
            Entry::Occupied(entry) => {
                let attrib = entry.into_mut();
                // Extend the existing attribute if necessary.
                if attrib.len() < num_elements {
                    attrib.extend_by(num_elements - attrib.len());
                }
                // The additional attribute must have the correct number of elements.
                assert_eq!(other_attrib.len(), num_additional_elements);

                let buf = attrib.data_mut_slice();
                let other_buf = other_attrib.data_slice();

                // Overwrite any existing values.
                write_data(buf, other_buf);
            }
            Entry::Vacant(entry) => {
                entry.insert(other_attrib.duplicate_with_len(num_elements, write_data));
            }
        }
    }

    // Extend any attributes in dict that weren't present in `additional_dict` to make sure there
    // are the same number of elements in each attribute in `dict` after this function is
    // completed.

    for (_, attrib) in dict.iter_mut() {
        if attrib.len() < num_elements {
            attrib.extend_by(num_elements - attrib.len());
        }

        assert_eq!(attrib.len(), num_elements);
    }
}

impl<T: Real> Merge for TetMesh<T> {
    fn merge(&mut self, other: Self) -> &mut Self {
        let self_num_vertices = self.num_vertices();
        let other_num_vertices = other.num_vertices();
        let self_num_cells = self.num_cells();
        let other_num_cells = other.num_cells();
        let self_num_cell_vertices = self.num_cell_vertices();
        let other_num_cell_vertices = other.num_cell_vertices();
        let self_num_cell_faces = self.num_cell_faces();
        let other_num_cell_faces = other.num_cell_faces();

        // Deconstruct the other mesh explicitly since it will not be valid as soon as we start to
        // canibalize its contents.
        let TetMesh {
            vertex_positions: mut other_vertex_positions,
            indices: other_indices,
            vertex_attributes: other_vertex_attributes,
            cell_attributes: other_cell_attributes,
            cell_vertex_attributes: other_cell_vertex_attributes,
            cell_face_attributes: other_cell_face_attributes,
            attribute_value_cache: other_attribute_value_cache,
        } = other;

        self.vertex_positions
            .as_mut_vec()
            .append(other_vertex_positions.as_mut_vec());
        self.indices
            .as_mut_vec()
            .extend(other_indices.iter().map(|&cell| -> [usize; 4] {
                Vector4::from(cell).map(|i| i + self_num_vertices).into()
            }));

        // Transfer attributes
        merge_attribute_dicts(
            &mut self.vertex_attributes,
            self_num_vertices,
            other_vertex_attributes,
            other_num_vertices,
        );
        merge_attribute_dicts(
            &mut self.cell_attributes,
            self_num_cells,
            other_cell_attributes,
            other_num_cells,
        );
        merge_attribute_dicts(
            &mut self.cell_vertex_attributes,
            self_num_cell_vertices,
            other_cell_vertex_attributes,
            other_num_cell_vertices,
        );
        merge_attribute_dicts(
            &mut self.cell_face_attributes,
            self_num_cell_faces,
            other_cell_face_attributes,
            other_num_cell_faces,
        );
        for value in other_attribute_value_cache.into_iter() {
            self.attribute_value_cache.insert(value);
        }
        self
    }
}

impl<T: Real> Merge for PointCloud<T> {
    fn merge(&mut self, other: Self) -> &mut Self {
        let self_num_vertices = self.num_vertices();
        let other_num_vertices = other.num_vertices();

        // Deconstruct the other mesh explicitly since it will not be valid as soon as we start to
        // canibalize its contents.
        let PointCloud {
            vertex_positions: mut other_vertex_positions,
            vertex_attributes: other_vertex_attributes,
        } = other;

        self.vertex_positions
            .as_mut_vec()
            .append(other_vertex_positions.as_mut_vec());

        // Transfer attributes
        merge_attribute_dicts(
            &mut self.vertex_attributes,
            self_num_vertices,
            other_vertex_attributes,
            other_num_vertices,
        );
        self
    }
}

impl<T: Real> TetMesh<T> {
    /// Merge a iterator of meshes into a single distinct mesh.
    ///
    /// This version of `merge` accepts an attribute name for the source index on vertices.
    ///
    /// The mesh vertices will be merged in the order given by the source attribute. This
    /// is useful when merging previously split up meshes. The source attribute needs to
    /// have type `usize`.
    ///
    /// If the source attribute does not exist in at least one of the given meshes, then
    /// `None` is returned and the merge is aborted.
    ///
    /// This is a non-destructive merge --- both original meshes remain intact.
    /// This also means that this way of merging is somewhat more expensive than a merge
    /// without any source indices.
    pub fn merge_with_vertex_source<'a, I>(meshes: I, source_attrib: &str) -> Result<Self, Error>
    where
        I: IntoIterator<Item = &'a Self>,
    {
        Self::merge_with_vertex_source_impl(meshes.into_iter(), source_attrib)
    }

    fn merge_with_vertex_source_impl<'a>(
        mesh_iter: impl Iterator<Item = &'a Self>,
        source_attrib: &str,
    ) -> Result<Self, Error> {
        let mut vertex_positions = Vec::new();
        let mut indices = Vec::new();
        let mut vertex_attributes = AttribDict::new();
        let mut cell_attributes = AttribDict::new();
        let mut cell_vertex_attributes = AttribDict::new();
        let mut cell_face_attributes = AttribDict::new();
        let mut num_vertices = 0;
        let mut attribute_value_cache = AttribValueCache::default();

        for mesh in mesh_iter {
            let src = mesh.attrib_as_slice::<usize, VertexIndex>(source_attrib)?;
            if src.is_empty() {
                // Empty mesh detected.
                continue;
            }

            num_vertices = num_vertices.max(*src.iter().max().unwrap() + 1);

            vertex_positions.resize(num_vertices, [T::zero(); 3]);
            for (&i, &pos) in src.iter().zip(mesh.vertex_position_iter()) {
                vertex_positions[i] = pos;
            }

            // Transfer attributes
            merge_attribute_dicts_with_source(
                &mut vertex_attributes,
                num_vertices,
                &mesh.vertex_attributes,
                mesh.num_vertices(),
                src,
            );
            merge_attribute_dicts(
                &mut cell_attributes,
                indices.len(),
                mesh.cell_attributes.clone(),
                mesh.num_cells(),
            );
            merge_attribute_dicts(
                &mut cell_vertex_attributes,
                indices.len() * 4,
                mesh.cell_vertex_attributes.clone(),
                mesh.num_cell_vertices(),
            );
            merge_attribute_dicts(
                &mut cell_face_attributes,
                indices.len() * 4,
                mesh.cell_face_attributes.clone(),
                mesh.num_cell_faces(),
            );

            for value in mesh.attribute_value_cache.iter() {
                attribute_value_cache.insert(value.clone());
            }

            // Extend the indices AFTER attributes are transfered since
            // `merge_attribute_dicts` expects num_elements to be the number before
            // the merge.
            indices.extend(
                mesh.cell_iter()
                    .map(|&cell| -> [usize; 4] { Vector4::from(cell).map(|i| src[i]).into() }),
            );
        }

        Ok(Self {
            vertex_positions: IntrinsicAttribute::from_vec(vertex_positions),
            indices: IntrinsicAttribute::from_vec(indices),
            vertex_attributes,
            cell_attributes,
            cell_vertex_attributes,
            cell_face_attributes,
            attribute_value_cache,
        })
    }
}

impl<T: Real> Merge for TetMeshExt<T> {
    /// Attributes with the same name but different types won't be merged.
    fn merge(&mut self, other: Self) -> &mut Self {
        let self_num_cells = self.num_cells();
        let self_num_vertex_cells = self.num_vertex_cells();
        let other_num_vertex_cells = other.num_vertex_cells();

        // Deconstruct the other mesh explicitly since it will not be valid as soon as we start to
        // canibalize its contents.
        let TetMeshExt {
            tetmesh: other_tetmesh,
            cell_offsets: other_cell_offsets,
            cell_indices: other_cell_indices,
            vertex_cell_attributes: other_vertex_cell_attributes,
        } = other;

        self.tetmesh.merge(other_tetmesh);
        self.cell_offsets.extend(
            other_cell_offsets
                .iter()
                .skip(1)
                .map(|&i| i + self_num_vertex_cells),
        );
        self.cell_indices
            .extend(other_cell_indices.iter().map(|&i| i + self_num_cells));

        // Transfer attributes
        merge_attribute_dicts(
            &mut self.vertex_cell_attributes,
            self_num_vertex_cells,
            other_vertex_cell_attributes,
            other_num_vertex_cells,
        );
        self
    }
}

impl<T: Real> TetMeshExt<T> {
    /// Merge a iterator of meshes into a single distinct mesh.
    ///
    /// This version of `merge` accepts an attribute name for the source index on vertices.
    ///
    /// The mesh vertices will be merged in the order given by the source attribute. This
    /// is useful when merging previously split up meshes. The source attribute needs to
    /// have type `usize`.
    ///
    /// If the source attribute does not exist in at least one of the given meshes, then
    /// `None` is returned and the merge is aborted.
    ///
    /// This is a non-destructive merge --- both original meshes remain intact.
    /// This also means that this way of merging is somewhat more expensive than a merge
    /// without any source indices.
    pub fn merge_with_vertex_source<'a, I>(meshes: I, source_attrib: &str) -> Result<Self, Error>
    where
        I: IntoIterator<Item = &'a Self>,
        I::IntoIter: Clone,
    {
        Self::merge_with_vertex_source_impl(meshes.into_iter(), source_attrib)
    }

    fn merge_with_vertex_source_impl<'a>(
        mesh_iter: impl Iterator<Item = &'a Self> + Clone,
        source_attrib: &str,
    ) -> Result<Self, Error> {
        let tetmesh = TetMesh::merge_with_vertex_source_impl(
            mesh_iter.clone().map(|t| &t.tetmesh),
            source_attrib,
        )?;

        let mut cell_offsets = vec![0];
        let mut cell_indices = Vec::new();
        let mut vertex_cell_attributes = AttribDict::new();
        let mut num_cells = 0;

        for mesh in mesh_iter {
            let src = mesh.attrib_as_slice::<usize, VertexIndex>(source_attrib)?;
            if src.is_empty() {
                // Empty mesh detected.
                continue;
            }

            // Transfer attributes
            merge_attribute_dicts(
                &mut vertex_cell_attributes,
                cell_indices.len(),
                mesh.vertex_cell_attributes.clone(),
                mesh.num_vertex_cells(),
            );

            cell_offsets.extend(
                mesh.cell_offsets
                    .iter()
                    .skip(1)
                    .map(|&i| i + cell_indices.len()),
            );
            cell_indices.extend(mesh.cell_indices.iter().map(|&i| i + num_cells));

            num_cells += mesh.num_cells();
        }

        Ok(Self {
            tetmesh,
            cell_offsets,
            cell_indices,
            vertex_cell_attributes,
        })
    }
}

impl<T: Real> Merge for PolyMesh<T> {
    /// Attributes with the same name but different types won't be merged.
    fn merge(&mut self, other: Self) -> &mut Self {
        let self_num_vertices = self.num_vertices();
        let other_num_vertices = other.num_vertices();
        let self_num_faces = self.num_faces();
        let other_num_faces = other.num_faces();
        let self_num_face_vertices = self.num_face_vertices();
        let other_num_face_vertices = other.num_face_vertices();
        let self_num_face_edges = self.num_face_edges();
        let other_num_face_edges = other.num_face_edges();

        // Deconstruct the other mesh explicitly since it will not be valid as soon as we start to
        // canibalize its contents.
        let PolyMesh {
            vertex_positions: mut other_vertex_positions,
            indices: other_indices,
            offsets: other_offsets,
            vertex_attributes: other_vertex_attributes,
            face_attributes: other_face_attributes,
            face_vertex_attributes: other_face_vertex_attributes,
            face_edge_attributes: other_face_edge_attributes,
            attribute_value_cache: other_attribute_value_cache,
        } = other;

        self.vertex_positions
            .as_mut_vec()
            .append(other_vertex_positions.as_mut_vec());
        self.offsets.extend(
            other_offsets
                .iter()
                .skip(1)
                .map(|&i| i + self_num_face_vertices),
        );
        self.indices
            .extend(other_indices.iter().map(|&i| i + self_num_vertices));

        // Transfer attributes
        merge_attribute_dicts(
            &mut self.vertex_attributes,
            self_num_vertices,
            other_vertex_attributes,
            other_num_vertices,
        );
        merge_attribute_dicts(
            &mut self.face_attributes,
            self_num_faces,
            other_face_attributes,
            other_num_faces,
        );
        merge_attribute_dicts(
            &mut self.face_vertex_attributes,
            self_num_face_vertices,
            other_face_vertex_attributes,
            other_num_face_vertices,
        );
        merge_attribute_dicts(
            &mut self.face_edge_attributes,
            self_num_face_edges,
            other_face_edge_attributes,
            other_num_face_edges,
        );

        for value in other_attribute_value_cache.into_iter() {
            self.attribute_value_cache.insert(value);
        }
        self
    }
}

impl<T: Real> PolyMesh<T> {
    /// Merge a iterator of meshes into a single distinct mesh.
    ///
    /// This version of `merge` accepts an attribute name for the source index on vertices.
    ///
    /// The mesh vertices will be merged in the order given by the source attribute. This
    /// is useful when merging previously split up meshes. The source attribute needs to
    /// have type `usize`.
    ///
    /// If the source attribute does not exist in at least one of the given meshes, then
    /// `None` is returned and the merge is aborted.
    ///
    /// This is a non-destructive merge --- both original meshes remain intact.
    /// This also means that this way of merging is somewhat more expensive than a merge
    /// without any source indices.
    pub fn merge_with_vertex_source<'a, I>(meshes: I, source_attrib: &str) -> Result<Self, Error>
    where
        I: IntoIterator<Item = &'a Self>,
    {
        Self::merge_with_vertex_source_impl(meshes.into_iter(), source_attrib)
    }

    fn merge_with_vertex_source_impl<'a>(
        mesh_iter: impl Iterator<Item = &'a Self>,
        source_attrib: &str,
    ) -> Result<Self, Error> {
        let mut vertex_positions = Vec::new();
        let mut offsets = vec![0];
        let mut indices = Vec::new();
        let mut vertex_attributes = AttribDict::new();
        let mut face_attributes = AttribDict::new();
        let mut face_vertex_attributes = AttribDict::new();
        let mut face_edge_attributes = AttribDict::new();
        let mut attribute_value_cache = AttribValueCache::with_hasher(Default::default());
        let mut num_vertices = 0;

        for mesh in mesh_iter {
            let src = mesh.attrib_as_slice::<usize, VertexIndex>(source_attrib)?;
            if src.is_empty() {
                // Empty mesh detected.
                continue;
            }

            num_vertices = num_vertices.max(*src.iter().max().unwrap() + 1);

            vertex_positions.resize(num_vertices, [T::zero(); 3]);
            for (&i, &pos) in src.iter().zip(mesh.vertex_position_iter()) {
                vertex_positions[i] = pos;
            }

            // Transfer attributes
            merge_attribute_dicts_with_source(
                &mut vertex_attributes,
                num_vertices,
                &mesh.vertex_attributes,
                mesh.num_vertices(),
                src,
            );
            merge_attribute_dicts(
                &mut face_attributes,
                indices.len(),
                mesh.face_attributes.clone(),
                mesh.num_faces(),
            );
            merge_attribute_dicts(
                &mut face_vertex_attributes,
                indices.len(),
                mesh.face_vertex_attributes.clone(),
                mesh.num_face_vertices(),
            );
            merge_attribute_dicts(
                &mut face_edge_attributes,
                indices.len(),
                mesh.face_edge_attributes.clone(),
                mesh.num_face_edges(),
            );

            for value in mesh.attribute_value_cache.iter() {
                attribute_value_cache.insert(value.clone());
            }

            offsets.extend(mesh.offsets.iter().skip(1).map(|&i| i + indices.len()));
            indices.extend(mesh.indices.iter().map(|&i| src[i]));
        }

        Ok(Self {
            vertex_positions: IntrinsicAttribute::from_vec(vertex_positions),
            offsets,
            indices,
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            face_edge_attributes,
            attribute_value_cache,
        })
    }
}

macro_rules! impl_merge_for_uniform_mesh {
    ($mesh_type:ident, $base_type:ident, $verts_per_face:expr, $vec:ident) => {
        impl<T: Real> Merge for $base_type<T> {
            /// Attributes with the same name but different types won't be merged.
            fn merge(&mut self, other: Self) -> &mut Self {
                let self_num_vertices = self.num_vertices();
                let other_num_vertices = other.num_vertices();
                let self_num_faces = self.num_faces();
                let other_num_faces = other.num_faces();
                let self_num_face_vertices = self.num_face_vertices();
                let other_num_face_vertices = other.num_face_vertices();
                let self_num_face_edges = self.num_face_edges();
                let other_num_face_edges = other.num_face_edges();

                // Deconstruct the other mesh explicitly since it will not be valid as soon as we start to
                // canibalize its contents.
                let $base_type {
                    vertex_positions: mut other_vertex_positions,
                    indices: other_indices,
                    vertex_attributes: other_vertex_attributes,
                    face_attributes: other_face_attributes,
                    face_vertex_attributes: other_face_vertex_attributes,
                    face_edge_attributes: other_face_edge_attributes,
                    attribute_value_cache: other_attribute_value_cache,
                } = other;

                self.vertex_positions
                    .as_mut_vec()
                    .append(other_vertex_positions.as_mut_vec());
                self.indices.as_mut_vec().extend(other_indices.iter().map(
                    |&face| -> [usize; $verts_per_face] {
                        $vec::from(face).map(|i| i + self_num_vertices).into()
                    },
                ));

                // Transfer attributes
                merge_attribute_dicts(
                    &mut self.vertex_attributes,
                    self_num_vertices,
                    other_vertex_attributes,
                    other_num_vertices,
                );
                merge_attribute_dicts(
                    &mut self.face_attributes,
                    self_num_faces,
                    other_face_attributes,
                    other_num_faces,
                );
                merge_attribute_dicts(
                    &mut self.face_vertex_attributes,
                    self_num_face_vertices,
                    other_face_vertex_attributes,
                    other_num_face_vertices,
                );
                merge_attribute_dicts(
                    &mut self.face_edge_attributes,
                    self_num_face_edges,
                    other_face_edge_attributes,
                    other_num_face_edges,
                );
                for value in other_attribute_value_cache.into_iter() {
                    self.attribute_value_cache.insert(value);
                }
                self
            }
        }

        impl<T: Real> Merge for $mesh_type<T> {
            /// Attributes with the same name but different types won't be merged.
            fn merge(&mut self, other: Self) -> &mut Self {
                let self_num_faces = self.num_faces();
                let self_num_vertex_faces = self.num_vertex_faces();

                // Deconstruct the other mesh explicitly since it will not be valid as soon as we start to
                // canibalize its contents.
                let $mesh_type {
                    base_mesh: other_base_mesh,
                    face_offsets: other_face_offsets,
                    face_indices: other_face_indices,
                } = other;

                self.base_mesh.merge(other_base_mesh);

                self.face_offsets.extend(
                    other_face_offsets
                        .iter()
                        .skip(1)
                        .map(|&i| i + self_num_vertex_faces),
                );
                self.face_indices
                    .extend(other_face_indices.iter().map(|&i| i + self_num_faces));

                //merge_attribute_dicts(
                //    &mut self.vertex_face_attributes,
                //    self_num_vertex_faces,
                //    other_vertex_face_attributes,
                //    other_num_vertex_faces,
                //    );
                self
            }
        }

        impl<T: Real> $base_type<T> {
            /// Merge a iterator of meshes into a single distinct mesh.
            ///
            /// This version of `merge` accepts an attribute name for the source index on vertices.
            ///
            /// The mesh vertices will be merged in the order given by the source attribute. This
            /// is useful when merging previously split up meshes. The source attribute needs to
            /// have type `usize`.
            ///
            /// If the source attribute does not exist in at least one of the given meshes, then
            /// `None` is returned and the merge is aborted.
            ///
            /// This is a non-destructive merge --- both original meshes remain intact.
            /// This also means that this way of merging is somewhat more expensive than a merge
            /// without any source indices.
            pub fn merge_with_vertex_source<'a, I>(
                meshes: I,
                source_attrib: &str,
            ) -> Result<Self, Error>
            where
                I: IntoIterator<Item = &'a Self>,
            {
                Self::merge_with_vertex_source_impl(meshes.into_iter(), source_attrib)
            }

            fn merge_with_vertex_source_impl<'a>(
                mesh_iter: impl Iterator<Item = &'a Self>,
                source_attrib: &str,
            ) -> Result<Self, Error> {
                let mut vertex_positions = Vec::new();
                let mut indices = Vec::new();
                let mut vertex_attributes = AttribDict::new();
                let mut face_attributes = AttribDict::new();
                let mut face_vertex_attributes = AttribDict::new();
                let mut face_edge_attributes = AttribDict::new();
                let mut attribute_value_cache = AttribValueCache::default();
                let mut num_vertices = 0;

                for mesh in mesh_iter {
                    let src = mesh.attrib_as_slice::<usize, VertexIndex>(source_attrib)?;
                    if src.is_empty() {
                        // Empty mesh detected.
                        continue;
                    }

                    num_vertices = num_vertices.max(*src.iter().max().unwrap() + 1);

                    vertex_positions.resize(num_vertices, [T::zero(); 3]);
                    for (&i, &pos) in src.iter().zip(mesh.vertex_position_iter()) {
                        vertex_positions[i] = pos;
                    }

                    // Transfer attributes
                    merge_attribute_dicts_with_source(
                        &mut vertex_attributes,
                        num_vertices,
                        &mesh.vertex_attributes,
                        mesh.num_vertices(),
                        src,
                    );
                    merge_attribute_dicts(
                        &mut face_attributes,
                        indices.len(),
                        mesh.face_attributes.clone(),
                        mesh.num_faces(),
                    );
                    merge_attribute_dicts(
                        &mut face_vertex_attributes,
                        indices.len() * $verts_per_face,
                        mesh.face_vertex_attributes.clone(),
                        mesh.num_face_vertices(),
                    );
                    merge_attribute_dicts(
                        &mut face_edge_attributes,
                        indices.len() * $verts_per_face,
                        mesh.face_edge_attributes.clone(),
                        mesh.num_face_edges(),
                    );

                    for value in mesh.attribute_value_cache.iter() {
                        attribute_value_cache.insert(value.clone());
                    }

                    // Extend the indices AFTER attributes are transfered since
                    // `merge_attribute_dicts` expects num_elements to be the number before
                    // the merge.
                    indices.extend(mesh.face_iter().map(|&face| -> [usize; $verts_per_face] {
                        $vec::from(face).map(|i| src[i]).into()
                    }));
                }

                Ok(Self {
                    vertex_positions: IntrinsicAttribute::from_vec(vertex_positions),
                    indices: IntrinsicAttribute::from_vec(indices),
                    vertex_attributes,
                    face_attributes,
                    face_vertex_attributes,
                    face_edge_attributes,
                    attribute_value_cache,
                })
            }
        }
    };
}

impl_merge_for_uniform_mesh!(TriMeshExt, TriMesh, 3, Vector3);
impl_merge_for_uniform_mesh!(QuadMeshExt, QuadMesh, 4, Vector4);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::test_utils::*;
    use crate::mesh::TetMeshExt;

    fn build_tetmesh_sample() -> (TetMeshExt<f64>, TetMeshExt<f64>, TetMeshExt<f64>) {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ];

        // One connected component consisting of two tets connected at a face, and another
        // consisting of a single tet.
        let indices = vec![[0, 1, 2, 3], [8, 7, 4, 5], [6, 8, 4, 5]];

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

    fn add_attribs_to_tetmeshes(sample: &mut (TetMeshExt<f64>, TetMeshExt<f64>, TetMeshExt<f64>)) {
        // Add a sample vertex attribute.
        sample
            .0
            .add_attrib_data::<usize, VertexIndex>("v", (0..sample.0.num_vertices()).collect())
            .unwrap();
        sample
            .1
            .add_attrib_data::<usize, VertexIndex>("v", (0..4).collect())
            .unwrap();
        sample
            .2
            .add_attrib_data::<usize, VertexIndex>("v", (4..9).collect())
            .unwrap();

        // Add a sample cell attribute.
        sample
            .0
            .add_attrib_data::<usize, CellIndex>("c", (0..sample.0.num_cells()).collect())
            .unwrap();
        sample
            .1
            .add_attrib_data::<usize, CellIndex>("c", vec![0])
            .unwrap();
        sample
            .2
            .add_attrib_data::<usize, CellIndex>("c", vec![1, 2])
            .unwrap();

        // Add a sample cell vertex attribute.
        sample
            .0
            .add_attrib_data::<usize, CellVertexIndex>(
                "cv",
                (0..sample.0.num_cells() * 4).collect(),
            )
            .unwrap();
        sample
            .1
            .add_attrib_data::<usize, CellVertexIndex>("cv", (0..4).collect())
            .unwrap();
        sample
            .2
            .add_attrib_data::<usize, CellVertexIndex>("cv", (4..12).collect())
            .unwrap();

        // Add a sample cell face attribute.
        sample
            .0
            .add_attrib_data::<usize, CellFaceIndex>("cf", (0..sample.0.num_cells() * 4).collect())
            .unwrap();
        sample
            .1
            .add_attrib_data::<usize, CellFaceIndex>("cf", (0..4).collect())
            .unwrap();
        sample
            .2
            .add_attrib_data::<usize, CellFaceIndex>("cf", (4..12).collect())
            .unwrap();

        // Add a sample vertex cell attribute.
        sample
            .0
            .add_attrib_data::<usize, VertexCellIndex>(
                "vc",
                (0..sample.0.num_cells() * 4).collect(),
            )
            .unwrap();
        sample
            .1
            .add_attrib_data::<usize, VertexCellIndex>("vc", (0..4).collect())
            .unwrap();
        sample
            .2
            .add_attrib_data::<usize, VertexCellIndex>("vc", (4..12).collect())
            .unwrap();
    }

    #[test]
    fn tetmesh_merge() {
        // Generate sample meshes
        let mut sample = build_tetmesh_sample();

        // Add all the attributes
        add_attribs_to_tetmeshes(&mut sample);
        let (tetmesh, mut comp1, comp2) = sample;

        comp1.merge(comp2);
        assert_eq!(comp1, tetmesh);
    }

    #[test]
    fn tetmesh_merge_with_vertex_source() {
        // Generate sample meshes
        let mut sample = build_tetmesh_sample();

        // Add all the attributes
        add_attribs_to_tetmeshes(&mut sample);
        let (tetmesh, mut comp1, mut comp2) = sample;

        comp1
            .add_attrib_data::<usize, VertexIndex>("src", (0..comp1.num_vertices()).collect())
            .unwrap();
        comp2
            .add_attrib_data::<usize, VertexIndex>(
                "src",
                (comp1.num_vertices()..comp1.num_vertices() + comp2.num_vertices()).collect(),
            )
            .unwrap();

        let mut res = TetMeshExt::merge_with_vertex_source(&[comp1, comp2], "src").unwrap();
        res.remove_attrib::<VertexIndex>("src").unwrap();

        assert_eq!(res, tetmesh);
    }

    #[test]
    fn tetmesh_merge_with_vertex_source_conflicting_attributes() {
        // Generate sample meshes
        let mut sample = build_tetmesh_sample();

        // Add all the attributes
        add_attribs_to_tetmeshes(&mut sample);
        let (mut tetmesh, mut comp1, mut comp2) = sample;

        comp1
            .add_attrib_data::<usize, VertexIndex>("src", (0..comp1.num_vertices()).collect())
            .unwrap();
        comp2
            .add_attrib_data::<usize, VertexIndex>(
                "src",
                (comp1.num_vertices()..comp1.num_vertices() + comp2.num_vertices()).collect(),
            )
            .unwrap();

        // Add two attributes with same names but different types.
        comp1
            .add_attrib_data::<i32, VertexIndex>(
                "conflict",
                (0..comp1.num_vertices() as i32).collect(),
            )
            .unwrap();
        comp2
            .add_attrib_data::<u64, VertexIndex>(
                "conflict",
                (0..comp2.num_vertices() as u64).collect(),
            )
            .unwrap();

        let mut conflict_exp: Vec<i32> = (0..comp1.num_vertices() as i32).collect();

        // The second part of "conflict" from comp2 is ignored because it has the wrong type.
        conflict_exp.extend(std::iter::repeat(0).take(comp2.num_vertices()));

        tetmesh
            .add_attrib_data::<i32, VertexIndex>("conflict", conflict_exp)
            .unwrap();

        let mut res = TetMeshExt::merge_with_vertex_source(&[comp1, comp2], "src").unwrap();
        res.remove_attrib::<VertexIndex>("src").unwrap();

        assert_eq!(res, tetmesh);
    }

    #[test]
    fn tetmesh_merge_collection() {
        // Generate sample meshes
        let mut sample = build_tetmesh_sample();
        add_attribs_to_tetmeshes(&mut sample);
        let (mesh, comp1, comp2) = sample;

        let res = Merge::merge_vec(vec![comp1.clone(), comp2.clone()]);
        assert_eq!(res, mesh);

        let res = Merge::merge_slice(&[comp1, comp2]);
        assert_eq!(res, mesh);
    }

    #[test]
    fn polymesh_merge() {
        let mut sample = build_polymesh_sample();
        add_attribs_to_polymeshes(&mut sample);
        let (mesh, mut comp1, comp2) = sample;
        comp1.merge(comp2);
        assert_eq!(comp1, mesh);
    }

    #[test]
    fn polymesh_merge_collection() {
        let mut sample = build_polymesh_sample();
        add_attribs_to_polymeshes(&mut sample);
        let (mesh, comp1, comp2) = sample;
        let res = Merge::merge_vec(vec![comp1.clone(), comp2.clone()]);
        assert_eq!(res, mesh);

        let res = Merge::merge_slice(&[comp1, comp2]);
        assert_eq!(res, mesh);
    }

    #[test]
    fn trival_merges() {
        let sample = build_tetmesh_sample();

        // Merging with an empty mesh should return the same mesh back.
        assert_eq!(*TetMeshExt::default().merge(sample.0.clone()), sample.0);
        assert_eq!(*sample.0.clone().merge(TetMeshExt::default()), sample.0);

        // Do the same test with a polymesh
        let sample = build_polymesh_sample();

        assert_eq!(*PolyMesh::default().merge(sample.0.clone()), sample.0);
        assert_eq!(*sample.0.clone().merge(PolyMesh::default()), sample.0);
    }
}
