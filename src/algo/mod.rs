//pub mod intersections;
pub mod connectivity;
pub mod merge;
pub mod normals;
pub mod partition;
pub mod split;

pub use self::connectivity::*;
pub use self::merge::*;
pub use self::normals::*;
pub use self::partition::*;
pub use self::split::*;

/// Useful utilities for testing algorithms in this module.
#[cfg(test)]
pub(crate) mod test_utils {
    use crate::mesh::{attrib::*, topology::*};
    type PolyMesh = crate::mesh::PolyMesh<f64>;

    pub(crate) fn build_polymesh_sample() -> (PolyMesh, PolyMesh, PolyMesh) {
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

        // One connected component consisting of two triangles connected at an edge, and another
        // consisting of a single quad.
        let indices = vec![3, 0, 1, 2, 3, 2, 1, 3, 4, 4, 5, 7, 6];

        let mesh = PolyMesh::new(verts, &indices);
        let comp1 = PolyMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            &[3, 0, 1, 2, 3, 2, 1, 3],
        );
        let comp2 = PolyMesh::new(
            vec![
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            &[4, 0, 1, 3, 2],
        );
        (mesh, comp1, comp2)
    }

    pub(crate) fn add_vertex_attrib_to_polymeshes(sample: &mut (PolyMesh, PolyMesh, PolyMesh)) {
        sample
            .0
            .add_attrib_data::<usize, VertexIndex>("v", (0..sample.0.num_vertices()).collect())
            .unwrap();
        sample
            .1
            .add_attrib_data::<usize, VertexIndex>("v", vec![0, 1, 2, 3])
            .unwrap();
        sample
            .2
            .add_attrib_data::<usize, VertexIndex>("v", vec![4, 5, 6, 7])
            .unwrap();
    }

    pub(crate) fn add_face_attrib_to_polymeshes(sample: &mut (PolyMesh, PolyMesh, PolyMesh)) {
        sample
            .0
            .add_attrib_data::<usize, FaceIndex>("f", (0..sample.0.num_faces()).collect())
            .unwrap();
        sample
            .1
            .add_attrib_data::<usize, FaceIndex>("f", vec![0, 1])
            .unwrap();
        sample
            .2
            .add_attrib_data::<usize, FaceIndex>("f", vec![2])
            .unwrap();
    }

    pub(crate) fn add_face_vertex_attrib_to_polymeshes(
        sample: &mut (PolyMesh, PolyMesh, PolyMesh),
    ) {
        sample
            .0
            .add_attrib_data::<usize, FaceVertexIndex>(
                "fv",
                (0..sample.0.num_face_vertices()).collect(),
            )
            .unwrap();
        sample
            .1
            .add_attrib_data::<usize, FaceVertexIndex>("fv", vec![0, 1, 2, 3, 4, 5])
            .unwrap();
        sample
            .2
            .add_attrib_data::<usize, FaceVertexIndex>("fv", vec![6, 7, 8, 9])
            .unwrap();
    }

    pub(crate) fn add_face_edge_attrib_to_polymeshes(sample: &mut (PolyMesh, PolyMesh, PolyMesh)) {
        sample
            .0
            .add_attrib_data::<usize, FaceEdgeIndex>("fe", (0..sample.0.num_face_edges()).collect())
            .unwrap();
        sample
            .1
            .add_attrib_data::<usize, FaceEdgeIndex>("fe", vec![0, 1, 2, 3, 4, 5])
            .unwrap();
        sample
            .2
            .add_attrib_data::<usize, FaceEdgeIndex>("fe", vec![6, 7, 8, 9])
            .unwrap();
    }

    pub(crate) fn add_attribs_to_polymeshes(sample: &mut (PolyMesh, PolyMesh, PolyMesh)) {
        add_vertex_attrib_to_polymeshes(sample);
        add_face_attrib_to_polymeshes(sample);
        add_face_vertex_attrib_to_polymeshes(sample);
        add_face_edge_attrib_to_polymeshes(sample);
    }
}

#[cfg(test)]
mod tests {
    use super::SplitIntoConnectedComponents;
    use crate::mesh::{attrib::*, topology::*};
    type TriMesh = crate::mesh::TriMesh<f64>;

    // Verify that merging works for two meshes even in different orders.
    #[test]
    fn trimesh_split_and_merge() {
        // Construct three triangles with interleaved vertices.
        let verts = vec![
            [0.0, 0.0, 0.0],  // tri 0
            [0.0, 0.0, 1.0],  // tri 1
            [1.0, 0.0, 0.0],  // tri 0
            [1.0, 0.0, 1.0],  // tri 1
            [1.0, 1.0, 1.0],  // tri 1
            [1.0, -1.0, 1.0], // tri 1
            [1.0, 1.0, 0.0],  // tri 0
        ];

        let indices = vec![[0, 2, 4], [1, 3, 5], [1, 4, 3]];
        let mut mesh = TriMesh::new(verts, indices);

        // This also serves as the source index.
        mesh.add_attrib_data::<usize, VertexIndex>("v", (0..mesh.num_vertices()).collect())
            .unwrap();

        mesh.add_attrib_data::<usize, FaceIndex>("f", (0..mesh.num_faces()).collect())
            .unwrap();

        mesh.add_attrib_data::<usize, FaceVertexIndex>(
            "fv",
            (0..mesh.num_face_vertices()).collect(),
        )
        .unwrap();
        mesh.add_attrib_data::<usize, FaceEdgeIndex>("fe", (0..mesh.num_face_edges()).collect())
            .unwrap();

        let parts = mesh.clone().split_into_connected_components();
        let merged = TriMesh::merge_with_vertex_source(&parts, "v").unwrap();
        assert_eq!(mesh, merged);
    }
}
