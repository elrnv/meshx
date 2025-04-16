fn main() {
    /*
     * Building meshes
     */
    use meshx::TriMesh;

    // Four corners of a [0,0]x[1,1] square in the xy-plane.
    let vertices = vec![
        [0.0f64, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];

    // Two triangles making up a square.
    let indices = vec![[0, 1, 2], [1, 3, 2]];

    // Create the triangle mesh
    let mesh = TriMesh::new(vertices, indices);

    use meshx::topology::*;

    /*
     * Simple count queries
     */

    println!("number of vertices: {}", mesh.num_vertices());
    println!("number of faces: {}", mesh.num_faces());
    println!("number of face-edges: {}", mesh.num_face_edges());
    println!("number of face-vertices: {}", mesh.num_face_vertices());

    /*
     * Topology queries
     */

    // Get triangle indices:
    println!("face 1: {:?}", mesh.face(1));

    // Face-edge topology:
    println!(
        "face-edge index of edge 2 on face 1: {:?}",
        mesh.face_edge(1, 2)
    );
    println!("edge index of face-edge 5: {:?}", mesh.edge(5));
    println!(
        "edge index of edge 2 on face 1: {:?}",
        mesh.face_to_edge(1, 2)
    );

    // Face-vertex topology:
    println!(
        "face-vertex index of vertex 1 on face 0: {:?}",
        mesh.face_vertex(0, 1)
    );
    println!("vertex index of face-vertex 1: {:?}", mesh.vertex(1));
    println!(
        "vertex index of vertex 1 on face 0: {:?}",
        mesh.face_vertex(0, 1)
    );

    /*
     * Attributes
     */

    use meshx::attrib::*;

    let mut mesh = mesh; // Make mesh mutable.

    let vectors = vec![
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];

    mesh.insert_attrib_data::<[f32; 3], VertexIndex>("up_and_away", vectors)
        .expect("Failed to insert 'up_and_away' attribute");

    /*
     * IO: loading and saving meshes
     */

    #[cfg(feature = "io")]
    {
        meshx::io::save_trimesh(&mesh, "tests/artifacts/tutorial_trimesh.vtk")
            .expect("Failed to save the tutorial triangle mesh");

        let loaded_mesh = meshx::io::load_trimesh("tests/artifacts/tutorial_trimesh.vtk")
            .expect("Failed to load the tutorial triangle mesh");

        assert_eq!(mesh, loaded_mesh);
    }
}
