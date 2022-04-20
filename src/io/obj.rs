use objio::{self, Group, IndexTuple, Object, SimplePolygon};

use crate::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::{PointCloud, PolyMesh, VertexPositions};
use crate::Real;

use super::MeshExtractor;
use super::{NORMAL_ATTRIB_NAME, UV_ATTRIB_NAME};

pub use objio::ObjError;
pub use objio::{LoadConfig, Obj, ObjData, ObjMaterial};

pub use super::Error;

enum TopologyType {
    Vertex,
    FaceVertex,
}

impl<T: Real> MeshExtractor<T> for ObjData {
    fn extract_polymesh(&self) -> Result<PolyMesh<T>, Error> {
        // Get points.
        let pts: Vec<[T; 3]> = self
            .position
            .iter()
            .map(|[a, b, c]| {
                [
                    T::from(*a).unwrap(),
                    T::from(*b).unwrap(),
                    T::from(*c).unwrap(),
                ]
            })
            .collect();

        let obj_texture = &self.texture;
        let obj_normal = &self.normal;

        // We use a simple algorithm to determine if the uvs and normals have the same topology as
        // vertex positions.
        let mut uv_topo_indices = vec![None; pts.len()];
        let mut nml_topo_indices = vec![None; pts.len()];

        // Assume vertex topology until we fail to match to vertices.
        let mut uv_topo = TopologyType::Vertex;
        let mut nml_topo = TopologyType::Vertex;

        let update_topo =
            |v_idx, topo_idx, topo: &mut TopologyType, topo_indices: &mut [Option<usize>]| {
                if matches!(*topo, TopologyType::Vertex) {
                    if let Some(topo_idx) = topo_idx {
                        if let Some(t_idx) = topo_indices[v_idx] {
                            if t_idx != topo_idx {
                                *topo = TopologyType::FaceVertex;
                            }
                        } else {
                            topo_indices[v_idx] = Some(topo_idx);
                        }
                    }
                }
            };

        // Get faces
        let mut faces = Vec::new();
        let mut uv_indices = Vec::new();
        let mut normal_indices = Vec::new();
        for object in &self.objects {
            for group in &object.groups {
                for poly in &group.polys {
                    faces.push(poly.0.len());
                    for idx in &poly.0 {
                        let v_idx = idx.0;
                        faces.push(v_idx);
                        update_topo(v_idx, idx.1, &mut uv_topo, &mut uv_topo_indices);
                        uv_indices.push(idx.1);
                        update_topo(v_idx, idx.2, &mut nml_topo, &mut nml_topo_indices);
                        normal_indices.push(idx.2);
                    }
                }
            }
        }

        let mut polymesh = PolyMesh::new(pts, &faces);

        // Get attributes:
        // Obj has 2D vertex uvs and vertex normals.

        if uv_topo_indices.iter().any(Option::is_some) {
            match uv_topo {
                TopologyType::Vertex => {
                    // Reorganize uvs to have the same order as vertices (this may be an identity mapping)
                    let mut vertex_uvs = vec![[0.0f32; 2]; polymesh.num_vertices()];
                    for (&uv_topo_idx, out_uv) in uv_topo_indices.iter().zip(vertex_uvs.iter_mut())
                    {
                        if let Some(uv_topo_idx) = uv_topo_idx {
                            *out_uv = obj_texture[uv_topo_idx];
                        }
                    }
                    polymesh.insert_attrib_data::<_, VertexIndex>(UV_ATTRIB_NAME, vertex_uvs)?;
                }
                TopologyType::FaceVertex => {
                    // We couldn't find a vertex correspondence, so write uvs to the face vertex topology.
                    let face_vertex_uvs: Vec<_> = uv_indices
                        .iter()
                        .map(|idx| idx.map(|idx| obj_texture[idx]).unwrap_or([0.0f32; 2]))
                        .collect();
                    polymesh.insert_attrib_data::<_, FaceVertexIndex>(
                        UV_ATTRIB_NAME,
                        face_vertex_uvs,
                    )?;
                }
            }
        }

        if nml_topo_indices.iter().any(Option::is_some) {
            match nml_topo {
                TopologyType::Vertex => {
                    // Reorganize uvs to have the same order as vertices (this may be an identity mappin)
                    let mut vertex_normals = vec![[0.0f32; 3]; polymesh.num_vertices()];
                    for (&nml_topo_idx, out_nml) in
                        nml_topo_indices.iter().zip(vertex_normals.iter_mut())
                    {
                        if let Some(nml_topo_idx) = nml_topo_idx {
                            *out_nml = obj_normal[nml_topo_idx];
                        }
                    }
                    polymesh
                        .insert_attrib_data::<_, VertexIndex>(NORMAL_ATTRIB_NAME, vertex_normals)?;
                }
                TopologyType::FaceVertex => {
                    // We couldn't find a vertex correspondence, so write normals to the face vertex topology.
                    let face_vertex_normals: Vec<_> = normal_indices
                        .iter()
                        .map(|idx| idx.map(|idx| obj_normal[idx]).unwrap_or([0.0f32; 3]))
                        .collect();
                    polymesh.insert_attrib_data::<_, FaceVertexIndex>(
                        NORMAL_ATTRIB_NAME,
                        face_vertex_normals,
                    )?;
                }
            }
        }

        // Add names
        let num_polys = polymesh.num_faces();
        let cache = &mut polymesh.attribute_value_cache;
        let mut object_names = IndirectData::with_size(num_polys, "default".to_string());
        let mut group_names = IndirectData::with_size(num_polys, "default".to_string());
        let mut mtl_names = None;

        let mut poly_count = 0;
        for object in &self.objects {
            let object_name = HValue::new(Irc::new(object.name.clone()));
            for group in &object.groups {
                let group_name = HValue::new(Irc::new(group.name.clone()));
                let mtl_name = group
                    .material
                    .as_ref()
                    .map(|mat| match mat {
                        ObjMaterial::Ref(s) => s.clone(),
                        ObjMaterial::Mtl(s) => s.name.clone(),
                    })
                    .map(|m| HValue::new(Irc::new(m)));
                if mtl_name.is_some() && mtl_names.is_none() {
                    // Initialize material names
                    mtl_names = Some(IndirectData::with_size(num_polys, "default".to_string()));
                }
                for _ in &group.polys {
                    object_names
                        .set_value_at(poly_count, &object_name, cache)
                        .unwrap();
                    group_names
                        .set_value_at(poly_count, &group_name, cache)
                        .unwrap();
                    if let Some(mtl) = mtl_name.as_ref() {
                        if let Some(names) = mtl_names.as_mut() {
                            names.set_value_at(poly_count, mtl, cache).unwrap();
                        }
                    }
                    poly_count += 1;
                }
            }
        }

        polymesh.insert_indirect_attrib_data::<FaceIndex>("object", object_names)?;
        polymesh.insert_indirect_attrib_data::<FaceIndex>("group", group_names)?;
        if let Some(mtl_names) = mtl_names {
            polymesh.insert_indirect_attrib_data::<FaceIndex>("mtl", mtl_names)?;
        }

        Ok(polymesh)
    }

    fn extract_pointcloud(&self) -> Result<PointCloud<T>, Error> {
        // Get points.
        let pts: Vec<[T; 3]> = self
            .position
            .iter()
            .map(|[a, b, c]| {
                [
                    T::from(*a).unwrap(),
                    T::from(*b).unwrap(),
                    T::from(*c).unwrap(),
                ]
            })
            .collect();

        let mut pointcloud = PointCloud::new(pts);

        // TODO: Check if there are faces, which could give a different normal and uv topology to point clouds

        // Get attributes:
        // Obj has 2D vertex uvs and vertex normals.

        if !self.texture.is_empty() {
            pointcloud
                .insert_attrib_data::<_, VertexIndex>(UV_ATTRIB_NAME, self.texture.clone())?;
        }

        if !self.normal.is_empty() {
            pointcloud
                .insert_attrib_data::<_, VertexIndex>(NORMAL_ATTRIB_NAME, self.normal.clone())?;
        }

        // TODO: Add names (see polymesh)

        Ok(pointcloud)
    }
}

pub fn convert_polymesh_to_obj_format<T: Real>(mesh: &PolyMesh<T>) -> Result<ObjData, Error> {
    let position: Vec<[f32; 3]> = mesh
        .vertex_position_iter()
        .cloned()
        .map(|[x, y, z]| {
            [
                x.to_f32().unwrap(),
                y.to_f32().unwrap(),
                z.to_f32().unwrap(),
            ]
        })
        .collect();

    // Find UVs
    let uvs: Option<(Vec<_>, TopologyType)> = if let Ok(uvs) =
        mesh.direct_attrib_clone_into_vec::<[f32; 2], VertexIndex>(UV_ATTRIB_NAME)
    {
        Some((uvs, TopologyType::Vertex))
    } else if let Ok(uvs) = mesh.attrib_iter::<[f32; 3], VertexIndex>(UV_ATTRIB_NAME) {
        Some((uvs.map(|&[u, v, _]| [u, v]).collect(), TopologyType::Vertex))
    } else if let Ok(uvs) = mesh.attrib_iter::<[f64; 2], VertexIndex>(UV_ATTRIB_NAME) {
        Some((
            uvs.map(|&[u, v]| [u as f32, v as f32]).collect(),
            TopologyType::Vertex,
        ))
    } else if let Ok(uvs) = mesh.attrib_iter::<[f64; 3], VertexIndex>(UV_ATTRIB_NAME) {
        Some((
            uvs.map(|&[u, v, _]| [u as f32, v as f32]).collect(),
            TopologyType::Vertex,
        ))
    } else if let Ok(uvs) =
        mesh.direct_attrib_clone_into_vec::<[f32; 2], FaceVertexIndex>(UV_ATTRIB_NAME)
    {
        Some((uvs, TopologyType::FaceVertex))
    } else if let Ok(uvs) = mesh.attrib_iter::<[f32; 3], FaceVertexIndex>(UV_ATTRIB_NAME) {
        Some((
            uvs.map(|&[u, v, _]| [u, v]).collect(),
            TopologyType::FaceVertex,
        ))
    } else if let Ok(uvs) = mesh.attrib_iter::<[f64; 2], FaceVertexIndex>(UV_ATTRIB_NAME) {
        Some((
            uvs.map(|&[u, v]| [u as f32, v as f32]).collect(),
            TopologyType::FaceVertex,
        ))
    } else if let Ok(uvs) = mesh.attrib_iter::<[f64; 3], FaceVertexIndex>(UV_ATTRIB_NAME) {
        Some((
            uvs.map(|&[u, v, _]| [u as f32, v as f32]).collect(),
            TopologyType::FaceVertex,
        ))
    } else {
        None
    };

    // Find normals
    let normals: Option<(Vec<_>, TopologyType)> = if let Ok(normals) =
        mesh.direct_attrib_clone_into_vec::<[f32; 3], VertexIndex>(NORMAL_ATTRIB_NAME)
    {
        Some((normals, TopologyType::Vertex))
    } else if let Ok(normals) = mesh.attrib_iter::<[f64; 3], VertexIndex>(NORMAL_ATTRIB_NAME) {
        Some((
            normals
                .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
                .collect(),
            TopologyType::Vertex,
        ))
    } else if let Ok(normals) =
        mesh.direct_attrib_clone_into_vec::<[f32; 3], FaceVertexIndex>(NORMAL_ATTRIB_NAME)
    {
        Some((normals, TopologyType::FaceVertex))
    } else if let Ok(normals) = mesh.attrib_iter::<[f64; 3], FaceVertexIndex>(NORMAL_ATTRIB_NAME) {
        Some((
            normals
                .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
                .collect(),
            TopologyType::FaceVertex,
        ))
    } else {
        None
    };

    let polys: Vec<SimplePolygon> = mesh
        .face_iter()
        .enumerate()
        .map(|(face_idx, face)| {
            SimplePolygon(
                face.iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let fv_idx = || face.len() * face_idx + i;
                        let uv = uvs.as_ref().map(|(_, topo_type)| match topo_type {
                            TopologyType::Vertex => v,
                            TopologyType::FaceVertex => fv_idx(),
                        });
                        let nv = normals.as_ref().map(|(_, topo_type)| match topo_type {
                            TopologyType::Vertex => v,
                            TopologyType::FaceVertex => fv_idx(),
                        });
                        IndexTuple(v, uv, nv)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    let find_mode_for_string_face_attrib = |attrib_name, def| {
        if let Ok(str_attrib) = mesh.attrib::<FaceIndex>(attrib_name) {
            let mut mode = Irc::new(def);
            let mut max_count = 1;
            if let AttributeData::Indirect(indirect_data) = &str_attrib.data {
                if let Ok(rc_iter) = indirect_data.as_rc_slice::<String>() {
                    for str_rc in rc_iter.iter() {
                        let cur_count = Irc::strong_count(str_rc);
                        if cur_count > max_count {
                            max_count = cur_count;
                            mode = Irc::clone(str_rc);
                        }
                    }
                }
            }
            (&*mode).clone()
        } else {
            def
        }
    };

    Ok(ObjData {
        position,
        texture: uvs.map(|(v, _)| v).unwrap_or_else(Vec::new),
        normal: normals.map(|(v, _)| v).unwrap_or_else(Vec::new),
        objects: vec![Object {
            name: find_mode_for_string_face_attrib("object", "default".to_string()),
            groups: vec![Group {
                name: find_mode_for_string_face_attrib("group", "default".to_string()),
                index: 0,
                material: None,
                polys,
            }],
        }],
        material_libs: Vec::new(),
    })
}

pub fn convert_pointcloud_to_obj_format<T: Real>(mesh: &PointCloud<T>) -> Result<ObjData, Error> {
    let position: Vec<[f32; 3]> = mesh
        .vertex_position_iter()
        .cloned()
        .map(|[x, y, z]| {
            [
                x.to_f32().unwrap(),
                y.to_f32().unwrap(),
                z.to_f32().unwrap(),
            ]
        })
        .collect();

    // Find UVs
    let uvs: Vec<_> = if let Ok(uvs) =
        mesh.direct_attrib_clone_into_vec::<[f32; 2], VertexIndex>(UV_ATTRIB_NAME)
    {
        uvs
    } else if let Ok(uvs) = mesh.attrib_iter::<[f32; 3], VertexIndex>(UV_ATTRIB_NAME) {
        uvs.map(|&[u, v, _]| [u, v]).collect()
    } else if let Ok(uvs) = mesh.attrib_iter::<[f64; 2], VertexIndex>(UV_ATTRIB_NAME) {
        uvs.map(|&[u, v]| [u as f32, v as f32]).collect()
    } else if let Ok(uvs) = mesh.attrib_iter::<[f64; 3], VertexIndex>(UV_ATTRIB_NAME) {
        uvs.map(|&[u, v, _]| [u as f32, v as f32]).collect()
    } else {
        Vec::new()
    };

    // Find normals
    let normals: Vec<_> = if let Ok(normals) =
        mesh.direct_attrib_clone_into_vec::<[f32; 3], VertexIndex>(NORMAL_ATTRIB_NAME)
    {
        normals
    } else if let Ok(normals) = mesh.attrib_iter::<[f64; 3], VertexIndex>(NORMAL_ATTRIB_NAME) {
        normals
            .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
            .collect()
    } else {
        Vec::new()
    };

    // For compatibility with other loaders, we also emit explicit single vertex polygons.
    let polys: Vec<SimplePolygon> = (0..mesh.num_vertices())
        .map(|vtx_idx| {
            SimplePolygon(vec![IndexTuple(
                vtx_idx,
                if uvs.is_empty() { None } else { Some(vtx_idx) },
                if normals.is_empty() {
                    None
                } else {
                    Some(vtx_idx)
                },
            )])
        })
        .collect();

    Ok(ObjData {
        position,
        texture: uvs,
        normal: normals,
        objects: vec![Object {
            name: "default".to_string(),
            groups: vec![objio::Group {
                name: "default".to_string(),
                index: 0,
                material: None,
                polys,
            }],
        }],
        material_libs: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_positions() -> Vec<[f32; 3]> {
        vec![
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ]
    }

    fn make_uvs() -> Vec<[f32; 2]> {
        vec![
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    }

    fn make_normals() -> Vec<[f32; 3]> {
        vec![
            [0.577350318, -0.577350318, 0.577350318],
            [-0.577350318, -0.577350318, 0.577350318],
            [0.577350318, 0.577350318, 0.577350318],
            [-0.577350318, 0.577350318, 0.577350318],
            [-0.577350318, -0.577350318, -0.577350318],
            [0.577350318, -0.577350318, -0.577350318],
            [-0.577350318, 0.577350318, -0.577350318],
            [0.577350318, 0.577350318, -0.577350318],
        ]
    }

    fn make_polygons() -> Vec<Vec<usize>> {
        vec![
            vec![0, 2, 3, 1],
            vec![4, 6, 7, 5],
            vec![6, 3, 2, 7],
            vec![5, 0, 1, 4],
            vec![5, 7, 2, 0],
            vec![1, 3, 6, 4],
        ]
    }

    /// An obj with only vertices can be represented by a pointcloud.
    fn obj_pointcloud_example() -> ObjData {
        ObjData {
            position: make_positions(),
            texture: make_uvs(),
            normal: make_normals(),
            objects: Vec::new(),
            material_libs: Vec::new(),
        }
    }

    fn pointcloud_example() -> PointCloud<f32> {
        let pts = make_positions();
        let mut ptcloud = PointCloud::new(pts);

        let uvs = make_uvs();
        let normals = make_normals();

        ptcloud
            .insert_attrib_data::<_, VertexIndex>(UV_ATTRIB_NAME, uvs)
            .ok()
            .unwrap();

        ptcloud
            .insert_attrib_data::<_, VertexIndex>(NORMAL_ATTRIB_NAME, normals)
            .ok()
            .unwrap();

        ptcloud
    }

    fn obj_polymesh_example() -> ObjData {
        let pos = make_positions();
        let normals = make_normals();
        let uvs = make_uvs();
        let polys = make_polygons()
            .into_iter()
            .map(|poly| {
                SimplePolygon(
                    poly.into_iter()
                        .map(|v| IndexTuple(v, Some(v), Some(v)))
                        .collect(),
                )
            })
            .collect();

        ObjData {
            position: pos,
            texture: uvs,
            normal: normals,
            objects: vec![Object {
                name: "default".to_string(),
                groups: vec![Group {
                    name: "default".to_string(),
                    index: 0,
                    material: None,
                    polys,
                }],
            }],
            material_libs: Vec::new(),
        }
    }

    fn polymesh_example() -> PolyMesh<f32> {
        let pts = make_positions();
        let faces = make_polygons();
        let faces_flat: Vec<_> = faces
            .into_iter()
            .flat_map(|poly| std::iter::once(poly.len()).chain(poly.into_iter()))
            .collect();
        let mut polymesh = PolyMesh::new(pts, &faces_flat);

        let uvs = make_uvs();
        let normals = make_normals();

        polymesh
            .insert_attrib_data::<_, VertexIndex>(UV_ATTRIB_NAME, uvs)
            .ok()
            .unwrap();

        polymesh
            .insert_attrib_data::<_, VertexIndex>(NORMAL_ATTRIB_NAME, normals)
            .ok()
            .unwrap();

        polymesh
            .insert_indirect_attrib::<_, FaceIndex>("object", "default".to_string())
            .ok()
            .unwrap();

        polymesh
            .insert_indirect_attrib::<_, FaceIndex>("group", "default".to_string())
            .ok()
            .unwrap();

        polymesh
    }

    #[test]
    fn obj_to_polymesh_test() -> Result<(), Error> {
        let obj = obj_polymesh_example();
        let polymesh = polymesh_example();

        let converted_polymesh = obj.extract_polymesh()?;
        assert_eq!(converted_polymesh, polymesh);
        Ok(())
    }

    #[test]
    fn obj_to_pointcloud_test() {
        let obj = obj_pointcloud_example();
        let polymesh = pointcloud_example();

        let converted_polymesh = obj.extract_pointcloud().unwrap();
        assert_eq!(converted_polymesh, polymesh);
    }

    #[test]
    fn roundtrip_obj_polymesh_test() {
        let obj = obj_polymesh_example();
        let polymesh = polymesh_example();

        let converted_polymesh = obj.extract_polymesh().unwrap();
        assert_eq!(converted_polymesh.clone(), polymesh);
        let converted_obj = convert_polymesh_to_obj_format(&converted_polymesh).unwrap();
        assert_eq!(converted_obj, obj);

        // This one is a sanity check to make sure nothing has been corrupted in the underlying
        // Datastructures, and the Eq implementation works as expected.
        let converted_polymesh = converted_obj.extract_polymesh().unwrap();
        assert_eq!(converted_polymesh, polymesh);
    }

    #[test]
    fn roundtrip_obj_pointcloud_test() {
        let obj = obj_pointcloud_example();
        let ptcloud = pointcloud_example();

        let converted_ptcloud = obj.extract_pointcloud().unwrap();
        assert_eq!(converted_ptcloud.clone(), ptcloud);
        let converted_obj = convert_pointcloud_to_obj_format(&converted_ptcloud).unwrap();

        // The objs will be different since we didnt include faces for points in the original.
        // So we do one extra conversion to do that round trip.

        let converted_ptcloud = converted_obj.extract_pointcloud().unwrap();
        assert_eq!(converted_ptcloud.clone(), ptcloud);
        let converted_obj_2 = convert_pointcloud_to_obj_format(&converted_ptcloud).unwrap();
        assert_eq!(converted_obj_2, converted_obj);
    }
}
