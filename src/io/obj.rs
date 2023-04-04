use std::collections::HashMap;

use objio::{self, Group, IndexTuple, Mtl, Object, SimplePolygon};

use crate::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::{PointCloud, PolyMesh, VertexPositions};
use crate::Real;

use super::{MeshExtractor, GROUP_ATTRIB_NAME, MTL_ATTRIB_NAME, OBJECT_ATTRIB_NAME};
use super::{NORMAL_ATTRIB_NAME, UV_ATTRIB_NAME};

pub use objio::ObjError;
pub use objio::{LoadConfig, Obj, ObjData, ObjMaterial};
pub use ordered_float::NotNan;

pub use super::Error;

#[allow(non_camel_case_types)]
type f32h = NotNan<f32>;

/// The model of a single Material as defined in the wavefront .mtl spec.
///
/// This is identical to the original `obj::Material` type with the exception that it implements `Eq` and `Hash` with the help of `ordered_float`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Material {
    pub name: String,

    // Material color and illumination
    pub ka: Option<[f32h; 3]>,
    pub kd: Option<[f32h; 3]>,
    pub ks: Option<[f32h; 3]>,
    pub ke: Option<[f32h; 3]>,
    pub km: Option<f32h>,
    pub tf: Option<[f32h; 3]>,
    pub ns: Option<f32h>,
    pub ni: Option<f32h>,
    pub tr: Option<f32h>,
    pub d: Option<f32h>,
    pub illum: Option<i32>,

    // Texture and reflection maps
    pub map_ka: Option<String>,
    pub map_kd: Option<String>,
    pub map_ks: Option<String>,
    pub map_ke: Option<String>,
    pub map_ns: Option<String>,
    pub map_d: Option<String>,
    pub map_bump: Option<String>,
    pub map_refl: Option<String>,
}

// Display the material as if defined in the .mtl format, which should be familiar to most users.
impl std::fmt::Display for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Reference:
        // newmtl spot
        // Ns 250.000000
        // Ka 1.000000 1.000000 1.000000
        // Ks 0.500000 0.500000 0.500000
        // Ke 0.000000 0.000000 0.000000
        // Ni 1.450000
        // d 1.000000
        // illum 2
        // map_Kd spot_texture.png

        writeln!(f, "newmtl {}", self.name)?;
        if let Some([a, b, c]) = self.ka {
            writeln!(f, "Ka {a} {b} {c}")?;
        }
        if let Some([a, b, c]) = self.kd {
            writeln!(f, "Kd {a} {b} {c}")?;
        }
        if let Some([a, b, c]) = self.ks {
            writeln!(f, "Ks {a} {b} {c}")?;
        }
        if let Some([a, b, c]) = self.ke {
            writeln!(f, "Ke {a} {b} {c}")?;
        }
        if let Some(km) = self.km {
            writeln!(f, "Km {km}")?;
        }
        if let Some([a, b, c]) = self.tf {
            writeln!(f, "Km {a} {b} {c}")?;
        }
        if let Some(x) = self.ns {
            writeln!(f, "Ns {x}")?;
        }
        if let Some(x) = self.ni {
            writeln!(f, "Ni {x}")?;
        }
        if let Some(x) = self.tr {
            writeln!(f, "Tr {x}")?;
        }
        if let Some(x) = self.d {
            writeln!(f, "d {x}")?;
        }
        if let Some(x) = self.illum {
            writeln!(f, "illum {x}")?;
        }
        if let Some(x) = self.map_ka.as_ref() {
            writeln!(f, "map_Ka {x}")?;
        }
        if let Some(x) = self.map_kd.as_ref() {
            writeln!(f, "map_Kd {x}")?;
        }
        if let Some(x) = self.map_ks.as_ref() {
            writeln!(f, "map_Ks {x}")?;
        }
        if let Some(x) = self.map_ke.as_ref() {
            writeln!(f, "map_Ke {x}")?;
        }
        if let Some(x) = self.map_ns.as_ref() {
            writeln!(f, "map_Ns {x}")?;
        }
        if let Some(x) = self.map_d.as_ref() {
            writeln!(f, "map_d {x}")?;
        }
        if let Some(x) = self.map_bump.as_ref() {
            writeln!(f, "map_bump {x}")?;
        }
        if let Some(x) = self.map_refl.as_ref() {
            writeln!(f, "map_refl {x}")?;
        }
        Ok(())
    }
}

fn new_option_f32h3(orig: Option<[f32; 3]>) -> Option<[f32h; 3]> {
    orig.and_then(|[a, b, c]| {
        if let Ok(a) = NotNan::new(a) {
            if let Ok(b) = NotNan::new(b) {
                if let Ok(c) = NotNan::new(c) {
                    return Some([a, b, c]);
                }
            }
        }
        return None;
    })
}

fn new_option_f32_3(orig: Option<[f32h; 3]>) -> Option<[f32; 3]> {
    orig.map(|[a, b, c]| [a.into(), b.into(), c.into()])
}

fn new_option_f32h(orig: Option<f32>) -> Option<f32h> {
    orig.and_then(|a| NotNan::new(a).ok())
}

fn new_option_f32(orig: Option<f32h>) -> Option<f32> {
    orig.map(|a| a.into())
}

impl From<objio::Material> for Material {
    fn from(v: objio::Material) -> Self {
        Material {
            name: v.name,
            ka: new_option_f32h3(v.ka),
            kd: new_option_f32h3(v.kd),
            ks: new_option_f32h3(v.ks),
            ke: new_option_f32h3(v.ke),
            km: new_option_f32h(v.km),
            tf: new_option_f32h3(v.tf),
            ns: new_option_f32h(v.ns),
            ni: new_option_f32h(v.ni),
            tr: new_option_f32h(v.tr),
            d: new_option_f32h(v.d),
            illum: v.illum,
            map_ka: v.map_ka,
            map_kd: v.map_kd,
            map_ks: v.map_ks,
            map_ke: v.map_ke,
            map_ns: v.map_ns,
            map_d: v.map_d,
            map_bump: v.map_bump,
            map_refl: v.map_refl,
        }
    }
}

impl From<Material> for objio::Material {
    fn from(v: Material) -> Self {
        objio::Material {
            name: v.name,
            ka: new_option_f32_3(v.ka),
            kd: new_option_f32_3(v.kd),
            ks: new_option_f32_3(v.ks),
            ke: new_option_f32_3(v.ke),
            km: new_option_f32(v.km),
            tf: new_option_f32_3(v.tf),
            ns: new_option_f32(v.ns),
            ni: new_option_f32(v.ni),
            tr: new_option_f32(v.tr),
            d: new_option_f32(v.d),
            illum: v.illum,
            map_ka: v.map_ka,
            map_kd: v.map_kd,
            map_ks: v.map_ks,
            map_ke: v.map_ke,
            map_ns: v.map_ns,
            map_d: v.map_d,
            map_bump: v.map_bump,
            map_refl: v.map_refl,
        }
    }
}

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

        // Add object and group names and materials
        let num_polys = polymesh.num_faces();
        let cache = &mut polymesh.attribute_value_cache;
        let mut object_names = IndirectData::with_size(num_polys, "default".to_string());
        let mut group_names = IndirectData::with_size(num_polys, "default".to_string());
        let mut mtls = None;

        let mut poly_count = 0;
        for object in &self.objects {
            let object_name = HValue::new(Irc::new(object.name.clone()));
            for group in &object.groups {
                let group_name = HValue::new(Irc::new(group.name.clone()));
                let mtl = group
                    .material
                    .as_ref()
                    .map(|mat| {
                        Material::from(match mat {
                            ObjMaterial::Ref(s) => objio::Material::new(s.clone()),
                            ObjMaterial::Mtl(s) => (**s).clone(),
                        })
                    })
                    .map(|m| HValue::new(Irc::new(m)));
                if mtl.is_some() && mtls.is_none() {
                    // Initialize materials
                    mtls = Some(IndirectData::with_size(
                        num_polys,
                        Material {
                            name: "default".into(),
                            ..Material::default()
                        },
                    ));
                }
                for _ in &group.polys {
                    object_names
                        .set_value_at(poly_count, &object_name, cache)
                        .unwrap();
                    group_names
                        .set_value_at(poly_count, &group_name, cache)
                        .unwrap();
                    if let Some(mtl) = mtl.as_ref() {
                        if let Some(mtls) = mtls.as_mut() {
                            mtls.set_value_at(poly_count, mtl, cache).unwrap();
                        }
                    }
                    poly_count += 1;
                }
            }
        }

        polymesh.insert_indirect_attrib_data::<FaceIndex>(OBJECT_ATTRIB_NAME, object_names)?;
        polymesh.insert_indirect_attrib_data::<FaceIndex>(GROUP_ATTRIB_NAME, group_names)?;
        if let Some(mtls) = mtls {
            polymesh.insert_indirect_attrib_data::<FaceIndex>(MTL_ATTRIB_NAME, mtls)?;
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

    // A generic name allows this obj to be saved easily.
    let mut material_lib = Mtl::new("material.mtl".to_string());

    // Object hierarchy.
    // Each object contains a set of groups, each group contains polygons. Each polygon can reference a unique material, which is collected into a material library.
    let mut objects = HashMap::new();
    let def_str = "default".to_string();

    // Iterator over materials (each wrapped with some) or an iterator over none, if there is no material attribute.
    let mtls = mesh
        .attrib_iter::<Material, FaceIndex>(MTL_ATTRIB_NAME)
        .map(|iter| iter.map(|mtl| Some(mtl)))
        .into_iter()
        .flatten()
        .chain(std::iter::repeat(None));
    let object_names = mesh
        .attrib::<FaceIndex>(OBJECT_ATTRIB_NAME)
        .unwrap() // No panic: Polymeshes have attribute value caches.
        .indirect_iter::<String>()
        .ok()
        .into_iter()
        .flatten()
        .chain(std::iter::repeat(&def_str));
    let group_names = mesh
        .attrib::<FaceIndex>(GROUP_ATTRIB_NAME)
        .unwrap() // No panic: Polymeshes have attribute value caches.
        .indirect_iter::<String>()
        .ok()
        .into_iter()
        .flatten()
        .chain(std::iter::repeat(&def_str));

    for (face_idx, (face, ((object_name, group_name), mtl))) in mesh
        .face_iter()
        .zip(object_names.zip(group_names).zip(mtls))
        .enumerate()
    {
        let object = objects.entry(object_name).or_insert(HashMap::new());
        let mut group = object.entry(group_name).or_insert(Group {
            name: group_name.to_string(),
            index: 0,
            material: None,
            polys: Vec::new(),
        });
        if group.material.is_none() {
            group.material = mtl.to_owned().map(|mtl| {
                let arc_mtl = std::sync::Arc::new(mtl.clone().into());
                material_lib.materials.push(std::sync::Arc::clone(&arc_mtl));
                ObjMaterial::Mtl(arc_mtl)
            })
        }
        group.polys.push(SimplePolygon(
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
        ));
    }

    Ok(ObjData {
        position,
        texture: uvs.map(|(v, _)| v).unwrap_or_else(Vec::new),
        normal: normals.map(|(v, _)| v).unwrap_or_else(Vec::new),
        objects: objects
            .into_iter()
            .map(|(name, obj)| Object {
                name: name.to_string(),
                groups: obj.into_values().collect::<Vec<_>>(),
            })
            .collect::<Vec<_>>(),
        material_libs: if material_lib.materials.is_empty() {
            Vec::new()
        } else {
            vec![material_lib]
        },
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

    #[test]
    fn roundtrip_obj_with_mtl_and_texture_test() {
        let mut obj_spot = Obj::load("./assets/spot.obj").unwrap();
        obj_spot.load_mtls().unwrap();

        let spot_polymesh: PolyMesh<f32> = obj_spot.data.extract_polymesh().unwrap();
        let mut attrib_iter = spot_polymesh
            .attrib_iter::<Material, FaceIndex>("mtl")
            .unwrap();
        for mtl in &mut attrib_iter {
            assert_eq!(
                mtl,
                &Material {
                    name: "spot".to_string(),
                    ka: Some([NotNan::new(1.0).unwrap(); 3]),
                    ks: Some([NotNan::new(0.5).unwrap(); 3]),
                    ke: Some([NotNan::new(0.0).unwrap(); 3]),
                    ns: Some(NotNan::new(250.0).unwrap()),
                    ni: Some(NotNan::new(1.45).unwrap()),
                    d: Some(NotNan::new(1.0).unwrap()),
                    illum: Some(2),
                    map_kd: Some("spot_texture.png".to_string()),
                    ..Default::default()
                }
            )
        }
        let obj_spot_roundtrip = convert_polymesh_to_obj_format(&spot_polymesh).unwrap();
        let spot_polymesh2: PolyMesh<f32> = obj_spot_roundtrip.extract_polymesh().unwrap();
        assert_eq!(spot_polymesh, spot_polymesh2);
    }
}
