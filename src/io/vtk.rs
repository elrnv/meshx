use crate::algo::merge::Merge;
use crate::attrib::{Attrib, AttribDict, AttribIndex, Attribute, AttributeValue};
use crate::mesh::topology::*;
use crate::mesh::{CellType, Mesh, PointCloud, PolyMesh, TetMesh, VertexPositions};
use ahash::HashSet;
use flatk::{
    consts::{U10, U11, U12, U13, U14, U15, U16, U2, U3, U4, U5, U6, U7, U8, U9},
    U,
};

use super::MeshExtractor;
use super::Real;
use super::{NORMAL_ATTRIB_NAME, UV_ATTRIB_NAME};

pub use vtkio::Error as VtkError;
pub use vtkio::*;

pub use super::Error;

/// The name of the Field Data in vtk files used to store face vertex attributes.
/// This is used to add support for things like face vertex attributes in vtk files as recommended
/// by the Vtk file format documentation.
const FACE_VERTEX_ATTRIBUTES_FIELD: &str = "face_vertex_attributes";

// TODO: Add support for cell vertex attributes in the same way.
//const CELL_VERTEX_ATTRIBUTES_FIELD: &'static str = "cell_vertex_attributes";

/// Populate populate this function with special field attribute names to be ignored when
/// transferring (reading) field attributes onto primary element topologies.
fn special_field_attributes() -> &'static [&'static str] {
    &[FACE_VERTEX_ATTRIBUTES_FIELD]
}

/// An enum indicating how polygon data should be exported in VTK format.
///
/// Polygon data can be represented by the designated `PolyData` VTK data set type or the more
/// general `UnstructuredGrid` type.
///
/// Note that both styles are supported in XML as well as Legacy VTK files.
pub enum VTKPolyExportStyle {
    /// Use `PolyData` VTK type for exporting polygons.
    PolyData,
    /// Use `UnstructuredGrid` VTK type for exporting polygons.
    UnstructuredGrid,
}

pub fn convert_mesh_to_vtk_format<T: Real>(mesh: &Mesh<T>) -> Result<model::Vtk, Error> {
    let points: Vec<T> = mesh
        .vertex_positions()
        .iter()
        .flat_map(|x| x.iter().cloned())
        .collect();
    let mut vertices = Vec::new();
    for cell in mesh.cell_iter() {
        vertices.push(cell.len() as u32);
        for &vtx in cell.iter() {
            vertices.push(vtx as u32);
        }
    }

    let cell_types: Vec<_> = mesh
        .cell_type_iter()
        .map(|cell_type| match cell_type {
            CellType::Line => model::CellType::Line,
            CellType::Tetrahedron => model::CellType::Tetra,
            CellType::Triangle => model::CellType::Triangle,
            CellType::Pyramid => model::CellType::Pyramid,
            CellType::Hexahedron => model::CellType::Hexahedron,
            CellType::Wedge => model::CellType::Wedge,
            CellType::Quad => model::CellType::Quad,
        })
        .collect();

    let point_attribs = mesh
        .attrib_dict::<VertexIndex>()
        .iter()
        .filter_map(|(name, attrib)| mesh_to_vtk_named_attrib(name, attrib))
        .collect();

    let cell_attribs = mesh
        .attrib_dict::<CellIndex>()
        .iter()
        .filter_map(|(name, attrib)| mesh_to_vtk_named_attrib(name, attrib))
        .collect();

    Ok(model::Vtk {
        version: model::Version::new((0, 1)),
        title: String::from("Unstructured Mesh"),
        byte_order: model::ByteOrder::BigEndian,
        file_path: None,
        data: model::DataSet::inline(model::UnstructuredGridPiece {
            points: points.into(),
            cells: model::Cells {
                cell_verts: model::VertexNumbers::Legacy {
                    num_cells: mesh.num_cells() as u32,
                    vertices,
                },
                types: cell_types,
            },
            data: model::Attributes {
                point: point_attribs,
                cell: cell_attribs,
            },
        }),
    })
}

pub fn convert_polymesh_to_vtk_format<T: Real>(
    mesh: &PolyMesh<T>,
    style: VTKPolyExportStyle,
) -> Result<model::Vtk, Error> {
    let points: Vec<T> = mesh
        .vertex_positions()
        .iter()
        .flat_map(|x| x.iter().cloned())
        .collect();
    let mut vertices = Vec::new();
    for face in mesh.face_iter() {
        vertices.push(face.len() as u32);
        for &vtx in face.iter() {
            vertices.push(vtx as u32);
        }
    }

    let point_attribs = mesh
        .attrib_dict::<VertexIndex>()
        .iter()
        .filter_map(|(name, attrib)| mesh_to_vtk_named_attrib(name, attrib))
        .collect();

    let face_attribs = mesh
        .attrib_dict::<FaceIndex>()
        .iter()
        .filter_map(|(name, attrib)| mesh_to_vtk_named_attrib(name, attrib))
        .chain(
            // In addition to Face attributes, we can load face vertex attributes into a Vtk file
            // via Vtk FIELD attributes.
            mesh_to_vtk_named_field_attribs(
                FACE_VERTEX_ATTRIBUTES_FIELD,
                mesh.attrib_dict::<FaceVertexIndex>(),
            ),
        )
        .collect();

    Ok(model::Vtk {
        version: model::Version::new((0, 1)),
        title: String::from("Polygonal Mesh"),
        byte_order: model::ByteOrder::BigEndian,
        file_path: None,
        data: match style {
            VTKPolyExportStyle::UnstructuredGrid => {
                model::DataSet::inline(model::UnstructuredGridPiece {
                    points: IOBuffer::new(points),
                    cells: model::Cells {
                        cell_verts: model::VertexNumbers::Legacy {
                            num_cells: mesh.num_faces() as u32,
                            vertices,
                        },
                        types: vec![model::CellType::Polygon; mesh.num_faces()],
                    },
                    data: model::Attributes {
                        point: point_attribs,
                        cell: face_attribs,
                    },
                })
            }
            VTKPolyExportStyle::PolyData => model::DataSet::inline(model::PolyDataPiece {
                points: IOBuffer::new(points),
                polys: Some(model::VertexNumbers::Legacy {
                    num_cells: mesh.num_faces() as u32,
                    vertices,
                }),
                data: model::Attributes {
                    point: point_attribs,
                    cell: face_attribs,
                },
                ..Default::default()
            }),
        },
    })
}

pub fn convert_tetmesh_to_vtk_format<T: Real>(tetmesh: &TetMesh<T>) -> Result<model::Vtk, Error> {
    let points: Vec<T> = tetmesh
        .vertex_positions()
        .iter()
        .flat_map(|x| x.iter().cloned())
        .collect();
    let mut vertices = Vec::new();
    for cell in tetmesh.cell_iter() {
        vertices.push(cell.len() as u32);
        for &vtx in cell.iter() {
            vertices.push(vtx as u32);
        }
    }

    let point_attribs = tetmesh
        .attrib_dict::<VertexIndex>()
        .iter()
        .filter_map(|(name, attrib)| mesh_to_vtk_named_attrib(name, attrib))
        .collect();

    let cell_attribs = tetmesh
        .attrib_dict::<CellIndex>()
        .iter()
        .filter_map(|(name, attrib)| mesh_to_vtk_named_attrib(name, attrib))
        .collect();

    Ok(model::Vtk {
        version: model::Version::new((0, 1)),
        title: String::from("Tetrahedral Mesh"),
        byte_order: model::ByteOrder::BigEndian,
        file_path: None,
        data: model::DataSet::inline(model::UnstructuredGridPiece {
            points: points.into(),
            cells: model::Cells {
                cell_verts: model::VertexNumbers::Legacy {
                    num_cells: tetmesh.num_cells() as u32,
                    vertices,
                },
                types: vec![model::CellType::Tetra; tetmesh.num_cells()],
            },
            data: model::Attributes {
                point: point_attribs,
                cell: cell_attribs,
            },
        }),
    })
}

pub fn convert_pointcloud_to_vtk_format<T: Real>(
    ptcloud: &PointCloud<T>,
    style: VTKPolyExportStyle,
) -> Result<model::Vtk, Error> {
    let num_verts = ptcloud.num_vertices() as u32;
    let points: Vec<T> = ptcloud
        .vertex_positions()
        .iter()
        .flat_map(|x| x.iter().cloned())
        .collect();

    let point_attribs = ptcloud
        .attrib_dict::<VertexIndex>()
        .iter()
        .filter_map(|(name, attrib)| mesh_to_vtk_named_attrib(name, attrib))
        .collect();

    Ok(model::Vtk {
        version: model::Version::new((0, 1)),
        title: String::from("Point Cloud"),
        byte_order: model::ByteOrder::BigEndian,
        file_path: None,
        data: match style {
            VTKPolyExportStyle::PolyData => {
                model::DataSet::inline(model::PolyDataPiece {
                    points: IOBuffer::new(points),
                    // A single VERTICES entry containing all points
                    verts: Some(model::VertexNumbers::Legacy {
                        num_cells: 1,
                        vertices: std::iter::once(num_verts)
                            .chain(0..num_verts)
                            .collect::<Vec<_>>(),
                    }),
                    data: model::Attributes {
                        point: point_attribs,
                        cell: Vec::new(),
                    },
                    ..Default::default()
                })
            }
            VTKPolyExportStyle::UnstructuredGrid => {
                model::DataSet::inline(model::UnstructuredGridPiece {
                    points: IOBuffer::new(points),
                    cells: model::Cells {
                        cell_verts: model::VertexNumbers::Legacy {
                            num_cells: 1,
                            vertices: std::iter::once(num_verts)
                                .chain(0..num_verts)
                                .collect::<Vec<_>>(),
                        },
                        types: vec![model::CellType::Vertex; ptcloud.num_vertices()],
                    },
                    data: model::Attributes {
                        point: point_attribs,
                        cell: Vec::new(),
                    },
                })
            }
        },
    })
}

// TODO: Refactor the functions below to reuse code.
impl<T: Real> MeshExtractor<T> for model::Vtk {
    /// Constructs an unstructured Mesh from this VTK model.
    ///
    /// This function will clone the given model as necessary.
    fn extract_mesh(&self) -> Result<Mesh<T>, Error> {
        let model::Vtk {
            file_path, data, ..
        } = &self;
        let mut cell_type_list = HashSet::<usize>::default();
        match data {
            model::DataSet::UnstructuredGrid { pieces, .. } => {
                let mesh = Mesh::merge_iter(pieces.iter().filter_map(|piece| {
                    let model::UnstructuredGridPiece {
                        points,
                        cells: model::Cells { cell_verts, types },
                        data,
                    } = piece
                        .load_piece_data(file_path.as_ref().map(AsRef::as_ref))
                        .ok()?;
                    // Get points.
                    let pt_coords: Vec<T> = points.cast_into()?;
                    let mut pts = Vec::with_capacity(pt_coords.len() / 3);
                    for coords in pt_coords.chunks_exact(3) {
                        pts.push([coords[0], coords[1], coords[2]]);
                    }

                    let num_cells = cell_verts.num_cells();
                    let (connectivity, offsets) = cell_verts.into_xml();

                    // Mapping to original topology. This is used when the vtk file has elements
                    // not supported by our Mesh.
                    let mut orig_cell_idx = Vec::with_capacity(num_cells);

                    // Get contiguous indices (4 vertex indices for each tet or 3 for triangles).
                    let mut begin = 0usize;
                    let mut indices = Vec::new();
                    let mut counts = Vec::new();
                    let mut cell_types = Vec::new();
                    for (c, &end) in offsets.iter().enumerate() {
                        let n = end as usize - begin;
                        let cell_type = match types[c] {
                            model::CellType::Triangle if n == 3 => CellType::Triangle,
                            model::CellType::Tetra if n == 4 => CellType::Tetrahedron,
                            model::CellType::Pyramid if n == 5 => CellType::Pyramid,
                            model::CellType::Hexahedron if n == 8 => CellType::Hexahedron,
                            model::CellType::Wedge if n == 6 => CellType::Wedge,
                            _ => {
                                cell_type_list.insert(types[c] as usize);
                                // Not a valid cell type, skip it.
                                begin = end as usize;
                                continue;
                            }
                        };

                        if cell_types.is_empty() || *cell_types.last().unwrap() != cell_type {
                            // Start a new block.
                            cell_types.push(cell_type);
                            counts.push(1);
                        } else if let Some(last) = counts.last_mut() {
                            *last += 1;
                        } else {
                            // Bug in the code. Counts must have the same size as cell_types.
                            return None;
                        }

                        orig_cell_idx.push(c);
                        for i in 0..n {
                            indices.push(connectivity[begin + i] as usize);
                        }
                        begin = end as usize;
                    }

                    let mut mesh =
                        Mesh::from_cells_counts_and_types(pts, indices, counts, cell_types);

                    // Don't bother transferring attributes if there are no vertices or cells.
                    // This supresses some needless size mismatch warnings when the dataset has an
                    // unstructured grid representing something other than a recognizable Mesh.

                    if mesh.num_vertices() > 0 {
                        // Populate point attributes.
                        vtk_to_mesh_attrib::<_, VertexIndex>(data.point, &mut mesh, None);
                    }

                    if mesh.num_cells() > 0 {
                        // Populate tet attributes
                        vtk_to_mesh_attrib::<_, CellIndex>(
                            data.cell,
                            &mut mesh,
                            Some(orig_cell_idx.as_slice()),
                        );
                    }

                    Some(mesh)
                }));
                if cell_type_list.len() > 0 {
                    use num_traits::FromPrimitive;
                    return Err(Error::UnsupportedCellTypes(
                        cell_type_list
                            .into_iter()
                            .map(|c| model::CellType::from_usize(c).unwrap())
                            .collect::<Vec<model::CellType>>(),
                    ));
                } else {
                    Ok(mesh)
                }
            }
            _ => Err(Error::UnsupportedDataFormat),
        }
    }
    /// Constructs a PolyMesh from this VTK model.
    ///
    /// This function will clone the given model as necessary.
    fn extract_polymesh(&self) -> Result<PolyMesh<T>, Error> {
        let model::Vtk {
            file_path, data, ..
        } = &self;
        match data {
            model::DataSet::UnstructuredGrid { pieces, .. } => {
                let mesh = PolyMesh::merge_iter(pieces.iter().filter_map(|piece| {
                    let model::UnstructuredGridPiece {
                        points,
                        cells: model::Cells { cell_verts, types },
                        data,
                    } = piece
                        .load_piece_data(file_path.as_ref().map(AsRef::as_ref))
                        .ok()?;
                    // Get points.
                    let pt_coords: Vec<T> = points.cast_into()?; // None is returned in case of overflow.
                    let mut pts = Vec::with_capacity(pt_coords.len() / 3);
                    for coords in pt_coords.chunks_exact(3) {
                        pts.push([coords[0], coords[1], coords[2]]);
                    }

                    // We use this counter to determine if the vtk file should actually be parsed as a
                    // different type of mesh.
                    let mut count_non_polymesh_faces = 0;

                    let num_cells = cell_verts.num_cells();
                    // TODO: it would be more efficient to have an implementation for xml and legacy formats separately.
                    let (connectivity, offsets) = cell_verts.into_xml();

                    // Mappings to original topology. This is used when the topology must be modified when converting to PolyMesh.
                    let mut orig_cell_idx = Vec::with_capacity(num_cells);
                    let mut orig_cell_vtx_idx = Vec::with_capacity(connectivity.len());

                    let mut begin = 0usize;
                    let mut faces = Vec::new();
                    for c in 0..num_cells {
                        let end = offsets[c] as usize;
                        let n = end - begin;

                        // Skip geometry we can't represent as a polygon mesh.
                        let skip = match types[c] {
                            model::CellType::Line => n != 1,
                            model::CellType::Triangle => n != 3,
                            model::CellType::Quad => n != 4,
                            model::CellType::Polygon | model::CellType::PolyLine => false,
                            _ => true,
                        };
                        if skip {
                            count_non_polymesh_faces += 1;
                            begin = end;
                            continue;
                        }

                        if types[c] == model::CellType::PolyLine {
                            // Each polyline is broken into multiple 2-vertex line segments.
                            for i in begin..end - 1 {
                                orig_cell_idx.push(c);
                                orig_cell_vtx_idx.push(i - begin);
                                orig_cell_vtx_idx.push(i + 1 - begin);
                                faces.push(2);
                                faces.push(connectivity[i] as usize);
                                faces.push(connectivity[i + 1] as usize);
                            }
                        } else {
                            orig_cell_idx.push(c);
                            faces.push(n);
                            orig_cell_vtx_idx.extend(0..end - begin);
                            faces.extend(connectivity[begin..end].iter().map(|&i| i as usize));
                        }

                        begin = end;
                    }

                    if faces.is_empty() && count_non_polymesh_faces > 0 {
                        // This should be parsed as another type of mesh.
                        // We opt to not interpret it as a point cloud mesh, which would be the case if
                        // there were no other types of faces.
                        return None;
                    } // Otherwise we output what we have found, whether or not some other faces were ignored.

                    let mut polymesh = PolyMesh::new(pts, &faces);

                    // Populate point attributes.
                    vtk_to_mesh_attrib::<_, VertexIndex>(data.point, &mut polymesh, None);

                    // Populate face attributes.
                    let remainder = vtk_to_mesh_attrib::<_, FaceIndex>(
                        data.cell,
                        &mut polymesh,
                        Some(orig_cell_idx.as_slice()),
                    );

                    // Populate face vertex attributes.
                    vtk_field_to_mesh_attrib(
                        remainder,
                        &mut polymesh,
                        Some(orig_cell_vtx_idx.as_slice()),
                    );

                    Some(polymesh)
                }));

                // If we try to build a polymesh out of an unstructured grid with no polygon data,
                // interpret this as an error, since it is most likely unintentional.
                // Clients could also very well be relying on this behaviour when trying to interpret a
                // vtk file.
                if mesh.num_faces() == 0 {
                    Err(Error::MeshTypeMismatch)
                } else {
                    Ok(mesh)
                }
            }
            model::DataSet::PolyData { pieces, .. } => {
                Ok(PolyMesh::merge_iter(pieces.iter().filter_map(|piece| {
                    let model::PolyDataPiece {
                        points,
                        lines,
                        polys,
                        strips,
                        // PolyData with vertices is best represented by a PointCloud
                        data,
                        ..
                    } = piece
                        .load_piece_data(file_path.as_ref().map(AsRef::as_ref))
                        .ok()?;
                    // Get points.
                    let pt_coords: Vec<T> = points.cast_into()?; // None is returned in case of overflow.
                    let mut pts = Vec::with_capacity(pt_coords.len() / 3);
                    for coords in pt_coords.chunks_exact(3) {
                        pts.push([coords[0], coords[1], coords[2]]);
                    }

                    // Mappings to original topology. This is used when the topology must be modified when converting to PolyMesh.
                    let mut cell_idx_map = Vec::new();
                    let mut cell_vtx_idx_map = Vec::new();

                    let mut num_faces = 0;

                    let mut faces = Vec::new();

                    let mut append_topo = |topo: model::VertexNumbers| {
                        cell_idx_map.extend(num_faces..num_faces + topo.num_cells());
                        num_faces += topo.num_cells();
                        let (connectivity, offsets) = topo.into_xml();
                        let mut begin = 0;
                        for &offset in offsets.iter() {
                            let end = offset as usize;
                            cell_vtx_idx_map.extend(begin..end);
                            faces.push(end - begin);
                            faces.extend(connectivity[begin..end].iter().map(|&i| i as usize));
                            begin = end;
                        }
                    };

                    polys.map(&mut append_topo);
                    strips.map(&mut append_topo);

                    if let Some(topo) = lines {
                        let (connectivity, offsets) = topo.into_xml();
                        let mut begin = 0;
                        for (orig_face_idx, &offset) in offsets.iter().enumerate() {
                            // Each polyline is broken into multiple 2-vertex line segments.
                            for i in begin..offset as usize - 1 {
                                cell_idx_map.push(num_faces + orig_face_idx);
                                cell_vtx_idx_map.push(i - begin);
                                cell_vtx_idx_map.push(i + 1 - begin);
                                faces.push(2);
                                faces.push(connectivity[i] as usize);
                                faces.push(connectivity[i + 1] as usize);
                            }
                            begin = offset as usize;
                        }
                    }

                    let mut polymesh = PolyMesh::new(pts, &faces);

                    // Populate point attributes.
                    vtk_to_mesh_attrib::<_, VertexIndex>(data.point, &mut polymesh, None);

                    // Populate face attributes
                    let remainder = vtk_to_mesh_attrib::<_, FaceIndex>(
                        data.cell,
                        &mut polymesh,
                        Some(cell_idx_map.as_slice()),
                    );

                    // Populate face vertex attributes.
                    vtk_field_to_mesh_attrib(
                        remainder,
                        &mut polymesh,
                        Some(cell_vtx_idx_map.as_slice()),
                    );

                    Some(polymesh)
                })))
            }
            _ => Err(Error::UnsupportedDataFormat),
        }
    }

    /// Constructs a TetMesh from this VTK model.
    ///
    /// This function will clone the given model as necessary.
    fn extract_tetmesh(&self) -> Result<TetMesh<T>, Error> {
        let model::Vtk {
            file_path, data, ..
        } = &self;
        match data {
            model::DataSet::UnstructuredGrid { pieces, .. } => {
                Ok(TetMesh::merge_iter(pieces.iter().filter_map(|piece| {
                    let model::UnstructuredGridPiece {
                        points,
                        cells: model::Cells { cell_verts, types },
                        data,
                    } = piece
                        .load_piece_data(file_path.as_ref().map(AsRef::as_ref))
                        .ok()?;
                    // Get points.
                    let pt_coords: Vec<T> = points.cast_into()?;
                    let mut pts = Vec::with_capacity(pt_coords.len() / 3);
                    for coords in pt_coords.chunks_exact(3) {
                        pts.push([coords[0], coords[1], coords[2]]);
                    }

                    let num_cells = cell_verts.num_cells();
                    let (connectivity, offsets) = cell_verts.into_xml();

                    // Mapping to original topology. This is used when the vtk file has elements other than tets.
                    let mut orig_cell_idx = Vec::with_capacity(num_cells);

                    // Get contiguous indices (4 vertex indices for each tet).
                    let mut begin = 0usize;
                    let mut indices = Vec::new();
                    for (c, &end) in offsets.iter().enumerate() {
                        let n = end as usize - begin;

                        if n != 4 || types[c] != model::CellType::Tetra {
                            // Not a tetrahedron, skip it.
                            begin = end as usize;
                            continue;
                        }

                        orig_cell_idx.push(c);
                        indices.push([
                            connectivity[begin] as usize,
                            connectivity[begin + 1] as usize,
                            connectivity[begin + 2] as usize,
                            connectivity[begin + 3] as usize,
                        ]);
                        begin = end as usize;
                    }

                    let mut tetmesh = TetMesh::new(pts, indices);

                    // Don't bother transferring attributes if there are no vertices or cells.
                    // This supresses some needless size mismatch warnings when the dataset has an
                    // unstructuredgrid representing something other than a tetmesh.

                    if tetmesh.num_vertices() > 0 {
                        // Populate point attributes.
                        vtk_to_mesh_attrib::<_, VertexIndex>(data.point, &mut tetmesh, None);
                    }

                    if tetmesh.num_cells() > 0 {
                        // Populate tet attributes
                        vtk_to_mesh_attrib::<_, CellIndex>(
                            data.cell,
                            &mut tetmesh,
                            Some(orig_cell_idx.as_slice()),
                        );
                    }

                    Some(tetmesh)
                })))
            }
            _ => Err(Error::UnsupportedDataFormat),
        }
    }

    /// Constructs a PointCloud from this VTK model.
    ///
    /// This function will clone the given model as necessary.
    fn extract_pointcloud(&self) -> Result<PointCloud<T>, Error> {
        let model::Vtk {
            file_path, data, ..
        } = &self;
        let mut pts = Vec::new();
        let mut vertices = Vec::new();
        match data {
            model::DataSet::UnstructuredGrid { pieces, .. } => {
                let ptcloud = PointCloud::merge_iter(pieces.iter().filter_map(|piece| {
                    let model::UnstructuredGridPiece {
                        points,
                        cells: model::Cells { cell_verts, types },
                        data,
                    } = piece
                        .load_piece_data(file_path.as_ref().map(AsRef::as_ref))
                        .ok()?;

                    pts.clear();
                    vertices.clear();

                    // Get points.
                    let pt_coords: Vec<T> = points.cast_into()?;
                    pts.reserve(pt_coords.len() / 3);
                    for coords in pt_coords.chunks_exact(3) {
                        pts.push([coords[0], coords[1], coords[2]]);
                    }

                    // We use this counter to determine if the vtk file should actually be parsed as a
                    // different type of mesh.
                    let mut count_non_vertex_cells = 0;

                    let (num_cells, cell_vertices) = cell_verts.into_legacy();

                    let mut i = 0usize;
                    for c in 0..num_cells {
                        if i >= cell_vertices.len() {
                            break;
                        }

                        let n = cell_vertices[i] as usize;
                        // Skip geometry we can't represent as a point cloud.
                        if types[c as usize] == model::CellType::Vertex {
                            if n != 1 {
                                i += n + 1;
                                count_non_vertex_cells += 1;
                                continue;
                            }
                        } else if types[c as usize] != model::CellType::PolyVertex {
                            i += n + 1;
                            count_non_vertex_cells += 1;
                            continue;
                        }

                        i += 1; // Skipping the size of the cell

                        for _ in 0..=n {
                            vertices.push(cell_vertices[i] as usize);
                            i += 1;
                        }
                    }

                    if vertices.is_empty() && count_non_vertex_cells > 0 {
                        // This should be parsed as another type of mesh.
                        // We opt to not interpret it as a point cloud mesh, which would be the case if
                        // there were no other types of faces.
                        return None;
                    } // Otherwise we output what we have found, whether or not some other faces were ignored.
                    let referenced_points: Vec<_> = vertices.iter().map(|&vtx| pts[vtx]).collect();
                    let mut pointcloud = PointCloud::new(referenced_points);

                    // Populate point attributes.
                    vtk_to_mesh_attrib::<_, VertexIndex>(data.point, &mut pointcloud, None);

                    Some(pointcloud)
                }));

                // If we try to build a point cloud out of an unstructured grid with no vertex objects,
                // interpret this as an error, since it is most likely unintentional.
                // Clients could also very well be relying on this behaviour when trying to interpret a
                // vtk file.
                if ptcloud.num_vertices() == 0 {
                    Err(Error::MeshTypeMismatch)
                } else {
                    Ok(ptcloud)
                }
            }
            model::DataSet::PolyData { pieces, .. } => {
                Ok(PointCloud::merge_iter(pieces.iter().filter_map(|piece| {
                    let model::PolyDataPiece {
                        points,
                        verts,
                        data,
                        ..
                    } = piece.load_piece_data(None).ok()?;
                    pts.clear();
                    vertices.clear();

                    // Get points.
                    let pt_coords: Vec<T> = points.cast_into()?;
                    pts.reserve(pt_coords.len() / 3);
                    for coords in pt_coords.chunks_exact(3) {
                        pts.push([coords[0], coords[1], coords[2]]);
                    }

                    // If vertex topology is given use it, otherwise load all vertices.
                    let mut ptcloud = if let Some(topo) = verts {
                        let (_, cell_vertices) = topo.into_legacy();
                        vertices.extend(cell_vertices.into_iter().skip(1).map(|x| x as usize));
                        let referenced_points: Vec<_> =
                            vertices.iter().map(|&vtx| pts[vtx]).collect();
                        PointCloud::new(referenced_points)
                    } else {
                        PointCloud::new(pts.clone())
                    };

                    // Populate point attributes.
                    vtk_to_mesh_attrib::<_, VertexIndex>(data.point, &mut ptcloud, None);

                    Some(ptcloud)
                })))
            }
            _ => Err(Error::UnsupportedDataFormat),
        }
    }
}

fn flatten2<T: Clone>(vec: Vec<[T; 2]>) -> Vec<T> {
    vec.iter().flat_map(|x| x.iter().cloned()).collect()
}
fn flatten3<T: Clone>(vec: Vec<[T; 3]>) -> Vec<T> {
    vec.iter().flat_map(|x| x.iter().cloned()).collect()
}
fn flatten4<T: Clone>(vec: Vec<[T; 4]>) -> Vec<T> {
    vec.iter().flat_map(|x| x.iter().cloned()).collect()
}
fn flatten33<T: Clone>(vec: Vec<[[T; 3]; 3]>) -> Vec<T> {
    vec.iter()
        .flat_map(|x| x.iter().flat_map(|y| y.iter().cloned()))
        .collect()
}

/// Transfer a `uv` attribute from this attribute to the `vtk` model.
fn into_vtk_attrib_uv<I>(name: &str, attrib: &Attribute<I>) -> Option<model::Attribute> {
    // Try 2d texture coordinates
    let mut maybe_iobuf = attrib
        .direct_clone_into_vec::<[f32; 2]>()
        .map(|y| IOBuffer::from(flatten2(y)));
    if maybe_iobuf.is_err() {
        // try with f64
        maybe_iobuf = attrib
            .direct_clone_into_vec::<[f64; 2]>()
            .map(|y| IOBuffer::from(flatten2(y)));
    }

    if let Ok(data) = maybe_iobuf {
        return Some(model::Attribute::tcoords(name, 2).with_data(data));
    }

    // Try 3d texture coordinates
    maybe_iobuf = attrib
        .direct_clone_into_vec::<[f32; 3]>()
        .map(|y| flatten3(y).into());
    if maybe_iobuf.is_err() {
        // try with f64
        maybe_iobuf = attrib
            .direct_clone_into_vec::<[f64; 3]>()
            .map(|y| flatten3(y).into());
    }

    if let Ok(data) = maybe_iobuf {
        return Some(model::Attribute::tcoords(name, 3).with_data(data));
    }

    None
}

macro_rules! try_interpret_attrib {
    (@direct $attrib:ident, $t:ident, $f:expr) => {
            $attrib.direct_clone_into_vec::<$t>().map($f)
    };
    (@build $attrib:ident, ($t:ident $($ts:ident)*), $f:expr) => {
        $attrib.direct_clone_into_vec::<$t>().map($f)
        $(
            .or_else(|_| try_interpret_attrib!(@direct $attrib, $ts, $f))
        )*
    };
    (@direct $attrib:ident, $n:expr, $t:ident, $f:expr) => {
            $attrib.direct_clone_into_vec::<[$t; $n]>().map($f)
    };
    (@build $attrib:ident, $n:expr, ($t:ident $($ts:ident)*), $f:expr) => {
        $attrib.direct_clone_into_vec::<[$t; $n]>().map($f)
        $(
            .or_else(|_| try_interpret_attrib!(@direct $attrib, $n, $ts, $f))
        )*
    };
    (@direct $attrib:ident, $n:expr, $m:expr, $t:ident, $f:expr) => {
            $attrib.direct_clone_into_vec::<[[$t; $n]; $m]>().map($f)
    };
    (@build $attrib:ident, $n:expr, $m:expr, ($t:ident $($ts:ident)*), $f:expr) => {
        $attrib.direct_clone_into_vec::<[[$t; $n]; $m]>().map($f)
        $(
            .or_else(|_| try_interpret_attrib!(@direct $attrib, $n, $m, $ts, $f))
        )*
    };
    ($attrib:ident, $f:expr) => {
        {
            try_interpret_attrib!(@build $attrib, (u8 i8 u16 i16 u32 i32 u64 i64 f32 f64), $f)
        }
    };
    ($attrib:ident, $n:expr, $f:expr) => {
        {
            try_interpret_attrib!(@build $attrib, $n, (u8 i8 u16 i16 u32 i32 u64 i64 f32 f64), $f)
        }
    };
    ($attrib:ident, $n:expr, $m:expr, $f:expr) => {
        {
            try_interpret_attrib!(@build $attrib, $n, $m, (u8 i8 u16 i16 u32 i32 u64 i64 f32 f64), $f)
        }
    }
}

macro_rules! try_interpret_generic_attrib {
    ($attrib:ident, $name:ident $(,$n:expr)*) => {
        {
            $(
                if let Ok(data) = try_interpret_attrib!($attrib, $n, |x| IOBuffer::from(
                        x.iter().flat_map(|x| x.iter().cloned()).collect::<Vec<_>>()
                )) {
                    return Some(model::Attribute::generic($name, $n).with_data(data));
                }
            )*
        }
    }
}

fn mesh_to_vtk_attrib_impl<I>(name: &str, attrib: &Attribute<I>) -> Option<model::Attribute> {
    // Try to match a scalar field.
    if let Ok(data) = try_interpret_attrib!(attrib, IOBuffer::from) {
        return Some(model::Attribute::scalars(name, 1).with_data(data));
    }

    // Try to match a vector field.
    if let Ok(data) = try_interpret_attrib!(attrib, 2, |x| IOBuffer::from(flatten2(x))) {
        return Some(model::Attribute::scalars(name, 2).with_data(data));
    }

    // Try to match a vector field.
    if let Ok(data) = try_interpret_attrib!(attrib, 3, |x| IOBuffer::from(flatten3(x))) {
        return Some(model::Attribute::vectors(name).with_data(data));
    }

    // Try to match a vector field.
    if let Ok(data) = try_interpret_attrib!(attrib, 4, |x| IOBuffer::from(flatten4(x))) {
        return Some(model::Attribute::scalars(name, 4).with_data(data));
    }

    // Try to match a tensor field.
    if let Ok(data) = try_interpret_attrib!(attrib, 3, 3, |x| IOBuffer::from(flatten33(x))) {
        return Some(model::Attribute::tensors(name).with_data(data));
    }

    // Try to match a generic field for any size up to 16.
    try_interpret_generic_attrib!(attrib, name, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    None
}

fn mesh_to_vtk_named_attrib<I>(name: &str, attrib: &Attribute<I>) -> Option<model::Attribute> {
    // Try to match special attributes
    if name == UV_ATTRIB_NAME {
        let attrib = into_vtk_attrib_uv(name, attrib);
        if attrib.is_some() {
            return attrib;
        }
    } else if name == NORMAL_ATTRIB_NAME {
        let mut maybe_iobuf: Result<IOBuffer, _> = attrib
            .direct_clone_into_vec::<[f32; 3]>()
            .map(|y| flatten3(y).into());
        if maybe_iobuf.is_err() {
            // try with f64
            maybe_iobuf = attrib
                .direct_clone_into_vec::<[f64; 3]>()
                .map(|y| flatten3(y).into());
        }

        if let Ok(data) = maybe_iobuf {
            return Some(model::Attribute::normals(name).with_data(data));
        }
    }

    // Match with other vtk attributes.
    mesh_to_vtk_attrib_impl(name, attrib)
}

/// Transfer attribute data from `attrib_dict` to a vtk FIELD attribute. This is useful for storing
/// attributes for topologies that Vtk doesn't directly support like `FaceVertex` or `CellVertex`
/// attributes which are important for passing through texture coordinates with seams.
fn mesh_to_vtk_named_field_attribs<I>(
    field_data_name: &str,
    attrib_dict: &AttribDict<I>,
) -> Option<model::Attribute> {
    let data_array: Vec<_> = attrib_dict
        .iter()
        .filter_map(|(name, attrib)| {
            // Try to match a scalar field.
            if let Ok(data) = try_interpret_attrib!(attrib, IOBuffer::from) {
                return Some(model::FieldArray::new(name, 1).with_data(data));
            }

            // Try to match a 2D vector field.
            if let Ok(data) = try_interpret_attrib!(attrib, 2, |x| IOBuffer::from(flatten2(x))) {
                return Some(model::FieldArray::new(name, 2).with_data(data));
            }

            // Try to match a 3D vector field.
            if let Ok(data) = try_interpret_attrib!(attrib, 3, |x| IOBuffer::from(flatten3(x))) {
                return Some(model::FieldArray::new(name, 3).with_data(data));
            }

            None
        })
        .collect();

    if !data_array.is_empty() {
        Some(model::Attribute::field(field_data_name).with_field_data(data_array))
    } else {
        None
    }
}

fn insert_2d_array_attrib<'a, T, M, I>(
    buf: &[T],
    name: &'a str,
    mesh: &mut M,
    remap: Option<&[usize]>,
) -> Result<(), Error>
where
    T: AttributeValue + Copy + Default,
    I: AttribIndex<M>,
    M: Attrib,
{
    let n = 9;
    let mut vecs = Vec::with_capacity(buf.len() / n);
    let mut count_comp = 0;
    let mut cur = [[T::default(); 3]; 3];
    let mut push_val = |val| {
        cur[count_comp / 3][count_comp % 3] = val; // row-major -> col-major
        count_comp += 1;
        if count_comp == n {
            vecs.push(cur);
            count_comp = 0;
        }
    };
    if let Some(remap) = remap {
        remap.iter().for_each(|&i| push_val(buf[i]));
    } else {
        buf.iter().cloned().for_each(push_val);
    }
    mesh.insert_attrib_data::<_, I>(name, vecs)?;
    Ok(())
}

fn insert_array_attrib<'a, T, M, I>(
    buf: &[T],
    name: &'a str,
    mesh: &mut M,
    remap: Option<&[usize]>,
) -> Result<(), Error>
where
    T: AttributeValue + Default,
    I: AttribIndex<M>,
    M: Attrib,
{
    let remapped_buf = if let Some(remap) = remap {
        remap.iter().map(|&i| buf[i].clone()).collect()
    } else {
        buf.to_vec()
    };
    mesh.insert_attrib_data::<_, I>(name, remapped_buf)?;
    Ok(())
}

fn insert_array_attrib_n<'a, T, M, I: AttribIndex<M>, N>(
    buf: &[T],
    name: &'a str,
    mesh: &mut M,
    remap: Option<&[usize]>,
) -> Result<(), Error>
where
    T: bytemuck::Pod + AttributeValue + Default,
    M: Attrib,
    N: flatk::Unsigned + Default + flatk::Array<T>,
    <N as flatk::Array<T>>::Array: Default + PartialEq + std::fmt::Debug + Send + Sync,
{
    let remapped_buf = if let Some(remap) = remap {
        remap
            .iter()
            .flat_map(|&i| (0..N::to_usize()).map(move |j| buf[N::to_usize() * i + j]))
            .collect()
    } else {
        buf.to_vec()
    };
    let chunked = flatk::UniChunked::<_, U<N>>::from_flat(remapped_buf);
    mesh.insert_attrib_data::<_, I>(name, chunked.into_arrays())?;
    Ok(())
}

/// Adds VTK attributes to the given mesh, and returns any unprocessed attributes that can be
/// processed further.
///
/// If the reason an attribute is not processed is because it
/// has an unsupported type, we leave it out of the remainder.
#[allow(clippy::cognitive_complexity)]
fn vtk_to_mesh_attrib<M, I>(
    attribs: Vec<model::Attribute>,
    mesh: &mut M,
    orig_map: Option<&[usize]>,
) -> Vec<model::Attribute>
where
    M: Attrib,
    I: AttribIndex<M>,
{
    // We populate another vector instead of using filter_map to allow for errors to propagate.
    let mut remainder = Vec::with_capacity(attribs.len());

    for attrib in attribs {
        match attrib {
            model::Attribute::DataArray(model::DataArray { name, elem, data }) => {
                let name = name.as_str();
                match elem {
                    model::ElementType::Scalars { num_comp: dim, .. } | model::ElementType::TCoords(dim) => {
                        match dim {
                            // Note that only the first found attribute with the same name and location
                            // will be inserted.
                            1 => match_buf!( &data, v => insert_array_attrib::<_,M,I>(v, name, mesh, orig_map) ),
                            2 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U2>(v, name, mesh, orig_map) ),
                            3 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U3>(v, name, mesh, orig_map) ),
                            4 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U4>(v, name, mesh, orig_map) ),
                            // Other values for dim are not supported by the vtk standard
                            // at the time of this writing.
                             _ => continue,
                        }
                    }
                    model::ElementType::Vectors | model::ElementType::Normals => {
                        match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U3>(v, name, mesh, orig_map) )
                    }
                    model::ElementType::Tensors => {
                        match_buf!( &data, v => insert_2d_array_attrib::<_,M,I>(v, name, mesh, orig_map) )
                    }
                    model::ElementType::Generic(dim) => {
                        match dim {
                            1 => match_buf!( &data, v => insert_array_attrib::<_,M,I>(v, name, mesh, orig_map) ),
                            2 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U2>(v, name, mesh, orig_map) ),
                            3 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U3>(v, name, mesh, orig_map) ),
                            4 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U4>(v, name, mesh, orig_map) ),
                            5 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U5>(v, name, mesh, orig_map) ),
                            6 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U6>(v, name, mesh, orig_map) ),
                            7 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U7>(v, name, mesh, orig_map) ),
                            8 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U8>(v, name, mesh, orig_map) ),
                            9 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U9>(v, name, mesh, orig_map) ),
                            10 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U10>(v, name, mesh, orig_map) ),
                            11 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U11>(v, name, mesh, orig_map) ),
                            12 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U12>(v, name, mesh, orig_map) ),
                            13 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U13>(v, name, mesh, orig_map) ),
                            14 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U14>(v, name, mesh, orig_map) ),
                            15 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U15>(v, name, mesh, orig_map) ),
                            16 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U16>(v, name, mesh, orig_map) ),
                            _ => continue,
                        }
                    }
                    _ => continue, // LookupTable and ColorScalars attributes ignored
                }
            }
            model::Attribute::Field { data_array, name } => {
                if special_field_attributes().contains(&name.as_str()) {
                    remainder.push(model::Attribute::Field { name, data_array });
                    continue;
                }
                for model::FieldArray {
                    name,
                    elem,
                    data,
                } in data_array
                {
                    let name = name.as_str();
                    // Field attributes do not necessarily have the right size. We check it here.
                    match elem {
                        // Note that only the first found attribute with the same name and location
                        // will be inserted.
                        1 => match_buf!( &data, v => insert_array_attrib::<_,M,I>(v, name, mesh, orig_map) ),
                        2 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U2>(v, name, mesh, orig_map) ),
                        3 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U3>(v, name, mesh, orig_map) ),
                        4 => match_buf!( &data, v => insert_array_attrib_n::<_,M,I,U4>(v, name, mesh, orig_map) ),
                        _ => continue,
                    }
                    .unwrap_or_else(|err| eprintln!("WARNING: Field attribute transfer error: {}", err));
                }
                continue;
            }
        }
        // Attribute transfer might fail, but we shouldn't stop trying the rest of the attributes.
        // Simply issue a warning and continue;
        .unwrap_or_else(|err| {
            #[cfg(feature = "unstable")]
            {
                eprintln!("WARNING: Attribute transfer error at {}: {}", std::intrinsics::type_name::<I>(), err)
            }
            #[cfg(not(feature = "unstable"))]
            {
                eprintln!("WARNING: Attribute transfer error: {}", err)
            }
        })
    }
    remainder
}

/// Populate face vertex attributes from field attributes.
#[allow(clippy::cognitive_complexity)]
fn vtk_field_to_mesh_attrib<M>(
    attribs: Vec<model::Attribute>,
    mesh: &mut M,
    orig_map: Option<&[usize]>,
) where
    M: Attrib + FaceVertex,
    FaceVertexIndex: AttribIndex<M>,
{
    for attrib in attribs {
        if !special_field_attributes().contains(&attrib.name()) {
            continue;
        }
        if let model::Attribute::Field { data_array, .. } = attrib {
            for model::FieldArray { name, elem, data } in data_array {
                let name = name.as_str();
                match elem {
                    // Note that only the first found attribute with the same name and location
                    // will be inserted.
                    1 => match_buf!( &data, v => insert_array_attrib::<_, _, FaceVertexIndex>(v, name, mesh, orig_map) ),
                    2 => match_buf!( &data, v => insert_array_attrib_n::<_, _, FaceVertexIndex,U2>(v, name, mesh, orig_map) ),
                    3 => match_buf!( &data, v => insert_array_attrib_n::<_, _, FaceVertexIndex,U3>(v, name, mesh, orig_map) ),
                    4 => match_buf!( &data, v => insert_array_attrib_n::<_, _, FaceVertexIndex,U4>(v, name, mesh, orig_map) ),
                    _ => continue,
                }
                .unwrap_or_else(|err| eprintln!("WARNING: Face Vertex Attribute transfer error for \"{}\": {}", name, err))
            }
        } // Ignore all other attributes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_test() {
        let vtk = model::Vtk {
            version: model::Version::new((0, 1)),
            title: String::from("Tetrahedral Mesh"),
            byte_order: model::ByteOrder::BigEndian,
            file_path: None,
            data: model::DataSet::inline(model::UnstructuredGridPiece {
                points: vec![
                    2., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 2., 1., 1., 0., 1., 2., 2., 1.,
                    2., 1., 1., 4., 2., 1., 4., 1., 1., 5., 2., 1., 5.,
                ]
                .into(),
                cells: model::Cells {
                    cell_verts: model::VertexNumbers::Legacy {
                        num_cells: 3,
                        vertices: vec![4, 1, 3, 2, 5, 4, 0, 4, 3, 6, 4, 9, 10, 8, 7],
                    },
                    types: vec![model::CellType::Tetra; 3],
                },
                data: model::Attributes {
                    point: vec![
                        model::Attribute::scalars("scalars", 1).with_data(vec![
                            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0f32,
                        ]),
                        model::Attribute::vectors("vectors").with_data(vec![
                            1., 0., 0., 1., 1., 0., 0., 2., 0., 1., 0., 0., 1., 1., 0., 0., 2., 0.,
                            1., 0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 1., 0., 0., 1.,
                        ]),
                        model::Attribute::tensors("tensors").with_data(vec![
                            1., 0., 0., 1., 1., 0., 0., 2., 0., 1., 0., 0., 1., 1., 0., 0., 2., 0.,
                            1., 0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0.,
                            1., 1., 0., 0., 2., 0., 1., 0., 0., 1., 1., 0., 0., 2., 0., 1., 0., 0.,
                            1., 1., 0., 0., 2., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0.,
                            0., 2., 0., 1., 0., 0., 1., 1., 0., 0., 2., 0., 1., 0., 0., 1., 1., 0.,
                            0., 2., 0., 0., 0., 1., 0., 0., 1.,
                        ]),
                    ],
                    cell: vec![],
                },
            }),
        };

        let vtktetmesh = vtk.extract_tetmesh().unwrap();

        let pts = vec![
            [2., 1., 0.],
            [0., 0., 1.],
            [0., 1., 1.],
            [1., 1., 1.],
            [2., 1., 1.],
            [0., 1., 2.],
            [2., 1., 2.],
            [1., 1., 4.],
            [2., 1., 4.],
            [1., 1., 5.],
            [2., 1., 5.],
        ];
        let indices = vec![[1, 3, 2, 5], [0, 4, 3, 6], [9, 10, 8, 7]];
        let mut tetmesh = TetMesh::new(pts, indices);

        let scalars = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0f32];
        let vectors = vec![
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
        ];
        let tensors = vec![
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[0., 0., 1.], [0., 0., 1.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [0., 0., 1.]],
            [[0., 0., 1.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [0., 0., 1.], [0., 0., 1.]],
        ];

        tetmesh
            .insert_attrib_data::<_, VertexIndex>("scalars", scalars)
            .ok()
            .unwrap();

        tetmesh
            .insert_attrib_data::<_, VertexIndex>("tensors", tensors)
            .ok()
            .unwrap();
        tetmesh
            .insert_attrib_data::<_, VertexIndex>("vectors", vectors)
            .ok()
            .unwrap();

        // vtk -> tetmesh test
        assert_eq!(vtktetmesh, tetmesh);

        // tetmesh -> vtk test
        let tetmeshvtk = convert_tetmesh_to_vtk_format(&tetmesh).unwrap();
        let vtktetmesh = tetmeshvtk.extract_tetmesh().unwrap();
        assert_eq!(vtktetmesh, tetmesh);
    }

    fn vtk_polymesh_example_data() -> (IOBuffer, model::VertexNumbers, model::Attributes) {
        let (buf, mut attrib) = vtk_pointcloud_example_data();
        let cell_verts = model::VertexNumbers::Legacy {
            num_cells: 3,
            vertices: vec![4, 1, 3, 2, 5, 4, 0, 4, 3, 6, 4, 9, 10, 8, 7],
        };
        attrib.cell = vec![model::Attribute::scalars("scalars", 1).with_data(vec![0.0, 1.0, 2.0])];

        (buf, cell_verts, attrib)
    }
    fn vtk_polymesh_example_mixed_polydata() -> (
        IOBuffer,
        model::VertexNumbers,
        model::VertexNumbers,
        model::Attributes,
    ) {
        let (buf, mut attrib) = vtk_pointcloud_example_data();
        let polys = model::VertexNumbers::Legacy {
            num_cells: 2,
            vertices: vec![4, 1, 3, 2, 5, 4, 0, 4, 3, 6],
        };
        let lines = model::VertexNumbers::Legacy {
            num_cells: 1,
            vertices: vec![4, 9, 10, 8, 7],
        };
        attrib.cell =
            vec![model::Attribute::scalars("scalars", 1).with_data(vec![0.0, 1.0, 2.0, 2.0, 2.0])];

        (buf, polys, lines, attrib)
    }

    /// Produce an example polymesh for testing that corresponds to the vtk model returned by
    /// `vtk_polymesh_example_data`.
    fn polymesh_example() -> PolyMesh<f64> {
        let pts = vec![
            [2., 1., 0.],
            [0., 0., 1.],
            [0., 1., 1.],
            [1., 1., 1.],
            [2., 1., 1.],
            [0., 1., 2.],
            [2., 1., 2.],
            [1., 1., 4.],
            [2., 1., 4.],
            [1., 1., 5.],
            [2., 1., 5.],
        ];
        let faces: Vec<usize> = vec![4, 1, 3, 2, 5, 4, 0, 4, 3, 6, 4, 9, 10, 8, 7];
        let mut polymesh = PolyMesh::new(pts, &faces);

        let face_scalars = vec![0.0, 1.0, 2.0];
        let scalars = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0f32];
        let vectors = vec![
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
        ];
        let tensors = vec![
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[0., 0., 1.], [0., 0., 1.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [0., 0., 1.]],
            [[0., 0., 1.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [0., 0., 1.], [0., 0., 1.]],
        ];

        polymesh
            .insert_attrib_data::<_, FaceIndex>("scalars", face_scalars)
            .unwrap();

        polymesh
            .insert_attrib_data::<_, VertexIndex>("scalars", scalars)
            .unwrap();

        polymesh
            .insert_attrib_data::<_, VertexIndex>("tensors", tensors)
            .unwrap();

        polymesh
            .insert_attrib_data::<_, VertexIndex>("vectors", vectors)
            .unwrap();

        polymesh
    }

    #[test]
    fn unstructured_data_polymesh_test() {
        let (points, cell_verts, data) = vtk_polymesh_example_data();
        let vtk = model::Vtk {
            version: model::Version::new((0, 1)),
            title: String::from("Polygonal Mesh"),
            byte_order: model::ByteOrder::BigEndian,
            file_path: None,
            data: model::DataSet::inline(model::UnstructuredGridPiece {
                points,
                cells: model::Cells {
                    cell_verts: cell_verts.clone(),
                    types: vec![model::CellType::Polygon; cell_verts.num_cells() as usize],
                },
                data,
            }),
        };

        let vtkpolymesh = vtk.extract_polymesh().unwrap();
        let polymesh = polymesh_example();

        // vtk -> polymesh test
        assert_eq!(vtkpolymesh, polymesh);

        // polymesh -> vtk test
        let polymeshvtk =
            convert_polymesh_to_vtk_format(&polymesh, VTKPolyExportStyle::UnstructuredGrid)
                .unwrap();
        let vtkpolymesh = polymeshvtk.extract_polymesh().unwrap();
        assert_eq!(vtkpolymesh, polymesh);
    }

    /// Produce an example polymesh for testing that corresponds to the vtk model returned by
    /// `vtk_polymesh_example_data`. This version interprets the last poly in `vtk_polymesh_example_data` as a polyline.
    fn polymesh_example_with_polyline() -> PolyMesh<f64> {
        let pts = vec![
            [2., 1., 0.],
            [0., 0., 1.],
            [0., 1., 1.],
            [1., 1., 1.],
            [2., 1., 1.],
            [0., 1., 2.],
            [2., 1., 2.],
            [1., 1., 4.],
            [2., 1., 4.],
            [1., 1., 5.],
            [2., 1., 5.],
        ];
        let faces: Vec<usize> = vec![4, 1, 3, 2, 5, 4, 0, 4, 3, 6, 2, 9, 10, 2, 10, 8, 2, 8, 7];
        let mut polymesh = PolyMesh::new(pts, &faces);

        let face_scalars = vec![0.0, 1.0, 2.0, 2.0, 2.0];
        let scalars = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0f32];
        let vectors = vec![
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
        ];
        let tensors = vec![
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[0., 0., 1.], [0., 0., 1.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [0., 0., 1.]],
            [[0., 0., 1.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [0., 0., 1.], [0., 0., 1.]],
        ];

        polymesh
            .insert_attrib_data::<_, FaceIndex>("scalars", face_scalars)
            .unwrap();

        polymesh
            .insert_attrib_data::<_, VertexIndex>("scalars", scalars)
            .unwrap();

        polymesh
            .insert_attrib_data::<_, VertexIndex>("tensors", tensors)
            .unwrap();

        polymesh
            .insert_attrib_data::<_, VertexIndex>("vectors", vectors)
            .unwrap();

        polymesh
    }

    #[test]
    fn mixed_unstructured_data_polymesh_test() {
        use model::CellType;
        let (points, cell_verts, data) = vtk_polymesh_example_data();
        let vtk = model::Vtk {
            version: model::Version::new((0, 1)),
            title: String::from("Polygonal Mesh"),
            byte_order: model::ByteOrder::BigEndian,
            file_path: None,
            data: model::DataSet::inline(model::UnstructuredGridPiece {
                points,
                cells: model::Cells {
                    cell_verts: cell_verts.clone(),
                    types: vec![CellType::Polygon, CellType::Polygon, CellType::PolyLine],
                },
                data: model::Attributes {
                    point: data.point,
                    // Override the cell data since the polyline is split into separate line segments.
                    cell: vec![model::Attribute::scalars("scalars", 1)
                        .with_data(vec![0.0, 1.0, 2.0, 2.0, 2.0])],
                },
            }),
        };

        let vtkpolymesh = vtk.extract_polymesh().unwrap();
        let polymesh = polymesh_example_with_polyline();

        // vtk -> polymesh test
        assert_eq!(vtkpolymesh, polymesh);

        // polymesh -> vtk test
        let polymeshvtk =
            convert_polymesh_to_vtk_format(&polymesh, VTKPolyExportStyle::UnstructuredGrid)
                .unwrap();
        let vtkpolymesh = polymeshvtk.extract_polymesh().unwrap();
        assert_eq!(vtkpolymesh, polymesh);
    }

    #[test]
    fn poly_data_polymesh_test() {
        let (points, polys, data) = vtk_polymesh_example_data();
        let vtk = model::Vtk {
            version: model::Version::new((0, 1)),
            title: String::from("Polygonal Mesh"),
            byte_order: model::ByteOrder::BigEndian,
            file_path: None,
            data: model::DataSet::inline(model::PolyDataPiece {
                points,
                polys: Some(polys),
                data,
                ..Default::default()
            }),
        };

        let vtkpolymesh = vtk.extract_polymesh().unwrap();
        let polymesh = polymesh_example();

        // vtk -> polymesh test
        assert_eq!(vtkpolymesh, polymesh);

        // polymesh -> vtk test
        let polymeshvtk =
            convert_polymesh_to_vtk_format(&polymesh, VTKPolyExportStyle::PolyData).unwrap();
        let vtkpolymesh = polymeshvtk.extract_polymesh().unwrap();
        assert_eq!(vtkpolymesh, polymesh);
    }

    #[test]
    fn mixed_poly_data_polymesh_test() {
        let (points, polys, lines, data) = vtk_polymesh_example_mixed_polydata();
        let vtk = model::Vtk {
            version: model::Version::new((0, 1)),
            title: String::from("Polygonal Mesh"),
            byte_order: model::ByteOrder::BigEndian,
            file_path: None,
            data: model::DataSet::inline(model::PolyDataPiece {
                points,
                polys: Some(polys),
                lines: Some(lines),
                data,
                ..Default::default()
            }),
        };

        let vtkpolymesh = vtk.extract_polymesh().unwrap();
        let polymesh = polymesh_example_with_polyline();

        // vtk -> polymesh test
        assert_eq!(vtkpolymesh, polymesh);

        // polymesh -> vtk test
        let polymeshvtk =
            convert_polymesh_to_vtk_format(&polymesh, VTKPolyExportStyle::PolyData).unwrap();
        let vtkpolymesh = polymeshvtk.extract_polymesh().unwrap();
        assert_eq!(vtkpolymesh, polymesh);
    }

    fn vtk_pointcloud_example_data() -> (IOBuffer, model::Attributes) {
        (
            vec![
                2., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 2., 1., 1., 0., 1., 2., 2., 1., 2.,
                1., 1., 4., 2., 1., 4., 1., 1., 5., 2., 1., 5.,
            ]
            .into(),
            model::Attributes {
                point: vec![
                    model::Attribute::scalars("scalars", 1).with_data(vec![
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0f32,
                    ]),
                    model::Attribute::vectors("vectors").with_data(vec![
                        1., 0., 0., 1., 1., 0., 0., 2., 0., 1., 0., 0., 1., 1., 0., 0., 2., 0., 1.,
                        0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 1., 0., 0., 1.,
                    ]),
                    model::Attribute::tensors("tensors").with_data(vec![
                        1., 0., 0., 1., 1., 0., 0., 2., 0., 1., 0., 0., 1., 1., 0., 0., 2., 0., 1.,
                        0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1.,
                        0., 0., 2., 0., 1., 0., 0., 1., 1., 0., 0., 2., 0., 1., 0., 0., 1., 1., 0.,
                        0., 2., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 2., 0., 1.,
                        0., 0., 1., 1., 0., 0., 2., 0., 1., 0., 0., 1., 1., 0., 0., 2., 0., 0., 0.,
                        1., 0., 0., 1.,
                    ]),
                ],
                cell: vec![],
            },
        )
    }

    /// Produce an example pointcloud for testing that corresponds to the vtk model returned by
    /// `vtk_pointcloud_example_data`.
    fn pointcloud_example() -> PointCloud<f64> {
        let pts = vec![
            [2., 1., 0.],
            [0., 0., 1.],
            [0., 1., 1.],
            [1., 1., 1.],
            [2., 1., 1.],
            [0., 1., 2.],
            [2., 1., 2.],
            [1., 1., 4.],
            [2., 1., 4.],
            [1., 1., 5.],
            [2., 1., 5.],
        ];
        let mut pointcloud = PointCloud::new(pts);

        let scalars = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0f32];
        let vectors = vec![
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
        ];
        let tensors = vec![
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[1., 0., 0.], [1., 1., 0.], [0., 2., 0.]],
            [[0., 0., 1.], [0., 0., 1.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [1., 0., 0.]],
            [[1., 1., 0.], [0., 2., 0.], [0., 0., 1.]],
            [[0., 0., 1.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 2., 0.], [0., 0., 1.], [0., 0., 1.]],
        ];

        pointcloud
            .insert_attrib_data::<_, VertexIndex>("scalars", scalars)
            .unwrap();

        pointcloud
            .insert_attrib_data::<_, VertexIndex>("tensors", tensors)
            .unwrap();
        pointcloud
            .insert_attrib_data::<_, VertexIndex>("vectors", vectors)
            .unwrap();

        pointcloud
    }

    #[test]
    fn poly_data_pointcloud_test() {
        let (points, data) = vtk_pointcloud_example_data();
        let num_vertices = (points.len() / 3) as u32;
        let verts = Some(model::VertexNumbers::Legacy {
            num_cells: 1,
            vertices: std::iter::once(num_vertices)
                .chain(0..num_vertices)
                .collect(),
        });

        let vtk = model::Vtk {
            version: model::Version::new((0, 1)),
            title: String::from("Point Cloud"),
            byte_order: model::ByteOrder::BigEndian,
            file_path: None,
            data: model::DataSet::inline(model::PolyDataPiece {
                points,
                verts,
                data,
                ..Default::default()
            }),
        };

        let vtkpointcloud = vtk.extract_pointcloud().unwrap();
        let pointcloud = pointcloud_example();

        // vtk -> pointcloud test
        assert_eq!(vtkpointcloud, pointcloud);

        // pointcloud -> vtk test
        let pointcloudvtk =
            convert_pointcloud_to_vtk_format(&pointcloud, VTKPolyExportStyle::PolyData).unwrap();
        let vtkpointcloud = pointcloudvtk.extract_pointcloud().unwrap();
        assert_eq!(vtkpointcloud, pointcloud);
    }
}
