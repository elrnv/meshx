//! IO module for mesh files.
//!
//! Supported formats for loading:
//!  - `msh` via [`mshio`](https://crates.io/mshio).
//!  - `obj` via [`obj`](https://crates.io/obj).
//!  - `vtk` via [`vtkio`](https://crates.io/vtkio).
//!
//! Supported formats for saving:
//!  - `obj` via [`obj`](https://crates.io/obj).
//!  - `vtk` via [`vtkio`](https://crates.io/vtkio).
use std::path::Path;

pub use vtkio::Vtk;

use crate::attrib;
use crate::mesh::{Mesh, PointCloud, PolyMesh, TetMesh, TriMesh};

#[cfg(feature = "mshio")]
pub mod msh;
pub mod obj;
pub mod vtk;

#[cfg(feature = "vtkio")]
pub trait Real: vtkio::model::Scalar + std::str::FromStr + crate::Real {}
#[cfg(feature = "vtkio")]
impl<T> Real for T where T: vtkio::model::Scalar + std::str::FromStr + crate::Real {}

#[cfg(not(feature = "vtkio"))]
pub trait Real: crate::Real + std::str::FromStr {}
#[cfg(not(feature = "vtkio"))]
impl<T> Real for T where T: crate::Real + std::str::FromStr {}

// These names are chosen to be rather short to reduce the const of comparisons.
// Although code that relies on this is not idiomatic, it can sometimes be simpler.
const UV_ATTRIB_NAME: &str = "uv";
const NORMAL_ATTRIB_NAME: &str = "N";
const MTL_ATTRIB_NAME: &str = "mtl";
const OBJECT_ATTRIB_NAME: &str = "object";
const GROUP_ATTRIB_NAME: &str = "group";

/// A trait for specific scene, object or mesh models to extract mesh data from.
///
/// All methods are optional and default implementations simply return an `UnsupportedDataFormat` error.
/// This trait defines an API for converting file specific object models to `meshx` mesh formats.
pub trait MeshExtractor<T: crate::Real> {
    /// Constructs an unstructured Mesh from this VTK model.
    ///
    /// This function may clone the given model as necessary.
    fn extract_mesh(&self) -> Result<Mesh<T>, Error> {
        Err(Error::UnsupportedDataFormat)
    }
    /// Constructs a PolyMesh from this VTK model.
    ///
    /// This function may clone the given model as necessary.
    fn extract_polymesh(&self) -> Result<PolyMesh<T>, Error> {
        Err(Error::UnsupportedDataFormat)
    }
    /// Constructs a TetMesh from this VTK model.
    ///
    /// This function may clone the given model as necessary.
    fn extract_tetmesh(&self) -> Result<TetMesh<T>, Error> {
        Err(Error::UnsupportedDataFormat)
    }
    /// Constructs a PointCloud from this VTK model.
    ///
    /// This function may clone the given model as necessary.
    fn extract_pointcloud(&self) -> Result<PointCloud<T>, Error> {
        Err(Error::UnsupportedDataFormat)
    }
}

/*
 * IO calls for unstructured Meshes
 */

/// Load a tetrahedral mesh from a given file.
pub fn load_mesh<T: Real, P: AsRef<Path>>(file: P) -> Result<Mesh<T>, Error> {
    load_mesh_impl(file.as_ref())
}

fn load_mesh_impl<T: Real>(file: &Path) -> Result<Mesh<T>, Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") | Some("vtu") | Some("pvtu") => {
            let vtk = Vtk::import(file)?;
            vtk.extract_mesh()
        }
        #[cfg(feature = "mshio")]
        Some("msh") => {
            let msh_bytes = std::fs::read(file)?;
            let msh = mshio::parse_msh_bytes(msh_bytes.as_slice()).map_err(msh::MshError::from)?;
            msh.extract_mesh()
        }
        // NOTE: wavefront obj files don't support unstructured meshes.
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/// Save a mesh to a file.
pub fn save_mesh<T: Real, P: AsRef<Path>>(mesh: &Mesh<T>, file: P) -> Result<(), Error> {
    save_mesh_impl(mesh, file.as_ref())
}

fn save_mesh_impl<T: Real>(mesh: &Mesh<T>, file: &Path) -> Result<(), Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") | Some("vtu") | Some("pvtu") => {
            let vtk = vtk::convert_mesh_to_vtk_format(mesh)?;
            vtk.export_be(file)?;
            Ok(())
        }
        // NOTE: wavefront obj files don't support unstructured meshes.
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/// Save a mesh to a file in ASCII format.
pub fn save_mesh_ascii<T: Real>(mesh: &Mesh<T>, file: impl AsRef<Path>) -> Result<(), Error> {
    save_mesh_ascii_impl(mesh, file.as_ref())
}

fn save_mesh_ascii_impl<T: Real>(mesh: &Mesh<T>, file: &Path) -> Result<(), Error> {
    match file.extension() {
        Some(ext) if ext.to_str() == Some("vtk") => {
            // NOTE: Currently writing to ascii is supported only for Legacy VTK files.
            let vtk = vtk::convert_mesh_to_vtk_format(mesh)?;
            vtk.export_ascii(file)?;
            Ok(())
        }
        // NOTE: wavefront obj files don't support unstructured meshes.
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/*
 * IO calls for TetMeshes
 */

/// Load a tetrahedral mesh from a given file.
pub fn load_tetmesh<T: Real, P: AsRef<Path>>(file: P) -> Result<TetMesh<T>, Error> {
    load_tetmesh_impl(file.as_ref())
}

fn load_tetmesh_impl<T: Real>(file: &Path) -> Result<TetMesh<T>, Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") | Some("vtu") | Some("pvtu") => {
            let vtk = Vtk::import(file)?;
            vtk.extract_tetmesh()
        }
        // NOTE: wavefront obj files don't support tetrahedral meshes.
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/// Save a tetrahedral mesh to a file.
pub fn save_tetmesh<T: Real, P: AsRef<Path>>(tetmesh: &TetMesh<T>, file: P) -> Result<(), Error> {
    save_tetmesh_impl(tetmesh, file.as_ref())
}

fn save_tetmesh_impl<T: Real>(tetmesh: &TetMesh<T>, file: &Path) -> Result<(), Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") | Some("vtu") | Some("pvtu") => {
            let vtk = vtk::convert_tetmesh_to_vtk_format(tetmesh)?;
            vtk.export_be(file)?;
            Ok(())
        }
        // NOTE: wavefront obj files don't support tetrahedral meshes.
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/// Save a tetrahedral mesh to a file.
pub fn save_tetmesh_ascii<T: Real>(
    tetmesh: &TetMesh<T>,
    file: impl AsRef<Path>,
) -> Result<(), Error> {
    save_tetmesh_ascii_impl(tetmesh, file.as_ref())
}

fn save_tetmesh_ascii_impl<T: Real>(tetmesh: &TetMesh<T>, file: &Path) -> Result<(), Error> {
    match file.extension() {
        Some(ext) if ext.to_str() == Some("vtk") => {
            // NOTE: Currently writing to ascii is supported only for Legacy VTK files.
            let vtk = vtk::convert_tetmesh_to_vtk_format(tetmesh)?;
            vtk.export_ascii(file)?;
            Ok(())
        }
        // NOTE: wavefront obj files don't support tetrahedral meshes.
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/*
 * IO calls for PolyMeshes
 */

/// Load a polygonal mesh from a given file.
pub fn load_polymesh<T: Real, P: AsRef<Path>>(file: P) -> Result<PolyMesh<T>, Error> {
    load_polymesh_impl(file.as_ref())
}

fn load_polymesh_impl<T: Real>(file: &Path) -> Result<PolyMesh<T>, Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") | Some("vtu") | Some("vtp") | Some("pvtu") | Some("pvtp") => {
            let vtk = Vtk::import(file)?;
            vtk.extract_polymesh()
        }
        Some("obj") => {
            let obj = obj::Obj::load_with_config(file, obj::LoadConfig { strict: false })?;
            obj.data.extract_polymesh()
        }
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/// Save a polygonal mesh to a file.
pub fn save_polymesh<T: Real, P: AsRef<Path>>(
    polymesh: &PolyMesh<T>,
    file: P,
) -> Result<(), Error> {
    save_polymesh_impl(polymesh, file.as_ref())
}

fn save_polymesh_impl<T: Real>(polymesh: &PolyMesh<T>, file: &Path) -> Result<(), Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") | Some("vtu") | Some("vtp") | Some("pvtu") | Some("pvtp") => {
            let vtk =
                vtk::convert_polymesh_to_vtk_format(polymesh, vtk::VTKPolyExportStyle::PolyData)?;
            vtk.export_be(file)?;
            Ok(())
        }
        Some("obj") => {
            let obj = obj::convert_polymesh_to_obj_format(polymesh)?;
            obj.save(file)?;
            Ok(())
        }
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/// Save a polygonal mesh to a file.
pub fn save_polymesh_ascii<T: Real, P: AsRef<Path>>(
    polymesh: &PolyMesh<T>,
    file: P,
) -> Result<(), Error> {
    save_polymesh_ascii_impl(polymesh, file.as_ref())
}

fn save_polymesh_ascii_impl<T: Real>(polymesh: &PolyMesh<T>, file: &Path) -> Result<(), Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") => {
            // NOTE: Currently writing to ascii is supported only for Legacy VTK files.
            let vtk =
                vtk::convert_polymesh_to_vtk_format(polymesh, vtk::VTKPolyExportStyle::PolyData)?;
            vtk.export_ascii(file)?;
            Ok(())
        }
        Some("obj") => {
            let obj = obj::convert_polymesh_to_obj_format(polymesh)?;
            obj.save(file)?;
            Ok(())
        }
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/*
 * IO calls for TriMeshes
 *
 * NOTE: These functions simply call into PolyMesh IO API, since it allows for
 * more flexibility (e.g. can load polymesh as a trimesh by triangulating) and
 * code reuse.
 */

/// Loads a triangle mesh from a given file.
pub fn load_trimesh<T: Real, P: AsRef<Path>>(file: P) -> Result<TriMesh<T>, Error> {
    load_polymesh_impl(file.as_ref()).map(TriMesh::from)
}

/// Saves a triangle mesh to a file.
pub fn save_trimesh<T: Real, P: AsRef<Path>>(trimesh: &TriMesh<T>, file: P) -> Result<(), Error> {
    save_polymesh_impl(&PolyMesh::from(trimesh.clone()), file.as_ref())
}

/// Saves a triangle mesh to a file.
pub fn save_trimesh_ascii<T: Real, P: AsRef<Path>>(
    trimesh: &TriMesh<T>,
    file: P,
) -> Result<(), Error> {
    save_polymesh_ascii_impl(&PolyMesh::from(trimesh.clone()), file.as_ref())
}

/*
 * IO calls for Point clouds
 */

/// Load a point cloud from a given file.
pub fn load_pointcloud<T: Real, P: AsRef<Path>>(file: P) -> Result<PointCloud<T>, Error> {
    load_pointcloud_impl(file.as_ref())
}

fn load_pointcloud_impl<T: Real>(file: &Path) -> Result<PointCloud<T>, Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") | Some("vtu") | Some("vtp") | Some("pvtu") | Some("pvtp") => {
            let vtk = Vtk::import(file)?;
            vtk.extract_pointcloud()
        }
        Some("obj") => {
            let obj = obj::Obj::load_with_config(file, obj::LoadConfig { strict: false })?;
            obj.data.extract_pointcloud()
        }
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/// Save a point cloud to a file.
pub fn save_pointcloud<T: Real, P: AsRef<Path>>(
    ptcloud: &PointCloud<T>,
    file: P,
) -> Result<(), Error> {
    save_pointcloud_impl(ptcloud, file.as_ref())
}

pub fn save_pointcloud_impl<T: Real>(ptcloud: &PointCloud<T>, file: &Path) -> Result<(), Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") | Some("vtu") | Some("vtp") | Some("pvtu") | Some("pvtp") => {
            let vtk =
                vtk::convert_pointcloud_to_vtk_format(ptcloud, vtk::VTKPolyExportStyle::PolyData)?;
            vtk.export_be(file)?;
            Ok(())
        }
        Some("obj") => {
            let obj = obj::convert_pointcloud_to_obj_format(ptcloud)?;
            obj.save(file)?;
            Ok(())
        }
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/// Save a point cloud to a file.
pub fn save_pointcloud_ascii<T: Real, P: AsRef<Path>>(
    ptcloud: &PointCloud<T>,
    file: P,
) -> Result<(), Error> {
    save_pointcloud_ascii_impl(ptcloud, file.as_ref())
}

fn save_pointcloud_ascii_impl<T: Real>(ptcloud: &PointCloud<T>, file: &Path) -> Result<(), Error> {
    match file.extension().and_then(|ext| ext.to_str()) {
        Some("vtk") => {
            let vtk =
                vtk::convert_pointcloud_to_vtk_format(ptcloud, vtk::VTKPolyExportStyle::PolyData)?;
            vtk.export_ascii(file)?;
            Ok(())
        }
        Some("obj") => {
            let obj = obj::convert_pointcloud_to_obj_format(ptcloud)?;
            obj.save(file)?;
            Ok(())
        }
        _ => Err(Error::UnsupportedFileFormat),
    }
}

/*
 * Low-level IO operations
 */

#[derive(Debug)]
pub enum MeshIOError {
    #[cfg(feature = "mshio")]
    Msh {
        source: msh::MshError,
    },
    Vtk {
        source: vtk::VtkError,
    },
    Obj {
        source: obj::ObjError,
    },
}

impl std::error::Error for MeshIOError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MeshIOError::Msh { source } => Some(source),
            MeshIOError::Vtk { source } => Some(source),
            MeshIOError::Obj { source } => Some(source),
        }
    }
}

impl std::fmt::Display for MeshIOError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MeshIOError::Msh { source } => write!(f, "A Msh IO error occurred: {}", source),
            MeshIOError::Vtk { source } => write!(f, "A Vtk IO error occurred: {}", source),
            MeshIOError::Obj { source } => write!(f, "An Obj IO error occurred: {}", source),
        }
    }
}

#[derive(Debug)]
pub enum Error {
    IO { source: std::io::Error },
    MeshIO { source: MeshIOError },
    Attrib { source: attrib::Error },
    UnsupportedFileFormat,
    UnsupportedDataFormat,
    MeshTypeMismatch,
    MissingMeshData,
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::IO { source } => Some(source),
            Error::MeshIO { source } => Some(source),
            Error::Attrib { .. } => {
                None // Implement when attrib::Error implements std::error::Error.
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::IO { source } => write!(f, "IO Error: {}", source),
            Error::MeshIO { source } => write!(f, "An error in mesh IO occurred: {}", source),
            Error::Attrib { source } => write!(f, "An attribute error occurred: {}", source),
            Error::UnsupportedFileFormat => write!(f, "Unsupported file format specified"),
            Error::UnsupportedDataFormat => write!(f, "Unsupported data format specified"),
            Error::MeshTypeMismatch => write!(f, "Mesh type doesn't match expected type"),
            Error::MissingMeshData => write!(f, "Missing mesh data"),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::IO { source: err }
    }
}

#[cfg(feature = "mshio")]
impl From<msh::MshError> for Error {
    fn from(err: msh::MshError) -> Error {
        Error::MeshIO {
            source: MeshIOError::Msh { source: err },
        }
    }
}

impl From<obj::ObjError> for Error {
    fn from(err: obj::ObjError) -> Error {
        Error::MeshIO {
            source: MeshIOError::Obj { source: err },
        }
    }
}

impl From<vtk::VtkError> for Error {
    fn from(err: vtk::VtkError) -> Error {
        Error::MeshIO {
            source: MeshIOError::Vtk { source: err },
        }
    }
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Error {
        Error::Attrib { source: err }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
     * Test round trip file loading and saving
     */

    #[test]
    fn polycube() -> Result<(), Error> {
        let polymesh: PolyMesh<f32> = load_polymesh("assets/cube.obj")?;
        save_polymesh(&polymesh, "tests/artifacts/cube.obj")?;
        let reloaded_polymesh = load_polymesh("tests/artifacts/cube.obj")?;
        assert_eq!(polymesh, reloaded_polymesh);
        Ok(())
    }

    // The following test verifies data integrity during conversion. This is useful because there
    // are many things happening here including indirect attribute transfer that can potentially be
    // botched.
    #[test]
    fn polycube_convert() -> Result<(), Error> {
        use crate::mesh::TriMesh;
        let _: TriMesh<f32> = load_polymesh("assets/cube.obj")?.into();
        Ok(())
    }

    // Test that loading a vtu file works as expected.
    #[test]
    fn tet_vtu() -> Result<(), Error> {
        use crate::mesh::TetMesh;
        let mesh: TetMesh<f32> = load_tetmesh("assets/tet.vtu")?;
        dbg!(&mesh);
        Ok(())
    }

    #[test]
    fn cloth_argus() -> Result<(), Error> {
        let polymesh: PolyMesh<f32> = load_polymesh("assets/cloth_argus.obj")?;
        dbg!(&polymesh);
        Ok(())
    }

    // Ensure that trying to load a tetmesh as a polymesh throws an error.
    // To get a PolyMesh out of a VTK tetmesh, a TetMesh must first be constructed and later
    // converted to a TriMesh (and then PolyMesh).
    #[test]
    fn tet_vtk_as_polymesh_error() {
        assert!(load_polymesh::<f64, _>("assets/tet.vtk").is_err());
    }

    // A regression test loading a real world poly mesh example.
    #[test]
    fn unstructured_data_polymesh_real_test() {
        assert!(load_polymesh::<f64, _>("./assets/tube.vtk").is_ok());
    }

    // Msh file loading test
    #[test]
    fn sphere_msh() -> Result<(), Error> {
        use crate::mesh::topology::*;
        let mesh: Mesh<f64> = load_mesh("assets/sphere_coarse.msh")?;
        assert_eq!(mesh.num_vertices(), 183);
        assert_eq!(mesh.num_cells(), 593);
        Ok(())
    }
}
