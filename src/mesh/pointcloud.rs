//!
//! PointCloud module. Describes a point cloud data structure and possible operations on it. This
//! is just a collection of points, onto which you may attach arbitrary application specific
//! attributes.
//!

use crate::mesh::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::{VertexMesh, VertexPositions};
use crate::Real;

/// A collection of disconnected points, possibly but not necessarily representing some geometry.
/// The points may have arbitrary attributes assigned to them such as orientation.
#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct PointCloud<T: Real> {
    /// Vertex positions.
    #[intrinsic(VertexPositions)]
    pub vertex_positions: IntrinsicAttribute<[T; 3], VertexIndex>,
    /// Vertex attribute data.
    pub vertex_attributes: AttribDict<VertexIndex>,
}

impl<T: Real> PointCloud<T> {
    /// Construct a `PointCloud` from an array of vertices.
    /// # Examples
    /// ```
    /// use meshx::mesh::{PointCloud, VertexPositions};
    /// let points = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0]];
    ///
    /// let ptcloud = PointCloud::new(points.clone());
    ///
    /// let positions = ptcloud.vertex_positions().to_vec();
    /// assert_eq!(positions, points);
    /// ```
    #[inline]
    pub fn new(verts: Vec<[T; 3]>) -> PointCloud<T> {
        PointCloud {
            vertex_positions: IntrinsicAttribute::from_vec(verts),
            vertex_attributes: AttribDict::new(),
        }
    }
}

impl<T: Real> Default for PointCloud<T> {
    /// Produce an empty `PointCloud`.
    ///
    /// This is not particularly useful on its own, however it can be
    /// used as a null case for various mesh algorithms.
    fn default() -> Self {
        PointCloud::new(vec![])
    }
}

impl<T: Real> NumVertices for PointCloud<T> {
    #[inline]
    fn num_vertices(&self) -> usize {
        self.vertex_positions.len()
    }
}

/// Convert a borrow of any `VertexMesh` into a `PointCloud`.
/// Since we can't destructure a generic type, it's sufficient to do the conversion from a
/// reference to a mesh.
impl<T: Real, M: VertexMesh<T>> From<&M> for PointCloud<T> {
    fn from(mesh: &M) -> PointCloud<T> {
        let vertex_attributes = mesh.attrib_dict::<VertexIndex>().clone();
        PointCloud {
            vertex_positions: IntrinsicAttribute::from_slice(mesh.vertex_positions()),
            vertex_attributes,
        }
    }
}

/// Convert a polygon mesh to a point cloud by erasing all polygon data.
impl<T: Real> From<super::PolyMesh<T>> for PointCloud<T> {
    fn from(polymesh: super::PolyMesh<T>) -> PointCloud<T> {
        let super::PolyMesh {
            vertex_positions,
            vertex_attributes,
            ..
        } = polymesh;

        PointCloud {
            vertex_positions,
            vertex_attributes,
        }
    }
}

/// Convert a triangle mesh to a point cloud by erasing all triangle data.
impl<T: Real> From<super::TriMesh<T>> for PointCloud<T> {
    fn from(mesh: super::TriMesh<T>) -> PointCloud<T> {
        let super::TriMesh {
            vertex_positions,
            vertex_attributes,
            ..
        } = mesh;

        PointCloud {
            vertex_positions,
            vertex_attributes,
        }
    }
}

/// Convert a quad mesh to a point cloud by erasing all quad data.
impl<T: Real> From<super::QuadMesh<T>> for PointCloud<T> {
    fn from(mesh: super::QuadMesh<T>) -> PointCloud<T> {
        let super::QuadMesh {
            vertex_positions,
            vertex_attributes,
            ..
        } = mesh;

        PointCloud {
            vertex_positions,
            vertex_attributes,
        }
    }
}

/// Convert a tet mesh to a point cloud by erasing all tet data.
impl<T: Real> From<super::TetMesh<T>> for PointCloud<T> {
    fn from(mesh: super::TetMesh<T>) -> PointCloud<T> {
        let super::TetMesh {
            vertex_positions,
            vertex_attributes,
            ..
        } = mesh;

        PointCloud {
            vertex_positions,
            vertex_attributes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointcloud_test() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];

        let mut ptcloud = PointCloud::new(pts.clone());
        assert_eq!(ptcloud.num_vertices(), 4);

        for (pt1, pt2) in ptcloud.vertex_position_iter().zip(pts.iter()) {
            assert_eq!(*pt1, *pt2);
        }
        for (pt1, pt2) in ptcloud.vertex_position_iter_mut().zip(pts.iter()) {
            assert_eq!(*pt1, *pt2);
        }
        for (pt1, pt2) in ptcloud.vertex_positions().iter().zip(pts.iter()) {
            assert_eq!(*pt1, *pt2);
        }
        for (pt1, pt2) in ptcloud.vertex_positions_mut().iter().zip(pts.iter()) {
            assert_eq!(*pt1, *pt2);
        }
    }

    #[test]
    fn convert_test() {
        use crate::mesh::{PolyMesh, QuadMesh, TetMesh, TriMesh};

        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];

        let ptcloud = PointCloud::new(pts.clone());

        let polymesh = PolyMesh::new(pts.clone(), &vec![]);
        let trimesh = TriMesh::new(pts.clone(), vec![]);
        let quadmesh = QuadMesh::new(pts.clone(), vec![]);
        let tetmesh = TetMesh::new(pts.clone(), vec![]);
        // Test converting from a reference
        assert_eq!(PointCloud::from(&polymesh), ptcloud);
        assert_eq!(PointCloud::from(&trimesh), ptcloud);
        assert_eq!(PointCloud::from(&quadmesh), ptcloud);
        assert_eq!(PointCloud::from(&tetmesh), ptcloud);

        // Test consuming conversions
        assert_eq!(PointCloud::from(polymesh), ptcloud);
        assert_eq!(PointCloud::from(trimesh), ptcloud);
        assert_eq!(PointCloud::from(quadmesh), ptcloud);
        assert_eq!(PointCloud::from(tetmesh), ptcloud);
    }
}
