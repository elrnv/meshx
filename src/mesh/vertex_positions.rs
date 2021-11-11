//!
//! This module defines the `VertexPositions` trait, which defines a core intrinsic attribute for
//! most standard meshes.
//!

use crate::bbox::BBox;
use crate::index::*;
use crate::mesh::topology::*;
use crate::ops::*;
use crate::Real;
use std::slice::{Iter, IterMut};

/// An "intrinsic" trait for accessing vertex positions on a mesh.
/// This trait can be implemented automatically by deriving the virtual "Intrinsic" trait and
/// taggin the field with the vertex positions field with the `#[intrinsic(VertexPositions)]`
/// attribute. Make sure that `VertexPositions` is in scope, or specify the path in the argument to
/// the `intrinsic` attribute directly.
pub trait VertexPositions {
    type Element: Copy;

    /// Vertex positions as a slice of triplets.
    fn vertex_positions(&self) -> &[Self::Element];

    /// Vertex positions as a mutable slice of triplets.
    fn vertex_positions_mut(&mut self) -> &mut [Self::Element];

    /// Vertex iterator.
    #[inline]
    fn vertex_position_iter(&self) -> Iter<Self::Element> {
        self.vertex_positions().iter()
    }

    /// Mutable vertex iterator.
    #[inline]
    fn vertex_position_iter_mut(&mut self) -> IterMut<Self::Element> {
        self.vertex_positions_mut().iter_mut()
    }

    /// Vertex accessor.
    #[inline]
    fn vertex_position<VI>(&self, vidx: VI) -> Self::Element
    where
        VI: Into<VertexIndex>,
    {
        let idx: Index = vidx.into().into();
        self.vertex_positions()[idx.unwrap()]
    }
}

impl<M, T: Real> BoundingBox<T> for M
where
    M: VertexPositions<Element = [T; 3]>,
{
    /// Compute the bounding box of this object.
    fn bounding_box(&self) -> BBox<T> {
        let mut bbox = BBox::empty();
        for &pos in self.vertex_position_iter() {
            bbox.absorb(pos);
        }
        bbox
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::TriMesh;

    #[test]
    fn triangle_bounding_box() {
        let pos = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let trimesh = TriMesh::new(pos, vec![[0, 1, 2]]);

        let bbox = trimesh.bounding_box();

        assert_eq!(bbox.min_corner(), [0.0, 0.0, 0.0]);
        assert_eq!(bbox.max_corner(), [1.0, 0.0, 1.0]);
    }
}
