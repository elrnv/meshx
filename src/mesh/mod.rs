mod bench;

pub mod builder;
pub mod topology;
pub mod transform_impl;

pub mod pointcloud;
pub mod polymesh;
/// Macro to implement constant size slice iterator.
/// This can be used to create an iterator over cells of size 4 for instance like in TetMesh.
//macro_rules! impl_const_slice_iter {
//    ($iter:ident, $iter_mut:ident, [$type:ty;$n:expr]) => {
//        pub struct $iter<'a> {
//            data: &'a [$type],
//        }
//
//        impl<'a> $iter<'a> {
//            pub fn new(data: &'a [$type]) -> Self {
//                $iter {
//                    data
//                }
//            }
//        }
//
//        pub struct $iter_mut<'a> {
//            data: &'a mut [$type],
//        }
//
//        impl<'a> $iter_mut<'a> {
//            pub fn new(data: &'a mut [$type]) -> Self {
//                $iter_mut {
//                    data
//                }
//            }
//        }
//
//        impl<'a> Iterator for $iter<'a> {
//            type Item = &'a [$type;$n];
//
//            #[inline]
//            fn next(&mut self) -> Option<&'a [$type;$n]> {
//                if self.data.is_empty() {
//                    return None;
//                }
//                let (l, r) = self.data.split_at($n);
//                // Convert a slice reference to an array reference.
//                let l_arr = unsafe { &*(l.as_ptr() as *const [$type;$n]) };
//                self.data = r;
//                Some(l_arr)
//            }
//        }
//
//        impl<'a> Iterator for $iter_mut<'a> {
//            type Item = &'a mut [$type;$n];
//
//            #[inline]
//            fn next(&mut self) -> Option<&'a mut [$type;$n]> {
//                use std::mem;
//                if self.data.is_empty() {
//                    return None;
//                }
//
//                // Get a unique mutable reference for data.
//                let slice = mem::replace(&mut self.data, &mut []);
//                let (l, r) = slice.split_at_mut($n);
//                // Convert a slice reference to an array reference.
//                let l_arr = unsafe { &mut *(l.as_mut_ptr() as *mut [$type;$n]) };
//                self.data = r;
//                Some(l_arr)
//            }
//        }
//    }
//}
pub mod tetmesh;
pub mod uniform_poly_mesh;
pub mod unstructured_mesh;
pub mod vertex_positions;

// Re-export meshes and traits
pub use self::pointcloud::*;
pub use self::polymesh::*;
pub use self::tetmesh::*;
pub use self::uniform_poly_mesh::*;
pub use self::unstructured_mesh::*;
pub use self::vertex_positions::*; // reexport intrinsic attribute

use self::topology::NumVertices;
use super::attrib::{Attrib, VertexAttrib};

/// VertexMesh is a marker trait to allow user code to be generic over vertex centric meshes with
/// intrinsic vertex positions attributes.
pub trait VertexMesh<T>:
    Attrib + VertexAttrib + NumVertices + VertexPositions<Element = [T; 3]>
{
}
impl<M, T> VertexMesh<T> for M where
    M: Attrib + VertexAttrib + NumVertices + VertexPositions<Element = [T; 3]>
{
}
