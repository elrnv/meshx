/*!
 * This module defines routines to compute normals on various meshes.
 */

use crate::prim::Triangle;
use math::Vector3;

pub fn compute_vertex_area_weighted_normals<T, V3>(
    vertex_positions: &[V3],
    topo: &[[usize; 3]],
    normals: &mut [[T; 3]],
) where
    T: Copy + math::SimdRealField,
    V3: Into<[T; 3]> + Clone,
{
    // Clear the normals.
    for nml in normals.iter_mut() {
        *nml = [T::zero(); 3];
    }

    for tri_indices in topo.iter() {
        let tri = Triangle::from_indexed_slice(tri_indices, vertex_positions);
        let area_nml = Vector3::from(tri.area_normal());
        normals[tri_indices[0]] = (Vector3::from(normals[tri_indices[0]]) + area_nml).into();
        normals[tri_indices[1]] = (Vector3::from(normals[tri_indices[1]]) + area_nml).into();
        normals[tri_indices[2]] = (Vector3::from(normals[tri_indices[2]]) + area_nml).into();
    }
}
