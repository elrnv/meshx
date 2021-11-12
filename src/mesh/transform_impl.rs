//!
//! This module implements the transform ops from `meshx::ops::transform` for common meshes.
//!

use super::vertex_positions::VertexPositions;
use crate::ops::transform::*;

use math::{Matrix3, RealField, Vector3};
use num_traits::Float;

impl<T: RealField, M: VertexPositions<Element = [T; 3]>> Scale<T> for M {
    /// Scale a mesh in 3D by a given vector of scale factors.
    /// `s = [1.0; 3]` corresponds to a noop.
    fn scale(&mut self, [x, y, z]: [T; 3]) {
        for p in self.vertex_position_iter_mut() {
            p[0] *= x.clone();
            p[1] *= y.clone();
            p[2] *= z.clone();
        }
    }
}

impl<T: RealField + Float, M: VertexPositions<Element = [T; 3]>> Rotate<T> for M {
    /// Rotate the mesh using the given column-major rotation matrix.
    fn rotate_by_matrix(&mut self, mtx: [[T; 3]; 3]) {
        let mtx = Matrix3::from(mtx);
        for p in self.vertex_position_iter_mut() {
            let pos = Vector3::from(*p);
            *p = (mtx * pos).into();
        }
    }
}

impl<T: RealField, M: VertexPositions<Element = [T; 3]>> Translate<T> for M {
    /// Translate the mesh by the given translation vector (displacement) `t`.
    fn translate(&mut self, [x, y, z]: [T; 3]) {
        for p in self.vertex_position_iter_mut() {
            p[0] += x.clone();
            p[1] += y.clone();
            p[2] += z.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::builder::*;
    use crate::ops::*;
    use approx::*;

    #[test]
    fn rotate_grid() {
        let mut grid = GridBuilder {
            rows: 1,
            cols: 1,
            orientation: AxisPlaneOrientation::ZX,
        }
        .build();
        grid.rotate([0.0, 1.0, 0.0], std::f64::consts::PI * 0.25);
        let bbox = grid.bounding_box();

        let bound = 2.0_f64.sqrt();
        let minb = bbox.min_corner();
        let maxb = bbox.max_corner();

        assert_relative_eq!(minb[0], -bound);
        assert_eq!(minb[1], 0.0);
        assert_relative_eq!(minb[2], -bound);
        assert_relative_eq!(maxb[0], bound);
        assert_eq!(maxb[1], 0.0);
        assert_relative_eq!(maxb[2], bound);
    }
}
