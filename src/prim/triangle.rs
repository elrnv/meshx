#![allow(dead_code)]
use crate::ops::*;
use crate::Pod;
use math::{convert, ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, Matrix3, Scalar, Vector3};
use num_traits::Zero;
use std::ops::Neg;

/// Generic triangle with three points
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Triangle<T: Scalar>(pub Vector3<T>, pub Vector3<T>, pub Vector3<T>);

impl<T: Scalar> Triangle<T> {
    /// Build a Triangle from a triplet of indices.
    #[inline]
    pub fn from_indexed_slice<V: Into<[T; 3]> + Clone>(
        indices: &[usize; 3],
        slice: &[V],
    ) -> Triangle<T> {
        Triangle(
            slice[indices[0]].clone().into().into(),
            slice[indices[1]].clone().into().into(),
            slice[indices[2]].clone().into().into(),
        )
    }

    /// Build a new triangle from an array of vertex positions.
    #[inline]
    pub fn new([a, b, c]: [[T; 3]; 3]) -> Self {
        Triangle(a.into(), b.into(), c.into())
    }
}

impl<T: Scalar + Pod> Triangle<T> {
    /// Return this triangle as an array of vertex positions.
    ///
    /// This can be useful for performing custom arithmetic on the triangle positions.
    #[inline]
    pub fn as_array(&self) -> &[[T; 3]; 3] {
        debug_assert_eq!(
            std::mem::size_of::<[[T; 3]; 3]>(),
            std::mem::size_of::<Triangle<T>>(),
        );

        unsafe { std::mem::transmute_copy(&self) }
    }
}

impl<T: Scalar> Triangle<T> {
    /// Convert this triangle into an array of vertex positions.
    #[inline]
    pub fn into_array(self) -> [[T; 3]; 3] {
        [self.0.into(), self.1.into(), self.2.into()]
    }
}

impl<T> Triangle<T>
where
    T: Scalar + Zero + ClosedAdd<T> + ClosedMul<T> + ClosedSub<T> + Neg<Output = T>,
{
    /// Compute the area weighted normal of this triangle. This is the standard way to compute the
    /// normal and the area of the triangle.
    ///
    /// Note the area of the triangle is half of the norm of this vector.
    #[inline]
    pub fn area_normal(&self) -> [T; 3] {
        let tri = self.clone();
        (tri.0.clone() - tri.1).cross(&(tri.0 - tri.2)).into()
    }

    /// Compute the gradient of the area weighted normal with respect to the given vertex.
    /// The returned matrix is in column-major format.
    #[inline]
    pub fn area_normal_gradient(&self, wrt: usize) -> [[T; 3]; 3] {
        let tri = self.clone();
        match wrt {
            0 => (tri.1.clone() - tri.2.clone()).skew(),
            1 => (tri.2 - tri.0.clone()).skew(),
            2 => (tri.0 - tri.1).skew(),
            _ => panic!("Triangle has only 3 vertices"),
        }
        .into()
    }

    /// Compute the hessian of the area weighted normal with respect to the given vertex (row
    /// index) at a given vertex (column index) multiplied by a given vector (`lambda`).
    ///
    /// The returned matrix is in column-major format.
    #[inline]
    pub fn area_normal_hessian_product(
        wrt_row: usize,
        at_col: usize,
        lambda: [T; 3],
    ) -> [[T; 3]; 3] {
        let zero = Matrix3::zero();
        let lambda = Vector3::from(lambda);
        match wrt_row {
            0 => match at_col {
                0 => zero,
                1 => (-lambda).skew(),
                2 => lambda.skew(),
                _ => panic!("Triangle has only 3 vertices"),
            },
            1 => match at_col {
                0 => lambda.skew(),
                1 => zero,
                2 => (-lambda).skew(),
                _ => panic!("Triangle has only 3 vertices"),
            },
            2 => match at_col {
                0 => (-lambda).skew(),
                1 => lambda.skew(),
                2 => zero,
                _ => panic!("Triangle has only 3 vertices"),
            },
            _ => panic!("Triangle has only 3 vertices"),
        }
        .into()
    }
}

impl<'a, T> Centroid<Vector3<T>> for &'a Triangle<T>
where
    T: Scalar + ClosedAdd<T> + ClosedDiv<T> + num_traits::FromPrimitive,
{
    #[inline]
    fn centroid(self) -> Vector3<T> {
        let tri = self.clone();
        (tri.0 + tri.1 + tri.2) / T::from_f64(3.0).unwrap()
    }
}

impl<'a, T> Normal<Vector3<T>> for &'a Triangle<T>
where
    T: math::SimdRealField,
{
    #[inline]
    fn normal(self) -> Vector3<T> {
        let nml: Vector3<T> = self.area_normal().into();
        let norm = nml.norm();
        nml / norm
    }
}

impl<'a, T> Centroid<[T; 3]> for &'a Triangle<T>
where
    T: Scalar + ClosedAdd<T> + ClosedDiv<T> + num_traits::FromPrimitive,
{
    #[inline]
    fn centroid(self) -> [T; 3] {
        <Self as Centroid<Vector3<T>>>::centroid(self).into()
    }
}

impl<'a, T> Normal<[T; 3]> for &'a Triangle<T>
where
    T: Scalar + ClosedAdd<T> + ClosedDiv<T> + num_traits::FromPrimitive,
{
    #[inline]
    fn normal(self) -> [T; 3] {
        <Self as Centroid<Vector3<T>>>::centroid(self).into()
    }
}

impl<T: Scalar> std::ops::Index<usize> for Triangle<T> {
    type Output = Vector3<T>;

    fn index(&self, i: usize) -> &Self::Output {
        match i {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => panic!("Index out of bounds: triangle has only 3 vertices."),
        }
    }
}

impl<T: Scalar> std::ops::IndexMut<usize> for Triangle<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        match i {
            0 => &mut self.0,
            1 => &mut self.1,
            2 => &mut self.2,
            _ => panic!("Index out of bounds: triangle has only 3 vertices."),
        }
    }
}

/// Returns a column major 3x2 matrix.
impl<'a, T: Scalar + ClosedSub<T>> ShapeMatrix<[[T; 3]; 2]> for &'a Triangle<T> {
    #[inline]
    fn shape_matrix(self) -> [[T; 3]; 2] {
        let tri = self.clone();
        [(tri.0 - tri.2.clone()).into(), (tri.1 - tri.2).into()]
    }
}

/// Returns a column major 3x2 matrix.
impl<T: Scalar + ClosedSub<T>> ShapeMatrix<[[T; 3]; 2]> for Triangle<T> {
    #[inline]
    fn shape_matrix(self) -> [[T; 3]; 2] {
        [(self.0 - self.2.clone()).into(), (self.1 - self.2).into()]
    }
}

impl<'a, T> Area<T> for &'a Triangle<T>
where
    T: math::SimdRealField,
{
    #[inline]
    fn area(self) -> T {
        self.signed_area()
    }
    #[inline]
    fn signed_area(self) -> T {
        Vector3::from(self.area_normal()).norm() / convert(2.0)
    }
}

impl<T> Area<T> for Triangle<T>
where
    T: math::SimdRealField,
{
    #[inline]
    fn area(self) -> T {
        // A triangle in 3D space can't be inverted.
        self.signed_area()
    }
    #[inline]
    fn signed_area(self) -> T {
        Vector3::from(self.area_normal()).norm() / convert(2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use autodiff::*;

    fn tri() -> Triangle<f64> {
        Triangle(
            Vector3::from([0.0, 0.0, 0.0]),
            Vector3::from([0.0, 0.0, 1.0]),
            Vector3::from([0.0, 1.0, 0.0]),
        )
    }

    #[test]
    fn area_test() {
        assert_relative_eq!(tri().area(), 0.5);
        assert_relative_eq!(tri().signed_area(), 0.5);
    }

    #[test]
    fn area_normal_gradient_test() {
        // Triangle in the XY plane.
        let trif = Triangle(
            Vector3::from([0.0, 0.0, 0.0]),
            Vector3::from([1.0, 0.0, 0.0]),
            Vector3::from([0.0, 1.0, 0.0]),
        );

        let mut tri = Triangle(
            trif[0].map(|x| F1::cst(x)),
            trif[1].map(|x| F1::cst(x)),
            trif[2].map(|x| F1::cst(x)),
        );

        // for each vertex
        for dvtx in 0..3 {
            let grad = trif.area_normal_gradient(dvtx);
            // for each component
            for i in 0..3 {
                tri[dvtx][i] = F::var(tri[dvtx][i]); // convert to variable
                let nml = tri.area_normal();
                tri[dvtx][i] = F::cst(tri[dvtx][i]); // convert to constant
                for j in 0..3 {
                    assert_relative_eq!(nml[j].deriv(), grad[j][i], max_relative = 1e-9);
                }
            }
        }
    }

    #[test]
    fn area_normal_hessian_test() {
        // Triangle in the XY plane.
        let trif = Triangle(
            Vector3::from([0.0, 0.0, 0.0]),
            Vector3::from([1.0, 0.0, 0.0]),
            Vector3::from([0.0, 1.0, 0.0]),
        );

        let mut tri = Triangle(
            trif[0].map(|x| F1::cst(x)),
            trif[1].map(|x| F1::cst(x)),
            trif[2].map(|x| F1::cst(x)),
        );

        let lambda = Vector3::from([0.2, 3.1, 42.0]); // some random multiplier

        // for each vertex
        for wrt_vtx in 0..3 {
            for i in 0..3 {
                tri[wrt_vtx][i] = F1::var(tri[wrt_vtx][i]); // convert to variable
                for at_vtx in 0..3 {
                    let hess_prod = Matrix3::from(Triangle::area_normal_hessian_product(
                        wrt_vtx,
                        at_vtx,
                        lambda.into(),
                    ));
                    // for each component
                    let ad_lambda = lambda.map(|x| F1::cst(x));
                    let grad_prod = Matrix3::from(tri.area_normal_gradient(at_vtx)) * ad_lambda;
                    for j in 0..3 {
                        assert_relative_eq!(
                            grad_prod[j].deriv(),
                            hess_prod[(i, j)],
                            max_relative = 1e-9
                        );
                    }
                }
                tri[wrt_vtx][i] = F1::cst(tri[wrt_vtx][i]); // convert to constant
            }
        }
    }
}
