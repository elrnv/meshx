use crate::ops::{Centroid, ShapeMatrix, Volume};
use crate::Pod;
use math::{ClosedAddAssign, ClosedMulAssign, ClosedSubAssign, Matrix3, RealField, Scalar, Vector3};
use num_traits::FromPrimitive;
use std::ops::{Add, Mul, Sub};

/// Generic tetrahedron with four points
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Tetrahedron<T: Scalar>(
    pub Vector3<T>,
    pub Vector3<T>,
    pub Vector3<T>,
    pub Vector3<T>,
);

impl<T: Scalar> Tetrahedron<T> {
    /// Build a Tetrahedron from a quadruple of indices.
    #[inline]
    pub fn from_indexed_slice<V: Into<[T; 3]> + Clone>(
        indices: &[usize; 4],
        slice: &[V],
    ) -> Tetrahedron<T> {
        Tetrahedron(
            slice[indices[0]].clone().into().into(),
            slice[indices[1]].clone().into().into(),
            slice[indices[2]].clone().into().into(),
            slice[indices[3]].clone().into().into(),
        )
    }

    /// Build a new tetrahedron from an array of vertex positions.
    #[inline]
    pub fn new([a, b, c, d]: [[T; 3]; 4]) -> Self {
        Tetrahedron(a.into(), b.into(), c.into(), d.into())
    }
}

impl<T: Scalar + Pod> Tetrahedron<T> {
    /// Return this tetrahedron as an array of vertex positions.
    ///
    /// This can be useful for performing custom arithmetic on the tetrahedron positions.
    #[inline]
    pub fn as_array(&self) -> &[[T; 3]; 4] {
        debug_assert_eq!(
            std::mem::size_of::<[[T; 3]; 4]>(),
            std::mem::size_of::<Tetrahedron<T>>(),
        );

        unsafe { std::mem::transmute_copy(&self) }
    }
}

impl<T: Scalar> Tetrahedron<T> {
    /// Convert this tet into an array of vertex positions.
    #[inline]
    pub fn into_array(self) -> [[T; 3]; 4] {
        [self.0.into(), self.1.into(), self.2.into(), self.3.into()]
    }
}

impl<T: Scalar + ClosedAddAssign<T>> Add for Tetrahedron<T> {
    type Output = Tetrahedron<T>;

    fn add(self, other: Tetrahedron<T>) -> Tetrahedron<T> {
        Tetrahedron(
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2,
            self.3 + other.3,
        )
    }
}

impl<'a, T: Scalar + ClosedAddAssign<T>> Add for &'a Tetrahedron<T> {
    type Output = Tetrahedron<T>;

    fn add(self, other: &Tetrahedron<T>) -> Tetrahedron<T> {
        Tetrahedron(
            self.0.clone() + other.0.clone(),
            self.1.clone() + other.1.clone(),
            self.2.clone() + other.2.clone(),
            self.3.clone() + other.3.clone(),
        )
    }
}

impl<T: Scalar + ClosedSubAssign<T>> Sub for Tetrahedron<T> {
    type Output = Tetrahedron<T>;

    fn sub(self, other: Tetrahedron<T>) -> Tetrahedron<T> {
        Tetrahedron(
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2,
            self.3 - other.3,
        )
    }
}

impl<'a, T: Scalar + ClosedSubAssign<T>> Sub for &'a Tetrahedron<T> {
    type Output = Tetrahedron<T>;

    fn sub(self, other: &Tetrahedron<T>) -> Tetrahedron<T> {
        Tetrahedron(
            self.0.clone() - other.0.clone(),
            self.1.clone() - other.1.clone(),
            self.2.clone() - other.2.clone(),
            self.3.clone() - other.3.clone(),
        )
    }
}

impl<T: Scalar + ClosedMulAssign<T>> Mul<T> for Tetrahedron<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Tetrahedron(
            self.0 * rhs.clone(),
            self.1 * rhs.clone(),
            self.2 * rhs.clone(),
            self.3 * rhs,
        )
    }
}

impl<'a, T: Scalar + ClosedMulAssign<T>> Mul<T> for &'a Tetrahedron<T> {
    type Output = Tetrahedron<T>;

    fn mul(self, rhs: T) -> Tetrahedron<T> {
        let tet = self.clone();
        Tetrahedron(
            tet.0 * rhs.clone(),
            tet.1 * rhs.clone(),
            tet.2 * rhs.clone(),
            tet.3 * rhs,
        )
    }
}

impl<'a, T: Scalar + ClosedAddAssign<T> + ClosedMulAssign<T> + FromPrimitive> Centroid<[T; 3]>
    for &'a Tetrahedron<T>
{
    #[inline]
    fn centroid(self) -> [T; 3] {
        let tet = self.clone();
        ((tet.0 + tet.1 + tet.2 + tet.3) * T::from_f64(0.25).unwrap()).into()
    }
}

/// Column-major array matrix.
impl<'a, T: Scalar + ClosedSubAssign<T>> ShapeMatrix<[[T; 3]; 3]> for &'a Tetrahedron<T> {
    #[inline]
    fn shape_matrix(self) -> [[T; 3]; 3] {
        let tet = self.clone();
        [
            (tet.0 - tet.3.clone()).into(),
            (tet.1 - tet.3.clone()).into(),
            (tet.2 - tet.3).into(),
        ]
    }
}

impl<T: Scalar + ClosedSubAssign<T>> ShapeMatrix<[[T; 3]; 3]> for Tetrahedron<T> {
    #[inline]
    fn shape_matrix(self) -> [[T; 3]; 3] {
        [
            (self.0 - self.3.clone()).into(),
            (self.1 - self.3.clone()).into(),
            (self.2 - self.3).into(),
        ]
    }
}

impl<'a, T: RealField> Volume<T> for &'a Tetrahedron<T> {
    #[inline]
    fn volume(self) -> T {
        self.signed_volume().abs()
    }
    #[inline]
    fn signed_volume(self) -> T {
        Matrix3::from(self.shape_matrix()).determinant() / math::convert(6.0)
    }
}

impl<T: RealField> Volume<T> for Tetrahedron<T> {
    #[inline]
    fn volume(self) -> T {
        self.signed_volume().abs()
    }
    #[inline]
    fn signed_volume(self) -> T {
        Matrix3::from(self.shape_matrix()).determinant() / math::convert(6.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn tet() -> Tetrahedron<f64> {
        Tetrahedron(
            Vector3::from([0.0, 1.0, 0.0]),
            Vector3::from([-0.94281, -0.33333, 0.0]),
            Vector3::from([0.471405, -0.33333, 0.816498]),
            Vector3::from([0.471405, -0.33333, -0.816498]),
        )
    }

    #[test]
    fn volume_test() {
        assert_relative_eq!(tet().volume(), 0.5132, epsilon = 1e-4);
        assert_relative_eq!(tet().signed_volume(), 0.5132, epsilon = 1e-4);
    }

    #[test]
    fn shape_matrix_test() {
        let mtx = tet().shape_matrix();
        assert_relative_eq!(mtx[0][0], -0.471405, epsilon = 1e-4);
        assert_relative_eq!(mtx[0][1], 1.33333, epsilon = 1e-4);
        assert_relative_eq!(mtx[0][2], 0.816498, epsilon = 1e-4);
        assert_relative_eq!(mtx[1][0], -1.41421, epsilon = 1e-4);
        assert_relative_eq!(mtx[1][1], 0.0, epsilon = 1e-4);
        assert_relative_eq!(mtx[1][2], 0.816498, epsilon = 1e-4);
        assert_relative_eq!(mtx[2][0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(mtx[2][1], 0.0, epsilon = 1e-4);
        assert_relative_eq!(mtx[2][2], 1.633, epsilon = 1e-4);
    }

    #[test]
    fn centroid_test() {
        let c = tet().centroid();
        assert_relative_eq!(c[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(c[1], 0.0, epsilon = 1e-4);
        assert_relative_eq!(c[2], 0.0, epsilon = 1e-4);
    }
}
