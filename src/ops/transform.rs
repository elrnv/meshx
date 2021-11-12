//!
//! This module defines a number of transformations that can be applied to geometrical objects like
//! meshes or point clouds.
//!

use crate::ops::Skew;
use math::{Matrix3, RealField, Vector3};
use num_traits::Zero;

pub trait Scale<T: Clone> {
    /// Scale the given object in 3D by a given vector of scale factors.
    /// `s = [1.0; 3]` corresponds to a noop.
    fn scale(&mut self, s: [T; 3]);
    /// Uniformly scale the given object by the given factor in all dimensions.
    fn uniform_scale(&mut self, s: T) {
        self.scale([s.clone(), s.clone(), s.clone()]);
    }
}

/// Rotate a given object by a certain amount. All functions rotate the object using the
/// right-hand-rule.
pub trait Rotate<T: RealField> {
    /// Rotate the object using the given column-major rotation matrix.
    fn rotate_by_matrix(&mut self, mtx: [[T; 3]; 3]);

    /// Rotate the object around the given unit vector `u` by the given angle `theta` (in radians).
    ///
    /// Note that it is assumed that `u` is indeed a unit vector, no further normalization should
    /// be performed.
    fn rotate(&mut self, axis: [T; 3], theta: T) {
        let u = Vector3::from(axis.clone());
        let [x, y, z] = axis;
        let id = Matrix3::identity();
        let u_skew = u.clone().skew();
        let cos_theta = theta.clone().cos();

        // Compute rotation matrix
        // R = cos(theta) * I + sin(theta)*[u]_X + (1 - cos(theta))(uu^T)
        // Compute outer product
        let u_v_t = {
            let [a, b, c]: [T; 3] = (u * (T::one() - cos_theta.clone())).into();
            Matrix3::from([
                [x.clone() * a.clone(), x.clone() * b.clone(), x * c.clone()],
                [y.clone() * a.clone(), y.clone() * b.clone(), y * c.clone()],
                [z.clone() * a, z.clone() * b, z * c],
            ])
        };
        let mtx = id * cos_theta + u_skew * theta.sin() + u_v_t;
        self.rotate_by_matrix(mtx.into());
    }

    /// Rotate the object using the given Euler vector (or rotation vector) `e`. The direction of
    /// `e` specifies the axis of rotation and its magnitude is the angle in radians.
    fn rotate_by_vector(&mut self, e: [T; 3])
    where
        T: Zero,
    {
        let e = Vector3::from(e);
        let theta = e.norm();
        if theta == T::zero() {
            return;
        }

        let u = e / theta.clone();
        self.rotate(u.into(), theta);
    }
}

pub trait Translate<T> {
    /// Translate the object by the given translation vector (displacement) `t`.
    fn translate(&mut self, t: [T; 3]);
}

/*
 * Functional variants of the above traits and their blanket implementations.
 */

pub trait Scaled<T>
where
    Self: Sized,
{
    /// Return a scaled version of `self`.
    fn scaled(self, s: [T; 3]) -> Self;
    /// Return a uniformly scaled version of `self`.
    fn uniformly_scaled(self, s: T) -> Self;
}

pub trait Rotated<T>
where
    Self: Sized,
{
    /// Return a version of `self` rotated about the unit vector `u` by the given angle `theta` (in
    /// radians).
    ///
    /// Note that it is assumed that `u` is indeed a unit vector, no further normalization should
    /// be performed.
    fn rotated(self, u: [T; 3], theta: T) -> Self;
    /// Return a version of `self` rotated using the given column-major rotation matrix
    fn rotated_by_matrix(self, mtx: [[T; 3]; 3]) -> Self;
    /// Return a version of `self` rotated about the Euler vector `e`.
    fn rotated_by_vector(self, e: [T; 3]) -> Self;
}

pub trait Translated<T>
where
    Self: Sized,
{
    /// Return a version of `self` translated by the given translation vector `t`.
    fn translated(self, t: [T; 3]) -> Self;
}

impl<S, T: Copy> Scaled<T> for S
where
    S: Scale<T> + Sized,
{
    fn scaled(mut self, s: [T; 3]) -> Self {
        self.scale(s);
        self
    }
    fn uniformly_scaled(mut self, s: T) -> Self {
        self.uniform_scale(s);
        self
    }
}

impl<S, T: RealField> Rotated<T> for S
where
    S: Rotate<T> + Sized,
{
    fn rotated(mut self, u: [T; 3], theta: T) -> Self {
        self.rotate(u, theta);
        self
    }
    fn rotated_by_matrix(mut self, mtx: [[T; 3]; 3]) -> Self {
        self.rotate_by_matrix(mtx);
        self
    }
    fn rotated_by_vector(mut self, e: [T; 3]) -> Self {
        self.rotate_by_vector(e);
        self
    }
}

impl<S, T> Translated<T> for S
where
    S: Translate<T> + Sized,
{
    fn translated(mut self, t: [T; 3]) -> Self {
        self.translate(t);
        self
    }
}
