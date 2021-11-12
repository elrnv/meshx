pub mod transform;

pub use transform::*;

use crate::bbox::BBox;

pub trait Empty {
    /// Construct an empty object.
    fn empty() -> Self;
    /// Check if this object is empty.
    fn is_empty(&self) -> bool;
}

pub trait Contains<RHS = Self> {
    /// Check if this object contains another.
    fn contains(&self, obj: RHS) -> bool;
}

pub trait Absorb<RHS = Self> {
    type Output;

    /// Absorb another object.
    /// For example if a = [-1, 2] and b = [3, 4] is are closed intervals,
    /// then a.absorb(b) == [-1 4].
    fn absorb(self, rhs: RHS) -> Self::Output;
}

/// Intersection trait, describes the intersection operation between two objects.
pub trait Intersect<RHS = Self> {
    type Output;

    /// Intersect on one object with another, producing the resulting intersection.
    /// For example if [-1, 2] and [0, 4] are two closed intervals, then their
    /// intersection is a closed interval [0, 2].
    /// Note that the intersection object can be of a different type
    fn intersect(self, rhs: RHS) -> Self::Output;

    /// Check if this object intersects another.
    fn intersects(self, rhs: RHS) -> bool;
}

pub trait Centroid<T> {
    /// Compute the centroid of the object.
    fn centroid(self) -> T;
}

pub trait Area<T> {
    /// Compute the area of the object.
    fn area(self) -> T;

    /// Compute the signed area of the object. The area is negative when
    /// the object is inverted.
    fn signed_area(self) -> T;
}

pub trait Volume<T> {
    /// Compute the volume of the object.
    fn volume(self) -> T;

    /// Compute the signed volume of the object. The volume is negative when
    /// the object is inverted.
    fn signed_volume(self) -> T;
}

/// Shape matrices are useful for finite element analysis.
pub trait ShapeMatrix<M> {
    /// Return a shape matrix of the given type `M`.
    fn shape_matrix(self) -> M;
}

pub trait Normal<T> {
    /// Compute the unit normal of this object.
    fn normal(self) -> T;
}

pub trait BoundingBox<T> {
    /// Compute the bounding box of this object.
    fn bounding_box(&self) -> BBox<T>;
}

pub trait Skew {
    type Output;
    /// Produce a skew form of self. For instance a 3D vector can be rearranged in a skew symmetric
    /// matrix, that corresponds to the cross product operator.
    fn skew(&self) -> Self::Output;
}
