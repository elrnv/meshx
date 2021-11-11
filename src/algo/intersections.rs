use ops::{Intersect};
use math::{Vector2};

/// Trait defining the necessary operations to resolve an intersection between two objects.
pub trait ResolveIntersection {
    type Output;

    fn resolve_intersection(self, rhs: RHS) -> Self::Output;
}

/// 2D triangle intersection resolution
impl ResolveIntersection<Output = Mesh> for Triangle<Vector2> {

}
