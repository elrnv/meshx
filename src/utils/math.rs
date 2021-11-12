use crate::ops::Skew;
use math::{Matrix3, Scalar, Vector3};
use num_traits::Zero;
use std::ops::Neg;

impl<T: Zero + Neg<Output = T> + Clone> Skew for [T; 3] {
    type Output = [[T; 3]; 3];

    /// A skew symmetric representation of the vector that represents the cross product operator.
    fn skew(&self) -> Self::Output {
        let [x, y, z] = self;
        [
            [T::zero(), z.clone(), -y.clone()],
            [-z.clone(), T::zero(), x.clone()],
            [y.clone(), -x.clone(), T::zero()],
        ]
    }
}

impl<T: Scalar + Zero + Neg<Output = T>> Skew for Vector3<T> {
    type Output = Matrix3<T>;

    /// A skew symmetric representation of the vector that represents the cross product operator.
    fn skew(&self) -> Self::Output {
        let arr: [T; 3] = self.clone().into();
        Matrix3::from(arr.skew())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::{Matrix3, Vector3};

    #[test]
    fn skew() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(
            Matrix3::from(a.skew()) * Vector3::from(b),
            Vector3::from(a).cross(&Vector3::from(b))
        );
    }
}
