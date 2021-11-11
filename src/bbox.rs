use crate::interval::{ClosedInterval, EndPoint};
use crate::ops::*;
use math::Vector3;
use num_traits::Float;

/// Structure for accessing corner positions of a bounding box.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Corner {
    pub x: EndPoint,
    pub y: EndPoint,
    pub z: EndPoint,
}

/// Corner can be encoded by 3 bits, identifying the positions of all 8 corners of a box.
/// # Examples
/// ```
/// # use meshx::bbox::Corner;
/// use meshx::interval::EndPoint::*;
///
/// assert_eq!(Corner::from(0b000), Corner { x: Lower, y: Lower, z: Lower });
/// assert_eq!(Corner::from(0b001), Corner { x: Upper, y: Lower, z: Lower });
/// assert_eq!(Corner::from(0b011), Corner { x: Upper, y: Upper, z: Lower });
/// ```
impl From<u8> for Corner {
    fn from(bits: u8) -> Corner {
        debug_assert!(bits < 8); // We only read the first 3 bits.
        Corner {
            x: if bits & 0b001 != 0 {
                EndPoint::Upper
            } else {
                EndPoint::Lower
            },
            y: if bits & 0b010 != 0 {
                EndPoint::Upper
            } else {
                EndPoint::Lower
            },
            z: if bits & 0b100 != 0 {
                EndPoint::Upper
            } else {
                EndPoint::Lower
            },
        }
    }
}

/// Corner can be encoded by 3 bits, identifying the positions of all 8 corners of a box.
/// # Examples
/// ```
/// # use meshx::bbox::Corner;
/// use meshx::interval::EndPoint::*;
///
/// assert_eq!(0b000u8, Corner { x: Lower, y: Lower, z: Lower }.into());
/// assert_eq!(0b001u8, Corner { x: Upper, y: Lower, z: Lower }.into());
/// assert_eq!(0b011u8, Corner { x: Upper, y: Upper, z: Lower }.into());
/// ```
impl From<Corner> for u8 {
    fn from(c: Corner) -> u8 {
        let mut bits = if c.x == EndPoint::Upper { 0b001 } else { 0 };
        bits |= if c.y == EndPoint::Upper { 0b010 } else { 0 };
        bits |= if c.z == EndPoint::Upper { 0b100 } else { 0 };
        bits
    }
}

/// General purpose bounding box structure.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct BBox<T> {
    interval_x: ClosedInterval<T>,
    interval_y: ClosedInterval<T>,
    interval_z: ClosedInterval<T>,
}

impl<T: Float> BBox<T> {
    /// Construct a bounding box containing points `minb` and `maxb`.
    /// # Examples
    /// ```
    /// # use meshx::bbox::*;
    /// # use meshx::interval::ClosedInterval;
    /// let min_p = [0.0; 3];
    /// let max_p = [1.5, 0.5, 1.0];
    /// let sample_box = BBox::<f32>::new(min_p, max_p);
    ///
    /// assert_eq!(sample_box.min_corner(), min_p);
    /// assert_eq!(sample_box.max_corner(), max_p);
    /// ```
    pub fn new(minb: [T; 3], maxb: [T; 3]) -> Self {
        let mut empty_bbox = BBox::empty();
        empty_bbox.absorb(minb).absorb(maxb);
        empty_bbox
    }

    /// Construct a unit bounding box from zero to one in each dimension.
    /// # Examples
    /// ```
    /// # use meshx::bbox::*;
    /// # use meshx::interval::ClosedInterval;
    /// let unit_box = BBox::<f32>::new(
    ///     [0.0; 3],
    ///     [1.0; 3]);
    /// assert_eq!(BBox::<f32>::unit(), unit_box);
    /// ```
    pub fn unit() -> BBox<T> {
        let zero_to_one = ClosedInterval::new(T::zero(), T::one());
        BBox {
            interval_x: zero_to_one,
            interval_y: zero_to_one,
            interval_z: zero_to_one,
        }
    }

    /// Get the size of the box in each dimension.
    /// # Examples
    /// ```
    /// # use meshx::bbox::*;
    /// # use meshx::interval::ClosedInterval;
    /// let unit_box = BBox::<f32>::unit();
    /// assert_eq!(unit_box.size(), [1.0, 1.0, 1.0]);
    /// ```
    pub fn size(&self) -> [T; 3] {
        [
            self.interval_x.length(),
            self.interval_y.length(),
            self.interval_z.length(),
        ]
    }

    /// Determine the axis of the maximum side length of the box and return
    /// both: the axis index (x => 0, y => 1, z => 2) and the length.
    /// # Examples
    /// ```
    /// # use meshx::bbox::*;
    /// # use meshx::interval::ClosedInterval;
    /// let min_p = [0.0; 3];
    /// let max_p = [1.5, 0.5, 1.0];
    /// let sample_box = BBox::<f32>::new(min_p, max_p);
    ///
    /// assert_eq!(sample_box.max_axis(), (0, 1.5));
    /// ```
    pub fn max_axis(&self) -> (u8, T) {
        let size = self.size();
        if size[0] > size[1] {
            if size[0] > size[2] {
                (0, size[0])
            } else {
                (2, size[2])
            }
        } else if size[1] > size[2] {
            (1, size[1])
        } else {
            (2, size[2])
        }
    }

    pub fn min_corner(&self) -> [T; 3] {
        self.corner(0b000)
    }

    pub fn max_corner(&self) -> [T; 3] {
        self.corner(0b111)
    }

    /// Diameter of the bounding box is the distance between the min and max corners.
    pub fn diameter(&self) -> T
    where
        T: math::RealField,
    {
        (Vector3::from(self.max_corner()) - Vector3::from(self.min_corner())).norm()
    }

    pub fn corner<C: Copy + Into<Corner>>(&self, which: C) -> [T; 3] {
        [
            self.interval_x.endpoint(which.into().x),
            self.interval_y.endpoint(which.into().y),
            self.interval_z.endpoint(which.into().z),
        ]
    }
}

impl<T: Float> Empty for BBox<T> {
    fn empty() -> Self {
        BBox {
            interval_x: ClosedInterval::empty(),
            interval_y: ClosedInterval::empty(),
            interval_z: ClosedInterval::empty(),
        }
    }
    fn is_empty(&self) -> bool {
        self.interval_x.is_empty() || self.interval_y.is_empty() || self.interval_z.is_empty()
    }
}

impl<T: PartialOrd + Copy> Contains<[T; 3]> for BBox<T> {
    fn contains(&self, p: [T; 3]) -> bool {
        self.interval_x.contains(p[0])
            && self.interval_y.contains(p[1])
            && self.interval_z.contains(p[2])
    }
}

impl<'a, T: Float, P> Absorb<P> for &'a mut BBox<T>
where
    P: Into<[T; 3]>,
{
    type Output = &'a mut BBox<T>;

    fn absorb(self, p: P) -> Self::Output {
        let p = p.into();
        self.interval_x = self.interval_x.absorb(p[0]);
        self.interval_y = self.interval_y.absorb(p[1]);
        self.interval_z = self.interval_z.absorb(p[2]);
        self
    }
}

impl<'a, T: Float> Absorb<BBox<T>> for &'a mut BBox<T> {
    type Output = &'a mut BBox<T>;

    fn absorb(self, other: BBox<T>) -> Self::Output {
        if !other.is_empty() {
            self.interval_x = self.interval_x.absorb(other.interval_x);
            self.interval_y = self.interval_y.absorb(other.interval_y);
            self.interval_z = self.interval_z.absorb(other.interval_z);
        }
        self
    }
}

impl<T: Float> Intersect<Self> for BBox<T> {
    type Output = Self;

    fn intersect(self, other: Self) -> Self {
        BBox {
            interval_x: self.interval_x.intersect(other.interval_x),
            interval_y: self.interval_y.intersect(other.interval_y),
            interval_z: self.interval_z.intersect(other.interval_z),
        }
    }

    #[inline]
    fn intersects(self, other: Self) -> bool {
        self.interval_x.intersects(other.interval_x)
            && self.interval_y.intersects(other.interval_y)
            && self.interval_z.intersects(other.interval_z)
    }
}

impl<'a, T: Float> Centroid<Option<[T; 3]>> for &'a BBox<T> {
    fn centroid(self) -> Option<[T; 3]> {
        if let Some(c_x) = self.interval_x.centroid() {
            if let Some(c_y) = self.interval_y.centroid() {
                if let Some(c_z) = self.interval_z.centroid() {
                    return Some([c_x, c_y, c_z]);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_construct_test() {
        let unit_box = BBox::<f32>::new([0.0; 3], [1.0; 3]);
        assert_eq!(unit_box, BBox::<f32>::unit());
    }

    #[test]
    fn bbox_empty_test() {
        let empty_box = BBox::<f32>::empty();
        assert_eq!(
            empty_box,
            BBox {
                interval_x: ClosedInterval::empty(),
                interval_y: ClosedInterval::empty(),
                interval_z: ClosedInterval::empty(),
            }
        );
    }

    #[test]
    fn bbox_contains_test() {
        let empty_box = BBox::<f32>::empty();
        assert!(!empty_box.contains([0.0; 3]));
        assert!(!empty_box.contains([1.0; 3]));

        let unit_box = BBox::<f32>::unit();
        assert!(unit_box.contains([0.0; 3]));
        assert!(unit_box.contains([1.0; 3]));
        assert!(unit_box.contains([0.5, 0.5, 1.0]));
        assert!(unit_box.contains([0.5; 3]));
        assert!(!unit_box.contains([1.5; 3]));
    }

    #[test]
    fn bbox_absorb_test() {
        let mut bbox = BBox::<f32>::empty();
        bbox.absorb([-1.0; 3]);
        bbox.absorb([1.0; 3]);
        assert!(bbox.contains([0.0; 3]));
        assert!(bbox.contains([1.0; 3]));
        assert!(bbox.contains([0.5, 0.5, 1.0]));
        assert!(bbox.contains([0.5; 3]));
        assert!(bbox.contains([-0.5; 3]));
        assert!(!bbox.contains([-1.5; 3]));
    }

    #[test]
    fn bbox_centroid_test() {
        let bbox = BBox::<f32>::unit();
        assert_eq!(bbox.centroid(), Some([0.5; 3]));
        let empty_bbox = BBox::<f32>::empty();
        assert_eq!(empty_bbox.centroid(), None);
    }

    // TODO: intersect test
}
