#![allow(dead_code)]

use crate::ops::*;
use num_traits::Float;
use std::ops::Sub;

/// Enum for accessing the endpoints of an interval.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub enum EndPoint {
    Lower,
    Upper,
}

/// Interval from a to b that contains both endponts,
/// where a < b on the floating point line.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub struct ClosedInterval<T> {
    /// The lower limit of the interval
    pub a: T,
    /// The upper limit of the interval
    pub b: T,
}

impl<T: PartialOrd + PartialEq + Copy> ClosedInterval<T> {
    pub fn new(a: T, b: T) -> ClosedInterval<T> {
        assert!(a <= b); // we shouldn't be creating empty intervals this way
        ClosedInterval { a, b }
    }

    /// Check if this interval contains just one point
    pub fn is_singleton(&self) -> bool {
        self.a == self.b
    }

    pub fn endpoint(&self, which: EndPoint) -> T {
        match which {
            EndPoint::Lower => self.a,
            EndPoint::Upper => self.b,
        }
    }
}

impl<T: Sub<Output = T> + Copy + PartialOrd> ClosedInterval<T> {
    pub fn length(&self) -> T {
        assert!(self.a <= self.b);
        self.b - self.a
    }
}

impl<T: Float> Default for ClosedInterval<T> {
    /// Create an empty interval.
    /// The choice of enpoints here is arbitrary as long as b < a.
    fn default() -> Self {
        ClosedInterval::empty()
    }
}

impl<T: Float> Empty for ClosedInterval<T> {
    #[inline]
    fn empty() -> Self {
        ClosedInterval {
            a: T::infinity(),
            b: T::neg_infinity(),
        }
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.a > self.b
    }
}

impl<T: PartialOrd> Contains<T> for ClosedInterval<T> {
    #[inline]
    fn contains(&self, x: T) -> bool {
        self.a <= x && x <= self.b
    }
}

impl<T: Float> Absorb<T> for ClosedInterval<T> {
    type Output = Self;

    #[inline]
    fn absorb(self, x: T) -> Self {
        ClosedInterval {
            a: T::min(x, self.a),
            b: T::max(x, self.b),
        }
    }
}

impl<T: Float> Absorb<Self> for ClosedInterval<T> {
    type Output = Self;

    #[inline]
    fn absorb(self, other: Self) -> Self {
        if other.is_empty() {
            self
        } else {
            ClosedInterval {
                a: T::min(self.a, other.a),
                b: T::max(self.b, other.b),
            }
        }
    }
}

impl<T: Float> Intersect<Self> for ClosedInterval<T> {
    type Output = Self;

    #[inline]
    fn intersect(self, interval: Self) -> Self {
        ClosedInterval {
            a: T::max(self.a, interval.a),
            b: T::min(self.b, interval.b),
        }
    }

    #[inline]
    fn intersects(self, interval: Self) -> bool {
        !(self.b < interval.a || self.a > interval.b)
    }
}

impl<T: Float> Centroid<Option<T>> for ClosedInterval<T> {
    fn centroid(self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(T::from(0.5).unwrap() * (self.a + self.b))
        }
    }
}

// -------------------------------------------------------------------------------------

/// Interval from a to b that doesn't contain the endpoints,
/// where a < b on the floating point line.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub struct OpenInterval<T> {
    /// The lower limit of the interval
    pub a: T,
    /// The upper limit of the interval
    pub b: T,
}

impl<T: PartialOrd + Copy> OpenInterval<T> {
    pub fn new(a: T, b: T) -> OpenInterval<T> {
        assert!(a < b); // we shouldn't be creating empty intervals this way
        OpenInterval { a, b }
    }

    pub fn endpoint(&self, which: EndPoint) -> T {
        match which {
            EndPoint::Lower => self.a,
            EndPoint::Upper => self.b,
        }
    }
}

impl<T: PartialOrd + Sub<Output = T> + Copy> OpenInterval<T> {
    pub fn length(&self) -> T {
        assert!(self.a <= self.b);
        self.b - self.a
    }
}

impl<T: Float> Default for OpenInterval<T> {
    /// Create an empty interval.
    /// The choice of endpoints here is arbitrary as long as b < a.
    fn default() -> Self {
        OpenInterval::empty()
    }
}

impl<T: Float> Empty for OpenInterval<T> {
    #[inline]
    fn empty() -> Self {
        OpenInterval {
            a: T::infinity(),
            b: T::neg_infinity(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.a >= self.b
    }
}

impl<T: PartialOrd> Contains<T> for OpenInterval<T> {
    #[inline]
    fn contains(&self, x: T) -> bool {
        self.a < x && x < self.b
    }
}

impl<T: Float> Absorb<Self> for OpenInterval<T> {
    type Output = Self;

    #[inline]
    fn absorb(self, other: Self) -> Self {
        if other.is_empty() {
            self
        } else {
            OpenInterval {
                a: T::min(self.a, other.a),
                b: T::max(self.b, other.b),
            }
        }
    }
}

impl<T: Float> Intersect<Self> for OpenInterval<T> {
    type Output = Self;

    #[inline]
    fn intersect(self, other: Self) -> Self {
        OpenInterval {
            a: T::max(self.a, other.a),
            b: T::min(self.b, other.b),
        }
    }

    #[inline]
    fn intersects(self, other: Self) -> bool {
        self.contains(other.a) || self.contains(other.b)
    }
}
