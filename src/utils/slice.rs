use dync::{traits::HasDrop, SliceMut};

// A helper trait for applying permutations for different collection types
pub trait Swap {
    fn len(&self) -> usize;
    fn swap(&mut self, i: usize, j: usize);
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Swap for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        self.as_mut_slice().swap(i, j);
    }
}

impl<T> Swap for [T] {
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        self.swap(i, j);
    }
}

impl<'a, V: HasDrop> Swap for SliceMut<'a, V> {
    #[inline]
    fn len(&self) -> usize {
        SliceMut::len(self)
    }
    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        self.swap(i, j);
    }
}

/// Apply a given `permutation` of indices to the given `array` of values in place.
///
/// # Example
///
/// ```
/// use meshx::utils::slice::apply_permutation;
/// let perm = vec![7, 8, 2, 3, 4, 1, 6, 5, 0];
/// let mut values = String::from("tightsemi");
/// // SAFETY: we are just permuting ASCII chars here for demonstrative purposes.
/// apply_permutation(&perm, unsafe { values.as_bytes_mut() });
/// assert_eq!(values, "mightiest");
/// ```
#[inline]
pub fn apply_permutation<A: Swap + ?Sized>(permutation: &[usize], array: &mut A) {
    let mut seen = vec![false; array.len()];
    apply_permutation_with_stride_and_seen(permutation, array, 1, &mut seen);
}

/// Apply a given `permutation` of indices to the given `array` of values in place.
///
/// This version of `apply_permutation` accepts a workspace `seen` vector of `bool`s.
/// This is useful when the allocation of `seen` affects performance.
#[inline]
pub fn apply_permutation_with_seen<A: Swap + ?Sized>(
    permutation: &[usize],
    array: &mut A,
    seen: &mut [bool],
) {
    apply_permutation_with_stride_and_seen(permutation, array, 1, seen);
}

/// Apply a given `permutation` of indices to the given `array` of values in place.
///
/// This version of `apply_permutation_with_seen` accepts a stride which interprets
/// `array` as an array of chunks with size `stride`.
pub fn apply_permutation_with_stride_and_seen<A: Swap + ?Sized>(
    permutation: &[usize],
    array: &mut A,
    stride: usize,
    seen: &mut [bool],
) {
    // Total number of elements being tracked.
    let nelem = seen.len();

    assert!(permutation.iter().all(|&i| i < nelem));
    assert_eq!(permutation.len(), nelem);
    debug_assert_eq!(nelem * stride, array.len());

    for unseen_i in 0..nelem {
        // SAFETY: unseen_i is explicitly between 0 and seen.len()
        if unsafe { *seen.get_unchecked(unseen_i) } {
            continue;
        }

        let mut i = unseen_i;
        loop {
            let idx = unsafe { *permutation.get_unchecked(i) };
            // SAFETY: checked permutation element bounds in the assert above.
            if unsafe { *seen.get_unchecked(idx) } {
                break;
            }

            for off in 0..stride {
                array.swap(off + stride * i, off + stride * idx);
            }

            // SAFETY: i is guaranteed to be < nelem
            unsafe { *seen.get_unchecked_mut(i) = true };
            i = idx;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_permutation_test() {
        // Checks inner loop in apply_permutation
        let perm = vec![7, 8, 2, 3, 4, 1, 6, 5, 0];
        let mut values = String::from("tightsemi");
        // SAFETY: we are just permuting ASCII chars here for demonstrative purposes.
        apply_permutation(&perm, unsafe { values.as_bytes_mut() });
        assert_eq!(values, "mightiest");

        // Checks the outer loop in apply_permutation
        let perm = vec![7, 8, 4, 3, 2, 1, 6, 5, 0];
        let mut values = String::from("tightsemi");
        let mut seen = vec![false; 9];
        // SAFETY: we are just permuting ASCII chars here for demonstrative purposes.
        apply_permutation_with_seen(&perm, unsafe { values.as_bytes_mut() }, &mut seen);
        assert_eq!(values, "mithgiest");

        let mut pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        seen.resize(5, false);
        seen.iter_mut().for_each(|b| *b = false);
        let order = [3, 2, 1, 4, 0];
        apply_permutation_with_seen(&order, &mut pts, &mut seen);
        assert_eq!(
            pts.as_slice(),
            &[
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        );
    }
}
