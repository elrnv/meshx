/*!
 * This module defines routines for dealing with meshes composed of multiple connected components.
 * # Implementation Notes
 *
 * Currently due to the limitations of the Rust language, it is not straightforward (or impossible)
 * to generalize the partition function over types that don't already implement `Eq` and `Hash`, if
 * we wanted to use a `HashMap`. That is we can't generalize the implementation to work with
 * floats or types that contain floats (e.g. arrays and tuples.
 *
 * For this reason there exist two variations of partition functions that work with different
 * assumptions on types.
 */

use crate::attrib::*;
use ahash::AHashMap as HashMap;
use std::hash::Hash;

/*
 * Partition function implementations
 */

/// Partition a given slice by unique values.
///
/// It may be more efficient to implement this function by hand, especially when the number of
/// partitions is known ahead of time.
pub fn partition<'a, T: Hash + Eq + 'a>(iter: impl Iterator<Item = &'a T>) -> (Vec<usize>, usize) {
    let mut partition = Vec::new();
    let mut map: HashMap<&'a T, usize> = HashMap::default();

    let mut part_counter = 0;
    for val in iter {
        let part = map.entry(val).or_insert_with(|| {
            let part = part_counter;
            part_counter += 1;
            part
        });
        partition.push(*part);
    }
    (partition, part_counter)
}

/// Partition a given slice by unique values.
///
/// This version of `partition` is useful when `T` doesn't implement `Eq` and `Hash` or
/// `PartitionHashEq` but has `PartialOrd`. For the majority of use cases it is better to use
/// `partition`.
pub fn partition_slice<T: PartialOrd>(slice: &[T]) -> (Vec<usize>, usize) {
    use std::cmp::Ordering;

    // Sort attrib via an explicit permutation.
    // The permutation then acts as a map from sorted back to unsorted attribs.
    let mut permutation: Vec<_> = (0..slice.len()).collect();

    // SAFETY: permutation indices are guaranteed to be below slice.len();
    permutation.sort_by(|&i, &j| unsafe {
        slice
            .get_unchecked(i)
            .partial_cmp(slice.get_unchecked(j))
            .unwrap_or(Ordering::Less)
    });

    let mut permutation_iter = permutation
        .iter()
        .map(|&i| (i, unsafe { slice.get_unchecked(i) }))
        .peekable();
    let mut partition = vec![0; slice.len()];
    let mut part_counter = 0;

    while let Some((pi, val)) = permutation_iter.next() {
        unsafe { *partition.get_unchecked_mut(pi) = part_counter };
        if permutation_iter.peek().map_or(true, |next| val != next.1) {
            part_counter += 1;
        }
    }

    (partition, part_counter)
}

pub trait Partition
where
    Self: Sized,
{
    /// Returns a partitioning by unique values of the given attribute.
    ///
    /// The returned vector consists of a ID assigned to each `Src` element identifying which
    /// partition it belongs to along with the total number of partitions.
    ///
    /// The attribute values must have type `T`.
    ///
    /// If `attrib` doesn't exist at the `Src` topology, the returned vector will consist of all
    /// zeros and the number of partitions will be 1.
    fn partition_by_attrib<T: AttributeValueHash, Src: AttribIndex<Self>>(
        &self,
        attrib: &str,
    ) -> (Vec<usize>, usize);

    /// Returns a partitioning by unique values of the given attribute.
    ///
    /// The returned vector consists of a ID assigned to each `Src` element identifying which
    /// partition it belongs to along with the total number of partitions.
    ///
    /// The attribute values must have type `T`.
    ///
    /// If `attrib` doesn't exist at the `Src` topology, the returned vector will consist of all
    /// zeros and the number of partitions will be 1.
    ///
    /// This version of `partition_by_attrib` uses sorting to determine unique values instead of a
    /// `HashMap`, and therefore only relies on `T` being `PartialOrd` but not `Eq` and `Hash`.
    fn partition_by_attrib_by_sort<T: PartialOrd + AttributeValue, Src: AttribIndex<Self>>(
        &self,
        attrib: &str,
    ) -> (Vec<usize>, usize);
}

impl<M> Partition for M
where
    Self: Attrib + Sized,
{
    #[inline]
    fn partition_by_attrib<T: AttributeValueHash, Src: AttribIndex<Self>>(
        &self,
        attrib: &str,
    ) -> (Vec<usize>, usize) {
        if let Ok(attrib_iter) = self.attrib_iter::<T, Src>(attrib) {
            partition(attrib_iter)
        } else {
            (vec![0; self.attrib_size::<Src>()], 1)
        }
    }

    #[inline]
    fn partition_by_attrib_by_sort<T: PartialOrd + AttributeValue, Src: AttribIndex<Self>>(
        &self,
        attrib: &str,
    ) -> (Vec<usize>, usize) {
        match self.attrib_as_slice::<T, Src>(attrib) {
            Ok(attrib) => partition_slice(attrib),
            Err(_) => (vec![0; self.attrib_size::<Src>()], 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::topology::*;
    use crate::mesh::{PointCloud, TetMeshExt};

    #[test]
    fn tetmesh_partition_by_attrib() {
        // The vertex positions are actually unimportant here.
        let verts = vec![[0.0; 3]; 12];

        // Topology is also unimportant for partitioning by attributes.
        let indices = vec![[0, 1, 2, 3], [1, 2, 3, 4], [5, 6, 7, 8], [8, 9, 10, 11]];

        let mut tetmesh = TetMeshExt::new(verts, indices);

        // Add an attribute that partitions out the first and last 2 vertices, which correspond to a whole
        // tet.
        tetmesh
            .insert_attrib_data::<usize, VertexIndex>(
                "attrib",
                vec![1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
            )
            .unwrap();

        // The resulting mesh must be identical
        assert_eq!(
            tetmesh.partition_by_attrib::<usize, VertexIndex>("attrib"),
            (vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 2)
        );
    }

    fn partition_data(size: usize, nbins: usize) -> Vec<[usize; 3]> {
        use rand::distributions::Uniform;
        use rand::prelude::*;

        let seed = [3u8; 32];
        let mut rng = StdRng::from_seed(seed);

        let index_bins = Uniform::from(0..nbins);

        let bins: Vec<[usize; 3]> = (0..nbins)
            .map(|_| [rng.gen(), rng.gen(), rng.gen()])
            .collect();

        (0..size)
            .map(|_| bins[index_bins.sample(&mut rng)])
            .collect()
    }

    #[test]
    fn partition_by_attrib_complex() {
        use rand::prelude::*;
        let size = 10;
        let seed = [3u8; 32];
        let mut rng = StdRng::from_seed(seed);

        // The vertex positions are actually unimportant here.
        let verts: Vec<[f64; 3]> = (0..size)
            .map(|_| [rng.gen(), rng.gen(), rng.gen()])
            .collect();
        let mut ptcld = PointCloud::new(verts);

        let data = partition_data(size, 10);
        ptcld
            .insert_attrib_data::<_, VertexIndex>("attrib", data)
            .unwrap();

        let (_, num_parts1) = ptcld.partition_by_attrib::<[usize; 3], VertexIndex>("attrib");
        let (_, num_parts2) =
            ptcld.partition_by_attrib_by_sort::<[usize; 3], VertexIndex>("attrib");
        assert_eq!(num_parts1, num_parts2);
    }

    #[test]
    fn partition_by_attrib_by_hash() {
        use rand::prelude::*;
        let size = 100_000;
        let seed = [3u8; 32];
        let mut rng = StdRng::from_seed(seed);

        // The vertex positions are actually unimportant here.
        let verts: Vec<[f64; 3]> = (0..size)
            .map(|_| [rng.gen(), rng.gen(), rng.gen()])
            .collect();
        let mut ptcld = PointCloud::new(verts);

        let data = partition_data(size, 100);
        ptcld
            .insert_attrib_data::<_, VertexIndex>("attrib", data)
            .unwrap();

        let now = std::time::Instant::now();
        let (_, num_parts) = ptcld.partition_by_attrib::<[usize; 3], VertexIndex>("attrib");
        eprintln!("hash time = {}", now.elapsed().as_millis());
        eprintln!("{}", num_parts);
    }

    #[test]
    fn partition_by_attrib_by_sort() {
        use rand::prelude::*;
        let size = 100_000;
        let seed = [3u8; 32];
        let mut rng = StdRng::from_seed(seed);

        // The vertex positions are actually unimportant here.
        let verts: Vec<[f64; 3]> = (0..size)
            .map(|_| [rng.gen(), rng.gen(), rng.gen()])
            .collect();
        let mut ptcld = PointCloud::new(verts);

        let data = partition_data(size, 100);
        ptcld
            .insert_attrib_data::<_, VertexIndex>("attrib", data)
            .unwrap();

        let now = std::time::Instant::now();
        let (_, num_parts) = ptcld.partition_by_attrib_by_sort::<[usize; 3], VertexIndex>("attrib");
        eprintln!("sort time = {}", now.elapsed().as_millis());
        eprintln!("{}", num_parts);
    }
}
