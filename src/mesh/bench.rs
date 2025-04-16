/**
 * This module benchmarks the performance of the Attribute implementation.
 */

#[cfg(all(feature = "unstable", test))]
mod bench {
    extern crate test;
    use crate::mesh::*;
    use crate::Real;
    use dync::VecDrop;
    use rand::distr::{Distribution, StandardUniform};
    use rand::prelude::*;
    use test::Bencher;
    use topology::*;

    static SEED: [u8; 32] = [3; 32];
    static BUF_SIZE: usize = 1000;

    #[inline]
    fn transform_point<T: Real>(pt: &mut [T; 3]) {
        pt[0] += T::from(0.1).unwrap();
        pt[1] += T::from(0.2).unwrap();
        pt[2] += T::from(0.3).unwrap();
    }

    #[inline]
    fn transform_point_vec<T: Real>(vec: &mut Vec<[T; 3]>) {
        for v in vec.iter_mut() {
            transform_point(v);
        }
    }

    #[inline]
    fn prepare_points<T>() -> Vec<[T; 3]>
    where
        StandardUniform: Distribution<T>,
    {
        let n = BUF_SIZE;

        let mut pts = Vec::new();
        let mut rng = StdRng::from_seed(SEED);
        for _ in 0..n {
            pts.push([rng.random(), rng.random(), rng.random()]);
        }

        pts
    }

    #[inline]
    fn prepare_trimesh<T>() -> TriMesh<T>
    where
        T: 'static + Real + Default,
        StandardUniform: Distribution<T>,
    {
        let pts = prepare_points::<T>();

        let mut trimesh = TriMesh::new(pts.clone(), Vec::new());

        trimesh.insert_attrib_data::<_, VertexIndex>("P", pts).ok();

        trimesh
    }

    /// Measure the performance of iterating over a standard `Vec<T>`.
    /// This serves as a benchmark we aim to approach.
    #[bench]
    fn transform_position_vec(b: &mut Bencher) {
        let mut pts = prepare_points::<f64>();

        b.iter(|| {
            transform_point_vec(&mut pts);
        });
    }

    /// Measure the performance of iterating over our `VecDrop` type.
    #[bench]
    fn transform_point_buffer(b: &mut Bencher) {
        let pts = prepare_points::<f64>();

        let mut data: VecDrop = VecDrop::from_vec(pts.clone());
        b.iter(|| {
            for v in data.iter_mut_as::<[f64; 3]>().unwrap() {
                transform_point(v);
            }
        });
    }

    /// Measure the performance of iterating over a direct attribute.
    #[bench]
    fn transform_point_attribute(b: &mut Bencher) {
        let mut trimesh = prepare_trimesh::<f64>();

        let attrib = trimesh.attrib_mut::<VertexIndex>("P").unwrap();
        b.iter(|| {
            for v in attrib.direct_iter_mut::<[f64; 3]>().unwrap() {
                transform_point(v);
            }
        });
    }

    /// Measure the performance of iterating over indices of the attribute and
    /// accessing and modifying its contents directly via `get_mut`.
    /// This is expected to be slower, so for most applications we should aim
    /// to use an iterator.
    #[bench]
    fn transform_point_attribute_with_get_mut(b: &mut Bencher) {
        let mut trimesh = prepare_trimesh::<f64>();

        let attrib = trimesh.attrib_mut::<VertexIndex>("P").unwrap();

        b.iter(|| {
            for i in 0..attrib.len() {
                let v = attrib.get_mut::<[f64; 3], _>(i).unwrap();
                transform_point(v);
            }
        });
    }

    #[bench]
    fn transform_point_attribute_with_get_unchecked_mut(b: &mut Bencher) {
        let mut trimesh = prepare_trimesh::<f64>();

        let attrib = trimesh
            .attrib_mut::<VertexIndex>("P")
            .unwrap()
            .data
            .direct_data_mut()
            .unwrap();

        b.iter(|| {
            for i in 0..attrib.len() {
                let v = unsafe { attrib.get_unchecked_mut::<[f64; 3]>(i) };
                transform_point(v);
            }
        });
    }
}
