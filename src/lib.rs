#![cfg_attr(feature = "unstable", feature(test, core_intrinsics, trace_macros))]

#[macro_use]
extern crate meshx_derive;

#[macro_use]
pub mod index;

pub mod bbox;
pub mod interval;
pub mod ops;
pub mod prim;

pub mod algo;
pub mod mesh;

#[cfg(feature = "io")]
pub mod io;

pub mod utils {
    pub mod math;
    pub mod slice;
}

// public re-exports
pub use self::index::Index;
pub use crate::mesh::topology::*;

/// Plain old data trait. Types that implement this trait contain no references and can be copied
/// with `memcpy`. The additional `Any` trait lets us inspect the type more easily.
pub trait Pod: 'static + Copy + Sized + Send + Sync + std::any::Any {}
impl<T> Pod for T where T: 'static + Copy + Sized + Send + Sync + std::any::Any {}

pub trait Real:
    math::ComplexField + num_traits::Float + ::std::fmt::Debug + std::iter::Sum + Pod
{
}
impl<T> Real for T where
    T: math::ComplexField + num_traits::Float + ::std::fmt::Debug + std::iter::Sum + Pod
{
}
