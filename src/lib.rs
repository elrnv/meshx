#![cfg_attr(feature = "unstable", feature(test, core_intrinsics, trace_macros))]

//! A mesh exchange library for providing convenient conversion utilities between popular mesh formats.
//!
//! # Overview
//!
//! This library is designed to simplify interoperability between different 3D applications using mesh data
//! structures. `meshx` also provides common mesh types and APIs to work with attributes.

#[macro_use]
extern crate meshx_derive;

#[macro_use]
pub mod index;

pub mod bbox;
pub mod interval;
pub mod ops;
pub mod prim;

pub mod attrib;

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
pub use crate::mesh::*;

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
