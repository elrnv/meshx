//!
//! This derive macro allows callers to automatically derive an intrinsic attribute for a mesh.
//! For instance, many meshes are build around vertex positions and primitive indices. This means
//! that many algorithms can be implemented in a mesh agnostic way, looking solely at their
//! intrinsic attributes. This macro lets the caller define a custom intrinsic attribute and derive
//! it automatically for a mesh with a field that provides this attribute. The intrinsic trait must
//! have a particular structure:
//! ```ignore
//! trait <name> {
//!     type Element;
//!     fn <field>(&self) -> &[Self::Element];
//!     fn <field>_mut(&mut self) -> &mut [Self::Element];
//!     ... (other automatically implemented methods) ...
//! }
//! ```
//! where `<name>` is the name of the desired intrinsic trait, and `<field>` is the name of the
//! desired field that must be present in the implementing struct.
//!
//! In the example of vertex positions, a trait may be defined as
//! ```
//! trait VertexPositions {
//!     type Element;
//!     fn vertex_positions(&self) -> &[Self::Element];
//!     fn vertex_positions_mut(&mut self) -> &mut [Self::Element];
//!     fn vertex_position_iter(&self) -> std::slice::Iter<Self::Element> {
//!         self.vertex_positions().iter()
//!     }
//! }
//! ```
//!
//! and derived on a mesh as follows
//! ```ignore
//! # extern crate meshx;
//! # use meshx_derive::Intrinsic;
//! # use meshx::IntrinsicAttribute;
//! #[derive(Intrinsic)]
//! struct Mesh {
//!     #[intrinsic(VertexPositions)]
//!     vertex_positions: IntrinsicAttribute<[f64; 3], VertexIndex>,
//!     indices: Vec<[usize; 3]>,
//! }
//! ```
//!
use proc_macro2::{Span, TokenStream};
use quote::{quote, TokenStreamExt};
use syn::{
    Data, DataStruct, DeriveInput, Fields, GenericArgument, Ident, Meta, MetaList, NestedMeta,
    Path, PathArguments, Type, TypePath,
};

pub(crate) fn impl_intrinsic(ast: &DeriveInput) -> TokenStream {
    let mut tokens = TokenStream::new();

    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    if let Data::Struct(DataStruct {
        fields: Fields::Named(ref fields),
        ..
    }) = ast.data
    {
        for field in fields.named.iter() {
            let mut intrinsic_trait = None;
            for attr in field.attrs.iter() {
                if let Ok(meta) = attr.parse_meta() {
                    match meta {
                        Meta::List(MetaList { path, nested, .. }) => {
                            if path != syn::parse::<Path>(quote!(intrinsic).into()).unwrap() {
                                continue;
                            }
                            if nested.len() != 1 {
                                panic!("Expected exactly one intrinsic trait path as a parameter to the `intrinsic` attribute");
                            }
                            let meta = nested.first().unwrap();
                            if let NestedMeta::Meta(Meta::Path(path)) = meta {
                                intrinsic_trait = Some(path.clone());
                            }
                        }
                        _ => continue,
                    }
                }
            }

            if let Some(intrinsic_trait) = intrinsic_trait {
                if let Type::Path(TypePath {
                    path: Path { ref segments, .. },
                    ..
                }) = field.ty
                {
                    if segments.last().is_none() {
                        continue;
                    }

                    let ty = segments.last().unwrap();

                    if ty.ident == Ident::new("IntrinsicAttribute", Span::call_site()) {
                        let field_name =
                            field.ident.clone().expect("Invalid intrinsic field name.");
                        let field_name_mut =
                            Ident::new(&format!("{}_mut", field_name), Span::call_site());

                        if let PathArguments::AngleBracketed(ref type_args) = ty.arguments {
                            // Static attribute implementation where AttribDict has a generic topo
                            // parameter.
                            if type_args.args.len() != 2 {
                                // looking for two generic arguments
                                panic!("Expecting two type arguments for IntrinsicAttribute, but {} were given", type_args.args.len());
                            }

                            if let GenericArgument::Type(ref ty) = type_args.args.first().unwrap() {
                                tokens.append_all(quote! {
                                    impl #impl_generics #intrinsic_trait for #name #ty_generics #where_clause {
                                        type Element = #ty;

                                        #[inline]
                                        fn #field_name(&self) -> &[Self::Element] {
                                            self.#field_name.as_slice()
                                        }
                                        #[inline]
                                        fn #field_name_mut(&mut self) -> &mut [Self::Element] {
                                            self.#field_name.as_mut_slice()
                                        }
                                    }
                                });
                            } else {
                                panic!("Invalid first type argument on the IntrinsicAttribute.");
                            }
                        }
                    }
                }
            } else {
                for attr in field.attrs.iter() {
                    if let Ok(meta) = attr.parse_meta() {
                        match meta {
                            Meta::List(MetaList { path, nested, .. }) => {
                                if path != syn::parse::<Path>(quote!(intrinsics).into()).unwrap() {
                                    continue;
                                }
                                for meta in nested.iter() {
                                    if let NestedMeta::Meta(Meta::Path(mut path)) = meta.clone() {
                                        if let Some(intrinsic_field) = path.segments.pop() {
                                            if path.segments.is_empty() {
                                                panic!("Expected a list of intrinsic trait and field pairs like `Trait::field`; found a single path segment");
                                            }

                                            let intrinsic_field_mut = Ident::new(
                                                &format!("{}_mut", intrinsic_field.value().ident),
                                                Span::call_site(),
                                            );
                                            let ty = &field.ty;
                                            let field_name = &field.ident;

                                            // Strip the last :: symbol in the path.
                                            let last_segment =
                                                path.segments.pop().unwrap().value().clone();
                                            path.segments.push(last_segment);

                                            tokens.append_all(quote! {
                                                impl #impl_generics #path for #name #ty_generics #where_clause {
                                                    type Element = <#ty as #path>::Element;

                                                    #[inline]
                                                    fn #intrinsic_field(&self) -> &[Self::Element] {
                                                        self.#field_name.#intrinsic_field()
                                                    }
                                                    #[inline]
                                                    fn #intrinsic_field_mut(&mut self) -> &mut [Self::Element] {
                                                        self.#field_name.#intrinsic_field_mut()
                                                    }
                                                }
                                            });
                                        }
                                    }
                                }
                            }
                            _ => continue,
                        }
                    }
                }
            }
        }
    }
    tokens
}
