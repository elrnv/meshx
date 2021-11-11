#![recursion_limit = "256"]
extern crate proc_macro;

mod attrib;
mod intrinsic;

use proc_macro2::{Span, TokenStream};
use quote::{quote, TokenStreamExt};
use syn::{
    Data, DataStruct, DeriveInput, Expr, Fields, GenericArgument, Generics, Ident, Index, Path,
    PathArguments, Type, TypeArray, TypePath,
};

/// Intrinsic derive macro. Intrinsic is not a real trait, but an indicator that intrinsic
/// intrinsic traits need to be implemented for this type. The name of the trait itself is
/// specified by the intrinsic attribute, which is to be specified in front of a field containing
/// the `IntrinsicAttribute` type.
#[proc_macro_derive(Intrinsic, attributes(intrinsics, intrinsic))]
pub fn intrinsic(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    let gen = intrinsic::impl_intrinsic(&input);

    gen.into()
}

/// Derive macro for implementing the `Attrib` trait for given struct.
#[proc_macro_derive(Attrib, attributes(attributes))]
pub fn attrib(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    let gen = attrib::impl_attrib(&input);

    gen.into()
}

#[proc_macro_derive(NewCollectionType)]
pub fn new_collection_type(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    let gen = impl_new_collection_type(&input);

    gen.into()
}

fn impl_new_array_collection(
    generics: &Generics,
    name: &Ident,
    parm: &Type,
    len: &Expr,
    field_idx: Index,
) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    quote! {
        //impl #impl_generics ::std::ops::Deref for #name #ty_generics #where_clause {
        //    type Target = [#parm;#len];
        //    #[inline]
        //    fn deref(&self) -> &[#parm;#len] {
        //        &self.#field_idx
        //    }
        //}
        //impl #impl_generics ::std::ops::DerefMut for #name #ty_generics #where_clause {
        //    #[inline]
        //    fn deref_mut(&mut self) -> &mut [#parm;#len] {
        //        &mut self.#field_idx
        //    }
        //}
        impl #impl_generics #name #ty_generics #where_clause {
            /// Get the underlying container.
            #[inline]
            pub fn get(&self) -> &[#parm;#len] {
                &self.#field_idx
            }
            #[inline]
            pub fn get_mut(&mut self) -> &mut [#parm;#len] {
                &mut self.#field_idx
            }
            #[inline]
            pub fn into_inner(self) -> [#parm;#len] {
                self.0
            }
        }

        impl #impl_generics Into<[#parm;#len]>for #name #ty_generics #where_clause {
            #[inline]
            fn into(self) -> [#parm;#len] {
                self.0
            }
        }

        impl #impl_generics ::std::ops::Index<usize>for #name #ty_generics #where_clause {
            type Output = #parm;
            #[inline]
            fn index(&self, index: usize) -> &#parm {
                &self.#field_idx[index]
            }
        }
        impl #impl_generics ::std::ops::IndexMut<usize>for #name #ty_generics #where_clause {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut #parm {
                &mut self.#field_idx[index]
            }
        }
    }
}

fn impl_new_vec_collection(
    generics: &Generics,
    name: &Ident,
    vec_type: &Type,
    parm: &GenericArgument,
    field_idx: Index,
) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    quote! {
        //impl #impl_generics ::std::ops::Deref for #name #ty_generics #where_clause {
        //    type Target = #vec_type;
        //    #[inline]
        //    fn deref(&self) -> &#vec_type {
        //        &self.#field_idx
        //    }
        //}
        //impl #impl_generics ::std::ops::DerefMut for #name #ty_generics #where_clause {
        //    #[inline]
        //    fn deref_mut(&mut self) -> &mut #vec_type {
        //        &mut self.#field_idx
        //    }
        //}
        impl #impl_generics #name #ty_generics #where_clause {
            /// Get the underlying container.
            #[inline]
            pub fn get(&self) -> &#vec_type {
                &self.#field_idx
            }
            #[inline]
            pub fn get_mut(&mut self) -> &mut #vec_type {
                &mut self.#field_idx
            }
            #[inline]
            pub fn capacity(&self) -> usize {
                self.#field_idx.capacity()
            }
            #[inline]
            pub fn reserve(&mut self, additional: usize) {
                self.#field_idx.reserve(additional)
            }
            #[inline]
            pub fn reserve_exact(&mut self, additional: usize) {
                self.#field_idx.reserve_exact(additional)
            }
            #[inline]
            pub fn shrink_to_fit(&mut self) {
                self.#field_idx.shrink_to_fit()
            }
            #[inline]
            pub fn into_boxed_slice(self) -> Box<[#parm]>{
                self.#field_idx.into_boxed_slice()
            }
            #[inline]
            pub fn truncate(&mut self, len: usize) {
                self.#field_idx.truncate(len)
            }
            #[inline]
            pub fn as_slice(&self) -> &[#parm] {
                self.#field_idx.as_slice()
            }
            #[inline]
            pub fn as_mut_slice(&mut self) -> &mut [#parm] {
                self.#field_idx.as_mut_slice()
            }
            // unsafe function set_len skipped
            #[inline]
            pub fn swap_remove(&mut self, index: usize) -> #parm {
                self.#field_idx.swap_remove(index)
            }
            #[inline]
            pub fn insert(&mut self, index: usize, element: #parm) {
                self.#field_idx.insert(index, element)
            }
            #[inline]
            pub fn remove(&mut self, index: usize) -> #parm {
                self.#field_idx.remove(index)
            }
            #[inline]
            pub fn retain<F>(&mut self, f: F) where F: FnMut(&#parm) -> bool {
                self.#field_idx.retain(f)
            }
            #[inline]
            pub fn push(&mut self, value: #parm) {
                self.#field_idx.push(value)
            }
            #[inline]
            pub fn pop(&mut self) -> Option<#parm> {
                self.#field_idx.pop()
            }
            #[inline]
            pub fn append(&mut self, other: &mut Vec<#parm>) {
                self.#field_idx.append(other)
            }
            // Skipped drain due to unstable API
            #[inline]
            pub fn clear(&mut self) {
                self.#field_idx.clear()
            }
            #[inline]
            pub fn len(&self) -> usize {
                self.#field_idx.len()
            }
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.#field_idx.is_empty()
            }
            #[inline]
            pub fn split_off(&mut self, at: usize) -> Vec<#parm> {
                self.#field_idx.split_off(at)
            }
        }

        impl #impl_generics Into<#vec_type>for #name #ty_generics #where_clause {
            #[inline]
            fn into(self) -> #vec_type {
                self.0
            }
        }

        impl #impl_generics ::std::ops::Index<usize>for #name #ty_generics #where_clause {
            type Output = #parm;
            #[inline]
            fn index(&self, index: usize) -> &#parm {
                &self.#field_idx[index]
            }
        }
        impl #impl_generics ::std::ops::IndexMut<usize>for #name #ty_generics #where_clause {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut #parm {
                &mut self.#field_idx[index]
            }
        }
    }
}

fn impl_new_collection_type(ast: &DeriveInput) -> TokenStream {
    let mut tokens = TokenStream::new();
    let name = &ast.ident;

    let supported_collection_types = vec!["Vec"];

    if let Data::Struct(DataStruct {
        fields: Fields::Unnamed(ref fields),
        ..
    }) = ast.data
    {
        for (i, field) in fields.unnamed.iter().enumerate() {
            if let Type::Path(TypePath {
                path: Path { ref segments, .. },
                ..
            }) = field.ty
            {
                if segments.last().is_none() {
                    continue;
                }

                let ty = segments.last().unwrap();

                if supported_collection_types
                    .iter()
                    .find(|&&x| Ident::new(x, Span::call_site()) == ty.ident)
                    .is_none()
                {
                    continue;
                }

                if let PathArguments::AngleBracketed(ref angled_args) = ty.arguments {
                    if angled_args.args.len() != 1 {
                        panic!(
                            "New collection type must wrap a collection that contains exactly\
                             one type."
                        );
                    }

                    let parm = angled_args.args.first().unwrap();

                    let field_idx = Index::from(i);
                    tokens.append_all(impl_new_vec_collection(
                        &ast.generics,
                        name,
                        &field.ty,
                        parm,
                        field_idx,
                    ));
                } else {
                    panic!("New collection type must wrap a collection that contains a type.");
                }
            } else if let Type::Array(TypeArray {
                ref elem, ref len, ..
            }) = field.ty
            {
                let parm = &**elem;

                let field_idx = Index::from(i);

                tokens.append_all(impl_new_array_collection(
                    &ast.generics,
                    name,
                    parm,
                    len,
                    field_idx,
                ));
            } else {
                panic!("New collection type must wrap a Vec or an array.");
            }
        }
    }
    tokens
}
