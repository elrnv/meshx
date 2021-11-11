use lazy_static::lazy_static;
use proc_macro2::{Span, TokenStream};
use quote::{quote, TokenStreamExt};
use std::collections::HashMap;
use syn::{
    parse_quote, Data, DataStruct, DeriveInput, Expr, Field, Fields, GenericArgument, Generics,
    Ident, Item, Meta, MetaList, NestedMeta, Path, PathArguments, Type, TypePath,
};

lazy_static! {
    static ref SUPPORTED_TOPO: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("Mesh", "num_meshes");
        m.insert("Vertex", "num_vertices");
        m.insert("Edge", "num_edges");
        m.insert("Face", "num_faces");
        m.insert("Cell", "num_cells");
        m.insert("EdgeVertex", "num_edge_vertices");
        m.insert("FaceVertex", "num_face_vertices");
        m.insert("FaceEdge", "num_face_edges");
        m.insert("CellVertex", "num_cell_vertices");
        m.insert("CellEdge", "num_cell_edges");
        m.insert("CellFace", "num_cell_faces");
        m.insert("VertexEdge", "num_vertex_edges");
        m.insert("VertexFace", "num_vertex_faces");
        m.insert("VertexCell", "num_vertex_cells");
        m.insert("EdgeFace", "num_edge_faces");
        m.insert("EdgeCell", "num_edge_cells");
        m.insert("FaceCell", "num_face_cells");
        m
    };
}

pub(crate) fn find_cache_field(data: &Data) -> Option<&Ident> {
    let mut cache_field = None;

    if let Data::Struct(DataStruct {
        fields: Fields::Named(ref fields),
        ..
    }) = data
    {
        for field in fields.named.iter() {
            if let Type::Path(TypePath {
                path: Path { ref segments, .. },
                ..
            }) = field.ty
            {
                if segments.last().is_none() {
                    continue;
                }

                let ty = segments.last().unwrap();

                let field_name = &field.ident;

                if ty.ident == "AttribValueCache" {
                    cache_field = field_name.as_ref();
                    break;
                }
            }
        }
    }

    cache_field
}

pub(crate) enum AttribType {
    Static,
    Dynamic, // Not currently supported
}

pub(crate) fn try_add_attrib_tokens(
    name: &Ident,
    attrib_dict_path: &Path,
    args: &PathArguments,
    cache_or_none: &Expr,
    field_name: &Ident,
    generics: &Generics,
) -> Option<(TokenStream, AttribType)> {
    if let PathArguments::AngleBracketed(ref type_args) = args {
        // Static attribute implementation where AttribDict has a generic topo
        // parameter.
        if type_args.args.len() != 1 {
            // looking for one generic argument
            return None;
        }

        if let GenericArgument::Type(Type::Path(TypePath { ref path, .. })) =
            type_args.args.first().unwrap()
        {
            path.segments.last()?;

            let gen_arg = &path.segments.last().unwrap().ident;

            let gen_arg_str = gen_arg.to_string();
            let topo = if gen_arg_str.ends_with("Index") {
                SUPPORTED_TOPO.get_key_value(&gen_arg_str[..gen_arg_str.len() - 5])
            } else {
                None
            }?;

            let topo_attrib = Ident::new(&format!("{}Attrib", topo.0), Span::call_site());
            let num_topo = Ident::new(topo.1, Span::call_site());

            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            return Some((
                quote! {
                    impl #impl_generics #topo_attrib for #name #ty_generics #where_clause {
                        #[inline]
                        fn topo_attrib_size(&self) -> usize {
                            self.#num_topo()
                        }
                        #[inline]
                        fn topo_attrib_dict(&self) -> &#attrib_dict_path {
                            &self.#field_name
                        }
                        #[inline]
                        fn topo_attrib_dict_mut(&mut self) -> &mut #attrib_dict_path {
                            &mut self.#field_name
                        }
                        #[inline]
                        fn topo_attrib_dict_and_cache_mut(&mut self) -> (&mut #attrib_dict_path, Option<&mut AttribValueCache>) {
                            (&mut self.#field_name, #cache_or_none)
                        }
                    }
                },
                AttribType::Static,
            ));
        } else {
            panic!(
                "Couldn't determine the type of the AttribDict field, \
                                 please provide an explicit index type from the topology module."
            );
        }
    } else if let PathArguments::None = args {
        // Dynamic topo detection.
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        return Some((
            quote! {
                impl #impl_generics Attrib for #name #ty_generics #where_clause {
                    #[inline]
                    fn attrib_dict(&self) -> &#attrib_dict_path {
                        &self.#field_name
                    }
                    #[inline]
                    fn attrib_dict_mut(&mut self) -> &mut #attrib_dict_path {
                        &mut self.#field_name
                    }
                }
            },
            AttribType::Dynamic,
        ));
    }
    None
}

pub(crate) fn try_add_inherited_attrib_tokens(
    name: &Ident,
    field: &Field,
    generics: &Generics,
) -> Vec<Item> {
    let mut items = Vec::new();
    let field_name = &field.ident;
    for attr in field.attrs.iter() {
        if let Ok(meta) = attr.parse_meta() {
            if let Meta::List(MetaList { path, nested, .. }) = meta {
                if path != syn::parse::<Path>(quote!(attributes).into()).unwrap() {
                    continue;
                }
                for meta in nested.iter() {
                    if let NestedMeta::Meta(Meta::Path(path)) = meta {
                        if let Some(ident) = path.get_ident() {
                            if SUPPORTED_TOPO.contains_key(ident.to_string().as_str()) {
                                let topo_attrib =
                                    Ident::new(&format!("{}Attrib", ident), Span::call_site());
                                let topo_index =
                                    Ident::new(&format!("{}Index", ident), Span::call_site());
                                let (impl_generics, ty_generics, where_clause) =
                                    generics.split_for_impl();
                                items.push(
                                    parse_quote! {
                                        impl #impl_generics #topo_attrib for #name #ty_generics #where_clause {
                                            #[inline]
                                            fn topo_attrib_size(&self) -> usize {
                                                #topo_attrib::topo_attrib_size(&self.#field_name)
                                            }
                                            #[inline]
                                            fn topo_attrib_dict(&self) -> &AttribDict<#topo_index> {
                                                #topo_attrib::topo_attrib_dict(&self.#field_name)
                                            }
                                            #[inline]
                                            fn topo_attrib_dict_mut(&mut self) -> &mut AttribDict<#topo_index> {
                                                #topo_attrib::topo_attrib_dict_mut(&mut self.#field_name)
                                            }
                                            #[inline]
                                            fn topo_attrib_dict_and_cache_mut(&mut self) -> (&mut AttribDict<#topo_index>, Option<&mut AttribValueCache>) {
                                                #topo_attrib::topo_attrib_dict_and_cache_mut(&mut self.#field_name)
                                            }
                                        }
                                    },
                                );
                            }
                        }
                    }
                }
            } else {
                continue;
            }
        }
    }
    items
}

pub(crate) fn impl_attrib(ast: &DeriveInput) -> TokenStream {
    let mut tokens = TokenStream::new();

    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let mut found_static_attrib = false;

    let cache_field = find_cache_field(&ast.data);

    // Find the cache if any
    // Cache accessor
    let cache_or_none: Expr = if let Some(cache_field) = cache_field {
        parse_quote! { Some(&mut self.#cache_field) }
    } else {
        parse_quote! { None }
    };

    // Process the individual attribute dictionaries
    if let Data::Struct(DataStruct {
        fields: Fields::Named(ref fields),
        ..
    }) = ast.data
    {
        for field in fields.named.iter() {
            if let Type::Path(TypePath {
                ref qself,
                ref path,
            }) = field.ty
            {
                if qself.is_some() || path.segments.last().is_none() {
                    // These are not what we are looking for.
                    continue;
                }

                let ty = path.segments.last().unwrap();

                let field_name = field.ident.as_ref();
                if field_name.is_none() {
                    continue;
                }

                if ty.ident == "AttribDict" {
                    if let Some((attrib_tokens, attrib_type)) = try_add_attrib_tokens(
                        &name,
                        &path,
                        &ty.arguments,
                        &cache_or_none,
                        field_name.unwrap(),
                        &ast.generics,
                    ) {
                        tokens.append_all(attrib_tokens);
                        if matches!(attrib_type, AttribType::Static) {
                            found_static_attrib = true;
                        }
                    }
                } else {
                    for item in
                        try_add_inherited_attrib_tokens(&name, &field, &ast.generics).into_iter()
                    {
                        tokens.append_all(quote! { #item });
                        found_static_attrib = true;
                    }
                }
            }
        }
    }

    // We found at least one attribute dict in the type, we can now derive the single
    // implementation for the Attrib trait
    if found_static_attrib {
        tokens.append_all(quote! {
            impl #impl_generics Attrib for #name #ty_generics #where_clause {
                #[inline]
                fn attrib_size<I: AttribIndex<Self>>(&self) -> usize {
                    I::attrib_size(self)
                }
                #[inline]
                fn attrib_dict<I: AttribIndex<Self>>(&self) -> &AttribDict<I> {
                    I::attrib_dict(self)
                }
                #[inline]
                fn attrib_dict_mut<I: AttribIndex<Self>>(&mut self) -> &mut AttribDict<I> {
                    I::attrib_dict_mut(self)
                }
                #[inline]
                fn attrib_dict_and_cache_mut<I: AttribIndex<Self>>(&mut self) -> (&mut AttribDict<I>, Option<&mut AttribValueCache>) {
                    I::attrib_dict_and_cache_mut(self)
                }
            }
        });
    }
    tokens
}
