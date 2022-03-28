use crate::mesh::{CellType, Mesh};
use ahash::AHashMap as HashMap;
use std::fmt::{Display, Formatter};

use super::Error;
use super::MeshExtractor;
use super::Real;

pub use mshio::*;

#[derive(Debug)]
pub struct MshError {
    message: String,
}

impl std::error::Error for MshError {}

impl<'a> From<mshio::MshParserError<&'a [u8]>> for MshError {
    fn from(source: mshio::MshParserError<&'a [u8]>) -> Self {
        Self {
            message: format!("{}", source),
        }
    }
}

impl Display for MshError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

// TODO: Refactor the functions below to reuse code.
impl<T: Real + std::str::FromStr> MeshExtractor<T> for MshFile<u64, i32, f64> {
    /// Constructs an unstructured Mesh from this Msh file.
    ///
    /// This function will clone the given model as necessary.
    fn extract_mesh(&self) -> Result<Mesh<T>, Error> {
        let MshFile { data, .. } = &self;
        let nodes = data.nodes.as_ref().ok_or_else(|| Error::MissingMeshData)?;

        // Output vertices.
        let mut vertices = Vec::new();

        // A map from node tag to its position in the output vertices vector.
        let mut point_map = HashMap::new();

        let mut add_node = |node_tag: u64, node: &Node<f64>| -> usize {
            *point_map.entry(node_tag).or_insert_with(|| {
                let new_index = vertices.len();
                let Node { x, y, z } = *node;
                vertices.push([
                    T::from(x).unwrap(),
                    T::from(y).unwrap(),
                    T::from(z).unwrap(),
                ]);
                new_index
            })
        };

        let mut find_and_add_node = |node_tag: u64| -> Option<usize> {
            let mut offset = 0;
            for block in nodes.node_blocks.iter() {
                if let Some(tags) = &block.node_tags {
                    if let Some(&index_in_block) = tags.get(&node_tag) {
                        // Node is in this block. Check if we have already have it, otherwise insert.
                        return Some(add_node(node_tag, &block.nodes[index_in_block - 1]));
                    }
                } else {
                    let mut node_index = node_tag as usize;
                    if node_index < offset {
                        break;
                    }
                    node_index -= offset;
                    if node_index <= block.nodes.len() {
                        return Some(add_node(node_tag, &block.nodes[node_index - 1]));
                    }
                }
                offset += block.nodes.len();
            }
            None
        };

        let elements = data
            .elements
            .as_ref()
            .ok_or_else(|| Error::MissingMeshData)?;
        let mut indices = Vec::new();
        let mut cell_types = Vec::new();
        let mut counts = Vec::new();

        for block in elements.element_blocks.iter() {
            if !matches!(block.element_type, ElementType::Tet4) {
                continue;
            }
            counts.push(block.elements.len());
            cell_types.push(CellType::Tetrahedron);
            for element in block.elements.iter() {
                for &node in element.nodes.iter() {
                    indices.push(find_and_add_node(node).ok_or(Error::MissingMeshData)?);
                }
            }
        }

        Ok(Mesh::from_cells_counts_and_types(
            vertices, indices, counts, cell_types,
        ))
    }
}
