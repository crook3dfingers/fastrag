//! NVD 2.0 feed parser for FastRAG.

pub mod metadata;
pub mod parser;
pub mod schema;

pub use parser::NvdFeedParser;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        assert_eq!(2 + 2, 4);
    }
}
