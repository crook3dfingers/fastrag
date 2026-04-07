use std::io;
use std::path::Path;

/// Hex-encoded blake3 digest of a file's contents, prefixed with `blake3:`.
pub fn hash_file(path: &Path) -> io::Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn same_bytes_same_hash() {
        let mut f1 = tempfile::NamedTempFile::new().unwrap();
        let mut f2 = tempfile::NamedTempFile::new().unwrap();
        f1.write_all(b"hello world").unwrap();
        f2.write_all(b"hello world").unwrap();
        assert_eq!(hash_file(f1.path()).unwrap(), hash_file(f2.path()).unwrap());
    }

    #[test]
    fn different_bytes_different_hash() {
        let mut f1 = tempfile::NamedTempFile::new().unwrap();
        let mut f2 = tempfile::NamedTempFile::new().unwrap();
        f1.write_all(b"alpha").unwrap();
        f2.write_all(b"beta").unwrap();
        assert_ne!(hash_file(f1.path()).unwrap(), hash_file(f2.path()).unwrap());
    }

    #[test]
    fn prefixed_with_scheme() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"x").unwrap();
        assert!(hash_file(f.path()).unwrap().starts_with("blake3:"));
    }
}
