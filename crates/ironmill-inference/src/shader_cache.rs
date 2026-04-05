//! Persistent shader pipeline cache (§11.4).
//!
//! Stores compiled shader binaries on disk keyed by content hash and
//! specialization constants, avoiding redundant GPU shader compilation
//! across runs.

use std::path::{Path, PathBuf};

/// Persistent shader pipeline cache.
pub struct ShaderCache {
    cache_dir: PathBuf,
    max_size_bytes: u64,
}

/// Key for looking up cached shader pipelines.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ShaderCacheKey {
    /// Hash identifying the shader source code.
    pub shader_hash: String,
    /// Specialization constants string for this pipeline variant.
    pub specialization: String,
}

impl ShaderCache {
    /// Open or create a shader cache in the given directory.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            cache_dir: dir,
            max_size_bytes: 512 * 1024 * 1024,
        })
    }

    /// Open the cache at the default platform-specific location.
    pub fn default_location() -> Result<Self, std::io::Error> {
        let dir = home_dir().join(".cache/ironmill/shaders");
        Self::open(dir)
    }

    /// Set the maximum cache size in bytes.
    pub fn with_max_size(mut self, bytes: u64) -> Self {
        self.max_size_bytes = bytes;
        self
    }

    /// Return the maximum cache size in bytes.
    pub fn max_size_bytes(&self) -> u64 {
        self.max_size_bytes
    }

    /// Look up a cached shader binary by key, returning `None` on miss.
    pub fn get(&self, key: &ShaderCacheKey) -> Option<Vec<u8>> {
        let path = self.key_path(key);
        match std::fs::read(&path) {
            Ok(data) => Some(data),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                eprintln!(
                    "Warning: shader cache read failed for {}: {e}",
                    path.display()
                );
                None
            }
        }
    }

    /// Store a compiled shader binary in the cache.
    pub fn put(&self, key: &ShaderCacheKey, binary: &[u8]) -> Result<(), std::io::Error> {
        let path = self.key_path(key);
        std::fs::write(&path, binary)
    }

    /// Remove all cached shaders from disk.
    pub fn clear(&self) -> Result<(), std::io::Error> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
            std::fs::create_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }

    /// Return the total size of all cached files in bytes.
    pub fn size(&self) -> u64 {
        dir_size(&self.cache_dir)
    }

    fn key_path(&self, key: &ShaderCacheKey) -> PathBuf {
        self.cache_dir
            .join(format!("{}_{}.bin", key.shader_hash, key.specialization))
    }
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn dir_size(dir: &Path) -> u64 {
    match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => {
            eprintln!(
                "Warning: failed to read shader cache directory {}: {e}",
                dir.display()
            );
            0
        }
    }
}
