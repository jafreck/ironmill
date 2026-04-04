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
    pub shader_hash: String,
    pub specialization: String,
}

impl ShaderCache {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            cache_dir: dir,
            max_size_bytes: 512 * 1024 * 1024,
        })
    }

    pub fn default_location() -> Result<Self, std::io::Error> {
        let dir = home_dir().join(".cache/ironmill/shaders");
        Self::open(dir)
    }

    pub fn with_max_size(mut self, bytes: u64) -> Self {
        self.max_size_bytes = bytes;
        self
    }

    pub fn max_size_bytes(&self) -> u64 {
        self.max_size_bytes
    }

    pub fn get(&self, key: &ShaderCacheKey) -> Option<Vec<u8>> {
        let path = self.key_path(key);
        std::fs::read(&path).ok()
    }

    pub fn put(&self, key: &ShaderCacheKey, binary: &[u8]) -> Result<(), std::io::Error> {
        let path = self.key_path(key);
        std::fs::write(&path, binary)
    }

    pub fn clear(&self) -> Result<(), std::io::Error> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
            std::fs::create_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }

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
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}
