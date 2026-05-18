use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bincode::{Decode, Encode};

const CACHE_FILE_NAME: &str = "embeddings.cache";

/// Compute a content hash for an embed text string.
pub fn compute_content_hash(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

#[derive(Debug, Clone, Encode, Decode)]
struct CacheEntry {
    key: String,
    content_hash: u64,
    embedding: Vec<f32>,
}

/// Persistent embedding cache with content-hash-based invalidation.
///
/// Stores `(doc_id → (content_hash, embedding))` pairs.
/// When the content of a document changes, its hash changes and the cached
/// embedding is automatically considered stale.
#[derive(Debug)]
pub struct EmbeddingCache {
    cache: HashMap<String, (u64, Vec<f32>)>,
    cache_path: PathBuf,
    dirty: bool,
}

impl EmbeddingCache {
    pub fn new(data_dir: &Path) -> Self {
        let cache_path = data_dir.join(CACHE_FILE_NAME);
        let cache = Self::load_from_file(&cache_path).unwrap_or_default();
        Self {
            cache,
            cache_path,
            dirty: false,
        }
    }

    /// Returns the cached embedding if the content hash matches.
    pub fn get(&self, key: &str, content_hash: u64) -> Option<&[f32]> {
        self.cache.get(key).and_then(|(hash, emb)| {
            if *hash == content_hash && Self::is_valid_embedding(emb) {
                Some(emb.as_slice())
            } else {
                None
            }
        })
    }

    /// Insert or update a cached embedding.
    pub fn insert(&mut self, key: String, content_hash: u64, embedding: Vec<f32>) {
        if Self::is_valid_embedding(&embedding) {
            self.cache.insert(key, (content_hash, embedding));
            self.dirty = true;
        }
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all entries (used for forced full rebuild).
    pub fn clear(&mut self) {
        self.cache.clear();
        self.dirty = true;
    }

    pub fn save(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        if let Some(parent) = self.cache_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {:?}", parent))?;
        }
        let entries: Vec<CacheEntry> = self
            .cache
            .iter()
            .map(|(k, (hash, v))| CacheEntry {
                key: k.clone(),
                content_hash: *hash,
                embedding: v.clone(),
            })
            .collect();
        let bytes = bincode::encode_to_vec(&entries, bincode::config::standard())
            .context("Failed to serialize embedding cache")?;
        std::fs::write(&self.cache_path, bytes)
            .with_context(|| format!("Failed to write cache file: {:?}", self.cache_path))?;
        self.dirty = false;
        Ok(())
    }

    fn load_from_file(path: &Path) -> Result<HashMap<String, (u64, Vec<f32>)>> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read cache file: {:?}", path))?;
        let (entries, _): (Vec<CacheEntry>, _) =
            bincode::decode_from_slice(&bytes, bincode::config::standard())
                .context("Failed to deserialize embedding cache")?;
        let mut map = HashMap::with_capacity(entries.len());
        for entry in entries {
            if Self::is_valid_embedding(&entry.embedding) {
                map.insert(entry.key, (entry.content_hash, entry.embedding));
            }
        }
        Ok(map)
    }

    fn is_valid_embedding(embedding: &[f32]) -> bool {
        if embedding.is_empty() {
            return false;
        }
        let mut has_non_zero = false;
        for &v in embedding {
            if !v.is_finite() {
                return false;
            }
            if v != 0.0 {
                has_non_zero = true;
            }
        }
        has_non_zero
    }
}
