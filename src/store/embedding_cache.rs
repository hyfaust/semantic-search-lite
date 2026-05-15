use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bincode::{Decode, Encode};

const CACHE_FILE_NAME: &str = "embeddings.cache";

#[derive(Debug, Clone, Encode, Decode)]
struct CacheEntry {
    key: String,
    embedding: Vec<f32>,
}

#[derive(Debug)]
pub struct EmbeddingCache {
    cache: HashMap<String, Vec<f32>>,
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

    pub fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: String, embedding: Vec<f32>) {
        if Self::is_valid_embedding(&embedding) {
            self.cache.insert(key, embedding);
            self.dirty = true;
        }
    }

    pub fn remove(&mut self, key: &str) -> Option<Vec<f32>> {
        let result = self.cache.remove(key);
        if result.is_some() {
            self.dirty = true;
        }
        result
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
            .map(|(k, v)| CacheEntry {
                key: k.clone(),
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

    fn load_from_file(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read cache file: {:?}", path))?;
        let (entries, _): (Vec<CacheEntry>, _) =
            bincode::decode_from_slice(&bytes, bincode::config::standard())
                .context("Failed to deserialize embedding cache")?;
        let mut map = HashMap::with_capacity(entries.len());
        for entry in entries {
            if Self::is_valid_embedding(&entry.embedding) {
                map.insert(entry.key, entry.embedding);
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
