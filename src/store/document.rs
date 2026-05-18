use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ai_core::embedding::EmbeddingModel;
use crate::store::embedding_cache::{compute_content_hash, EmbeddingCache};

const STORE_FILE_NAME: &str = "documents.bin";

// ── Document ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub title: Option<String>,
    pub source_path: Option<String>,
    pub embedding: Vec<f32>,
    pub created_at: i64,
    /// Hash of embed_text() at the time the embedding was computed.
    /// Used for incremental rebuild: if the hash matches, the embedding is still valid.
    #[serde(default)]
    pub content_hash: u64,
}

impl Document {
    pub fn new(id: String, text: String) -> Self {
        Self {
            id,
            text,
            title: None,
            source_path: None,
            embedding: Vec::new(),
            created_at: chrono::Utc::now().timestamp(),
            content_hash: 0,
        }
    }

    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    pub fn with_source_path(mut self, path: String) -> Self {
        self.source_path = Some(path);
        self
    }

    pub fn has_embedding(&self) -> bool {
        !self.embedding.is_empty() && self.embedding.iter().any(|&v| v != 0.0)
    }

    /// Returns true if the embedding's content hash matches the current embed_text().
    pub fn embedding_is_fresh(&self) -> bool {
        if !self.has_embedding() {
            return false;
        }
        self.content_hash == compute_content_hash(&self.embed_text())
    }

    pub fn embed_text(&self) -> String {
        match &self.title {
            Some(title) => format!("title: {} | text: {}", title, self.text),
            None => self.text.clone(),
        }
    }
}

#[derive(Debug)]
pub struct DocumentStore {
    documents: HashMap<String, Document>,
    store_path: PathBuf,
}

impl DocumentStore {
    pub fn new(data_dir: &Path) -> Self {
        let store_path = data_dir.join(STORE_FILE_NAME);
        let documents = Self::load_from_file(&store_path).unwrap_or_default();
        Self {
            documents,
            store_path,
        }
    }

    pub fn add(&mut self, doc: Document) {
        self.documents.insert(doc.id.clone(), doc);
    }

    pub fn remove(&mut self, id: &str) -> Option<Document> {
        self.documents.remove(id)
    }

    pub fn get(&self, id: &str) -> Option<&Document> {
        self.documents.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Document> {
        self.documents.get_mut(id)
    }

    pub fn all_documents(&self) -> Vec<&Document> {
        self.documents.values().collect()
    }

    pub fn documents_without_embeddings(&self) -> Vec<&Document> {
        self.documents
            .values()
            .filter(|d| !d.has_embedding())
            .collect()
    }

    pub fn count(&self) -> usize {
        self.documents.len()
    }

    pub fn count_with_embeddings(&self) -> usize {
        self.documents.values().filter(|d| d.has_embedding()).count()
    }

    /// Count documents whose embedding is stale (content changed since last computation).
    pub fn count_stale_embeddings(&self) -> usize {
        self.documents
            .values()
            .filter(|d| d.has_embedding() && !d.embedding_is_fresh())
            .count()
    }

    /// Compute embeddings for documents that need them.
    ///
    /// For each document, checks the `embedding_cache` first:
    /// - If cache hit with matching content hash → copy from cache (no inference)
    /// - Otherwise → run ORT inference, store result in both document and cache
    ///
    /// `session_reset_interval`: if > 0, recreate the model via `model_factory` every N
    /// documents to prevent ORT session memory accumulation.  Set to 0 to disable.
    /// `model_factory`: closure that creates a fresh model instance (used for session reset).
    pub fn compute_missing_embeddings<M, F>(
        &mut self,
        model: &mut M,
        embedding_cache: &mut EmbeddingCache,
        session_reset_interval: usize,
        model_factory: F,
    ) -> Result<usize>
    where
        M: EmbeddingModel,
        F: Fn() -> Result<M>,
    {
        // 1. Identify documents needing recomputation
        let pending: Vec<String> = self
            .documents
            .values()
            .filter(|d| !d.embedding_is_fresh())
            .map(|d| d.id.clone())
            .collect();

        if pending.is_empty() {
            return Ok(0);
        }

        let mut count = 0usize;
        let mut since_reset = 0usize;

        // 2. Try loading from cache first (no inference needed)
        let mut still_pending: Vec<String> = Vec::new();
        for id in &pending {
            let doc = &self.documents[id];
            let hash = compute_content_hash(&doc.embed_text());
            if let Some(cached_emb) = embedding_cache.get(id, hash) {
                if let Some(doc) = self.documents.get_mut(id) {
                    doc.embedding = cached_emb.to_vec();
                    doc.content_hash = hash;
                    count += 1;
                }
            } else {
                still_pending.push(id.clone());
            }
        }

        if count > 0 {
            println!("  Restored {} embedding(s) from cache", count);
        }

        if still_pending.is_empty() {
            return Ok(count);
        }

        // 3. Compute remaining via ORT inference
        let to_compute = still_pending.len();
        println!(
            "  Computing {} new/changed embedding(s)...",
            to_compute
        );

        for (i, id) in still_pending.iter().enumerate() {
            let embed_text = self.documents[id].embed_text();
            let hash = compute_content_hash(&embed_text);

            let emb = model
                .compute_embedding(&embed_text)
                .with_context(|| format!("Failed to compute embedding for: {}", id))?;

            let emb_vec: Vec<f32> = emb.iter().cloned().collect();
            embedding_cache.insert(id.clone(), hash, emb_vec.clone());

            if let Some(doc) = self.documents.get_mut(id) {
                doc.embedding = emb_vec;
                doc.content_hash = hash;
                count += 1;
            }

            since_reset += 1;

            // Session reset to prevent ORT memory accumulation
            if session_reset_interval > 0
                && since_reset >= session_reset_interval
                && i + 1 < still_pending.len()
            {
                print!("  [session reset after {} inferences...] ", since_reset);
                *model = model_factory()?;
                since_reset = 0;
                println!("ok");
            }

            let processed = i + 1;
            if processed % 50 == 0 || processed == to_compute {
                println!(
                    "  [{}/{}] embeddings computed...",
                    processed, to_compute
                );
            }
        }

        Ok(count)
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.store_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {:?}", parent))?;
        }
        let bytes = serde_json::to_vec_pretty(&self.documents)
            .context("Failed to serialize documents")?;
        std::fs::write(&self.store_path, bytes)
            .with_context(|| format!("Failed to write store file: {:?}", self.store_path))?;
        Ok(())
    }

    fn load_from_file(path: &Path) -> Result<HashMap<String, Document>> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read store file: {:?}", path))?;
        // Try JSON first (new format), fall back to bincode (legacy)
        if let Ok(docs) = serde_json::from_slice::<HashMap<String, Document>>(&bytes) {
            return Ok(docs);
        }
        let (docs, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .context("Failed to deserialize documents")?;
        Ok(docs)
    }
}
