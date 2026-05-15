use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ai_core::embedding::EmbeddingModel;

const STORE_FILE_NAME: &str = "documents.bin";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub title: Option<String>,
    pub source_path: Option<String>,
    pub embedding: Vec<f32>,
    pub created_at: i64,
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

    pub fn compute_missing_embeddings<M: EmbeddingModel>(
        &mut self,
        model: &mut M,
    ) -> Result<usize> {
        let pending: Vec<String> = self
            .documents
            .values()
            .filter(|d| !d.has_embedding())
            .map(|d| d.id.clone())
            .collect();

        if pending.is_empty() {
            return Ok(0);
        }

        let texts: Vec<String> = pending
            .iter()
            .map(|id| self.documents[id].embed_text())
            .collect();

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = model
            .compute_embeddings(&text_refs)
            .context("Failed to compute embeddings")?;

        let mut count = 0;
        for (id, emb) in pending.iter().zip(embeddings.iter()) {
            if let Some(doc) = self.documents.get_mut(id) {
                doc.embedding = emb.iter().cloned().collect();
                count += 1;
            }
        }

        Ok(count)
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.store_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {:?}", parent))?;
        }
        let bytes = bincode::serde::encode_to_vec(&self.documents, bincode::config::standard())
            .context("Failed to serialize documents")?;
        std::fs::write(&self.store_path, bytes)
            .with_context(|| format!("Failed to write store file: {:?}", self.store_path))?;
        Ok(())
    }

    fn load_from_file(path: &Path) -> Result<HashMap<String, Document>> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read store file: {:?}", path))?;
        let (docs, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .context("Failed to deserialize documents")?;
        Ok(docs)
    }
}
