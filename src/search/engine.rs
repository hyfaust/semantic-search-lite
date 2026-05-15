use anyhow::Result;
use ndarray::Array1;

use super::ranker::Ranker;
use super::types::{SearchQuery, SearchResult};
use crate::ai_core::embedding::EmbeddingModel;
use crate::store::document::DocumentStore;

pub struct SearchEngine {
    ranker: Ranker,
}

impl SearchEngine {
    pub fn new() -> Self {
        Self {
            ranker: Ranker::new(),
        }
    }

    pub fn with_ranker(ranker: Ranker) -> Self {
        Self { ranker }
    }

    /// Execute a semantic search against the document store.
    pub fn search<M: EmbeddingModel>(
        &self,
        model: &mut M,
        store: &DocumentStore,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>> {
        // Compute query embedding
        let query_text = query.embed_text();
        let query_embedding = model.compute_embedding(&query_text)?;

        // Score all documents that have embeddings
        let mut scored: Vec<SearchResult> = store
            .all_documents()
            .iter()
            .filter(|doc| doc.has_embedding())
            .filter_map(|doc| {
                let doc_emb = Array1::from_vec(doc.embedding.clone());
                let semantic_score =
                    Ranker::cosine_similarity(query_embedding.view(), doc_emb.view());

                let combined_score = self.ranker.score(
                    semantic_score,
                    &query.text,
                    &doc.text,
                    doc.title.as_deref(),
                    doc.source_path.as_deref(),
                );

                if combined_score >= query.threshold {
                    Some(SearchResult {
                        document_id: doc.id.clone(),
                        title: doc.title.clone(),
                        text: doc.text.clone(),
                        source_path: doc.source_path.clone(),
                        score: combined_score,
                        rank: 0,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks and truncate
        for (i, result) in scored.iter_mut().enumerate() {
            result.rank = i + 1;
        }
        scored.truncate(query.top_k);

        Ok(scored)
    }
}

impl Default for SearchEngine {
    fn default() -> Self {
        Self::new()
    }
}
