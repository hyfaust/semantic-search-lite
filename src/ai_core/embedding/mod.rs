pub mod gemma;

use ndarray::{Array1, ArrayView1};
use std::fmt::Debug;

use super::OnnxModelConfig;

/// Trait for embedding models.
pub trait EmbeddingModel: Debug + Send + Sync {
    fn new(config: OnnxModelConfig) -> anyhow::Result<Self>
    where
        Self: Sized;

    /// Compute embedding for a single text.
    fn compute_embedding(&mut self, text: &str) -> anyhow::Result<Array1<f32>> {
        let embeddings = self.compute_embeddings(&[text])?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    /// Compute embeddings for a batch of texts.
    fn compute_embeddings(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Array1<f32>>>;

    /// Compute similarity between two embeddings (returns percentage 0-100).
    fn compute_similarity(embedding1: ArrayView1<f32>, embedding2: ArrayView1<f32>) -> f32;

    fn get_default_config() -> OnnxModelConfig
    where
        Self: Sized;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingModelType {
    EmbeddingGemma,
}
