use crate::ai_core::ai_loader::{setup_session, setup_tokenizer};
use crate::ai_core::embedding::EmbeddingModel;
use crate::ai_core::OnnxModelConfig;
use ndarray::{Array1, Axis};
use ort::session::Session;
use ort::value::TensorRef;
use rayon::prelude::*;
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct EmbeddingGemmaModel {
    session: Session,
    tokenizer: Tokenizer,
}

impl EmbeddingModel for EmbeddingGemmaModel {
    fn new(config: OnnxModelConfig) -> anyhow::Result<Self> {
        let session = setup_session(&config)?;
        let mut tokenizer = setup_tokenizer(&config)?;

        // Configure padding from tokenizer_config.json
        let config_content = std::fs::read_to_string(&config.tokenizer_config_path)?;
        let tokenizer_config: Value = serde_json::from_str(&config_content)?;
        if let Some(pad_token) = tokenizer_config["pad_token"].as_str() {
            tokenizer.with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                direction: tokenizers::PaddingDirection::Right,
                pad_to_multiple_of: None,
                pad_id: 0,
                pad_type_id: 0,
                pad_token: pad_token.to_string(),
            }));
        }

        Ok(Self {
            session,
            tokenizer,
        })
    }

    fn compute_embeddings(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Array1<f32>>> {
        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let padded_len = encodings[0].len();

        // Prepare tensors (parallel collect for large batches)
        let ids: Vec<i64> = encodings
            .par_iter()
            .flat_map_iter(|e| e.get_ids().iter().map(|&i| i as i64))
            .collect();
        let mask: Vec<i64> = encodings
            .par_iter()
            .flat_map_iter(|e| e.get_attention_mask().iter().map(|&i| i as i64))
            .collect();

        let a_ids = TensorRef::from_array_view(([texts.len(), padded_len], &*ids))?;
        let a_mask = TensorRef::from_array_view(([texts.len(), padded_len], &*mask))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![a_ids, a_mask])?;
        let raw = outputs[0].try_extract_array::<f32>()?.to_owned();

        // Mean pooling
        let mut pooled = raw.mean_axis(Axis(1)).unwrap();

        // L2 normalization
        pooled.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.mapv_inplace(|x| x / norm);
            }
        });

        let result: Vec<Array1<f32>> = pooled
            .axis_iter(Axis(0))
            .map(|row| row.to_owned().into_dimensionality().unwrap())
            .collect();

        Ok(result)
    }

    fn compute_similarity(embedding1: ndarray::ArrayView1<f32>, embedding2: ndarray::ArrayView1<f32>) -> f32 {
        if embedding1.len() != embedding2.len() {
            return 0.0;
        }
        embedding1.dot(&embedding2) * 100.0
    }

    fn get_default_config() -> OnnxModelConfig {
        OnnxModelConfig::default()
    }
}
