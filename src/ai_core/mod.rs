pub mod ai_loader;
pub mod embedding;

/// ONNX model configuration.
#[derive(Debug, Clone)]
pub struct OnnxModelConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub tokenizer_config_path: String,
}

impl Default for OnnxModelConfig {
    fn default() -> Self {
        Self {
            model_path: "models/model.onnx".to_string(),
            tokenizer_path: "models/tokenizer.json".to_string(),
            tokenizer_config_path: "models/tokenizer_config.json".to_string(),
        }
    }
}
