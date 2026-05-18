use crate::ai_core::OnnxModelConfig;
use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokenizers::Tokenizer;
use tracing::info;

/// Create an ONNX session with CPU execution provider.
pub fn setup_session(config: &OnnxModelConfig) -> anyhow::Result<Session> {
    static ORT_INIT: std::sync::Once = std::sync::Once::new();
    ORT_INIT.call_once(|| {
        ort::init().commit();
        info!("ORT initialized");
    });

    let builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?;
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("Optimization level error: {}", e))?;
    let mut builder = builder
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .map_err(|e| anyhow::anyhow!("CPU provider error: {}", e))?;

    let session = builder.commit_from_file(&config.model_path)?;
    info!("Using execution provider: CPU");
    Ok(session)
}

/// Load tokenizer from file.
pub fn setup_tokenizer(config: &OnnxModelConfig) -> anyhow::Result<Tokenizer> {
    Tokenizer::from_file(&config.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
}
