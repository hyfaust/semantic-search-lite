use crate::ai_core::OnnxModelConfig;
use ort::execution_providers::{CPUExecutionProvider, XNNPACKExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokenizers::Tokenizer;
use tracing::{info, warn};

/// Create an ONNX session with XNNPACK priority → CPU fallback.
pub fn setup_session(config: &OnnxModelConfig) -> anyhow::Result<Session> {
    // Initialize ORT once globally (commit() returns bool in 2.0.0-rc.12)
    static ORT_INIT: std::sync::Once = std::sync::Once::new();
    ORT_INIT.call_once(|| {
        ort::init().commit();
        info!("ORT initialized");
    });

    // Attempt XNNPACK + CPU
    // Note: builder methods return Result<SessionBuilder, Error<SessionBuilder>>
    // where Error<SessionBuilder> is NOT Send+Sync, so we can't use `?` with anyhow.
    // Instead, chain manually and only use `?` on commit_from_file (which returns Error<()>).
    let builder = Session::builder();
    let builder = match builder {
        Ok(b) => b,
        Err(e) => return Err(anyhow::anyhow!("Failed to create session builder: {}", e)),
    };

    let builder = match builder.with_optimization_level(GraphOptimizationLevel::Level3) {
        Ok(b) => b,
        Err(e) => {
            warn!("Failed to set optimization level: {}", e);
            return Err(anyhow::anyhow!("Optimization level error: {}", e));
        }
    };

    let mut builder = match builder.with_execution_providers([
        XNNPACKExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(),
    ]) {
        Ok(b) => b,
        Err(e) => {
            warn!("Failed to set execution providers: {}", e);
            return Err(anyhow::anyhow!("Execution provider error: {}", e));
        }
    };

    // Try XNNPACK+CPU first
    match builder.commit_from_file(&config.model_path) {
        Ok(session) => {
            info!("Using execution providers: XNNPACK, CPU");
            Ok(session)
        }
        Err(e) => {
            warn!("XNNPACK+CPU failed ({}), trying CPU only", e);

            let builder = Session::builder()
                .map_err(|e| anyhow::anyhow!("Session builder failed: {}", e))?;
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
    }
}

/// Load tokenizer from file.
pub fn setup_tokenizer(config: &OnnxModelConfig) -> anyhow::Result<Tokenizer> {
    Tokenizer::from_file(&config.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
}
