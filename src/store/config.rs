use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const CONFIG_FILE_NAME: &str = "config.json";
const DEFAULT_DATA_DIR_NAME: &str = "semantic-search-lite";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
    pub search_result_count: usize,
    pub similarity_threshold: f32,
    pub log_level: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        let data_dir = Self::default_data_dir();
        let models_dir = Self::default_models_dir();
        Self {
            data_dir,
            models_dir,
            search_result_count: 5,
            similarity_threshold: 30.0,
            log_level: "info".to_string(),
        }
    }
}

impl AppConfig {
    pub fn load_or_default(config_path: Option<&Path>) -> Result<Self> {
        let path = match config_path {
            Some(p) => p.to_path_buf(),
            None => Self::default_config_path(),
        };

        if path.exists() {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read config: {:?}", path))?;
            let config: AppConfig = serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse config: {:?}", path))?;
            Ok(config)
        } else {
            let config = AppConfig::default();
            config.save_to(&path)?;
            Ok(config)
        }
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::default_config_path())
    }

    fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {:?}", parent))?;
        }
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize config")?;
        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config: {:?}", path))?;
        Ok(())
    }

    fn default_data_dir() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(DEFAULT_DATA_DIR_NAME)
    }

    fn default_models_dir() -> PathBuf {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("models")))
            .unwrap_or_else(|| PathBuf::from("models"))
    }

    fn default_config_path() -> PathBuf {
        Self::default_data_dir().join(CONFIG_FILE_NAME)
    }
}
