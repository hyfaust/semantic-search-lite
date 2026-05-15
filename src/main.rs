mod ai_core;
mod search;
mod store;

use std::path::{Path, PathBuf};

use ai_core::embedding::gemma::EmbeddingGemmaModel;
use ai_core::embedding::EmbeddingModel as _;
use ai_core::OnnxModelConfig;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use search::engine::SearchEngine;
use search::types::SearchQuery;
use store::document::{Document, DocumentStore};

#[derive(Parser)]
#[command(name = "ssl")]
#[command(about = "Lightweight semantic search RAG engine powered by EmbeddingGemma-300m")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to ONNX model directory
    #[arg(long, default_value = "models")]
    model_dir: String,

    /// Data directory for storing documents and cache
    #[arg(long)]
    data_dir: Option<String>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Add documents to the knowledge base
    Add {
        /// Path to a file or directory to index
        #[arg(short, long)]
        path: String,

        /// Recursive scan for directories
        #[arg(short, long, default_value = "false")]
        recursive: bool,
    },

    /// Search the knowledge base
    Search {
        /// Search query
        #[arg(short, long)]
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        top_k: usize,

        /// Minimum similarity threshold (0-100)
        #[arg(long, default_value = "30.0")]
        threshold: f32,
    },

    /// Show statistics about the knowledge base
    Stats,

    /// Recompute all embeddings
    Rebuild,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(&cli.log_level)
        .init();

    // Resolve data directory
    let data_dir = resolve_data_dir(cli.data_dir.as_deref())?;
    tracing::info!("Data directory: {:?}", data_dir);

    match cli.command {
        Commands::Add { path, recursive } => cmd_add(&data_dir, &cli.model_dir, &path, recursive),
        Commands::Search {
            query,
            top_k,
            threshold,
        } => cmd_search(&data_dir, &cli.model_dir, &query, top_k, threshold),
        Commands::Stats => cmd_stats(&data_dir),
        Commands::Rebuild => cmd_rebuild(&data_dir, &cli.model_dir),
    }
}

fn resolve_data_dir(override_path: Option<&str>) -> Result<PathBuf> {
    let path = match override_path {
        Some(p) => PathBuf::from(p),
        None => dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("semantic-search-lite"),
    };
    std::fs::create_dir_all(&path)
        .with_context(|| format!("Failed to create data directory: {:?}", path))?;
    Ok(path)
}

fn create_model(model_dir: &str) -> Result<EmbeddingGemmaModel> {
    let config = OnnxModelConfig {
        model_path: format!("{}/model.onnx", model_dir),
        tokenizer_path: format!("{}/tokenizer.json", model_dir),
        tokenizer_config_path: format!("{}/tokenizer_config.json", model_dir),
    };
    tracing::info!("Loading model from: {}", config.model_path);
    let model = EmbeddingGemmaModel::new(config)?;
    tracing::info!("Model loaded successfully");
    Ok(model)
}

// ── Add command ──────────────────────────────────────────────────────────────

fn cmd_add(data_dir: &Path, model_dir: &str, path: &str, recursive: bool) -> Result<()> {
    let input_path = PathBuf::from(path);
    let mut store = DocumentStore::new(data_dir);

    let files = collect_files(&input_path, recursive)?;
    if files.is_empty() {
        println!("No supported files found in: {}", path);
        return Ok(());
    }

    println!("Found {} file(s) to index", files.len());
    let mut added = 0;

    for file_path in &files {
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Skipping {:?}: {}", file_path, e);
                continue;
            }
        };

        let id = file_path.to_string_lossy().to_string();
        if store.get(&id).is_some() {
            tracing::info!("Already indexed: {:?}", file_path);
            continue;
        }

        let title = file_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string());

        let doc = Document::new(id, content)
            .with_source_path(file_path.to_string_lossy().to_string());

        let doc = match title {
            Some(t) => doc.with_title(t),
            None => doc,
        };

        store.add(doc);
        added += 1;
        println!("  + {:?}", file_path.file_name().unwrap_or_default());
    }

    if added > 0 {
        println!("\nComputing embeddings for {} new document(s)...", added);
        let mut model = create_model(model_dir)?;
        let computed = store.compute_missing_embeddings(&mut model)?;
        store.save()?;
        println!("Indexed {} document(s), computed {} embedding(s)", added, computed);
    } else {
        println!("No new documents to add.");
    }

    Ok(())
}

fn collect_files(path: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let supported_extensions = ["txt", "md", "rs", "py", "js", "ts", "toml", "json", "yaml", "yml", "csv", "log"];

    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        anyhow::bail!("Path does not exist: {:?}", path);
    }

    let mut files = Vec::new();
    if recursive {
        walk_dir_recursive(path, &mut files)?;
    } else {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();
            if file_path.is_file() {
                files.push(file_path);
            }
        }
    }

    // Filter by supported extensions
    let filtered: Vec<PathBuf> = files
        .into_iter()
        .filter(|p| {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            supported_extensions.contains(&ext)
        })
        .collect();

    Ok(filtered)
}

fn walk_dir_recursive(dir: &Path, results: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_dir_recursive(&path, results)?;
        } else {
            results.push(path);
        }
    }
    Ok(())
}

// ── Search command ───────────────────────────────────────────────────────────

fn cmd_search(
    data_dir: &Path,
    model_dir: &str,
    query_text: &str,
    top_k: usize,
    threshold: f32,
) -> Result<()> {
    let store = DocumentStore::new(data_dir);
    if store.count() == 0 {
        println!("Knowledge base is empty. Use 'ssl add --path <path>' to index documents.");
        return Ok(());
    }

    let mut model = create_model(model_dir)?;
    let engine = SearchEngine::new();

    let query = SearchQuery::new(query_text.to_string())
        .with_top_k(top_k)
        .with_threshold(threshold);

    let results = engine.search(&mut model, &store, &query)?;

    if results.is_empty() {
        println!("No results found for: '{}'", query_text);
        return Ok(());
    }

    println!("Found {} result(s) for: '{}'\n", results.len(), query_text);
    for result in &results {
        println!(
            "  #{}  [{:.1}]  {}",
            result.rank,
            result.score,
            truncate(&result.text, 120)
        );
        if let Some(ref path) = result.source_path {
            println!("       Source: {}", path);
        }
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

// ── Stats command ────────────────────────────────────────────────────────────

fn cmd_stats(data_dir: &Path) -> Result<()> {
    let store = DocumentStore::new(data_dir);
    println!("Knowledge base statistics:");
    println!("  Documents:          {}", store.count());
    println!("  With embeddings:    {}", store.count_with_embeddings());
    println!("  Missing embeddings: {}", store.count() - store.count_with_embeddings());
    println!("  Data directory:     {:?}", data_dir);
    Ok(())
}

// ── Rebuild command ──────────────────────────────────────────────────────────

fn cmd_rebuild(data_dir: &Path, model_dir: &str) -> Result<()> {
    let mut store = DocumentStore::new(data_dir);
    println!("Rebuilding embeddings for {} document(s)...", store.count());

    let mut model = create_model(model_dir)?;
    let computed = store.compute_missing_embeddings(&mut model)?;
    store.save()?;

    println!("Done. Recomputed {} embedding(s).", computed);
    Ok(())
}
