# semantic-search-lite

[English](README.md) | [简体中文](README_zh.md)

---

> Lightweight semantic search RAG engine powered by EmbeddingGemma-300m, running entirely on CPU.

semantic-search-lite is a local-first semantic search engine that uses the EmbeddingGemma-300m ONNX model for text embedding. It provides hybrid search combining vector similarity with fuzzy text matching, incremental embedding computation with content-hash caching, and JSONL-based document management.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Import Documents](#import-documents)
  - [Search](#search)
  - [Add Local Files](#add-local-files)
  - [Stats & Maintenance](#stats--maintenance)
- [Architecture](#architecture)
  - [ai_core](#ai_core)
  - [search](#search-module)
  - [store](#store)
- [Data Formats](#data-formats)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Build](#build)
- [License](#license)

## Features

- **ONNX Runtime inference** — EmbeddingGemma-300m on CPU with mean pooling and L2 normalization
- **Hybrid search** — semantic similarity (70%) + fuzzy text matching (30%) using skim algorithm
- **Incremental rebuild** — content-hash based cache invalidation, only recomputes changed documents
- **Session management** — automatic ORT session reset every 100 inferences to prevent memory leaks
- **JSONL import** — compatible with kb-builder output format
- **Batch processing** — parallel embedding computation via rayon

## Prerequisites

| Dependency | Version | Required |
|------------|---------|----------|
| Windows    | 10+     | Yes      |
| Rust       | >= 1.80 | Build only |
| MSVC Build Tools | Latest | Build only |

The ONNX model files must be placed in the `models/` directory:

- `model.onnx` — model graph
- `model.onnx_data` — model weights
- `tokenizer.json` — HuggingFace tokenizer
- `tokenizer_config.json` — tokenizer configuration

## Installation

### Build from source

```bash
cd semantic-search-lite
cargo build --release
```

The binary will be at `target/release/semantic-search-lite.exe`.

## Usage

### Import Documents

Import documents from a kb-builder JSONL file:

```bash
semantic-search-lite import --path documents.jsonl
```

Import without computing embeddings (for batch processing):

```bash
semantic-search-lite import --path documents.jsonl --skip-embeddings
```

### Search

```bash
# Basic search
semantic-search-lite search --query "web browser"

# Custom top-k and threshold
semantic-search-lite search --query "chat application" --top-k 10 --threshold 50.0
```

### Add Local Files

```bash
# Add a single file
semantic-search-lite add --path readme.md

# Add a directory recursively
semantic-search-lite add --path ~/Documents --recursive
```

Supported file types: `.txt`, `.md`, `.rs`, `.py`, `.js`, `.ts`, `.toml`, `.json`, `.yaml`, `.yml`, `.csv`, `.log`

### Stats & Maintenance

```bash
# Show knowledge base statistics
semantic-search-lite stats

# Incremental rebuild (recompute stale embeddings only)
semantic-search-lite rebuild

# Force full rebuild (clear cache)
semantic-search-lite rebuild --force
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   ai_core    │     │    search    │     │    store     │
│              │     │              │     │              │
│ ONNX Session │     │ SearchEngine │     │ DocumentStore│
│ Tokenizer    │     │ Ranker       │     │ EmbedCache   │
│ GemmaModel   │     │ FuzzyScorer  │     │ AppConfig    │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┴────────────────────┘
                  Search Pipeline
```

### ai_core

- **`ai_loader.rs`** — Creates ONNX Runtime session (CPU execution provider, Level3 optimization) and loads HuggingFace tokenizer
- **`embedding/gemma.rs`** — `EmbeddingGemmaModel` implementation:
  1. Tokenize input text with batch padding (BatchLongest strategy)
  2. Run ONNX inference
  3. Mean pooling over token dimension
  4. L2 normalization → 768-dimensional unit vectors

### search module

- **`engine.rs`** — `SearchEngine`: computes query embedding, scores all documents, filters by threshold, returns top-k
- **`ranker.rs`** — `Ranker`: hybrid scoring formula `semantic × 0.7 + fuzzy × 0.3`
- **`fuzzy.rs`** — `FuzzyScorer`: skim algorithm for fuzzy text matching, searches both text and title
- **`types.rs`** — `SearchQuery` with task-aware prefix: `"task: search result | query: {text}"`

### store

- **`document.rs`** — `Document` struct with content-hash tracking; `DocumentStore` with JSON persistence (bincode fallback for legacy)
- **`embedding_cache.rs`** — `EmbeddingCache` with bincode persistence and content-hash invalidation
- **`config.rs`** — `AppConfig` with configurable data directory, model paths, search parameters

## Data Formats

### documents.bin (JSON format)

```json
{
  "id": "path:c:\\...\\notepad.lnk",
  "text": "title: Notepad | text: 软件名字:Notepad，也叫做:n,notepad，启动地址:C:\\...\\Notepad.lnk",
  "title": "Notepad",
  "source_path": "C:\\...\\Notepad.lnk",
  "embedding": [0.012, -0.034, ...],
  "created_at": 1715990000,
  "content_hash": 1234567890
}
```

### embeddings.cache (bincode format)

Binary serialized `Vec<CacheEntry>` where each entry contains:
- `key: String` — document ID
- `content_hash: u64` — hash of `embed_text()` output
- `embedding: Vec<f32>` — 768-dimensional vector

### Input JSONL (from kb-builder)

```json
{
  "id": "path:c:\\...\\firefox.lnk",
  "title": "Firefox",
  "text": "title: Firefox | text: 软件名字:Firefox，也叫做:ff,firefox，启动地址:C:\\...\\Firefox.lnk",
  "source_type": "filesystem",
  "source_path": "C:\\...\\Firefox.lnk",
  "keywords": ["ff", "firefox"],
  "description": ""
}
```

## Configuration

Default data directory: `%APPDATA%/semantic-search-lite/`

Override with `--data-dir`:

```bash
semantic-search-lite --data-dir ./my-data stats
```

## Project Structure

```
semantic-search-lite/
├── src/
│   ├── main.rs                    # CLI entry point (clap)
│   ├── lib.rs                     # Library re-exports
│   ├── ai_core/
│   │   ├── mod.rs                 # OnnxModelConfig
│   │   ├── ai_loader.rs           # ONNX session & tokenizer loading
│   │   └── embedding/
│   │       ├── mod.rs             # EmbeddingModel trait
│   │       └── gemma.rs           # EmbeddingGemma implementation
│   ├── search/
│   │   ├── mod.rs                 # Re-exports
│   │   ├── engine.rs              # SearchEngine
│   │   ├── ranker.rs              # Hybrid ranker
│   │   ├── fuzzy.rs               # Fuzzy scorer (skim)
│   │   └── types.rs               # SearchQuery, SearchResult
│   └── store/
│       ├── mod.rs                 # Re-exports
│       ├── document.rs            # Document & DocumentStore
│       ├── embedding_cache.rs     # EmbeddingCache
│       └── config.rs              # AppConfig
├── models/                        # ONNX model files (not in repo)
├── Cargo.toml
└── LICENSE
```

## Build

```bash
# Debug build
cargo build

# Release build (LTO, single codegen unit, stripped)
cargo build --release
```

## Acknowledgments & Disclaimer

This project is built upon [ZeroLaunch](https://github.com/ghost-him/ZeroLaunch-rs), adopting nearly all of its original implementation logic. It is intended solely for learning and personal use.

## License

GPL-3.0
