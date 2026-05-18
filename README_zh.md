# semantic-search-lite

[English](README.md) | [简体中文](README_zh.md)

---

> 基于 EmbeddingGemma-300m 的轻量级语义搜索 RAG 引擎，完全运行在 CPU 上。


semantic-search-lite 是一个本地优先的语义搜索引擎，使用 EmbeddingGemma-300m ONNX 模型进行文本向量化。它结合了向量相似度与模糊文本匹配的混合搜索、基于内容哈希的增量向量嵌入计算，以及基于 JSONL 的文档管理。

## 目录

- [功能特性](#功能特性)
- [环境依赖](#环境依赖)
- [安装](#安装)
- [使用方法](#使用方法)
  - [导入文档](#导入文档)
  - [搜索](#搜索)
  - [添加本地文件](#添加本地文件)
  - [统计与维护](#统计与维护)
- [架构](#架构)
  - [ai_core](#ai_core)
  - [search 模块](#search-模块)
  - [store](#store)
- [数据格式](#数据格式)
- [配置](#配置)
- [项目结构](#项目结构)
- [构建](#构建)
- [许可证](#许可证)

## 功能特性

- **ONNX Runtime 推理** — 使用 EmbeddingGemma-300m 在 CPU 上进行均值池化和 L2 归一化
- **混合搜索** — 语义相似度 (70%) + 模糊文本匹配 (30%)，使用 skim 算法
- **增量重建** — 基于内容哈希的缓存失效，仅重新计算已变更的文档
- **会话管理** — 每 100 次推理自动重置 ORT 会话，防止内存泄漏
- **JSONL 导入** — 兼容 kb-builder 输出格式
- **批量处理** — 通过 rayon 实现并行向量嵌入计算

## 环境依赖

| 依赖项 | 版本 | 是否必需 |
|------------|---------|----------|
| Windows    | 10+     | 是      |
| Rust       | >= 1.80 | 仅构建时需要 |
| MSVC Build Tools | 最新版 | 仅构建时需要 |

ONNX 模型文件需放置在 `models/` 目录下：

- `model.onnx` — 模型图结构
- `model.onnx_data` — 模型权重
- `tokenizer.json` — HuggingFace 分词器
- `tokenizer_config.json` — 分词器配置

## 安装

### 从源码构建

```bash
cd semantic-search-lite
cargo build --release
```

构建产物位于 `target/release/semantic-search-lite.exe`。

## 使用方法

### 导入文档

从 kb-builder 的 JSONL 文件导入文档：

```bash
semantic-search-lite import --path documents.jsonl
```

导入时不计算向量嵌入（用于批量处理）：

```bash
semantic-search-lite import --path documents.jsonl --skip-embeddings
```

### 搜索

```bash
# 基础搜索
semantic-search-lite search --query "web browser"

# 自定义 top-k 和阈值
semantic-search-lite search --query "chat application" --top-k 10 --threshold 50.0
```

### 添加本地文件

```bash
# 添加单个文件
semantic-search-lite add --path readme.md

# 递归添加目录
semantic-search-lite add --path ~/Documents --recursive
```

支持的文件类型：`.txt`、`.md`、`.rs`、`.py`、`.js`、`.ts`、`.toml`、`.json`、`.yaml`、`.yml`、`.csv`、`.log`

### 统计与维护

```bash
# 显示知识库统计信息
semantic-search-lite stats

# 增量重建（仅重新计算过期的向量嵌入）
semantic-search-lite rebuild

# 强制全量重建（清空缓存）
semantic-search-lite rebuild --force
```

## 架构

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
                   搜索流水线
```

### ai_core

- **`ai_loader.rs`** — 创建 ONNX Runtime 会话（CPU 执行提供者，Level3 优化）并加载 HuggingFace 分词器
- **`embedding/gemma.rs`** — `EmbeddingGemmaModel` 实现：
  1. 使用批量填充（BatchLongest 策略）对输入文本进行分词
  2. 执行 ONNX 推理
  3. 在 token 维度上进行均值池化
  4. L2 归一化 → 768 维单位向量

### search 模块

- **`engine.rs`** — `SearchEngine`：计算查询向量嵌入，对所有文档评分，按阈值过滤，返回 top-k 结果
- **`ranker.rs`** — `Ranker`：混合评分公式 `语义 × 0.7 + 模糊 × 0.3`
- **`fuzzy.rs`** — `FuzzyScorer`：使用 skim 算法进行模糊文本匹配，同时搜索 text 和 title 字段
- **`types.rs`** — `SearchQuery`，带任务感知前缀：`"task: search result | query: {text}"`

### store

- **`document.rs`** — `Document` 结构体，带内容哈希追踪；`DocumentStore` 使用 JSON 持久化（兼容旧版 bincode 回退）
- **`embedding_cache.rs`** — `EmbeddingCache`，使用 bincode 持久化和内容哈希失效机制
- **`config.rs`** — `AppConfig`，可配置数据目录、模型路径、搜索参数

## 数据格式

### documents.bin（JSON 格式）

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

### embeddings.cache（bincode 格式）

二进制序列化的 `Vec<CacheEntry>`，每个条目包含：
- `key: String` — 文档 ID
- `content_hash: u64` — `embed_text()` 输出的哈希值
- `embedding: Vec<f32>` — 768 维向量

### 输入 JSONL（来自 kb-builder）

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

## 配置

默认数据目录：`%APPDATA%/semantic-search-lite/`

可通过 `--data-dir` 覆盖：

```bash
semantic-search-lite --data-dir ./my-data stats
```

## 项目结构

```
semantic-search-lite/
├── src/
│   ├── main.rs                    # CLI 入口点 (clap)
│   ├── lib.rs                     # 库导出
│   ├── ai_core/
│   │   ├── mod.rs                 # OnnxModelConfig
│   │   ├── ai_loader.rs           # ONNX 会话与分词器加载
│   │   └── embedding/
│   │       ├── mod.rs             # EmbeddingModel trait
│   │       └── gemma.rs           # EmbeddingGemma 实现
│   ├── search/
│   │   ├── mod.rs                 # 模块导出
│   │   ├── engine.rs              # SearchEngine
│   │   ├── ranker.rs              # 混合排序器
│   │   ├── fuzzy.rs               # 模糊评分器 (skim)
│   │   └── types.rs               # SearchQuery, SearchResult
│   └── store/
│       ├── mod.rs                 # 模块导出
│       ├── document.rs            # Document 与 DocumentStore
│       ├── embedding_cache.rs     # EmbeddingCache
│       └── config.rs              # AppConfig
├── models/                        # ONNX 模型文件（不在仓库中）
├── Cargo.toml
└── LICENSE
```

## 构建

```bash
# 调试构建
cargo build

# 发布构建（LTO，单代码生成单元，符号剥离）
cargo build --release
```

## 许可证

GPL-3.0
