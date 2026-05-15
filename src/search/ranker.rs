use ndarray::ArrayView1;

use super::fuzzy::FuzzyScorer;

/// Multi-dimensional ranker combining semantic similarity with fuzzy text matching.
///
/// Simplified from ZeroLaunch-rs formula:
///   base_score + fuzzy_score * fuzzy_weight + source_boost
///
/// ZeroLaunch-rs full formula (reference):
///   base_score + (history×0.8 + recent_habit×1.5 + temporal×0.5)
///     × suppression_factor + query_affinity×3.0
///
/// For CLI usage (no interaction history), we simplify to:
///   semantic_score × semantic_weight + fuzzy_score × fuzzy_weight + source_boost
pub struct Ranker {
    fuzzy: FuzzyScorer,
    pub semantic_weight: f32,
    pub fuzzy_weight: f32,
    pub source_boost: f32,
}

impl Ranker {
    pub fn new() -> Self {
        Self {
            fuzzy: FuzzyScorer::new(),
            semantic_weight: 0.7,
            fuzzy_weight: 0.3,
            source_boost: 0.0,
        }
    }

    /// Compute cosine similarity between two L2-normalized vectors, scaled to 0..100.
    pub fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        // Since vectors are L2-normalized, dot product is cosine similarity
        (dot * 100.0).clamp(0.0, 100.0)
    }

    /// Compute the final combined score for a document.
    ///
    /// - `semantic_score`: cosine similarity (0..100)
    /// - `query`: the original query text
    /// - `doc_text`: the document text (for fuzzy matching)
    /// - `doc_title`: optional document title (also used for fuzzy matching)
    /// - `source_path`: optional source path (used for source boost)
    pub fn score(
        &self,
        semantic_score: f32,
        query: &str,
        doc_text: &str,
        doc_title: Option<&str>,
        _source_path: Option<&str>,
    ) -> f32 {
        // Fuzzy score against text
        let fuzzy_text = self.fuzzy.score(query, doc_text);

        // Fuzzy score against title (higher weight if available)
        let fuzzy_title = doc_title
            .map(|t| self.fuzzy.score(query, t))
            .unwrap_or(0.0);

        // Take the better fuzzy score
        let fuzzy_score = fuzzy_text.max(fuzzy_title);

        // Combined score (simplified from ZeroLaunch-rs formula)
        semantic_score * self.semantic_weight
            + fuzzy_score * self.fuzzy_weight
            + self.source_boost
    }
}

impl Default for Ranker {
    fn default() -> Self {
        Self::new()
    }
}
