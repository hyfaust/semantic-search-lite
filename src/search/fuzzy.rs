use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;

pub struct FuzzyScorer {
    matcher: SkimMatcherV2,
}

impl FuzzyScorer {
    pub fn new() -> Self {
        Self {
            matcher: SkimMatcherV2::default(),
        }
    }

    /// Score how well `query` matches `text`. Returns a normalized score in 0..100 range.
    pub fn score(&self, query: &str, text: &str) -> f32 {
        if query.is_empty() || text.is_empty() {
            return 0.0;
        }

        // Try full match first
        if let Some(raw) = self.matcher.fuzzy_match(text, query) {
            return Self::normalize_score(raw, query.len(), text.len());
        }

        // Fallback: match against individual words, take best
        let best_word_score = text
            .split_whitespace()
            .filter_map(|word| self.matcher.fuzzy_match(word, query))
            .max()
            .unwrap_or(0);

        Self::normalize_score(best_word_score, query.len(), text.len())
    }

    fn normalize_score(raw: i64, query_len: usize, text_len: usize) -> f32 {
        // SkimMatcherV2 returns negative scores where closer to 0 is better
        // We normalize to 0..100
        if raw == 0 || query_len == 0 || text_len == 0 {
            return 0.0;
        }
        // raw is typically in range [-1000, 0] for good matches
        // Normalize: better matches have raw closer to 0
        let max_expected = (query_len as f64 * 10.0) as i64;
        let normalized = ((raw as f64 / max_expected as f64) + 1.0).clamp(0.0, 1.0);
        (normalized * 100.0) as f32
    }
}

impl Default for FuzzyScorer {
    fn default() -> Self {
        Self::new()
    }
}
