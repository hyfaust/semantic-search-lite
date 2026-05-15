#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub text: String,
    pub top_k: usize,
    pub threshold: f32,
}

impl SearchQuery {
    pub fn new(text: String) -> Self {
        Self {
            text,
            top_k: 10,
            threshold: 30.0,
        }
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn embed_text(&self) -> String {
        format!("task: search result | query: {}", self.text)
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub document_id: String,
    pub title: Option<String>,
    pub text: String,
    pub source_path: Option<String>,
    pub score: f32,
    pub rank: usize,
}

impl SearchResult {
    pub fn display_text(&self) -> String {
        match &self.title {
            Some(title) => format!("[{}] {}", title, self.text),
            None => self.text.clone(),
        }
    }
}
