//! Generation parameters for text generation (§10.2).

/// Controls sampling behaviour during text generation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenParams {
    pub temperature: f32,
    pub max_tokens: usize,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 512,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.0,
        }
    }
}

impl GenParams {
    /// Greedy decoding (temperature = 0).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }
}
