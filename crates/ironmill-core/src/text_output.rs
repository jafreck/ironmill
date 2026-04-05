//! Output types for text generation (§10.2).

use std::time::Duration;

/// The complete result of a non-streaming text generation call.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TextOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub token_count: usize,
    pub prompt_token_count: usize,
    pub finish_reason: String,
    pub elapsed: Duration,
}

/// A single incremental chunk emitted during streaming generation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub token: u32,
    pub position: usize,
    pub finished: bool,
    pub finish_reason: Option<String>,
}

/// An iterator that yields [`TextChunk`]s during streaming generation.
pub struct TextStream<'a> {
    _lifetime: std::marker::PhantomData<&'a ()>,
}
