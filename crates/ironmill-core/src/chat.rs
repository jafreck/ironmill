//! Multi-turn chat sessions (§10.5).

use crate::model::Model;
use crate::{ChatMessage, GenParams, ModelError, TextOutput};

/// A stateful multi-turn chat session bound to a [`Model`].
pub struct ChatSession<'m> {
    model: &'m Model,
    history: Vec<ChatMessage>,
    system_prompt: Option<String>,
    default_params: GenParams,
}

impl<'m> ChatSession<'m> {
    /// Create a new chat session backed by `model`.
    pub fn new(model: &'m Model) -> Self {
        Self {
            model,
            history: Vec::new(),
            system_prompt: None,
            default_params: GenParams::default(),
        }
    }

    /// Set a system prompt that will be prepended to every request.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Override the default generation parameters for this session.
    pub fn with_params(mut self, params: GenParams) -> Self {
        self.default_params = params;
        self
    }

    /// Send a user message and receive the assistant's reply.
    pub fn send(&mut self, message: impl Into<String>) -> Result<TextOutput, ModelError> {
        self.send_with_params(message, self.default_params.clone())
    }

    /// Send a user message with custom generation parameters.
    pub fn send_with_params(
        &mut self,
        message: impl Into<String>,
        _params: GenParams,
    ) -> Result<TextOutput, ModelError> {
        self.history.push(ChatMessage::user(message));
        let _model = self.model;
        let _sys = &self.system_prompt;
        let _history = &self.history;
        todo!("ChatSession::send requires ironmill-inference engine integration")
    }

    /// Return the conversation history accumulated so far.
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Clear the conversation history, keeping the system prompt.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}
