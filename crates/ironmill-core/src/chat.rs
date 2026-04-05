//! Multi-turn chat sessions (§10.5).

use crate::model::Model;
use crate::{ChatMessage, GenParams, ModelError, TextOutput};

/// A stateful multi-turn chat session bound to a [`Model`].
pub struct ChatSession<'m> {
    model: &'m mut Model,
    history: Vec<ChatMessage>,
    system_prompt: Option<String>,
    default_params: GenParams,
    max_context_tokens: usize,
}

impl<'m> ChatSession<'m> {
    /// Create a new chat session backed by `model`.
    pub(crate) fn new(
        model: &'m mut Model,
        history: Vec<ChatMessage>,
        params: GenParams,
        max_context_tokens: usize,
    ) -> Self {
        Self {
            model,
            history,
            system_prompt: None,
            default_params: params,
            max_context_tokens,
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
        let _model = &self.model;
        let _sys = &self.system_prompt;
        let _history = &self.history;
        let _max_ctx = self.max_context_tokens;
        todo!("ChatSession::send requires ironmill-inference engine integration")
    }

    /// Send a message and stream the response.
    pub fn say_stream(&mut self, _message: &str) -> Result<(), ModelError> {
        todo!("ChatSession::say_stream requires ironmill-inference engine integration")
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

/// Builder for creating a [`ChatSession`] with configuration.
pub struct ChatSessionBuilder<'a> {
    model: &'a mut Model,
    system_prompt: Option<String>,
    params: GenParams,
    max_context_tokens: usize,
}

impl<'a> ChatSessionBuilder<'a> {
    pub(crate) fn new(model: &'a mut Model) -> Self {
        let max_ctx = model.info().max_context_len;
        Self {
            model,
            system_prompt: None,
            params: GenParams::default(),
            max_context_tokens: max_ctx,
        }
    }

    /// Set the system prompt.
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set generation parameters.
    pub fn params(mut self, params: GenParams) -> Self {
        self.params = params;
        self
    }

    /// Set max context tokens.
    pub fn max_context_tokens(mut self, n: usize) -> Self {
        self.max_context_tokens = n;
        self
    }

    /// Build the chat session.
    pub fn build(self) -> ChatSession<'a> {
        let mut history = Vec::new();
        if let Some(sys) = self.system_prompt {
            history.push(ChatMessage::system(sys));
        }
        ChatSession::new(self.model, history, self.params, self.max_context_tokens)
    }
}
