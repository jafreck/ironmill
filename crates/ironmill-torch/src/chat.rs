//! Multi-turn chat sessions.

use ironmill_core::tokenizer::{ChatMessage, Role};

use crate::error::TorchError;
use crate::gen_params::GenParams;
use crate::model::Model;
use crate::text_output::{TextOutput, TextStream};

/// A stateful multi-turn chat session bound to a [`Model`].
///
/// Maintains conversation history and applies the tokenizer's chat
/// template to format prompts.
///
/// # Example
///
/// ```rust,no_run
/// use ironmill_torch::{Model, GenParams};
///
/// let mut model = Model::from_pretrained("./model/").build()?;
/// let mut chat = model.chat()
///     .system("You are a helpful assistant.")
///     .build();
///
/// let reply = chat.send("What is Rust?")?;
/// println!("{}", reply.text);
/// # Ok::<(), ironmill_torch::TorchError>(())
/// ```
pub struct ChatSession<'m> {
    model: &'m mut Model,
    history: Vec<ChatMessage>,
    default_params: GenParams,
}

impl<'m> ChatSession<'m> {
    pub(crate) fn new(model: &'m mut Model, history: Vec<ChatMessage>, params: GenParams) -> Self {
        Self {
            model,
            history,
            default_params: params,
        }
    }

    /// Send a user message and receive the assistant's reply.
    pub fn send(&mut self, message: impl Into<String>) -> Result<TextOutput, TorchError> {
        self.send_with_params(message, self.default_params.clone())
    }

    /// Send a user message with custom generation parameters.
    pub fn send_with_params(
        &mut self,
        message: impl Into<String>,
        params: GenParams,
    ) -> Result<TextOutput, TorchError> {
        let user_msg = ChatMessage::user(message);
        // Format the full conversation into a prompt
        let mut messages = self.history.clone();
        messages.push(user_msg.clone());
        let prompt = self.model.tokenizer().apply_chat_template(&messages)?;

        // Generate
        let output = self.model.generate(&prompt, &params)?;

        // Only append to history AFTER successful generation
        self.history.push(user_msg);
        self.history.push(ChatMessage::assistant(&output.text));

        Ok(output)
    }

    /// Send a message and stream the response token-by-token.
    ///
    /// Returns a [`TextStream`] iterator. The caller should consume the stream
    /// to completion, then call [`finish_stream`](Self::finish_stream) with the
    /// collected text to append the assistant reply to the conversation history.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ironmill_torch::{Model, GenParams};
    /// # let mut model = Model::from_pretrained("./model/").build().unwrap();
    /// # let mut chat = model.chat().build();
    /// let stream = chat.send_stream("Hello")?;
    /// let mut full_text = String::new();
    /// for chunk in stream {
    ///     let chunk = chunk?;
    ///     full_text.push_str(&chunk.text);
    /// }
    /// chat.finish_stream(full_text);
    /// # Ok::<(), ironmill_torch::TorchError>(())
    /// ```
    pub fn send_stream<'a>(&'a mut self, message: &str) -> Result<TextStream<'a>, TorchError> {
        let user_msg = ChatMessage::user(message);
        let mut messages = self.history.clone();
        messages.push(user_msg.clone());
        let prompt = self.model.tokenizer().apply_chat_template(&messages)?;

        // Append user message to history before streaming; the assistant
        // response is appended via `finish_stream` after the caller collects it.
        self.history.push(user_msg);
        self.model.stream(&prompt, &self.default_params)
    }

    /// Append the assistant's reply after a streaming response completes.
    ///
    /// Call this after consuming the [`TextStream`] returned by
    /// [`send_stream`](Self::send_stream) to keep the conversation history
    /// in sync.
    pub fn finish_stream(&mut self, assistant_text: impl Into<String>) {
        self.history.push(ChatMessage::assistant(assistant_text));
    }

    /// Return the conversation history.
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Clear the conversation history, keeping the system prompt.
    pub fn reset(&mut self) {
        let system = self
            .history
            .iter()
            .find(|m| m.role == Role::System)
            .cloned();
        self.history.clear();
        if let Some(sys) = system {
            self.history.push(sys);
        }
    }
}

/// Builder for [`ChatSession`].
pub struct ChatSessionBuilder<'a> {
    model: &'a mut Model,
    system_prompt: Option<String>,
    params: GenParams,
}

impl<'a> ChatSessionBuilder<'a> {
    pub(crate) fn new(model: &'a mut Model) -> Self {
        Self {
            model,
            system_prompt: None,
            params: GenParams::default(),
        }
    }

    /// Set the system prompt.
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set default generation parameters.
    pub fn params(mut self, params: GenParams) -> Self {
        self.params = params;
        self
    }

    /// Build the chat session.
    pub fn build(self) -> ChatSession<'a> {
        let mut history = Vec::new();
        if let Some(sys) = self.system_prompt {
            history.push(ChatMessage::system(sys));
        }
        ChatSession::new(self.model, history, self.params)
    }
}
