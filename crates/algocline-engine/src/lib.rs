//! Lua strategy execution engine.
//!
//! Owns the [`Session`] lifecycle (running / needs_response / completed),
//! the [`Executor`] loop that drives Lua coroutines and mediates
//! `alc.llm()` pauses, the [`FileCardStore`] for structured session
//! artifacts, [`bridge`] modules that expose Rust-backed globals
//! (`alc.*`) to Lua, and the resolver factory that wires strategies
//! to LLM providers.

pub mod bridge;
pub mod card;
pub mod execution;
mod executor;
mod llm_bridge;
mod resolver_factory;
pub mod session;
pub mod state;
mod variant_pkg;

pub use card::FileCardStore;
pub use executor::{Executor, SessionDirs};
pub use llm_bridge::{LlmRequest, QueryRequest};
pub use session::{
    ExecutionResult, FeedResult, PendingFilter, PromptProjection, Session, SessionRegistry,
    DEFAULT_PROMPT_PREVIEW_CHARS,
};
pub use state::JsonFileStore;
pub use variant_pkg::VariantPkg;
