//! MCP (Model Context Protocol) server layer.
//!
//! Exposes [`AlcService`] — the rmcp handler implementing MCP tools
//! (`alc_run` / `alc_continue` / `alc_status` / `alc_card_*` /
//! `alc_pkg_*` / `alc_hub_*` / `alc_eval_*` etc.), the [`PromptCatalog`]
//! and [`ResourceCatalog`], the request registry used for MCP sampling
//! continuations, and the [`progress_forwarder`] mapping engine
//! progress events to MCP progress notifications.

pub mod progress_forwarder;
pub mod prompts;
pub mod req_registry;
pub mod resources;
mod service;

pub use prompts::PromptCatalog;
pub use resources::ResourceCatalog;
pub use service::AlcService;
