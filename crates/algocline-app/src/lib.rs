//! Service layer sitting between `algocline-engine` and `algocline-mcp`.
//!
//! Hosts [`AppConfig`], [`AppService`], and the process pool ([`pool`])
//! providing UDS-based worker isolation for MCP tool calls. Also owns
//! the hub-projection preset resolution (`load_hub_projection_config`
//! and related config types) that translates preset selectors into
//! runtime-ready projection contexts.

pub mod pool;
mod service;

pub use service::hub_dist_preset::load_hub_projection_config;
pub use service::hub_dist_preset::{
    HubContext7Config, HubDevinConfig, HubProjectionConfig, ResolvedContext7, ResolvedDevin,
    PRESET_CATALOG_VERSION,
};
pub use service::{
    AppConfig, AppService, EngineApi, LogDirSource, PackOptions, QueryResponse, SearchPath,
    TokenUsage, UnpackMode, UnpackOptions,
};

#[doc(hidden)]
pub use service::gendoc::alc_shapes_codegen::gen_alc_shapes_dlua_contents;
