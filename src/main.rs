//! # algocline — LLM amplification engine
//!
//! algocline provides a Lua execution environment for structurally enhancing LLM
//! reasoning. Strategies are Pure Lua modules that call `alc.*` primitives
//! to orchestrate multi-step LLM interactions.
//!
//! ## Architecture: Three-Layer StdLib
//!
//! ```text
//! Layer 0: Runtime Primitives (Rust → alc.*)
//! │  Injected by bridge.rs into every Lua session.
//! │  These are the foundational building blocks that cannot be
//! │  expressed in Pure Lua (I/O, LLM calls, serialization).
//! │
//! │  alc.llm(prompt, opts?)       — Host LLM call via MCP Sampling
//! │  alc.json_encode / json_decode — serde_json bridge
//! │  alc.log(level, msg)          — tracing bridge
//! │  alc.state.get/set/keys/delete — persistent key-value store
//! │  alc.chunk(text, opts?)       — text segmentation
//! │
//! Layer 1: Prelude Combinators (Lua → alc.*)
//! │  Loaded from prelude.lua (embedded via include_str!).
//! │  Higher-order functions that compose Layer 0 primitives.
//! │  Auto-injected into alc.* namespace alongside Layer 0.
//! │
//! │  alc.map(items, fn)          — transform each element
//! │  alc.reduce(items, fn, init) — fold to single value
//! │  alc.vote(answers)           — majority aggregation
//! │  alc.filter(items, fn)       — conditional selection
//! │
//! Layer 2: Bundled Packages (require() from ~/.algocline/packages/)
//!    Installed to ~/.algocline/packages/ via `alc init`.
//!    Each is a self-contained Lua module built on Layer 0/1.
//!    Loaded explicitly via require("{name}").
//!
//!    explore  — UCB1 hypothesis space exploration    [selection]
//!    panel    — multi-perspective deliberation       [synthesis]
//!    chain    — iterative chain-of-thought           [reasoning]
//!    ensemble — independent sampling + majority vote [aggregation]
//!    verify   — draft-verify-revise cycle            [validation]
//! ```
//!
//! **Design rationale**: Layer 0/1 form the built-in library — always
//! available, no explicit import needed. Layer 2 packages are bundled
//! but opt-in via `require()`, analogous to how `tokio` relates to `std`
//! in the Rust ecosystem.

mod init;

use std::sync::Arc;

use algocline_app::{AppConfig, AppService};
use algocline_engine::Executor;
use algocline_mcp::AlcService;
use rmcp::{transport::stdio, ServiceExt};

fn setup_tracing(log_dir: Option<&std::path::Path>) -> anyhow::Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(false);

    let registry = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(stderr_layer);

    if let Some(dir) = log_dir {
        let file_appender = tracing_appender::rolling::daily(dir, "tracing.log");
        let file_layer = tracing_subscriber::fmt::layer()
            .with_writer(file_appender)
            .with_ansi(false);
        registry.with(file_layer).try_init()?;
    } else {
        registry.try_init()?;
    }

    Ok(())
}

fn resolve_lib_paths() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();

    // 1. ALC_PACKAGES_PATH env (colon-separated, highest priority)
    //    Set via .mcp.json env or user override
    if let Ok(env_paths) = std::env::var("ALC_PACKAGES_PATH") {
        for p in env_paths.split(':') {
            let path = std::path::PathBuf::from(p);
            if path.is_dir() {
                paths.push(path);
            }
        }
    }

    // 2. ~/.algocline/packages/ (installed packages)
    if let Some(home) = dirs::home_dir() {
        let packages = home.join(".algocline").join("packages");
        if packages.is_dir() {
            paths.push(packages);
        }
    }

    paths
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // `alc init` — install bundled packages (skip existing)
    if args.get(1).is_some_and(|a| a == "init") {
        return init::run(&args[2..], false).await;
    }

    // `alc update` — update all bundled packages (alias for init --force)
    if args.get(1).is_some_and(|a| a == "update") {
        return init::run(&args[2..], true).await;
    }

    // Default: MCP server mode
    let config = AppConfig::from_env();
    setup_tracing(config.log_dir.as_deref())?;

    tracing::info!("algocline server starting");

    let lib_paths = resolve_lib_paths();
    let executor = Arc::new(Executor::new(lib_paths).await?);
    let app = AppService::new(executor, config);
    let server = AlcService::new(app);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
