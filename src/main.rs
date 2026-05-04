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
mod pool_worker;

use std::sync::Arc;

use algocline_app::{AppConfig, AppService, EngineApi};
use algocline_engine::Executor;
use algocline_mcp::AlcService;
use clap::{Parser, Subcommand};
use rmcp::{transport::stdio, ServiceExt};

#[derive(Parser)]
#[command(
    name = "alc",
    version,
    about = "algocline — LLM amplification engine (MCP server)"
)]
#[command(
    long_about = "By default `alc` runs as an MCP stdio server, intended to be launched \
by an MCP client such as Claude Code. Use subcommands for one-shot maintenance tasks. \
For ad-hoc shell access to MCP tools, run `alc` from an agent-block harness that calls the \
MCP server directly."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Install bundled packages into ~/.algocline/packages/
    Init {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        rest: Vec<String>,
    },
    /// Re-install bundled packages with --force (alias for `init --force`)
    Update {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        rest: Vec<String>,
    },
    /// (Internal) Pool worker subprocess entrypoint — not for direct use.
    #[command(hide = true)]
    PoolWorker {
        /// Session ID assigned to this worker.
        #[arg(long)]
        sid: String,
        /// Unix-domain socket path to listen on.
        #[arg(long)]
        sock: std::path::PathBuf,
    },
}

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

fn resolve_lib_paths() -> Vec<algocline_app::SearchPath> {
    use algocline_app::SearchPath;
    let mut paths = Vec::new();

    // 1. ALC_PACKAGES_PATH env (colon-separated, highest priority)
    //    Set via .mcp.json env or user override
    if let Ok(env_paths) = std::env::var("ALC_PACKAGES_PATH") {
        for p in env_paths.split(':') {
            let path = std::path::PathBuf::from(p);
            if path.is_dir() {
                paths.push(SearchPath::env(path));
            }
        }
    }

    // 2. ~/.algocline/packages/ (installed packages)
    if let Some(home) = dirs::home_dir() {
        let packages = home.join(".algocline").join("packages");
        if packages.is_dir() {
            paths.push(SearchPath::default_global(packages));
        }
    }

    paths
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Init { rest }) => return init::run(&rest, false).await,
        Some(Commands::Update { rest }) => return init::run(&rest, true).await,
        Some(Commands::PoolWorker { sid, sock }) => {
            let config = AppConfig::from_env();
            setup_tracing(config.log_dir.as_deref())?;
            return pool_worker::run(sid, sock).await;
        }
        None => {} // fall through to MCP server mode
    }

    // Default: MCP server mode
    let config = AppConfig::from_env();
    setup_tracing(config.log_dir.as_deref())?;

    // Eager init of the card event bus so that ALC_CARD_SINKS
    // registration logs surface at startup rather than on the first
    // card write. Placed after setup_tracing so that tracing::info!
    // calls inside init_event_bus() reach the configured subscriber.
    algocline_engine::card::init_event_bus();

    // Distribute type stubs (non-fatal: log warning on failure)
    match init::distribute_types() {
        Ok(init::DistributedTypes { alc, alc_shapes }) => {
            tracing::debug!(
                "types installed: {} + {}",
                alc.display(),
                alc_shapes.display()
            )
        }
        Err(e) => tracing::warn!("failed to install type stubs: {e}"),
    }

    tracing::info!("algocline server starting");

    let search_paths = resolve_lib_paths();
    let lib_paths: Vec<_> = search_paths.iter().map(|sp| sp.path.clone()).collect();
    let executor = Arc::new(Executor::new(lib_paths).await?);
    let app_dir = config.app_dir(); // Arc<AppDir> — resolved before AppService construction
    let app = Arc::new(AppService::new(executor, config, search_paths));
    let server = AlcService::new(app as Arc<dyn EngineApi>, app_dir);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
