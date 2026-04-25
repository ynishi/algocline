# algocline — development task runner
# Usage: just <recipe>

_default:
    @just --list -u

# ─── Check ──────────────────────────────────────────────────────

# Run all checks (fmt, clippy, test, V0 invariants) — CI equivalent
[group: 'agent']
ci: fmt-check clippy test check-invariants

# Lint with clippy (warnings = errors)
[group: 'agent']
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Check formatting
[group: 'agent']
fmt-check:
    cargo fmt --all -- --check

# ─── Build ──────────────────────────────────────────────────────

# Type-check without codegen
[group: 'agent']
check:
    cargo check --workspace --all-targets

# Build release binary
[group: 'agent']
build:
    cargo build --release

# Install locally (for MCP server reload)
install:
    cargo install --path .

# ─── Test ───────────────────────────────────────────────────────

# Run all tests
[group: 'agent']
test:
    cargo test --workspace

# Run tests matching a pattern
[group: 'agent']
filter PATTERN:
    cargo test --workspace -- {{PATTERN}}

# Run e2e tests only
[group: 'agent']
e2e:
    cargo test --test e2e

# Review insta snapshots (interactive)
snapshots:
    cargo insta review

# ─── Format ─────────────────────────────────────────────────────

# Auto-format all code
[group: 'agent']
fmt:
    cargo fmt --all

# ─── Quality ────────────────────────────────────────────────────

# Full pre-commit check: format, lint, test
ready:
    just fmt
    just clippy
    just test

# ─── Invariants ─────────────────────────────────────────────────

# Check V0 AppDir-guard invariants:
#   Inv-1: Service layer (algocline-app) no longer reads HOME / ALC_HOME
#          directly — `AppConfig::resolve_app_dir` / `resolve_log_dir` in
#          `service/config.rs` are the single whitelisted resolvers.
#   Inv-2: Execution layer (algocline-engine, incl. `prelude.lua`) no
#          longer reads HOME / ALC_HOME directly.
#   Inv-3: `algocline_core::AppDir` / `AppConfig` are not referenced from
#          inside the engine crate (engine public API stays free of the
#          service-layer abstractions).
#   Inv-4: InstalledManifestStore encapsulation — no `installed.json` filesystem
#          calls live outside `service/manifest.rs` (the `FsInstalledManifestStore`
#          impl block is the single source). Added in Subtask 3b together
#          with the `InstalledManifestStore` trait extraction (Subtask 3a).
[group: 'agent']
check-invariants:
    #!/usr/bin/env bash
    set -euo pipefail
    fail=0
    # Inv-1: Service layer must route every HOME access through AppConfig.
    # Whitelist:
    #   `config.rs`     — single source for AppConfig::resolve_app_dir / resolve_log_dir.
    #   `test_support.rs` — `FakeHome` test fixture (軸 A defer; guards HOME for
    #                      integration tests while parallel isolation is not yet in place).
    if grep -rn -E 'dirs::home_dir\(\)|std::env::var(_os)?\("(HOME|ALC_HOME)"\)' \
            crates/algocline-app/src/service/ --include='*.rs' \
            | grep -v -E '^crates/algocline-app/src/service/(config|test_support)\.rs:'; then
        echo "Inv-1 FAILED: HOME/ALC_HOME read outside service/config.rs (or FakeHome)" >&2
        fail=1
    fi
    # Inv-2 (Rust): Execution layer (engine crate) must not read HOME.
    if grep -rn -E 'dirs::home_dir\(\)|std::env::var(_os)?\("(HOME|ALC_HOME)"\)' \
            crates/algocline-engine/src/ --include='*.rs'; then
        echo "Inv-2 (Rust) FAILED: HOME/ALC_HOME read in engine crate" >&2
        fail=1
    fi
    # Inv-2 (Lua): prod Lua (prelude.lua) must not call os.getenv("HOME"|"ALC_HOME").
    if grep -n -E 'os\.getenv\("(HOME|ALC_HOME)"\)' \
            crates/algocline-engine/src/prelude.lua; then
        echo "Inv-2 (Lua) FAILED: HOME/ALC_HOME read in prod Lua" >&2
        fail=1
    fi
    # Inv-3: engine crate must not import AppDir/AppConfig from core.
    if grep -rn -E 'algocline_core::(AppDir|AppConfig)' \
            crates/algocline-engine/src/ --include='*.rs'; then
        echo "Inv-3 FAILED: engine references algocline_core::AppDir/AppConfig" >&2
        fail=1
    fi
    # Inv-4: InstalledManifestStore encapsulation — every `std::fs::*` call that
    # touches `installed.json` / `installed.json.lock` / `installed.json.tmp`
    # must sit inside `service/manifest.rs` (the `FsInstalledManifestStore` impl
    # block). Call sites in sibling service files may still read the
    # *path* via `app_dir.installed_json()` for diagnostics — that is not
    # filesystem IO and does not surface here.
    #
    # Limitation: this grep is literal-only. An indirection pattern like
    #   let p = app_dir.installed_json();
    #   std::fs::write(p, ...)
    # would evade it because the `installed.json` literal is no longer
    # on the `std::fs::*` line. The real guard is the `FsInstalledManifestStore`
    # impl boundary itself (the trait confines IO); this grep is a
    # belt-and-braces sanity check. The follow-up that plumbs
    # `Arc<dyn InstalledManifestStore>` through `AppService` (alongside the
    # sibling `HubRepo` / `EvalRepo` extractions) will let us delete
    # this grep once the trait boundary is fully exercised.
    if grep -rn -E 'std::fs::[A-Za-z_]+[^;]*installed\.json' \
            crates/algocline-app/src/service/ --include='*.rs' \
            | grep -v -E '^crates/algocline-app/src/service/manifest\.rs:'; then
        echo "Inv-4 FAILED: installed.json filesystem access outside service/manifest.rs" >&2
        fail=1
    fi
    if [ "$fail" -ne 0 ]; then
        exit 1
    fi
    echo "All AppDir-guard invariants PASS"

# ─── Publish ────────────────────────────────────────────────────

# Dry-run publish check (dependency order)
#
# NOTE: only the leaf (algocline-core) actually verifies — intermediate crates
# (engine/app/mcp/root) fail on dry-run because they reference the new version
# of upstream crates not yet on crates.io (cargo-release issue #691). Use this
# for syntax / Cargo.toml metadata checks; real verification happens during
# `just publish`.
publish-dry:
    cargo publish -p algocline-core --dry-run
    cargo publish -p algocline-engine --dry-run
    cargo publish -p algocline-app --dry-run
    cargo publish -p algocline-mcp --dry-run
    cargo publish -p algocline --dry-run

# Publish to crates.io in dependency order, then tag + push.
#
# Usage: just publish 0.27.0
#
# Pre-check verifies the passed VERSION matches Cargo.toml workspace version
# to prevent typos. Each `cargo publish` is followed by a 60s sleep so the
# crates.io index propagates before the next dependent crate is uploaded
# (CLAUDE.md §crates.io公開).
#
# NOT allow-agent — irreversible upload + git push. Human-only execution.
publish VERSION:
    @ACTUAL=$(grep '^version = ' Cargo.toml | head -1 | cut -d'"' -f2); \
        if [ "$ACTUAL" != "{{VERSION}}" ]; then \
            echo "ERROR: Cargo.toml workspace.package.version is $ACTUAL, but you passed {{VERSION}}"; \
            exit 1; \
        fi
    cargo publish -p algocline-core
    sleep 60
    cargo publish -p algocline-engine
    sleep 60
    cargo publish -p algocline-app
    sleep 60
    cargo publish -p algocline-mcp
    sleep 60
    cargo publish
    git tag v{{VERSION}}
    git push origin v{{VERSION}}

# ─── Codegen ────────────────────────────────────────────────────

# Regenerate types/alc_shapes.d.lua from embedded alc_shapes Lua sources
[group('allow-agent')]
gen-shapes:
    cargo run -p algocline-app --example gen_alc_shapes_dlua
