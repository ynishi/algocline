# algocline — development task runner
# Usage: just <recipe>

_default:
    @just --list -u

# ─── Check ──────────────────────────────────────────────────────

# Run all checks (fmt, clippy, test, V0 invariants) — CI equivalent
[group('allow-agent')]
ci: fmt-check lua-fmt-check clippy clippy-nn test test-nn check-invariants check-agent-index

# Lint with clippy (warnings = errors)
[group('allow-agent')]
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Check formatting (Rust)
[group('allow-agent')]
fmt-check:
    cargo fmt --all -- --check

# Check formatting (Lua, stylua)
[group('allow-agent')]
lua-fmt-check:
    stylua --check .

# ─── Build ──────────────────────────────────────────────────────

# Type-check without codegen
[group('allow-agent')]
check:
    cargo check --workspace --all-targets

# Build release binary
[group('allow-agent')]
build:
    cargo build --release

# Install locally (for MCP server reload)
install:
    cargo install --path .

# ─── Test ───────────────────────────────────────────────────────

# Run all tests
[group('allow-agent')]
test:
    cargo test --workspace

# Run tests matching a pattern
[group('allow-agent')]
filter PATTERN:
    cargo test --workspace -- {{PATTERN}}

# Run e2e tests only
[group('allow-agent')]
e2e:
    cargo test --test e2e

# Run nn-feature tests (candle wrapper unit tests + feature-gated engine bridge
# smoke + feature-gated engine lib unit tests). `just test` does NOT cover the
# nn-feature layer because it lives behind `--features nn`; this recipe closes
# that gap so nn work can be verified with one command.
#
# The `--lib` line was added for L5b-S1 (`nn_wrap.rs` bridge tests). Inline
# `#[cfg(test)] mod ..._bridge_tests { ... }` modules under `--features nn`
# would otherwise be silently skipped by both `just test` and the previous
# `--test nn_bridge_smoke`-only invocation.
[group('allow-agent')]
test-nn:
    cargo test -p algocline-nn
    cargo test -p algocline-engine --features nn --lib
    cargo test -p algocline-engine --features nn --test nn_bridge_smoke
    cargo test -p algocline-engine --features nn --test nn_distill_teacher_card_e2e
    cargo test -p algocline-engine --features nn --test gameai_smoke_test

# Clippy for the nn-feature layer. `just clippy` runs the default (nn off)
# workspace clippy; this recipe closes the gap for the `--features nn` code
# paths (`crates/algocline-engine/src/bridge/nn_card.rs` +
# `crates/algocline-engine/src/bridge/nn_wrap.rs` + `algocline-nn` crate lib).
[group('allow-agent')]
clippy-nn:
    cargo clippy -p algocline-nn --all-targets -- -D warnings
    cargo clippy -p algocline-engine --features nn --all-targets -- -D warnings

# Install locally with the `nn` feature enabled (alc.nn Lua surface + candle).
# Same overwrite target as `just install` (~/.cargo/bin/alc); the default
# `just install` builds without nn to keep the MCP binary light.
[group('allow-agent')]
install-nn:
    cargo install --path . --features nn

# Review insta snapshots (interactive)
snapshots:
    cargo insta review

# ─── Format ─────────────────────────────────────────────────────

# Auto-format all code (Rust)
[group('allow-agent')]
fmt:
    cargo fmt --all

# Auto-format Lua code (stylua)
[group('allow-agent')]
lua-fmt:
    stylua .

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
[group('allow-agent')]
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
    #
    # Rustdoc intra-doc links inside doc comments (`///` / `//!`) reference
    # `algocline_core::AppDir::nn_dir` etc. as documentation about *what the
    # service layer resolves for us* — engine keeps the direct dependency in
    # Cargo.toml purely for these links. Those are not runtime references,
    # so we strip doc-comment lines before deciding failure. Ordinary `//`
    # code comments are left in the grep window on purpose so that "sneaking
    # a use behind a `// comment`" still gets caught.
    if grep -rn -E 'algocline_core::(AppDir|AppConfig)' \
            crates/algocline-engine/src/ --include='*.rs' \
            | grep -v -E '^[^:]+:[0-9]+:[[:space:]]*(///|//!)'; then
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

# ─── Docs ───────────────────────────────────────────────────────

# Verify docs/AGENT_INDEX.md: all linked paths exist + all docs/*.md are
# referenced (except AGENT_INDEX.md itself).
[group('allow-agent')]
check-agent-index:
    #!/usr/bin/env bash
    set -euo pipefail
    index=docs/AGENT_INDEX.md
    if [ ! -f "$index" ]; then
        echo "FAIL: $index not found" >&2
        exit 1
    fi
    fail=0
    # 1. Every markdown link target referenced from the index must exist.
    #    Match `](path)` for relative paths (skip http/https/anchor-only).
    while IFS= read -r target; do
        # strip optional #anchor
        path="${target%%#*}"
        [ -z "$path" ] && continue
        case "$path" in
            http://*|https://*) continue ;;
        esac
        resolved="docs/$path"
        case "$path" in ../*) resolved="${path#../}" ;; esac
        if [ ! -e "$resolved" ]; then
            echo "FAIL: linked path missing: $path (resolved=$resolved)" >&2
            fail=1
        fi
    done < <(grep -oE '\]\(([^)]+)\)' "$index" | sed -E 's/^\]\(//; s/\)$//')
    # 2. Every docs/*.md must be referenced from the index (except itself).
    for f in docs/*.md; do
        base=$(basename "$f")
        [ "$base" = "AGENT_INDEX.md" ] && continue
        if ! grep -qF "$base" "$index"; then
            echo "FAIL: docs/$base not referenced in $index" >&2
            fail=1
        fi
    done
    if [ "$fail" -ne 0 ]; then
        exit 1
    fi
    echo "AGENT_INDEX.md OK"

# Generate LLM-facing docs (llms.txt / llms-full.txt) under docs/aidoc/
[group('allow-agent')]
aidoc-gen:
    cargo aidoc

# CI drift + lint gate (exit 2 on drift, --strict promotes lint warnings to errors)
[group('allow-agent')]
aidoc-check:
    cargo aidoc --check --strict

# ─── Codegen ────────────────────────────────────────────────────

# Regenerate types/alc_shapes.d.lua from embedded alc_shapes Lua sources
[group('allow-agent')]
gen-shapes:
    cargo run -p algocline-app --example gen_alc_shapes_dlua

# ─── Vendor Sync ────────────────────────────────────────────────
#
# Sync vendored copies of upstream Lua packages into `crates/**/vendor/`.
# Each recipe wipes the destination dir, copies every `.lua` file from
# the source dir preserving subdir structure, and prepends a 2-line
# origin marker to each vendored file.
#
# Source dirs default to local checkouts under `$HOME/projects/`.
# Override per invocation via env vars:
#   ALC_SHAPES_SRC=<abs path> just vendor-alc-shapes
#   EVALFRAME_SRC=<abs path>  just vendor-evalframe
#
# After a sync, run `just ci` (fmt / clippy / test / lua-fmt-check /
# check-invariants) to verify no regressions before committing.

# Sync vendored alc_shapes.
# Source override: ALC_SHAPES_SRC=<abs path>
[group('allow-agent')]
vendor-alc-shapes:
    #!/usr/bin/env bash
    src="${ALC_SHAPES_SRC:-$HOME/projects/algocline-bundled-packages/alc_shapes}"
    just _vendor-lua-pkg alc_shapes "$src" crates/algocline-app/src/service/gendoc/alc_shapes

# Sync vendored evalframe.
# Source override: EVALFRAME_SRC=<abs path>
[group('allow-agent')]
vendor-evalframe:
    #!/usr/bin/env bash
    src="${EVALFRAME_SRC:-$HOME/projects/evalframe/evalframe}"
    just _vendor-lua-pkg evalframe "$src" crates/algocline-engine/src/vendor/evalframe

# Sync every vendored Lua package.
[group('allow-agent')]
vendor-sync: vendor-alc-shapes vendor-evalframe

# Internal: shared vendor-sync body used by every `vendor-*` recipe.
# Not marked `[group('allow-agent')]` because it is not meant to be
# invoked directly.
_vendor-lua-pkg NAME SRC DST:
    #!/usr/bin/env bash
    set -euo pipefail
    name="{{NAME}}"
    src="{{SRC}}"
    dst="{{DST}}"
    if [ ! -d "$src" ]; then
        echo "vendor-sync: source dir '$src' not found (set {{ uppercase(NAME) }}_SRC to override)" >&2
        exit 1
    fi
    # Extract semver from init.lua; falls back to "unknown" when the
    # file has no recognisable version field.
    version="$(
        if [ -f "$src/init.lua" ]; then
            grep -hE "(^\s*version\s*=|M\.VERSION\s*=)" "$src/init.lua" \
                | head -1 \
                | sed -E "s/.*['\"]([0-9]+\.[0-9]+\.[0-9]+[^'\"]*)['\"].*/\1/"
        else
            echo ""
        fi
    )"
    version="${version:-unknown}"
    # Wipe the destination so files removed upstream stop shipping.
    rm -rf "$dst"
    mkdir -p "$dst"
    count=0
    while IFS= read -r rel; do
        rel="${rel#./}"
        subdir=$(dirname "$rel")
        mkdir -p "$dst/$subdir"
        # Recipe names use hyphens; package names may use underscores
        # (e.g. `alc_shapes`). The origin marker points at the recipe.
        recipe_name="${name//_/-}"
        {
            printf -- "-- vendored from %s v%s\n" "$name" "$version"
            printf -- "-- DO NOT EDIT here — run \`just vendor-%s\` to resync\n" "$recipe_name"
            cat "$src/$rel"
        } > "$dst/$rel"
        count=$((count + 1))
    done < <(cd "$src" && find . -name '*.lua' -type f | sort)
    echo "vendor-sync: $name v$version -> $dst ($count files)"
