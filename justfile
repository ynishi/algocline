# algocline — development task runner
# Usage: just <recipe>

_default:
    @just --list -u

# ─── Check ──────────────────────────────────────────────────────

# Run all checks (fmt, clippy, test) — CI equivalent
[group: 'agent']
ci: fmt-check clippy test

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

# ─── Publish ────────────────────────────────────────────────────

# Dry-run publish check (dependency order)
publish-dry:
    cargo publish -p algocline-core --dry-run
    cargo publish -p algocline-engine --dry-run
    cargo publish -p algocline-app --dry-run
    cargo publish -p algocline-mcp --dry-run
    cargo publish -p algocline --dry-run
