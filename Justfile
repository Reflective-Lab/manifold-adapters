# manifold development commands
# Install: brew install just  |  cargo install just
# Usage:   just --list

set dotenv-load := true

# Show available recipes
default:
    @just --list

# Build the workspace
build:
    cargo build --workspace

# Build release artifacts
build-release:
    cargo build --workspace --release

# Check all workspace targets
check:
    cargo check --workspace --all-targets

# Check all storage adapter features
check-all:
    cargo check --workspace --all-targets --features manifold/all-storage

# Refresh the OpenRouter model catalog cache (~/.cache/manifold/openrouter-catalog.json).
# Pulls latest pricing and capabilities from https://openrouter.ai/api/v1/models.
refresh-model-catalog:
    cargo run -p converge-manifold-adapters --example refresh_catalog --features _http

# Run tests
test:
    cargo test --workspace --all-targets

# Run tests with all features
test-all:
    cargo test --workspace --all-targets --all-features

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Format code
fmt:
    cargo fmt --all

# Run clippy
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Run clippy with all features
clippy-all:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Formatting plus clippy
lint: fmt-check clippy

# Generate docs
doc:
    cargo doc --workspace --no-deps

# Generate and open docs
doc-open:
    cargo doc --workspace --no-deps --open

# Local release hygiene.
# cargo audit is given the same RUSTSEC ignore list as deny.toml, since it
# does not read deny.toml. Keep this list in lockstep with deny.toml.
security-audit:
    cargo audit --deny warnings \
        --ignore RUSTSEC-2023-0071 \
        --ignore RUSTSEC-2023-0089 \
        --ignore RUSTSEC-2025-0057 \
        --ignore RUSTSEC-2025-0119 \
        --ignore RUSTSEC-2024-0436 \
        --ignore RUSTSEC-2025-0134 \
        --ignore RUSTSEC-2026-0002 \
        --ignore RUSTSEC-2026-0194 \
        --ignore RUSTSEC-2026-0195 \
        --ignore RUSTSEC-2026-0173
    cargo deny check

# Session opener
focus: status check test

# Git status and recent commits
status:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d .git ]; then
      git status --short --branch
      git log --oneline -5
    else
      echo "no local .git"
    fi

# Remove build artifacts
clean:
    cargo clean

# ── Release-grade gates (appended from ~/dev/reflective-templates/converge-extension) ─
# Standard: https://github.com/Reflective-Lab/converge/blob/main/kb/Standards/Extension%20Release%20Checklist.md

# Gate 2: workspace coverage. ≥ 80% per crate, no regression.
coverage:
    #!/usr/bin/env bash
    set -euo pipefail
    out_dir="target/coverage"
    mkdir -p "${out_dir}/html"
    common=(--workspace --lib --tests
        --ignore-filename-regex '(^|/)(tests|benches|examples)/')
    cargo llvm-cov clean --workspace
    rm -rf target/tests/trybuild
    cargo llvm-cov "${common[@]}" --no-report
    cargo llvm-cov report \
        --json --summary-only --output-path "${out_dir}/converge-coverage.json"
    cargo llvm-cov report \
        --lcov --output-path "${out_dir}/lcov.info"
    cargo llvm-cov report \
        --html --output-dir "${out_dir}/html"
    pct=$(python3 -c "import json; d=json.load(open('${out_dir}/converge-coverage.json')); print(f\"{d['data'][0]['totals']['lines']['percent']:.1f}\")")
    echo "coverage: ${pct}%  json→${out_dir}/converge-coverage.json  lcov→${out_dir}/lcov.info  html→${out_dir}/html/index.html"
    awk -v p="${pct}" 'BEGIN { if (p+0 < 80) { print "FAIL: coverage " p "% below 80% floor"; exit 1 } }'

# Gate 3: Criterion baseline. Set PERF_BASELINE to the release tag.

# Gate 3: Criterion baseline. Set PERF_BASELINE to the release tag.
performance-profile:
    #!/usr/bin/env bash
    set -euo pipefail
    name="${PERF_BASELINE:-v0.1.0}"
    mode_flag="--save-baseline"
    if [ -d "target/criterion" ]; then
        existing="$(find target/criterion -mindepth 2 -maxdepth 3 -type d -name "${name}" -print -quit 2>/dev/null || true)"
        if [ -n "${existing}" ]; then
            mode_flag="--baseline"
        fi
    fi
    echo "performance-profile: ${mode_flag} ${name}"
    cargo bench --workspace -- "${mode_flag}" "${name}" || true
    if [ -f scripts/extract-criterion-baseline.py ]; then
        python3 scripts/extract-criterion-baseline.py || \
            echo "warn: baseline extraction failed (non-fatal)"
    fi
    echo "performance-profile: criterion→target/criterion/"

# Gate 4: bounded soak run. Configure with SOAK_DURATION_MIN (default 5).

# Gate 4: bounded soak run. Configure with SOAK_DURATION_MIN (default 5).
soak:
    #!/usr/bin/env bash
    set -euo pipefail
    duration_min="${SOAK_DURATION_MIN:-5}"
    out_dir="target/soak"
    mkdir -p "${out_dir}"
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    log="${out_dir}/soak-${stamp}.log"
    cycles=$(awk -v d="${duration_min}" 'BEGIN { printf "%d", 200 * d }')
    iterations=$(awk -v d="${duration_min}" 'BEGIN { printf "%d", 40 * d }')
    concurrency=100
    echo "soak: duration=${duration_min}min cycles=${cycles} concurrency=${concurrency} iterations=${iterations}" | tee "${log}"
    SOAK_CYCLES="${cycles}" \
    SOAK_CONCURRENCY="${concurrency}" \
    SOAK_ITERATIONS="${iterations}" \
    cargo test --workspace -- --include-ignored soak --nocapture 2>&1 | tee -a "${log}"
    ln -sf "soak-${stamp}.log" "${out_dir}/latest.log"
    echo "soak: log → ${log}"

# The five-command release ritual. All five must be green before tagging.

# The five-command release ritual. All five must be green before tagging.
release-check:
    just security-audit
    just coverage
    PERF_BASELINE="v$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"(.*)".*/\1/')" just performance-profile
    SOAK_DURATION_MIN=5 just soak
    just lint
    cargo test --workspace
