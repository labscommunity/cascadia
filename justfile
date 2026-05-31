# Cascadia task runner. Install `just` (https://github.com/casey/just), then
# run `just` to see this list. These recipes capture the non-obvious build
# incantations (the OpenVINO env var, the feature flag) so they aren't
# tribal knowledge.

# Show available recipes.
default:
    @just --list

# Build the binary in stub mode (no OpenVINO; mock engine only). Fast path
# for dev / CI on any platform.
build:
    cargo build --release -p cascadia

# Build with real OpenVINO. Requires INTEL_OPENVINO_DIR to point at an
# OpenVINO GenAI 2026.1+ SDK (see INSTALL.md). Usage:
#   just build-ov                       # uses $INTEL_OPENVINO_DIR
#   just build-ov /opt/intel/openvino   # overrides it
build-ov dir=env_var_or_default("INTEL_OPENVINO_DIR", ""):
    INTEL_OPENVINO_DIR="{{dir}}" cargo build --release -p cascadia --features openvino

# Environment + hardware self-check. Run after building. Add --features
# openvino to the build to have this enumerate real OV devices.
doctor:
    cargo run -p cascadia -- doctor

# Run a model on a single machine (OpenAI API on :8000). Stub build serves
# the mock engine; an --features openvino build serves the real model.
#   just run unsloth/Meta-Llama-3.1-8B-Instruct
run model:
    cargo run --release -p cascadia -- run {{model}}

# List Cascadia peers advertising on the LAN.
discover:
    cargo run -p cascadia -- discover

# The full pre-commit gate CI runs (stub mode).
check:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets
    cargo test --workspace --all-targets

# Auto-format the whole workspace.
fmt:
    cargo fmt --all

# Install a shell completion script. Usage: just completions zsh > _cascadia
completions shell:
    cargo run -p cascadia -- completions {{shell}}
