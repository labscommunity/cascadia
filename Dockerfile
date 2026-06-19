# Cascadia reference image — bundles the OpenVINO + Level-Zero GPU runtime
# stack so the hardest onboarding step (getting OpenVINO to see the Intel
# GPU) becomes `docker run` instead of a multi-package host install.
#
# This is a REFERENCE image, not a CI-tested artifact: GPU inference needs
# the host's Intel GPU passed into the container (`--device /dev/dri`) and
# a host driver new enough to match the in-image compute-runtime. CPU/stub
# use works anywhere. Pin OPENVINO_URL to the GenAI archive you want.
#
# Build (stub, no OpenVINO):
#   docker build --target stub -t cascadia:stub .
# Build (real OpenVINO):
#   docker build --target openvino \
#     --build-arg OPENVINO_URL=https://storage.openvinotoolkit.org/.../openvino_genai_ubuntu24_2026.2.0.0_x86_64.tar.gz \
#     -t cascadia:ov .
# Run (GPU):
#   docker run --rm --device /dev/dri -p 8000:8000 cascadia:ov \
#     run unsloth/Meta-Llama-3.1-8B-Instruct

# ── builder ──────────────────────────────────────────────────────────────
FROM rust:1.82-bookworm AS builder
WORKDIR /src

# OpenVINO GenAI SDK archive URL. Override with --build-arg. Empty = stub.
ARG OPENVINO_URL=""

# C++ toolchain for the FFI shim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fetch + unpack the OpenVINO GenAI SDK if a URL was given.
RUN if [ -n "$OPENVINO_URL" ]; then \
        mkdir -p /opt/intel && cd /opt/intel && \
        curl -fsSL "$OPENVINO_URL" -o ov.tgz && \
        tar -xzf ov.tgz && rm ov.tgz && \
        ln -s "$(find /opt/intel -maxdepth 1 -type d -name 'openvino*' | head -1)" /opt/intel/openvino ; \
    fi

COPY . .

# stub target — no OpenVINO link.
FROM builder AS build-stub
RUN cargo build --release -p cascadia

# openvino target — links against the bundled SDK.
FROM builder AS build-openvino
RUN INTEL_OPENVINO_DIR=/opt/intel/openvino \
        cargo build --release -p cascadia --features openvino

# ── runtime: stub ─────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS stub
COPY --from=build-stub /src/target/release/cascadia /usr/local/bin/cascadia
ENTRYPOINT ["cascadia"]
CMD ["doctor"]

# ── runtime: openvino (GPU) ────────────────────────────────────────────────
FROM debian:bookworm-slim AS openvino
# Intel GPU runtime: OpenCL ICD + Level-Zero + Compute Runtime. Without
# these the OpenVINO GPU plugin silently sees only the CPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero \
        libtbb12 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-openvino /opt/intel/openvino/runtime/lib/intel64 /opt/intel/openvino/runtime/lib/intel64
COPY --from=build-openvino /src/target/release/cascadia /usr/local/bin/cascadia
ENV LD_LIBRARY_PATH=/opt/intel/openvino/runtime/lib/intel64
ENTRYPOINT ["cascadia"]
CMD ["doctor"]
