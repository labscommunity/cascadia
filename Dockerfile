# Cascadia reference image — bundles the OpenVINO + Level-Zero GPU runtime
# stack so the hardest onboarding step (getting OpenVINO to see the Intel
# GPU) becomes `docker run` instead of a multi-package host install.
#
# This is a REFERENCE image, not a CI-tested artifact: GPU inference needs
# the host's Intel GPU passed into the container and a host driver new enough
# to match the in-image compute-runtime. CPU/stub use works anywhere.
#
# The `openvino` target is x86_64-only (the SDK archive and Intel's drivers are
# amd64); `stub` builds on any arch. GPU passthrough is `--device /dev/dri` on
# Linux; under WSL2 it is `--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl`, and
# the NPU is not exposed to containers at all.
#
# Build (stub, no OpenVINO):
#   docker build --target stub -t cascadia:stub .
# Build (real OpenVINO). Use the ubuntu22 archive: the builder below is Debian
# bookworm (glibc 2.36) and the ubuntu24 build of OpenVINO imports GLIBC_2.38
# symbols, so it cannot be linked here. The ubuntu22 libs run fine on the
# Ubuntu 24.04 runtime image.
#   docker build --target openvino \
#     --build-arg OPENVINO_URL=https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.2/linux/openvino_genai_ubuntu22_2026.2.0.0_x86_64.tar.gz \
#     -t cascadia:ov .
# Run (GPU). Mount a model directory — cascadia serves pre-exported models
# from disk (an OpenVINO IR, or a `cascadia shard` tree); it does not download
# or convert at run time:
#   docker run --rm --device /dev/dri -p 8000:8000 -v ~/models:/models \
#     cascadia:ov run /models/llama-3.1-8b-int4-ov

# ── builder ──────────────────────────────────────────────────────────────
FROM rust:1.89-bookworm AS builder
WORKDIR /src

# OpenVINO GenAI SDK archive URL. Override with --build-arg. Empty = stub.
ARG OPENVINO_URL=""

# C++ toolchain for the FFI shim. The OpenCL loader is needed at LINK time too:
# libopenvino_genai.so imports OpenCL, so without it the final link fails.
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ curl ca-certificates ocl-icd-libopencl1 \
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

# openvino target — links against the bundled SDK. TBB lives outside
# runtime/lib/intel64 and libopenvino.so imports it, so it must be on the
# linker's search path.
FROM builder AS build-openvino
RUN INTEL_OPENVINO_DIR=/opt/intel/openvino \
    LIBRARY_PATH=/opt/intel/openvino/runtime/3rdparty/tbb/lib:/opt/intel/openvino/runtime/lib/intel64 \
    LD_LIBRARY_PATH=/opt/intel/openvino/runtime/3rdparty/tbb/lib:/opt/intel/openvino/runtime/lib/intel64 \
        cargo build --release -p cascadia --features openvino

# ── runtime: stub ─────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS stub
COPY --from=build-stub /src/target/release/cascadia /usr/local/bin/cascadia
ENTRYPOINT ["cascadia"]
CMD ["doctor"]

# ── runtime: openvino (GPU) ────────────────────────────────────────────────
FROM ubuntu:24.04 AS openvino
# Intel GPU runtime: OpenCL ICD + Compute Runtime (+ Level-Zero for NPU).
# Without these the OpenVINO GPU plugin silently sees only the CPU.
#
# From Intel's repo, not the distro: Ubuntu ships Compute Runtime 23.43 and
# Debian 22.43, both older than the hardware cascadia targets (Lunar Lake, Arc
# B). Intel publishes that repo for Ubuntu only — hence the Ubuntu base.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget \
    && wget -qO- https://repositories.intel.com/gpu/intel-graphics.key \
         | gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu noble client" \
         > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 intel-opencl-icd libze-intel-gpu1 libze1 \
    # Don't leave Intel's repo (or the tools that added it) in the shipped image:
    # anything built FROM this would silently inherit it as a trusted apt source.
    && apt-get purge -y --auto-remove gnupg wget \
    && rm -f /etc/apt/sources.list.d/intel-gpu.list \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-openvino /opt/intel/openvino/runtime/lib/intel64 /opt/intel/openvino/runtime/lib/intel64
# TBB ships beside the runtime, not inside it — copy the SDK's own build rather
# than apt's libtbb12, which is a different oneTBB than OpenVINO was built with.
COPY --from=build-openvino /opt/intel/openvino/runtime/3rdparty/tbb/lib /opt/intel/openvino/runtime/3rdparty/tbb/lib
COPY --from=build-openvino /src/target/release/cascadia /usr/local/bin/cascadia
ENV LD_LIBRARY_PATH=/opt/intel/openvino/runtime/lib/intel64:/opt/intel/openvino/runtime/3rdparty/tbb/lib
ENTRYPOINT ["cascadia"]
CMD ["doctor"]
