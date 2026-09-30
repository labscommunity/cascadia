#!/bin/bash
# Build the release binary on the build host (ssh alias `miner`): ship `git archive HEAD`, build there, fetch the binary.
#   git archive --format=tar.gz -o /tmp/ms-src.tar.gz HEAD && scp /tmp/ms-src.tar.gz miner:inkling-build/ms-src.tar.gz
#   ssh miner 'bash -s' < autolab/bench/miner_build.sh
#   scp miner:inkling-build/target/release/cascadia ~/inkling-release/builds/cascadia-$(git rev-parse --short=8 HEAD)
# The build host's root disk is full: never add build trees there; tests go to its RAM disk (miner_test.sh).
set -eo pipefail
B=$HOME/inkling-build; export PATH=$HOME/.cargo/bin:$PATH; set +u
. $B/ov/openvino_genai_ubuntu24_2026.3.1.0_x86_64/setupvars.sh >/dev/null 2>&1
cd $B && rm -rf repo && mkdir repo && tar -xzf ms-src.tar.gz -C repo && tar -xzf dash-dist.tar.gz -C repo/crates/cascadia-dashboard/web && cd repo && \
INTEL_OPENVINO_DIR=$B/ov/openvino_genai_ubuntu24_2026.3.1.0_x86_64 CARGO_TARGET_DIR=$B/target \
nice -n 10 cargo build --release -p cascadia --features openvino,dashboard-embed 2>&1 | tail -5
ls -la $B/target/release/cascadia && sha256sum $B/target/release/cascadia
