#!/bin/bash
# Build and run the multi-stream pipeline tests on the build host's RAM disk (this Mac has no room for test binaries).
#   git archive --format=tar.gz -o /tmp/t.tar.gz HEAD && scp /tmp/t.tar.gz miner:inkling-build/ms-test-src.tar.gz
#   ssh miner 'bash -s' < autolab/bench/miner_test.sh
set -eo pipefail
B=$HOME/inkling-build; T=/dev/shm/inkling-test; export PATH=$HOME/.cargo/bin:$PATH; set +u
. $B/ov/openvino_genai_ubuntu24_2026.3.1.0_x86_64/setupvars.sh >/dev/null 2>&1
mkdir -p $T && cd $T && rm -rf repo && mkdir repo && tar -xzf $B/ms-test-src.tar.gz -C repo && cd repo && \
INTEL_OPENVINO_DIR=$B/ov/openvino_genai_ubuntu24_2026.3.1.0_x86_64 CARGO_TARGET_DIR=$T/target \
nice -n 10 cargo test -p cascadia-engine-sparse-moe --test inkling_streams_ready --test inkling_streams_wire --test inkling_streams_head_batch --test inkling_streams_return --test inkling_streams --test inkling_streams_spec -- --nocapture > "$T/test.log" 2>&1
grep -E "^test |test result|shared|panicked|error(\[|:)|FAILED|warning: unused" "$T/test.log"
du -sh $T/target 2>/dev/null | tail -1
