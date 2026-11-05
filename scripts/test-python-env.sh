#!/usr/bin/env bash
# End-to-end checks of how `cascadia shard` and `cascadia export` find or install
# Python. Starts with nothing on PATH, so what this machine has can't leak in,
# and keeps everything it installs under its work dir. Needs the network (uv,
# Python, PyPI, pytorch.org and a tiny model from Hugging Face), about 6 GB free
# and about 10 minutes.
#
#   scripts/test-python-env.sh target/release/cascadia
#
# Prints PASS or FAIL per check and ends with ALL PYTHON ENV CHECKS PASSED; exits
# non-zero if any failed. Knobs: WORK (work dir, default a new temp dir, kept on
# failure), DRY_RUN=1 (list the checks and stop).
set -uo pipefail

BIN="${1:?usage: $0 <path to the cascadia binary>}"
case "$BIN" in /*|[A-Za-z]:*) ;; *) BIN="$PWD/$BIN" ;; esac
MODEL="hf-internal-testing/tiny-random-LlamaForCausalLM"
CHECKS="no-prompt-outside-terminal doctor-only-looks terminal-prompt setup-from-nothing second-run-installs-nothing int4-means-int4 export-through-private skip-check caller-python-vars-ignored activated-venv-wins broken-venv-falls-through python-flag-wins killed-setup-recovers damaged-env-rebuilds reuses-system-uv-python-packages system-python-unchanged changed-package-restored moe-exporters-reachable old-env-removed remove-deletes-everything"

if [ "${DRY_RUN:-0}" = 1 ]; then
    for c in $CHECKS; do echo "would run: $c"; done
    exit 0
fi

WORK="${WORK:-$(mktemp -d)}"
mkdir -p "$WORK/empty"
ROOT="$WORK/home"
FAILED=0
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) WINDOWS=1 ;; *) WINDOWS=0 ;; esac
HOST_PATH="$PATH"

pass() { echo "PASS $1"; }
fail() {
    echo "FAIL $1: $2"
    FAILED=$((FAILED + 1))
    tail -5 "$WORK/out.log" 2>/dev/null | sed 's/^/    | /'
}
check() { # <name> <condition as a string> <why it failed>
    if eval "$2"; then pass "$1"; else fail "$1" "$3"; fi
}

# Run cascadia as a user with only $RUN_PATH on PATH, no terminal and a clean
# Python-related environment; output in $WORK/out.log, exit code in $RC.
RUN_PATH="$WORK/empty"
cas() {
    env -u VIRTUAL_ENV -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH -u PYTHONHOME \
        PATH="$RUN_PATH" CASCADIA_PYTHON_HOME="$ROOT" "$BIN" "$@" </dev/null >"$WORK/out.log" 2>&1
    RC=$?
}
said() { grep -qF -- "$1" "$WORK/out.log"; }
shard() { rm -rf "$WORK/tree"; cas shard --model "$MODEL" -o "$WORK/tree" --num-stages 2 --quantization fp16 "$@"; }
envdir() { find "$ROOT" -maxdepth 1 -type d -name '????????????' | head -1; }
pybin() { if [ "$WINDOWS" = 1 ]; then echo "$1/Scripts/python.exe"; else echo "$1/bin/python"; fi; }
uvbin() { find "$ROOT/uv" -type f \( -name uv -o -name uv.exe \) | head -1; }
tree_ok() { [ -f "$WORK/tree/pipeline_config.json" ] && [ -d "$WORK/tree/stage_1" ]; }

# --- Before anything is installed ------------------------------------------
shard
check no-prompt-outside-terminal '[ "$RC" != 0 ] && said "--yes" && [ -z "$(envdir)" ]' \
    "exit $RC, should refuse and name --yes, installing nothing"

cas doctor
check doctor-only-looks '[ -z "$(envdir)" ] && [ -z "$(uvbin 2>/dev/null)" ] && said "Private Python"' \
    "doctor should report and install nothing"

# A pty for the prompt; there is no `script` to do it with on Windows.
if [ "$WINDOWS" = 0 ]; then
    if [ "$(uname -s)" = Darwin ]; then
        pty() { script -q /dev/null "$@"; }
    else
        pty() { script -qec "$*" /dev/null; }
    fi
    printf 'n\n' | PTY_BIN="$BIN" PTY_ROOT="$ROOT" PTY_PATH="$WORK/empty" PTY_OUT="$WORK/tree" PTY_MODEL="$MODEL" \
        bash -c "$(declare -f pty); pty /usr/bin/env PATH=\"\$PTY_PATH\" CASCADIA_PYTHON_HOME=\"\$PTY_ROOT\" \"\$PTY_BIN\" shard --model \"\$PTY_MODEL\" -o \"\$PTY_OUT\" --num-stages 2" >"$WORK/out.log" 2>&1
    check terminal-prompt 'said "[Y/n]" && said "Not installing" && [ -z "$(envdir)" ]' \
        "a terminal should get the question, and n should install nothing"
fi

# --- Nothing on the machine: the whole thing ------------------------------
shard --yes
check setup-from-nothing 'tree_ok && said "Downloading uv" && said "Using Python:" && said "private" && [ -f "$(envdir)/cascadia-complete" ]' \
    "exit $RC; should download uv, install, and export with the private environment"
ENV_DIR="$(envdir)"
PY="$(pybin "$ENV_DIR")"
UV="$(uvbin)"

shard
check second-run-installs-nothing 'tree_ok && ! said "Downloading uv" && ! said "Exporting needs" && ! said "installed," && said "private"' \
    "exit $RC; the second run should just export"

# The default format: the tree must really hold compressed weights.
rm -rf "$WORK/tree"
cas shard --model "$MODEL" -o "$WORK/tree" --num-stages 2 --quantization int4
RC_INT4=$RC
cat >"$WORK/constant_types_probe.py" <<'PY'
import sys
if __name__ == "__main__":
    import openvino as ov
    m = ov.Core().read_model(sys.argv[1])
    print(sorted({n.get_output_element_type(0).get_type_name() for n in m.get_ordered_ops() if n.get_type_name() == "Constant"}))
PY
TYPES="$(env -u PYTHONPATH "$PY" "$WORK/constant_types_probe.py" "$WORK/tree/stage_0/openvino_model.xml" 2>/dev/null | tail -1)"
check int4-means-int4 '[ "$RC_INT4" = 0 ] && echo "$TYPES" | grep -q "i4"' "exit $RC_INT4, constants $TYPES"

rm -rf "$WORK/exp"
cas export --model "$MODEL" -o "$WORK/exp" --weight-format fp16
check export-through-private '[ "$RC" = 0 ] && [ -f "$WORK/exp/cascadia-export.json" ]' "exit $RC"

shard --skip-check
check skip-check 'tree_ok && said "private" && ! said "Checking Python"' "exit $RC"

# --- The caller's environment ----------------------------------------------
mkdir -p "$WORK/shadow"
echo 'raise SystemExit("HIJACKED")' >"$WORK/shadow/torch.py"
rm -rf "$WORK/tree"
env -u VIRTUAL_ENV PATH="$RUN_PATH" PYTHONPATH="$WORK/shadow" PYTHONHOME="$WORK/shadow" CASCADIA_PYTHON_HOME="$ROOT" \
    "$BIN" shard --model "$MODEL" -o "$WORK/tree" --num-stages 2 --quantization fp16 </dev/null >"$WORK/out.log" 2>&1
RC=$?
check caller-python-vars-ignored 'tree_ok && ! said HIJACKED' "exit $RC with PYTHONPATH and PYTHONHOME set"

# An activated venv is what the user asked for: not the private one.
VENV="$WORK/user-venv"
env -u VIRTUAL_ENV UV_CACHE_DIR="$ROOT/uv-cache" UV_PYTHON_INSTALL_DIR="$ROOT/cpython" PATH="$RUN_PATH" \
    "$UV" venv "$VENV" --no-config --no-project --python "$PY" --quiet
"$UV" pip freeze --no-config --python "$PY" 2>/dev/null >"$WORK/freeze.txt"
env UV_CACHE_DIR="$ROOT/uv-cache" PATH="$RUN_PATH" "$UV" pip install --no-config --quiet \
    --python "$(pybin "$VENV")" --no-deps --torch-backend cpu --only-binary :all: -r "$WORK/freeze.txt"
rm -rf "$WORK/tree"
env PATH="$RUN_PATH" VIRTUAL_ENV="$VENV" CASCADIA_PYTHON_HOME="$ROOT" \
    "$BIN" shard --model "$MODEL" -o "$WORK/tree" --num-stages 2 --quantization fp16 </dev/null >"$WORK/out.log" 2>&1
RC=$?
check activated-venv-wins 'tree_ok && said "activated venv"' "exit $RC"

# A venv without the packages isn't what they meant: keep working.
env -u VIRTUAL_ENV UV_CACHE_DIR="$ROOT/uv-cache" PATH="$RUN_PATH" \
    "$UV" venv "$WORK/empty-venv" --no-config --no-project --python "$PY" --quiet
rm -rf "$WORK/tree"
env PATH="$RUN_PATH" VIRTUAL_ENV="$WORK/empty-venv" CASCADIA_PYTHON_HOME="$ROOT" \
    "$BIN" shard --model "$MODEL" -o "$WORK/tree" --num-stages 2 --quantization fp16 </dev/null >"$WORK/out.log" 2>&1
RC=$?
check broken-venv-falls-through 'tree_ok && said "private"' "exit $RC"

shard --python "$(pybin "$VENV")"
check python-flag-wins 'tree_ok && said "(--python)"' "exit $RC"

# --- Interrupted and damaged -------------------------------------------------
rm -f "$ENV_DIR/cascadia-complete"
rm -rf "$ENV_DIR"
if [ "$WINDOWS" = 0 ]; then
    # In its own process group, so uv goes with it, as with Ctrl-C.
    ( set -m
      env PATH="$RUN_PATH" CASCADIA_PYTHON_HOME="$ROOT" "$BIN" python setup </dev/null >/dev/null 2>&1 &
      pid=$!
      for _ in $(seq 1 600); do
          [ -d "$ENV_DIR/lib" ] && [ "$(find "$ENV_DIR/lib" -name '*.py' 2>/dev/null | head -50 | wc -l)" -ge 50 ] && break
          sleep 0.1
      done
      kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid"
      wait "$pid" 2>/dev/null )
    MARKER_AFTER_KILL=$([ -f "$ENV_DIR/cascadia-complete" ] && echo yes || echo no)
else
    MARKER_AFTER_KILL=no
fi
cas python setup
RC_SETUP=$RC
cp "$WORK/out.log" "$WORK/after-kill.log"
"$PY" -c "import torch, openvino, nncf, optimum.intel" >/dev/null 2>&1
RC_IMPORT=$?
check killed-setup-recovers '[ "$MARKER_AFTER_KILL" = no ] && [ "$RC_SETUP" = 0 ] && [ "$RC_IMPORT" = 0 ] && [ -f "$ENV_DIR/cascadia-complete" ]' \
    "marker after kill: $MARKER_AFTER_KILL, setup exit $RC_SETUP, import exit $RC_IMPORT"

FUNCTIONAL="$("$PY" -c 'import importlib.util as u; print(u.find_spec("torch").submodule_search_locations[0])')/nn/functional.py"
rm -f "$FUNCTIONAL"
shard
RC_REFUSED=$RC
shard --yes
check damaged-env-rebuilds '[ "$RC_REFUSED" != 0 ] && tree_ok && [ -f "$FUNCTIONAL" ]' \
    "without --yes exit $RC_REFUSED; should refuse, then rebuild with --yes"

# --- Only what's missing -------------------------------------------------------
# A "system" Python holding torch at the locked version, numpy too, and
# transformers at a version the lock doesn't want. uv is on PATH as well.
SYS="$WORK/system-python"
cp -R "$(find "$ROOT/cpython" -maxdepth 1 -type d -name 'cpython-*' | head -1)" "$SYS"
find "$SYS" -name EXTERNALLY-MANAGED -delete
if [ "$WINDOWS" = 1 ]; then SYS_PY="$SYS/python.exe"; SYS_BIN="$SYS"; else SYS_PY="$SYS/bin/python3"; SYS_BIN="$SYS/bin"; fi
version() { "$PY" -c "import importlib.metadata as m; print(m.version('$1'))"; }
env UV_CACHE_DIR="$ROOT/uv-cache" PATH="$RUN_PATH" "$UV" pip install --no-config --quiet --python "$SYS_PY" \
    --break-system-packages --torch-backend cpu --only-binary :all: --no-deps \
    "torch==$(version torch)" "numpy==$(version numpy)" transformers==5.4.0
SYS_FILES_BEFORE="$(find "$SYS" -type f -not -name '*.pyc' | wc -l)"
mkdir -p "$WORK/uv-on-path"
cp "$UV" "$WORK/uv-on-path/"
rm -rf "$ENV_DIR" "$ROOT/uv"
RUN_PATH="$SYS_BIN:$WORK/uv-on-path"
cas python setup
TORCH_FROM="$("$PY" -c 'import torch; print(torch.__file__)')"
TRANSFORMERS_NOW="$(version transformers)"
check reuses-system-uv-python-packages '[ "$RC" = 0 ] && ! said "Downloading uv" && said "reused from" && [ ! -d "$ROOT/uv" ] && ! echo "$TORCH_FROM" | grep -qF "$ENV_DIR" && [ "$TRANSFORMERS_NOW" != 5.4.0 ]' \
    "exit $RC, torch from $TORCH_FROM, transformers $TRANSFORMERS_NOW"
shard
check system-python-unchanged 'tree_ok && [ "$(find "$SYS" -type f -not -name "*.pyc" | wc -l)" = "$SYS_FILES_BEFORE" ] && [ "$("$SYS_PY" -c "import importlib.metadata as m; print(m.version(\"transformers\"))")" = 5.4.0 ]' \
    "exit $RC; the system Python should be as it was"

# The user upgrades a package the environment was reusing. Imports still pass.
LOCKED_NUMPY="$(version numpy)"
env UV_CACHE_DIR="$ROOT/uv-cache" PATH="$RUN_PATH" "$WORK/uv-on-path/$(basename "$UV")" pip install --no-config --quiet --python "$SYS_PY" \
    --break-system-packages --only-binary :all: --no-deps "numpy<$LOCKED_NUMPY"
shard
check changed-package-restored 'tree_ok && said "restoring" && [ "$(version numpy)" = "$LOCKED_NUMPY" ]' \
    "exit $RC, numpy now $(version numpy), locked $LOCKED_NUMPY"

# Real checkpoints are hundreds of GB. A config alone proves the binary carries
# each exporter and its imports: it runs, and stops on the missing weights.
MOE_OK=yes
for t in glm_moe_dsa:export_glm5 deepseek_v4:export_deepseek_v4 minimax_m2:export_minimax_m2 inkling_mm_model:export_inkling; do
    mkdir -p "$WORK/moe/${t%%:*}"
    echo "{\"model_type\": \"${t%%:*}\"}" >"$WORK/moe/${t%%:*}/config.json"
    rm -rf "$WORK/tree"
    cas shard --model "$WORK/moe/${t%%:*}" -o "$WORK/tree" --num-stages 2
    if [ "$RC" = 0 ] || ! said "running ${t##*:}" || said "No module named" || said "does not support"; then
        MOE_OK="no (${t%%:*}, exit $RC)"
    fi
done
check moe-exporters-reachable '[ "$MOE_OK" = yes ]' "$MOE_OK"

# --- Cleaning up -----------------------------------------------------------------
mkdir -p "$ROOT/aaaaaaaaaaaa"
rm -f "$ENV_DIR/cascadia-complete"
cas python setup
check old-env-removed '[ "$RC" = 0 ] && [ ! -d "$ROOT/aaaaaaaaaaaa" ] && [ -d "$ENV_DIR" ]' "exit $RC"

cas python remove
check remove-deletes-everything '[ "$RC" = 0 ] && [ ! -e "$ROOT" ] && [ -x "$SYS_PY" ]' "exit $RC"

echo "reproduce: WORK=$WORK scripts/test-python-env.sh $BIN"
if [ "$FAILED" != 0 ]; then
    echo "$FAILED CHECK(S) FAILED; logs under $WORK"
    exit 1
fi
rm -rf "$WORK"
echo "ALL PYTHON ENV CHECKS PASSED"
