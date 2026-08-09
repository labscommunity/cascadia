// C ABI shim around openvino-genai's C++ LLMPipeline + GenerationConfig +
// Tokenizer + the lower-level Core/CompiledModel/InferRequest used by
// ov-runtime + ov-dist-spec engines. Exists because the upstream C API
// does not yet expose draft_model(), Tokenizer access, prompt_lookup
// property, or InferRequest::reset_state().
//
// Thread safety: all handles are NOT thread-safe; serialise calls from
// the Rust side. Tokenizer handles are owned by their pipeline and live
// only as long as the pipeline does.
//
// Error reporting: every entry point returns 0 on success, non-zero on
// failure. The most recent error message is retrievable via cascadia_last_error_message.
//
// Wire ownership:
//   - Strings passed in are copied; caller may free immediately.
//   - Strings returned via out-parameters are heap-allocated by the shim
//     and must be freed via cascadia_free_string.

#ifndef CASCADIA_OV_GENAI_SHIM_H
#define CASCADIA_OV_GENAI_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cascadia_pipeline_t cascadia_pipeline_t;
typedef struct cascadia_genconfig_t cascadia_genconfig_t;
typedef struct cascadia_tokenizer_t cascadia_tokenizer_t;
typedef struct cascadia_runtime_t cascadia_runtime_t;

/// ABI marker for this compiled shim's C SOURCE (cpp/shim.cpp + friends).
/// Bump on every symbol addition/change that a Rust binding depends on. Rust
/// asserts this against its own expected constant at startup (glm5 ov_attn's
/// `from_opts`) so a shim compiled from source that predates a required
/// symbol fails LOUDLY (event=ov_attn_unavailable) instead of silently
/// skipping a safety check.
///
/// Scope, stated honestly: `build.rs` compiles this file fresh alongside
/// every `--features openvino` Rust build (no separately-shipped shim
/// binary today), so in the CURRENT build model this check cannot actually
/// disagree — an old Rust binary simply lacks the new code entirely. It is
/// defense-in-depth against a future build-model change (e.g. a prebuilt-
/// object cache, or this shim shipped as its own dylib). It says NOTHING
/// about, and does not protect against, a stale OpenVINO RUNTIME install
/// (`libopenvino`/`libopenvino_genai` behind `INTEL_OPENVINO_DIR`) — that is
/// this fleet's actual stale-artifact hazard and a completely different
/// mechanism (dynamic linking against the SDK's own shared libraries, not
/// this header's version marker). Do not read this symbol as covering that.
/// Bumped to 2 for cascadia_runtime_get_property +
/// cascadia_runtime_compile_bf16_canary (glm5-attn-igpu-t3 Task 5).
#define CASCADIA_SHIM_ABI_VERSION 2
int32_t cascadia_shim_abi_version();

/// Get the last error message thrown anywhere in the shim. Static buffer;
/// not thread-safe — read it on the same thread that triggered the error.
const char* cascadia_last_error_message();

// Class of the last error on this thread: 0 = none, 1 = generic,
// 2 = resource exhaustion (EAGAIN/ENOMEM inside a plugin). Read it right
// after a failed call, before any other shim call on the same thread.
int32_t cascadia_last_error_code();

/// Test-only: run collect_properties() over `properties_kv` and report how
/// `key` was stored — 1 = int64_t (written to *out_i64), 0 = string, -1 =
/// absent. Exists so the Rust test can verify the NPU LLM keys are coerced to
/// int64 (ov::genai reads them via Any::as<int64_t>()) without a real model.
int32_t cascadia_debug_property_int64_kind(
    const char* const* properties_kv,
    size_t properties_count,
    const char* key,
    int64_t* out_i64);

// ---- Pipeline (LLMPipeline) construction ---------------------------------

int32_t cascadia_pipeline_create(
    const char* model_path,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    cascadia_pipeline_t** out_handle);

int32_t cascadia_pipeline_create_with_draft(
    const char* model_path,
    const char* device,
    const char* draft_model_path,
    const char* draft_device,
    const char* const* properties_kv,
    size_t properties_count,
    cascadia_pipeline_t** out_handle);

int32_t cascadia_pipeline_create_with_prompt_lookup(
    const char* model_path,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    cascadia_pipeline_t** out_handle);

/// VLMPipeline-backed pipeline for VLM-layout exports (e.g. Qwen3.5/3.6:
/// `openvino_language_model.xml` + separate embeddings/vision IRs), used
/// text-only. `enable_prompt_lookup` non-zero turns on prompt-lookup
/// decoding (OV GenAI >= 2026.2 extends it to VLM pipelines). The handle
/// is interchangeable with LLMPipeline handles for generate / tokenizer /
/// destroy.
int32_t cascadia_pipeline_create_vlm(
    const char* model_path,
    const char* device,
    int32_t enable_prompt_lookup,
    const char* const* properties_kv,
    size_t properties_count,
    cascadia_pipeline_t** out_handle);

void cascadia_pipeline_destroy(cascadia_pipeline_t* handle);

// ---- GenerationConfig -----------------------------------------------------

cascadia_genconfig_t* cascadia_genconfig_new();
void cascadia_genconfig_destroy(cascadia_genconfig_t* cfg);
void cascadia_genconfig_set_max_new_tokens(cascadia_genconfig_t* cfg, uint32_t v);
void cascadia_genconfig_set_temperature(cascadia_genconfig_t* cfg, float v);
void cascadia_genconfig_set_do_sample(cascadia_genconfig_t* cfg, int32_t enabled);
void cascadia_genconfig_set_num_assistant_tokens(cascadia_genconfig_t* cfg, uint32_t v);
void cascadia_genconfig_set_max_ngram_size(cascadia_genconfig_t* cfg, uint32_t v);
// When enabled==0 the pipeline skips its internal chat-template apply (caller
// pre-rendered the template, e.g. to honor enable_thinking). Default: apply.
void cascadia_genconfig_set_apply_chat_template(cascadia_genconfig_t* cfg, int32_t enabled);
// OpenAI-compatible sampling knobs (#14): map to ov::genai::GenerationConfig
// fields top_p / top_k / frequency_penalty / presence_penalty / rng_seed.
void cascadia_genconfig_set_top_p(cascadia_genconfig_t* cfg, float v);
void cascadia_genconfig_set_top_k(cascadia_genconfig_t* cfg, uint32_t v);
void cascadia_genconfig_set_frequency_penalty(cascadia_genconfig_t* cfg, float v);
void cascadia_genconfig_set_presence_penalty(cascadia_genconfig_t* cfg, float v);
void cascadia_genconfig_set_rng_seed(cascadia_genconfig_t* cfg, uint64_t v);

// ---- Generation -----------------------------------------------------------

int32_t cascadia_pipeline_generate(
    cascadia_pipeline_t* handle,
    const char* prompt,
    const cascadia_genconfig_t* cfg,
    char** out_text,
    uint32_t* out_token_count);

void cascadia_free_string(char* s);

// ---- Continuous batching (ContinuousBatchingPipeline, issue #20) ----------
//
// One pipeline serves many concurrent requests: add_request() enrolls a
// prompt into the running batch, step() advances the batch by one scheduler
// iteration, and per-request handles surface incremental text. NOT
// thread-safe (like everything else here) — serialise from Rust.

typedef struct cascadia_cb_pipeline_t cascadia_cb_pipeline_t;
typedef struct cascadia_cb_handle_t cascadia_cb_handle_t;

// Values for cascadia_cb_handle_read's out_status. IGNORED is distinct from
// FINISHED on purpose: it means OpenVINO could not continue the request (KV
// cache exhausted), which is a failure, not an answer.
#define CASCADIA_CB_STATUS_RUNNING 0
#define CASCADIA_CB_STATUS_FINISHED 1
#define CASCADIA_CB_STATUS_CANCELLED 2
#define CASCADIA_CB_STATUS_IGNORED 3

// Values for cascadia_cb_handle_read's out_finish_reason. NONE means OpenVINO
// has not reported one yet; the caller falls back to its own inference.
#define CASCADIA_CB_FINISH_NONE 0
#define CASCADIA_CB_FINISH_STOP 1
#define CASCADIA_CB_FINISH_LENGTH 2

/// Scheduler knobs: 0 keeps the ov-genai default for the size fields;
/// -1 keeps the default for the two tri-state booleans (0/1 = explicit).
int32_t cascadia_cb_pipeline_create(
    const char* model_path,
    const char* device,
    uint64_t cache_size_gb,
    uint64_t max_num_seqs,
    uint64_t max_num_batched_tokens,
    int32_t dynamic_split_fuse,
    int32_t enable_prefix_caching,
    const char* const* properties_kv,
    size_t properties_count,
    cascadia_cb_pipeline_t** out_handle);

void cascadia_cb_pipeline_destroy(cascadia_cb_pipeline_t* handle);

/// Enroll a prompt. `request_id` is caller-chosen; upstream requires it be
/// unique for EVERY add_request() call, not merely among live requests. The
/// returned handle must be destroyed AFTER the request finishes or is
/// cancelled, and before the pipeline is destroyed — the Rust wrapper enforces
/// the ordering by having each handle hold an Arc on its pipeline.
int32_t cascadia_cb_add_request(
    cascadia_cb_pipeline_t* handle,
    uint64_t request_id,
    const char* prompt,
    const cascadia_genconfig_t* cfg,
    cascadia_cb_handle_t** out_handle);

/// Advance the batch by one scheduler iteration. The scheduler picks which
/// enrolled requests run, subject to max_num_seqs / max_num_batched_tokens
/// and KV pressure, so a given request may not progress. A no-op (returns 0)
/// when the pipeline has no non-finished requests.
int32_t cascadia_cb_step(cascadia_cb_pipeline_t* handle);

/// Read one request's generated text so far.
///
/// `out_text` receives the FULL decode of every token generated by this
/// request (free via cascadia_free_string), with its byte length in
/// `out_text_len` — the buffer is NUL-terminated for convenience but the
/// length is authoritative, since a byte-level vocabulary can decode a token
/// to 0x00. The caller diffs against what it has already emitted and owns the
/// UTF-8 hold-back for a trailing incomplete codepoint.
///
/// `out_new_tokens` is how many tokens were appended to the accumulated
/// sequence by THIS call. It is not the token count of any text delta the
/// caller derives: a call can add tokens whose bytes are not yet emittable.
///
/// `out_status` is one of CASCADIA_CB_STATUS_*; `out_finish_reason` one of
/// CASCADIA_CB_FINISH_*, latched from GenerationOutput as soon as OpenVINO
/// reports it (the terminal read may carry no new output).
int32_t cascadia_cb_handle_read(
    cascadia_cb_pipeline_t* pipeline,
    cascadia_cb_handle_t* handle,
    char** out_text,
    size_t* out_text_len,
    uint32_t* out_new_tokens,
    int32_t* out_status,
    int32_t* out_finish_reason);

/// Abort a running request (client disconnect / cancel). Safe on finished
/// handles.
int32_t cascadia_cb_handle_cancel(cascadia_cb_handle_t* handle);

void cascadia_cb_handle_destroy(cascadia_cb_handle_t* handle);

/// Token count via the CB pipeline's tokenizer (usage reporting).
int32_t cascadia_cb_count_tokens(
    cascadia_cb_pipeline_t* handle, const char* text, uint32_t* out_count);

// ---- Tokenizer (workaround for missing C API) -----------------------------

/// Borrow the pipeline's tokenizer. The returned handle is heap-allocated
/// and the caller MUST free it via cascadia_tokenizer_destroy() before
/// destroying the parent pipeline. The Tokenizer object inside is a
/// refcount on the pipeline's model resources; outliving the pipeline
/// triggers undefined behaviour.
cascadia_tokenizer_t* cascadia_pipeline_get_tokenizer(cascadia_pipeline_t* handle);

void cascadia_tokenizer_destroy(cascadia_tokenizer_t* tok);

int32_t cascadia_tokenizer_count_tokens(
    cascadia_tokenizer_t* tok,
    const char* text,
    uint32_t* out_count);

// ---- Lower-level OV runtime (Core + CompiledModel + InferRequest) ---------
//
// Used by the ov-runtime + ov-dist-spec engines. Wraps a single per-stage
// IR; `reset_state()` is the key function the openvino-rs crate omits.

/// Compile a model from the given .xml file path, on the given device,
/// with optional plugin-config properties.
int32_t cascadia_runtime_compile(
    const char* model_xml_path,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    cascadia_runtime_t** out_handle);

void cascadia_runtime_destroy(cascadia_runtime_t* handle);

/// Compile like cascadia_runtime_compile, but first rewrite every NNCF
/// sym-INT4 decompress→MatMul into the CascadiaInt4Gemv extension op whose
/// weights stay backed by the read_model mmap of the original .bin — the
/// CPU plugin never makes its own resident repacked copy. CPU-class devices
/// only (the op executes via the evaluate() fallback). Do not pass CACHE_DIR
/// (the custom op's member tensors don't survive blob serialization).
/// `out_offloaded` (optional) receives the number of MatMuls rewritten.
int32_t cascadia_runtime_compile_gemv_offload(
    const char* model_xml_path,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    uint32_t* out_offloaded,
    cascadia_runtime_t** out_handle);

/// Import a precompiled blob (produced by ov::CompiledModel::export_model,
/// e.g. an AOT cross-compile on a big-RAM host with NPU_PLATFORM set) instead
/// of compiling from IR. Skips the compiler entirely — the NPU compile
/// transient (~5.5x INT4 bytes host RAM) never happens on this box; peak is
/// ~the blob size. Same handle contract as cascadia_runtime_compile (the
/// infer request is created eagerly, which forces device weight load).
int32_t cascadia_runtime_import_blob(
    const char* blob_path,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    cascadia_runtime_t** out_handle);

/// Serialize the last inference's per-node profiling info (requires the
/// model to have been compiled with the PERF_COUNT=YES plugin property) as
/// TSV lines "node_name\tnode_type\texec_type\treal_us\tcpu_us\n" into
/// `out_buf` (truncating at `buf_cap`); `out_len` receives the byte count.
int32_t cascadia_runtime_profiling(
    cascadia_runtime_t* handle, char* out_buf, size_t buf_cap, size_t* out_len);

/// Reset stateful KV-cache nodes to their initial value. No-op on
/// stateless models.
int32_t cascadia_runtime_reset_state(cascadia_runtime_t* handle);

size_t cascadia_runtime_input_count(cascadia_runtime_t* handle);
size_t cascadia_runtime_output_count(cascadia_runtime_t* handle);

/// Returns the FIRST tensor name (alias) of the input/output port at `idx`.
/// Caller-allocated buffer; pass `out_buf=NULL, out_cap=0` to size-query
/// (returns required capacity excluding the trailing NUL via *out_len).
int32_t cascadia_runtime_input_name(
    cascadia_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len);
int32_t cascadia_runtime_output_name(
    cascadia_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len);

/// All aliases of input/output port `idx`, joined by '\n'. Use this for
/// canonical-name matching against IRs that expose multiple aliases per
/// port (e.g. v5 stage_1 IRs where "hidden_states" is a secondary
/// alias of an internal node name).
int32_t cascadia_runtime_input_name_all(
    cascadia_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len);
int32_t cascadia_runtime_output_name_all(
    cascadia_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len);

/// Element type codes — keep in sync with cascadia-transport's DType.
/// 0 = f32, 1 = f16, 2 = i8, 3 = i32, 4 = i64, 5 = bf16.
#define CASCADIA_DTYPE_F32  0u
#define CASCADIA_DTYPE_F16  1u
#define CASCADIA_DTYPE_I8   2u
#define CASCADIA_DTYPE_I32  3u
#define CASCADIA_DTYPE_I64  4u
#define CASCADIA_DTYPE_BF16 5u

/// Bind an input tensor by name. `shape` is `rank` element counts (no
/// padding); `data` is row-major raw bytes whose total length must equal
/// product(shape) * dtype.bytes_per_element.
int32_t cascadia_runtime_set_input(
    cascadia_runtime_t* handle,
    const char* tensor_name,
    uint32_t dtype,
    const size_t* shape,
    size_t rank,
    const void* data,
    size_t data_size);

/// Run inference. Inputs must have been set via cascadia_runtime_set_input.
int32_t cascadia_runtime_infer(cascadia_runtime_t* handle);

/// Get the output tensor's shape rank. After this, call
/// cascadia_runtime_output_shape with a buffer of size `rank` to read the
/// actual dim values.
int32_t cascadia_runtime_output_rank(
    cascadia_runtime_t* handle, size_t output_idx, size_t* out_rank);

int32_t cascadia_runtime_output_shape(
    cascadia_runtime_t* handle, size_t output_idx,
    size_t* out_shape, size_t shape_cap);

int32_t cascadia_runtime_output_dtype(
    cascadia_runtime_t* handle, size_t output_idx, uint32_t* out_dtype);

/// Input-tensor introspection (rank / shape / dtype), mirroring the output
/// getters. Reports the compiled model's concrete input dims — used by
/// `cascadia profile-stages` to size the zeroed inputs it feeds when
/// timing a stage. Expects a static model; on a dynamic-shape port get_shape()
/// errors (returns non-zero), which the caller surfaces as an error.
int32_t cascadia_runtime_input_rank(
    cascadia_runtime_t* handle, size_t input_idx, size_t* out_rank);

int32_t cascadia_runtime_input_shape(
    cascadia_runtime_t* handle, size_t input_idx,
    size_t* out_shape, size_t shape_cap);

int32_t cascadia_runtime_input_dtype(
    cascadia_runtime_t* handle, size_t input_idx, uint32_t* out_dtype);

/// Copy the output tensor's raw bytes into `out_buf`. `out_buf_size` must
/// equal byte_size of the output tensor (call output_byte_size first).
int32_t cascadia_runtime_output_byte_size(
    cascadia_runtime_t* handle, size_t output_idx, size_t* out_byte_size);

int32_t cascadia_runtime_output_copy(
    cascadia_runtime_t* handle, size_t output_idx,
    void* out_buf, size_t out_buf_size);

/// Read a property back from a COMPILED model (as opposed to
/// cascadia_core_get_property, which queries a device before any compile) —
/// e.g. the EFFECTIVE INFERENCE_PRECISION_HINT / DYNAMIC_QUANTIZATION_GROUP_SIZE
/// after compilation, since a compile hint is not a guarantee. Same
/// size-query-then-fill convention as cascadia_core_get_property.
int32_t cascadia_runtime_get_property(
    cascadia_runtime_t* handle, const char* property,
    char* out_buf, size_t out_cap, size_t* out_len);

/// Compile a standalone `x[n] -> bf16-roundtrip -> y` canary graph: the exact
/// arithmetic bf16-rounding subgraph every attn_ov layer graph embeds after
/// each linear/RMSNorm/rope (see tools/glm5_attn_ov.py::_bf16_roundtrip and
/// cpp/bf16_canary.cpp). Built from arithmetic ops, not Convert, because a
/// plain Convert(f32->bf16)->Convert(bf16->f32) pair was observed silently
/// elided as an identity by the CPU plugin under INFERENCE_PRECISION_HINT=f32
/// — this is what the device canary (spec Sec9 gate 0) actually exercises.
/// The returned handle is driven via the ordinary cascadia_runtime_set_input /
/// cascadia_runtime_infer / cascadia_runtime_output_* calls (input "x",
/// output "y", both `[n]` f32).
int32_t cascadia_runtime_compile_bf16_canary(
    size_t n,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    cascadia_runtime_t** out_handle);

// ---- Core enumeration (no compile) ----------------------------------------
//
// Used by `cascadia profile-devices` to discover which OV plugins are
// available on the worker host without paying the cost of a model compile.
// Each call constructs a transient `ov::Core` (cheap; plugins are
// registered on first use, cached process-wide).

/// Get the list of OV plugins available on this host, joined by '\n'.
/// Use copy_name_to_buf semantics: pass `(NULL, 0, &len)` to size-query,
/// then allocate `len + 1` and call again to fill. Returns 0 on success.
int32_t cascadia_core_list_devices(
    char* out_buf, size_t out_cap, size_t* out_len);

/// Query a single OV property on a single device (e.g. "FULL_DEVICE_NAME",
/// "DEVICE_ARCHITECTURE"). The returned value is the property's
/// `to_string()` form. RO properties are safe to query; RW properties
/// return their current value. Pass `(NULL, 0, &len)` to size-query.
int32_t cascadia_core_get_property(
    const char* device, const char* property,
    char* out_buf, size_t out_cap, size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif  // CASCADIA_OV_GENAI_SHIM_H
