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
// failure. The most recent error message is retrievable via tahoma_last_error_message.
//
// Wire ownership:
//   - Strings passed in are copied; caller may free immediately.
//   - Strings returned via out-parameters are heap-allocated by the shim
//     and must be freed via tahoma_free_string.

#ifndef TAHOMA_OV_GENAI_SHIM_H
#define TAHOMA_OV_GENAI_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tahoma_pipeline_t tahoma_pipeline_t;
typedef struct tahoma_genconfig_t tahoma_genconfig_t;
typedef struct tahoma_tokenizer_t tahoma_tokenizer_t;
typedef struct tahoma_runtime_t tahoma_runtime_t;

/// Get the last error message thrown anywhere in the shim. Static buffer;
/// not thread-safe — read it on the same thread that triggered the error.
const char* tahoma_last_error_message();

// ---- Pipeline (LLMPipeline) construction ---------------------------------

int32_t tahoma_pipeline_create(
    const char* model_path,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    tahoma_pipeline_t** out_handle);

int32_t tahoma_pipeline_create_with_draft(
    const char* model_path,
    const char* device,
    const char* draft_model_path,
    const char* draft_device,
    const char* const* properties_kv,
    size_t properties_count,
    tahoma_pipeline_t** out_handle);

int32_t tahoma_pipeline_create_with_prompt_lookup(
    const char* model_path,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    tahoma_pipeline_t** out_handle);

void tahoma_pipeline_destroy(tahoma_pipeline_t* handle);

// ---- GenerationConfig -----------------------------------------------------

tahoma_genconfig_t* tahoma_genconfig_new();
void tahoma_genconfig_destroy(tahoma_genconfig_t* cfg);
void tahoma_genconfig_set_max_new_tokens(tahoma_genconfig_t* cfg, uint32_t v);
void tahoma_genconfig_set_temperature(tahoma_genconfig_t* cfg, float v);
void tahoma_genconfig_set_do_sample(tahoma_genconfig_t* cfg, int32_t enabled);
void tahoma_genconfig_set_num_assistant_tokens(tahoma_genconfig_t* cfg, uint32_t v);
void tahoma_genconfig_set_max_ngram_size(tahoma_genconfig_t* cfg, uint32_t v);

// ---- Generation -----------------------------------------------------------

int32_t tahoma_pipeline_generate(
    tahoma_pipeline_t* handle,
    const char* prompt,
    const tahoma_genconfig_t* cfg,
    char** out_text,
    uint32_t* out_token_count);

void tahoma_free_string(char* s);

// ---- Tokenizer (workaround for missing C API) -----------------------------

/// Borrow the pipeline's tokenizer. The returned handle is heap-allocated
/// and the caller MUST free it via tahoma_tokenizer_destroy() before
/// destroying the parent pipeline. The Tokenizer object inside is a
/// refcount on the pipeline's model resources; outliving the pipeline
/// triggers undefined behaviour.
tahoma_tokenizer_t* tahoma_pipeline_get_tokenizer(tahoma_pipeline_t* handle);

void tahoma_tokenizer_destroy(tahoma_tokenizer_t* tok);

int32_t tahoma_tokenizer_count_tokens(
    tahoma_tokenizer_t* tok,
    const char* text,
    uint32_t* out_count);

// ---- Lower-level OV runtime (Core + CompiledModel + InferRequest) ---------
//
// Used by the ov-runtime + ov-dist-spec engines. Wraps a single per-stage
// IR; `reset_state()` is the key function the openvino-rs crate omits.

/// Compile a model from the given .xml file path, on the given device,
/// with optional plugin-config properties.
int32_t tahoma_runtime_compile(
    const char* model_xml_path,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    tahoma_runtime_t** out_handle);

void tahoma_runtime_destroy(tahoma_runtime_t* handle);

/// Reset stateful KV-cache nodes to their initial value. No-op on
/// stateless models.
int32_t tahoma_runtime_reset_state(tahoma_runtime_t* handle);

size_t tahoma_runtime_input_count(tahoma_runtime_t* handle);
size_t tahoma_runtime_output_count(tahoma_runtime_t* handle);

/// Returns the FIRST tensor name (alias) of the input/output port at `idx`.
/// Caller-allocated buffer; pass `out_buf=NULL, out_cap=0` to size-query
/// (returns required capacity excluding the trailing NUL via *out_len).
int32_t tahoma_runtime_input_name(
    tahoma_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len);
int32_t tahoma_runtime_output_name(
    tahoma_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len);

/// All aliases of input/output port `idx`, joined by '\n'. Use this for
/// canonical-name matching against IRs that expose multiple aliases per
/// port (e.g. v5 stage_1 IRs where "hidden_states" is a secondary
/// alias of an internal node name).
int32_t tahoma_runtime_input_name_all(
    tahoma_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len);
int32_t tahoma_runtime_output_name_all(
    tahoma_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len);

/// Element type codes — keep in sync with tahoma-transport's DType.
/// 0 = f32, 1 = f16, 2 = i8, 3 = i32, 4 = i64, 5 = bf16.
#define TAHOMA_DTYPE_F32  0u
#define TAHOMA_DTYPE_F16  1u
#define TAHOMA_DTYPE_I8   2u
#define TAHOMA_DTYPE_I32  3u
#define TAHOMA_DTYPE_I64  4u
#define TAHOMA_DTYPE_BF16 5u

/// Bind an input tensor by name. `shape` is `rank` element counts (no
/// padding); `data` is row-major raw bytes whose total length must equal
/// product(shape) * dtype.bytes_per_element.
int32_t tahoma_runtime_set_input(
    tahoma_runtime_t* handle,
    const char* tensor_name,
    uint32_t dtype,
    const size_t* shape,
    size_t rank,
    const void* data,
    size_t data_size);

/// Run inference. Inputs must have been set via tahoma_runtime_set_input.
int32_t tahoma_runtime_infer(tahoma_runtime_t* handle);

/// Get the output tensor's shape rank. After this, call
/// tahoma_runtime_output_shape with a buffer of size `rank` to read the
/// actual dim values.
int32_t tahoma_runtime_output_rank(
    tahoma_runtime_t* handle, size_t output_idx, size_t* out_rank);

int32_t tahoma_runtime_output_shape(
    tahoma_runtime_t* handle, size_t output_idx,
    size_t* out_shape, size_t shape_cap);

int32_t tahoma_runtime_output_dtype(
    tahoma_runtime_t* handle, size_t output_idx, uint32_t* out_dtype);

/// Copy the output tensor's raw bytes into `out_buf`. `out_buf_size` must
/// equal byte_size of the output tensor (call output_byte_size first).
int32_t tahoma_runtime_output_byte_size(
    tahoma_runtime_t* handle, size_t output_idx, size_t* out_byte_size);

int32_t tahoma_runtime_output_copy(
    tahoma_runtime_t* handle, size_t output_idx,
    void* out_buf, size_t out_buf_size);

#ifdef __cplusplus
}
#endif

#endif  // TAHOMA_OV_GENAI_SHIM_H
