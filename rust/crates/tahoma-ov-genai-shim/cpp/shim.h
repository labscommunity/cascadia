// C ABI shim around openvino-genai's C++ LLMPipeline + GenerationConfig +
// Tokenizer. Exists because the upstream C API does not yet expose
// draft_model(), Tokenizer access, or the prompt_lookup property.
//
// Thread safety: pipeline handles are NOT thread-safe; serialise calls from
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
typedef struct tahoma_result_t tahoma_result_t;

/// Get the last error message thrown anywhere in the shim. Static buffer;
/// not thread-safe — read it on the same thread that triggered the error.
const char* tahoma_last_error_message();

// ---- Pipeline construction ------------------------------------------------

/// Create a plain LLMPipeline (no draft, no prompt_lookup).
/// `properties_kv` is a flat array of [key, value, key, value, ...] strings;
/// `properties_count` is the number of pairs (so the array has 2*N entries).
int32_t tahoma_pipeline_create(
    const char* model_path,
    const char* device,
    const char* const* properties_kv,
    size_t properties_count,
    tahoma_pipeline_t** out_handle);

/// Create an LLMPipeline with FastDraft companion. The draft model is loaded
/// onto `draft_device` and registered as the assistant model. Pass NULL or ""
/// for `draft_device` to use the same device as the target.
int32_t tahoma_pipeline_create_with_draft(
    const char* model_path,
    const char* device,
    const char* draft_model_path,
    const char* draft_device,
    const char* const* properties_kv,
    size_t properties_count,
    tahoma_pipeline_t** out_handle);

/// Create an LLMPipeline with prompt-lookup decoding enabled.
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

/// Run a single greedy/sampled generation. The result text is heap-allocated
/// and must be freed via tahoma_free_string. `out_token_count` is the number
/// of tokens emitted (per the pipeline's perf metrics, when available).
int32_t tahoma_pipeline_generate(
    tahoma_pipeline_t* handle,
    const char* prompt,
    const tahoma_genconfig_t* cfg,
    char** out_text,
    uint32_t* out_token_count);

void tahoma_free_string(char* s);

// ---- Tokenizer (workaround for missing C API) -----------------------------

/// Borrow the pipeline's tokenizer. The returned handle is invalidated when
/// the pipeline is destroyed; the caller does NOT free it.
tahoma_tokenizer_t* tahoma_pipeline_get_tokenizer(tahoma_pipeline_t* handle);

/// Encode `text` and return the number of resulting tokens. Used to fix the
/// perf-metrics token-count bug for short greedy decodes.
int32_t tahoma_tokenizer_count_tokens(
    tahoma_tokenizer_t* tok,
    const char* text,
    uint32_t* out_count);

#ifdef __cplusplus
}
#endif

#endif  // TAHOMA_OV_GENAI_SHIM_H
