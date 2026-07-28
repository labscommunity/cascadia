// C++ implementation of the C ABI declared in shim.h. Links against
// openvino-genai (libopenvino_genai) and openvino (libopenvino). All C++
// exceptions are caught and translated to int32_t error codes; the most
// recent message is stashed in a thread-local static buffer.

#include "shim.h"

#include "gemv_offload.hpp"

#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/variable_state.hpp>
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#include <openvino/genai/continuous_batching_pipeline.hpp>

namespace {

thread_local std::string g_last_error;

void set_last_error(const char* msg) {
    g_last_error = msg ? msg : "(null)";
}

void set_last_error(const std::exception& e) { g_last_error = e.what(); }

// Hard cap on plugin property pairs. OpenVINO plugin configs never
// exceed a few dozen keys in practice; this caps allocation at ~4 KB
// even if the caller passes a poisoned count.
static constexpr size_t MAX_PROPERTY_PAIRS = 256;

ov::AnyMap collect_properties(const char* const* kv, size_t count) {
    ov::AnyMap props;
    if (!kv) return props;
    if (count > MAX_PROPERTY_PAIRS) {
        set_last_error("properties_count exceeds MAX_PROPERTY_PAIRS=256");
        return props;
    }
    for (size_t i = 0; i < count; ++i) {
        const char* key = kv[2 * i];
        const char* val = kv[2 * i + 1];
        if (!key || !val) {
            continue;
        }
        std::string k(key);
        // A few properties are consumed by ov::genai's own pipeline code
        // (NPU static-LLM path), not the plugin config parser. genai reads
        // them via Any::as<int64_t>() and throws on a string Any
        // ("Failed to extract MAX_PROMPT_LEN. Type mismatch: expected int").
        // The Rust side sends every value as a string, so coerce these known
        // integer keys to int64_t here. Plugin-parsed properties (CACHE_DIR,
        // PERFORMANCE_HINT, INFERENCE_PRECISION_HINT, NUM_STREAMS, …) accept
        // strings fine and stay strings.
        if (k == "MAX_PROMPT_LEN" || k == "MIN_RESPONSE_LEN" ||
            k == "NPUW_LLM_PREFILL_CHUNK_SIZE") {
            try {
                // Require the WHOLE value to be an integer: std::stoll parses a
                // leading numeric prefix and silently drops the rest ("512abc"
                // -> 512), which would hand OV a wrong-but-valid int. Check pos
                // reached the end; otherwise fall through to the string branch
                // and let OV report the bad value.
                const std::string s(val);
                size_t pos = 0;
                long long parsed = std::stoll(s, &pos);
                if (pos != s.size()) {
                    throw std::invalid_argument("trailing chars after integer");
                }
                props[k] = static_cast<int64_t>(parsed);
            } catch (...) {
                // Not an integer — let OV report it against the string value.
                props[k] = std::string(val);
            }
        } else {
            props[k] = std::string(val);
        }
    }
    return props;
}

// Hard caps for input-tensor sanity. set_input is called with
// caller-supplied (rank, shape, data_size); validate they're sane to
// avoid wild reads on `shape`/`data` and gigantic allocations.
static constexpr size_t MAX_TENSOR_RANK = 16;
static constexpr size_t MAX_TENSOR_BYTES = 256 * 1024 * 1024; // 256 MiB

// Cap returned text size from generate(). At 4 bytes/UTF-8 char and
// 4096 max_new_tokens, even a UTF-8-heavy alphabet tops out around
// 16 KiB. 64 MiB is a generous overshoot but blocks OOM-by-model.
static constexpr size_t MAX_GENERATED_TEXT_BYTES = 64 * 1024 * 1024;

// Convert dtype code to OV element type. Throws on unknown codes so
// the caller's try/catch turns it into a typed error rather than
// silently using the wrong dtype (which would produce garbage output).
ov::element::Type dtype_from_code(uint32_t code) {
    switch (code) {
        case CASCADIA_DTYPE_F32:  return ov::element::f32;
        case CASCADIA_DTYPE_F16:  return ov::element::f16;
        case CASCADIA_DTYPE_I8:   return ov::element::i8;
        case CASCADIA_DTYPE_I32:  return ov::element::i32;
        case CASCADIA_DTYPE_I64:  return ov::element::i64;
        case CASCADIA_DTYPE_BF16: return ov::element::bf16;
        default:
            throw std::invalid_argument(
                "unknown dtype code in dtype_from_code");
    }
}

uint32_t code_from_dtype(const ov::element::Type& t) {
    if (t == ov::element::f16)  return CASCADIA_DTYPE_F16;
    if (t == ov::element::i8)   return CASCADIA_DTYPE_I8;
    if (t == ov::element::i32)  return CASCADIA_DTYPE_I32;
    if (t == ov::element::i64)  return CASCADIA_DTYPE_I64;
    if (t == ov::element::bf16) return CASCADIA_DTYPE_BF16;
    return CASCADIA_DTYPE_F32;
}

int32_t copy_name_to_buf(const std::string& name, char* out_buf,
                         size_t out_cap, size_t* out_len) {
    if (out_len) *out_len = name.size();
    if (out_cap == 0 || out_buf == nullptr) return 0;
    size_t copy_len = std::min(out_cap - 1, name.size());
    std::memcpy(out_buf, name.data(), copy_len);
    out_buf[copy_len] = 0;
    return 0;
}

// Join all aliases of a port with '\n' so Rust can split on the
// delimiter. OV node names can't legally contain newlines (per IR
// spec), so this is unambiguous. Templated since OV returns
// std::unordered_set<std::string> at 2026.1 but the API may move
// to std::set in future releases.
template <typename Set>
std::string join_port_names(const Set& names) {
    std::string out;
    for (const auto& n : names) {
        if (!out.empty()) out.push_back('\n');
        out += n;
    }
    return out;
}

}  // namespace

struct cascadia_pipeline_t {
    // Exactly one of `pipe` / `vlm` is set; generate / get_tokenizer
    // dispatch on which. VLM handles are text-only (no image inputs
    // through this ABI).
    std::unique_ptr<ov::genai::LLMPipeline> pipe;
    std::unique_ptr<ov::genai::VLMPipeline> vlm;
};
struct cascadia_genconfig_t {
    ov::genai::GenerationConfig cfg;
};
struct cascadia_tokenizer_t {
    ov::genai::Tokenizer tok;
};

struct cascadia_runtime_t {
    ov::Core core;
    std::shared_ptr<ov::CompiledModel> compiled;
    std::shared_ptr<ov::InferRequest> request;
    std::vector<std::string> input_names;     // first/any name per port
    std::vector<std::string> output_names;
    std::vector<std::string> input_aliases;   // ALL aliases joined by '\n'
    std::vector<std::string> output_aliases;
};

extern "C" {

const char* cascadia_last_error_message() {
    return g_last_error.c_str();
}

// Test-only introspection over collect_properties(): run it on `properties_kv`
// / `properties_count` and report how `key` landed in the resulting AnyMap.
// Returns 1 and writes *out_i64 if stored as int64_t, 0 if stored as a string,
// -1 if absent. Lets the Rust test pin the string->int64 coercion of the NPU
// LLM keys (which ov::genai reads via Any::as<int64_t>()) without a real model.
int32_t cascadia_debug_property_int64_kind(
    const char* const* properties_kv, size_t properties_count,
    const char* key, int64_t* out_i64) {
    if (!key) return -1;
    auto props = collect_properties(properties_kv, properties_count);
    auto it = props.find(std::string(key));
    if (it == props.end()) return -1;
    if (it->second.is<int64_t>()) {
        if (out_i64) *out_i64 = it->second.as<int64_t>();
        return 1;
    }
    return 0;
}

// ===================== LLMPipeline (genai) =====================

int32_t cascadia_pipeline_create(
    const char* model_path, const char* device,
    const char* const* properties_kv, size_t properties_count,
    cascadia_pipeline_t** out_handle) {
    if (!model_path || !device || !out_handle) {
        set_last_error("null arg in pipeline_create"); return 1;
    }
    try {
        auto props = collect_properties(properties_kv, properties_count);
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(
            std::filesystem::path(model_path), std::string(device), props);
        *out_handle = new cascadia_pipeline_t{std::move(pipe)};
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_create"); return 1;
    }
}

int32_t cascadia_pipeline_create_with_draft(
    const char* model_path, const char* device,
    const char* draft_model_path, const char* draft_device,
    const char* const* properties_kv, size_t properties_count,
    cascadia_pipeline_t** out_handle) {
    if (!model_path || !device || !draft_model_path || !out_handle) {
        set_last_error("null arg in pipeline_create_with_draft"); return 1;
    }
    try {
        auto props = collect_properties(properties_kv, properties_count);
        const std::string draft_dev =
            (draft_device && *draft_device) ? std::string(draft_device)
                                            : std::string(device);
        auto draft_kv = ov::genai::draft_model(
            std::filesystem::path(draft_model_path), draft_dev);
        props.emplace(draft_kv.first, draft_kv.second);
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(
            std::filesystem::path(model_path), std::string(device), props);
        *out_handle = new cascadia_pipeline_t{std::move(pipe)};
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_create_with_draft");
        return 1;
    }
}

int32_t cascadia_pipeline_create_with_prompt_lookup(
    const char* model_path, const char* device,
    const char* const* properties_kv, size_t properties_count,
    cascadia_pipeline_t** out_handle) {
    if (!model_path || !device || !out_handle) {
        set_last_error("null arg in pipeline_create_with_prompt_lookup"); return 1;
    }
    try {
        auto props = collect_properties(properties_kv, properties_count);
        props[ov::genai::prompt_lookup.name()] = true;
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(
            std::filesystem::path(model_path), std::string(device), props);
        *out_handle = new cascadia_pipeline_t{std::move(pipe)};
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_create_with_prompt_lookup");
        return 1;
    }
}

int32_t cascadia_pipeline_create_vlm(
    const char* model_path, const char* device, int32_t enable_prompt_lookup,
    const char* const* properties_kv, size_t properties_count,
    cascadia_pipeline_t** out_handle) {
    if (!model_path || !device || !out_handle) {
        set_last_error("null arg in pipeline_create_vlm"); return 1;
    }
    try {
        auto props = collect_properties(properties_kv, properties_count);
        if (enable_prompt_lookup) {
            props[ov::genai::prompt_lookup.name()] = true;
        }
        auto vlm = std::make_unique<ov::genai::VLMPipeline>(
            std::filesystem::path(model_path), std::string(device), props);
        auto* h = new cascadia_pipeline_t{};
        h->vlm = std::move(vlm);
        *out_handle = h;
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_create_vlm");
        return 1;
    }
}

void cascadia_pipeline_destroy(cascadia_pipeline_t* handle) { delete handle; }

cascadia_genconfig_t* cascadia_genconfig_new() {
    try { return new cascadia_genconfig_t{ov::genai::GenerationConfig{}}; }
    catch (...) { return nullptr; }
}
void cascadia_genconfig_destroy(cascadia_genconfig_t* cfg) { delete cfg; }
void cascadia_genconfig_set_max_new_tokens(cascadia_genconfig_t* cfg, uint32_t v) {
    if (cfg) cfg->cfg.max_new_tokens = v;
}
void cascadia_genconfig_set_temperature(cascadia_genconfig_t* cfg, float v) {
    if (cfg) cfg->cfg.temperature = v;
}
void cascadia_genconfig_set_do_sample(cascadia_genconfig_t* cfg, int32_t enabled) {
    if (cfg) cfg->cfg.do_sample = enabled != 0;
}
void cascadia_genconfig_set_num_assistant_tokens(cascadia_genconfig_t* cfg, uint32_t v) {
    if (cfg) cfg->cfg.num_assistant_tokens = v;
}
void cascadia_genconfig_set_max_ngram_size(cascadia_genconfig_t* cfg, uint32_t v) {
    if (cfg) cfg->cfg.max_ngram_size = v;
}
void cascadia_genconfig_set_apply_chat_template(cascadia_genconfig_t* cfg, int32_t enabled) {
    if (cfg) cfg->cfg.apply_chat_template = enabled != 0;
}
// OpenAI-compatible sampling knobs (#14). All map to standard public fields
// of ov::genai::GenerationConfig; trivial assignments that cannot throw.
void cascadia_genconfig_set_top_p(cascadia_genconfig_t* cfg, float v) {
    if (cfg) cfg->cfg.top_p = v;
}
void cascadia_genconfig_set_top_k(cascadia_genconfig_t* cfg, uint32_t v) {
    if (cfg) cfg->cfg.top_k = static_cast<size_t>(v);
}
void cascadia_genconfig_set_frequency_penalty(cascadia_genconfig_t* cfg, float v) {
    if (cfg) cfg->cfg.frequency_penalty = v;
}
void cascadia_genconfig_set_presence_penalty(cascadia_genconfig_t* cfg, float v) {
    if (cfg) cfg->cfg.presence_penalty = v;
}
void cascadia_genconfig_set_rng_seed(cascadia_genconfig_t* cfg, uint64_t v) {
    if (cfg) cfg->cfg.rng_seed = static_cast<size_t>(v);
}

int32_t cascadia_pipeline_generate(
    cascadia_pipeline_t* handle, const char* prompt, const cascadia_genconfig_t* cfg,
    char** out_text, uint32_t* out_token_count) {
    if (!handle || (!handle->pipe && !handle->vlm) || !prompt || !out_text) {
        set_last_error("null arg in pipeline_generate"); return 1;
    }
    try {
        std::string text;
        uint32_t generated = 0;
        if (handle->vlm) {
            // Text-only generate on a VLM-layout export; no image tensors.
            ov::genai::GenerationConfig gen_cfg =
                cfg ? cfg->cfg : ov::genai::GenerationConfig{};
            // StreamerVariant{} default-constructs its FIRST alternative —
            // an empty std::function — which fast-fails (0xc0000409) when
            // invoked. Pass monostate explicitly: "no streamer".
            ov::genai::VLMDecodedResults results = handle->vlm->generate(
                std::string(prompt), std::vector<ov::Tensor>{}, gen_cfg,
                ov::genai::StreamerVariant{std::monostate{}});
            text = results.texts.empty() ? std::string{} : results.texts[0];
            try {
                generated = static_cast<uint32_t>(
                    results.perf_metrics.get_num_generated_tokens());
            } catch (...) { generated = 0; }
        } else {
            ov::genai::DecodedResults results = cfg
                ? handle->pipe->generate(std::string(prompt), cfg->cfg)
                : handle->pipe->generate(std::string(prompt));
            text = results;
            try {
                generated = static_cast<uint32_t>(
                    results.perf_metrics.get_num_generated_tokens());
            } catch (...) { generated = 0; }
        }
        if (text.size() > MAX_GENERATED_TEXT_BYTES) {
            set_last_error(
                "generate produced text larger than MAX_GENERATED_TEXT_BYTES "
                "(64 MiB) — refusing to allocate");
            return 1;
        }
        char* buf = static_cast<char*>(std::malloc(text.size() + 1));
        if (!buf) { set_last_error("malloc failure for output text"); return 1; }
        std::memcpy(buf, text.data(), text.size());
        buf[text.size()] = 0;
        *out_text = buf;
        if (out_token_count) {
            *out_token_count = generated;
        }
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_generate"); return 1;
    }
}

void cascadia_free_string(char* s) { std::free(s); }

// ---- Continuous batching (issue #20) ---------------------------------------

struct cascadia_cb_pipeline_t {
    std::unique_ptr<ov::genai::ContinuousBatchingPipeline> pipe;
    std::unique_ptr<ov::genai::Tokenizer> tok;
};

struct cascadia_cb_handle_t {
    ov::genai::GenerationHandle handle;
    // Accumulated token ids + how many bytes of their decode we have already
    // handed to the caller. Re-decoding the full sequence each read (instead
    // of decoding the delta ids alone) keeps multi-byte codepoints whole when
    // a codepoint's tokens land in different reads.
    std::vector<int64_t> acc_ids;
    size_t emitted_bytes = 0;
};

namespace {

// Hold back a trailing incomplete UTF-8 sequence (and the U+FFFD the
// detokenizer substitutes for a half-decoded codepoint) while the request
// is still running — the completing bytes arrive with the next tokens.
size_t utf8_hold_back(const std::string& s) {
    size_t hold = 0;
    static const char kReplacement[] = "\xEF\xBF\xBD";
    if (s.size() >= 3 && s.compare(s.size() - 3, 3, kReplacement) == 0) {
        hold += 3;
    }
    size_t end = s.size() - hold;
    size_t cont = 0;
    while (cont < 3 && cont < end &&
           (static_cast<unsigned char>(s[end - 1 - cont]) & 0xC0) == 0x80) {
        ++cont;
    }
    if (cont < end) {
        unsigned char lead = static_cast<unsigned char>(s[end - 1 - cont]);
        size_t need = (lead & 0xF8) == 0xF0   ? 4
                      : (lead & 0xF0) == 0xE0 ? 3
                      : (lead & 0xE0) == 0xC0 ? 2
                                              : 1;
        if (need > cont + 1) {
            hold += cont + 1;
        }
    }
    return hold;
}

}  // namespace

int32_t cascadia_cb_pipeline_create(
    const char* model_path, const char* device, uint64_t cache_size_gb,
    uint64_t max_num_seqs, uint64_t max_num_batched_tokens,
    int32_t dynamic_split_fuse, int32_t enable_prefix_caching,
    const char* const* properties_kv, size_t properties_count,
    cascadia_cb_pipeline_t** out_handle) {
    if (!model_path || !device || !out_handle) {
        set_last_error("null arg in cb_pipeline_create");
        return 1;
    }
    try {
        ov::genai::SchedulerConfig sched;
        if (cache_size_gb > 0) sched.cache_size = static_cast<size_t>(cache_size_gb);
        if (max_num_seqs > 0) sched.max_num_seqs = static_cast<size_t>(max_num_seqs);
        if (max_num_batched_tokens > 0)
            sched.max_num_batched_tokens = static_cast<size_t>(max_num_batched_tokens);
        if (dynamic_split_fuse >= 0) sched.dynamic_split_fuse = dynamic_split_fuse != 0;
        if (enable_prefix_caching >= 0)
            sched.enable_prefix_caching = enable_prefix_caching != 0;

        ov::AnyMap props = collect_properties(properties_kv, properties_count);
        auto p = std::make_unique<cascadia_cb_pipeline_t>();
        p->pipe = std::make_unique<ov::genai::ContinuousBatchingPipeline>(
            std::filesystem::path(model_path), sched, std::string(device), props);
        p->tok = std::make_unique<ov::genai::Tokenizer>(p->pipe->get_tokenizer());
        *out_handle = p.release();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in cb_pipeline_create");
        return 1;
    }
}

void cascadia_cb_pipeline_destroy(cascadia_cb_pipeline_t* handle) { delete handle; }

int32_t cascadia_cb_add_request(
    cascadia_cb_pipeline_t* handle, uint64_t request_id, const char* prompt,
    const cascadia_genconfig_t* cfg, cascadia_cb_handle_t** out_handle) {
    if (!handle || !handle->pipe || !prompt || !cfg || !out_handle) {
        set_last_error("null arg in cb_add_request");
        return 1;
    }
    try {
        auto h = std::make_unique<cascadia_cb_handle_t>();
        h->handle =
            handle->pipe->add_request(request_id, std::string(prompt), cfg->cfg);
        *out_handle = h.release();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in cb_add_request");
        return 1;
    }
}

int32_t cascadia_cb_step(cascadia_cb_pipeline_t* handle) {
    if (!handle || !handle->pipe) {
        set_last_error("null arg in cb_step");
        return 1;
    }
    try {
        if (handle->pipe->has_non_finished_requests()) {
            handle->pipe->step();
        }
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in cb_step");
        return 1;
    }
}

int32_t cascadia_cb_has_unfinished(cascadia_cb_pipeline_t* handle, int32_t* out_has) {
    if (!handle || !handle->pipe || !out_has) {
        set_last_error("null arg in cb_has_unfinished");
        return 1;
    }
    try {
        *out_has = handle->pipe->has_non_finished_requests() ? 1 : 0;
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in cb_has_unfinished");
        return 1;
    }
}

int32_t cascadia_cb_handle_read(
    cascadia_cb_pipeline_t* pipeline, cascadia_cb_handle_t* handle,
    char** out_text_delta, uint32_t* out_new_tokens, int32_t* out_status) {
    if (!pipeline || !pipeline->tok || !handle || !handle->handle ||
        !out_text_delta || !out_new_tokens || !out_status) {
        set_last_error("null arg in cb_handle_read");
        return 1;
    }
    try {
        const size_t before = handle->acc_ids.size();
        if (handle->handle->can_read()) {
            // read() surfaces tokens generated since the previous read().
            // Single sequence per request (n / best_of are not exposed), so
            // take the first (only) sequence output.
            ov::genai::GenerationOutputs outs = handle->handle->read();
            if (!outs.empty()) {
                const auto& ids = outs.begin()->second.generated_ids;
                handle->acc_ids.insert(handle->acc_ids.end(), ids.begin(), ids.end());
            }
        }

        const auto status = handle->handle->get_status();
        const bool running = status == ov::genai::GenerationStatus::RUNNING;
        *out_status = running ? 0
                      : status == ov::genai::GenerationStatus::FINISHED ? 1
                                                                        : 2;

        std::string full = pipeline->tok->decode(handle->acc_ids);
        std::string delta;
        if (full.size() > handle->emitted_bytes) {
            delta = full.substr(handle->emitted_bytes);
            if (running) {
                const size_t hold = utf8_hold_back(delta);
                delta.resize(delta.size() - hold);
            }
            handle->emitted_bytes += delta.size();
        }
        if (delta.size() > MAX_GENERATED_TEXT_BYTES) {
            set_last_error("cb read delta exceeds MAX_GENERATED_TEXT_BYTES");
            return 1;
        }
        char* buf = static_cast<char*>(std::malloc(delta.size() + 1));
        if (!buf) {
            set_last_error("malloc failure for cb delta");
            return 1;
        }
        std::memcpy(buf, delta.data(), delta.size());
        buf[delta.size()] = 0;
        *out_text_delta = buf;
        *out_new_tokens = static_cast<uint32_t>(handle->acc_ids.size() - before);
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in cb_handle_read");
        return 1;
    }
}

int32_t cascadia_cb_handle_cancel(cascadia_cb_handle_t* handle) {
    if (!handle || !handle->handle) {
        set_last_error("null arg in cb_handle_cancel");
        return 1;
    }
    try {
        handle->handle->drop();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in cb_handle_cancel");
        return 1;
    }
}

void cascadia_cb_handle_destroy(cascadia_cb_handle_t* handle) { delete handle; }

int32_t cascadia_cb_count_tokens(
    cascadia_cb_pipeline_t* handle, const char* text, uint32_t* out_count) {
    if (!handle || !handle->tok || !text || !out_count) {
        set_last_error("null arg in cb_count_tokens");
        return 1;
    }
    try {
        auto enc = handle->tok->encode(std::string(text));
        const auto& shape = enc.input_ids.get_shape();
        *out_count = shape.empty() ? 0 : static_cast<uint32_t>(shape.back());
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in cb_count_tokens");
        return 1;
    }
}

// Tokenizer ownership: the returned handle is heap-allocated and the
// caller MUST free it via cascadia_tokenizer_destroy() before destroying
// the parent pipeline. The Tokenizer object inside is a refcount on
// model resources; outliving the pipeline triggers UB.
cascadia_tokenizer_t* cascadia_pipeline_get_tokenizer(cascadia_pipeline_t* handle) {
    if (!handle || (!handle->pipe && !handle->vlm)) return nullptr;
    try {
        return new cascadia_tokenizer_t{handle->vlm
            ? handle->vlm->get_tokenizer()
            : handle->pipe->get_tokenizer()};
    } catch (const std::exception& e) {
        set_last_error(e); return nullptr;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_get_tokenizer");
        return nullptr;
    }
}

void cascadia_tokenizer_destroy(cascadia_tokenizer_t* tok) { delete tok; }

int32_t cascadia_tokenizer_count_tokens(
    cascadia_tokenizer_t* tok, const char* text, uint32_t* out_count) {
    if (!tok || !text || !out_count) {
        set_last_error("null arg in tokenizer_count_tokens"); return 1;
    }
    try {
        auto enc = tok->tok.encode(std::string(text));
        const auto& shape = enc.input_ids.get_shape();
        *out_count = shape.empty() ? 0 : static_cast<uint32_t>(shape.back());
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in tokenizer_count_tokens");
        return 1;
    }
}

// ===================== ov-runtime (Core/CompiledModel/InferRequest) =====================

int32_t cascadia_runtime_compile(
    const char* model_xml_path, const char* device,
    const char* const* properties_kv, size_t properties_count,
    cascadia_runtime_t** out_handle) {
    if (!model_xml_path || !device || !out_handle) {
        set_last_error("null arg in runtime_compile"); return 1;
    }
    try {
        auto handle = std::make_unique<cascadia_runtime_t>();
        auto props = collect_properties(properties_kv, properties_count);
        auto compiled = handle->core.compile_model(
            std::string(model_xml_path), std::string(device), props);
        handle->compiled = std::make_shared<ov::CompiledModel>(std::move(compiled));
        handle->request = std::make_shared<ov::InferRequest>(
            handle->compiled->create_infer_request());

        for (const auto& port : handle->compiled->inputs()) {
            std::string name;
            try {
                name = port.get_any_name();
            } catch (...) {
                const auto& names = port.get_names();
                name = names.empty() ? std::string{} : *names.begin();
            }
            handle->input_aliases.push_back(join_port_names(port.get_names()));
            handle->input_names.push_back(std::move(name));
        }
        for (const auto& port : handle->compiled->outputs()) {
            std::string name;
            try {
                name = port.get_any_name();
            } catch (...) {
                const auto& names = port.get_names();
                name = names.empty() ? std::string{} : *names.begin();
            }
            handle->output_aliases.push_back(join_port_names(port.get_names()));
            handle->output_names.push_back(std::move(name));
        }

        *out_handle = handle.release();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in runtime_compile"); return 1;
    }
}

void cascadia_runtime_destroy(cascadia_runtime_t* handle) { delete handle; }

int32_t cascadia_runtime_import_blob(
    const char* blob_path, const char* device,
    const char* const* properties_kv, size_t properties_count,
    cascadia_runtime_t** out_handle) {
    if (!blob_path || !device || !out_handle) {
        set_last_error("null arg in runtime_import_blob"); return 1;
    }
    try {
        auto handle = std::make_unique<cascadia_runtime_t>();
        auto props = collect_properties(properties_kv, properties_count);

        std::ifstream f(blob_path, std::ios::binary | std::ios::ate);
        if (!f) {
            set_last_error((std::string("cannot open blob: ") + blob_path).c_str());
            return 1;
        }
        const auto pos = f.tellg();
        if (pos < 0 || pos == std::ifstream::pos_type(0)) {
            set_last_error(
                (std::string("cannot size blob (empty or unreadable): ") + blob_path).c_str());
            return 1;
        }
        const auto size = static_cast<size_t>(pos);
        f.seekg(0);
        ov::Tensor blob(ov::element::u8, ov::Shape{size});
        if (!f.read(reinterpret_cast<char*>(blob.data()),
                    static_cast<std::streamsize>(size))) {
            set_last_error((std::string("short read on blob: ") + blob_path).c_str());
            return 1;
        }
        f.close();

        auto compiled =
            handle->core.import_model(blob, std::string(device), props);
        handle->compiled = std::make_shared<ov::CompiledModel>(std::move(compiled));
        handle->request = std::make_shared<ov::InferRequest>(
            handle->compiled->create_infer_request());

        for (const auto& port : handle->compiled->inputs()) {
            std::string name;
            try {
                name = port.get_any_name();
            } catch (...) {
                const auto& names = port.get_names();
                name = names.empty() ? std::string{} : *names.begin();
            }
            handle->input_aliases.push_back(join_port_names(port.get_names()));
            handle->input_names.push_back(std::move(name));
        }
        for (const auto& port : handle->compiled->outputs()) {
            std::string name;
            try {
                name = port.get_any_name();
            } catch (...) {
                const auto& names = port.get_names();
                name = names.empty() ? std::string{} : *names.begin();
            }
            handle->output_aliases.push_back(join_port_names(port.get_names()));
            handle->output_names.push_back(std::move(name));
        }

        *out_handle = handle.release();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in runtime_import_blob"); return 1;
    }
}

int32_t cascadia_runtime_compile_gemv_offload(
    const char* model_xml_path, const char* device,
    const char* const* properties_kv, size_t properties_count,
    uint32_t* out_offloaded,
    cascadia_runtime_t** out_handle) {
    if (!model_xml_path || !device || !out_handle) {
        set_last_error("null arg in runtime_compile_gemv_offload"); return 1;
    }
    try {
        auto handle = std::make_unique<cascadia_runtime_t>();
        auto props = collect_properties(properties_kv, properties_count);
        // read_model keeps the .bin mmapped; the offload pass moves the
        // sym-INT4 weight constants into CascadiaInt4Gemv op members so the
        // plugin compile below never repacks them into a resident copy.
        auto model = handle->core.read_model(std::string(model_xml_path));
        const uint32_t offloaded = cascadia_gemv::offload_int4_gemv(handle->core, model);
        if (out_offloaded) *out_offloaded = offloaded;
        auto compiled = handle->core.compile_model(model, std::string(device), props);
        handle->compiled = std::make_shared<ov::CompiledModel>(std::move(compiled));
        handle->request = std::make_shared<ov::InferRequest>(
            handle->compiled->create_infer_request());
        for (const auto& port : handle->compiled->inputs()) {
            std::string name;
            try {
                name = port.get_any_name();
            } catch (...) {
                const auto& names = port.get_names();
                name = names.empty() ? std::string{} : *names.begin();
            }
            handle->input_aliases.push_back(join_port_names(port.get_names()));
            handle->input_names.push_back(std::move(name));
        }
        for (const auto& port : handle->compiled->outputs()) {
            std::string name;
            try {
                name = port.get_any_name();
            } catch (...) {
                const auto& names = port.get_names();
                name = names.empty() ? std::string{} : *names.begin();
            }
            handle->output_aliases.push_back(join_port_names(port.get_names()));
            handle->output_names.push_back(std::move(name));
        }
        *out_handle = handle.release();
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in runtime_compile_gemv_offload"); return 1;
    }
}

int32_t cascadia_runtime_reset_state(cascadia_runtime_t* handle) {
    if (!handle || !handle->request) {
        set_last_error("null runtime handle"); return 1;
    }
    try {
        // OV C++ API: query each variable_state and reset it. Equivalent
        // to InferRequest::reset_state() (which is sugar over this loop).
        for (auto& state : handle->request->query_state()) {
            state.reset();
        }
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in runtime_reset_state"); return 1;
    }
}

int32_t cascadia_runtime_profiling(
    cascadia_runtime_t* handle, char* out_buf, size_t buf_cap, size_t* out_len) {
    if (!handle || !handle->request || !out_buf || !out_len) {
        set_last_error("null arg in runtime_profiling"); return 1;
    }
    try {
        std::string acc;
        for (const auto& pi : handle->request->get_profiling_info()) {
            acc += pi.node_name;
            acc += '\t';
            acc += pi.node_type;
            acc += '\t';
            acc += pi.exec_type;
            acc += '\t';
            acc += std::to_string(pi.real_time.count());
            acc += '\t';
            acc += std::to_string(pi.cpu_time.count());
            acc += '\n';
        }
        const size_t len = acc.size() < buf_cap ? acc.size() : buf_cap;
        std::memcpy(out_buf, acc.data(), len);
        *out_len = len;
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in runtime_profiling"); return 1;
    }
}

size_t cascadia_runtime_input_count(cascadia_runtime_t* handle) {
    return handle ? handle->input_names.size() : 0;
}
size_t cascadia_runtime_output_count(cascadia_runtime_t* handle) {
    return handle ? handle->output_names.size() : 0;
}

int32_t cascadia_runtime_input_name(
    cascadia_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len) {
    if (!handle || idx >= handle->input_names.size()) {
        set_last_error("input idx out of range"); return 1;
    }
    return copy_name_to_buf(handle->input_names[idx], out_buf, out_cap, out_len);
}

int32_t cascadia_runtime_output_name(
    cascadia_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len) {
    if (!handle || idx >= handle->output_names.size()) {
        set_last_error("output idx out of range"); return 1;
    }
    return copy_name_to_buf(handle->output_names[idx], out_buf, out_cap, out_len);
}

int32_t cascadia_runtime_input_name_all(
    cascadia_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len) {
    if (!handle || idx >= handle->input_aliases.size()) {
        set_last_error("input idx out of range"); return 1;
    }
    return copy_name_to_buf(handle->input_aliases[idx], out_buf, out_cap, out_len);
}

int32_t cascadia_runtime_output_name_all(
    cascadia_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len) {
    if (!handle || idx >= handle->output_aliases.size()) {
        set_last_error("output idx out of range"); return 1;
    }
    return copy_name_to_buf(handle->output_aliases[idx], out_buf, out_cap, out_len);
}

int32_t cascadia_runtime_set_input(
    cascadia_runtime_t* handle, const char* tensor_name,
    uint32_t dtype, const size_t* shape, size_t rank,
    const void* data, size_t data_size) {
    if (!handle || !handle->request || !tensor_name) {
        set_last_error("null arg in set_input"); return 1;
    }
    if (rank > MAX_TENSOR_RANK) {
        set_last_error("set_input: rank exceeds MAX_TENSOR_RANK=16"); return 1;
    }
    if (rank > 0 && !shape) {
        set_last_error("set_input: null shape with rank>0"); return 1;
    }
    if (data_size > 0 && !data) {
        set_last_error("set_input: null data with data_size>0"); return 1;
    }
    if (data_size > MAX_TENSOR_BYTES) {
        set_last_error("set_input: data_size exceeds MAX_TENSOR_BYTES=256MiB");
        return 1;
    }
    // Bound the shape product so we don't construct a tensor that
    // would attempt a multi-terabyte allocation.
    size_t total_elems = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) { total_elems = 0; break; }
        if (total_elems > SIZE_MAX / shape[i]) {
            set_last_error("set_input: shape product overflows size_t");
            return 1;
        }
        total_elems *= shape[i];
    }
    try {
        ov::Shape ov_shape(shape, shape + rank);
        ov::element::Type elem = dtype_from_code(dtype);
        // Allocate a Tensor and copy the bytes in. Using void-cast allocate
        // is the safe path (no aliasing of caller buffer beyond the call).
        ov::Tensor tensor(elem, ov_shape);
        if (data_size != tensor.get_byte_size()) {
            set_last_error("data_size does not match tensor.get_byte_size()");
            return 1;
        }
        std::memcpy(tensor.data(), data, data_size);
        handle->request->set_tensor(std::string(tensor_name), tensor);
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in set_input"); return 1;
    }
}

int32_t cascadia_runtime_infer(cascadia_runtime_t* handle) {
    if (!handle || !handle->request) { set_last_error("null runtime"); return 1; }
    try { handle->request->infer(); return 0; }
    catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) {
        set_last_error("unknown C++ exception in infer"); return 1;
    }
}

int32_t cascadia_runtime_output_rank(
    cascadia_runtime_t* handle, size_t output_idx, size_t* out_rank) {
    if (!handle || !handle->request || !out_rank) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_output_tensor(output_idx);
        *out_rank = t.get_shape().size();
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in output_rank"); return 1; }
}

int32_t cascadia_runtime_output_shape(
    cascadia_runtime_t* handle, size_t output_idx,
    size_t* out_shape, size_t shape_cap) {
    if (!handle || !handle->request || (shape_cap > 0 && !out_shape)) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_output_tensor(output_idx);
        auto shp = t.get_shape();
        if (shape_cap < shp.size()) {
            set_last_error("shape_cap too small"); return 1;
        }
        for (size_t i = 0; i < shp.size(); ++i) out_shape[i] = shp[i];
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in output_shape"); return 1; }
}

int32_t cascadia_runtime_output_dtype(
    cascadia_runtime_t* handle, size_t output_idx, uint32_t* out_dtype) {
    if (!handle || !handle->request || !out_dtype) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_output_tensor(output_idx);
        *out_dtype = code_from_dtype(t.get_element_type());
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in output_dtype"); return 1; }
}

// Input introspection — mirrors the output_* getters but over
// get_input_tensor. The pre-allocated input tensor carries the compiled
// model's shape, so these report concrete dims for a static (NPU-export)
// model; on a dynamic model a dim may be reported as 0.
int32_t cascadia_runtime_input_rank(
    cascadia_runtime_t* handle, size_t input_idx, size_t* out_rank) {
    if (!handle || !handle->request || !out_rank) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_input_tensor(input_idx);
        *out_rank = t.get_shape().size();
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in input_rank"); return 1; }
}

int32_t cascadia_runtime_input_shape(
    cascadia_runtime_t* handle, size_t input_idx,
    size_t* out_shape, size_t shape_cap) {
    if (!handle || !handle->request || (shape_cap > 0 && !out_shape)) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_input_tensor(input_idx);
        auto shp = t.get_shape();
        if (shape_cap < shp.size()) {
            set_last_error("shape_cap too small"); return 1;
        }
        for (size_t i = 0; i < shp.size(); ++i) out_shape[i] = shp[i];
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in input_shape"); return 1; }
}

int32_t cascadia_runtime_input_dtype(
    cascadia_runtime_t* handle, size_t input_idx, uint32_t* out_dtype) {
    if (!handle || !handle->request || !out_dtype) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_input_tensor(input_idx);
        *out_dtype = code_from_dtype(t.get_element_type());
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in input_dtype"); return 1; }
}

int32_t cascadia_runtime_output_byte_size(
    cascadia_runtime_t* handle, size_t output_idx, size_t* out_byte_size) {
    if (!handle || !handle->request || !out_byte_size) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_output_tensor(output_idx);
        *out_byte_size = t.get_byte_size();
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in output_byte_size"); return 1; }
}

int32_t cascadia_runtime_output_copy(
    cascadia_runtime_t* handle, size_t output_idx,
    void* out_buf, size_t out_buf_size) {
    if (!handle || !handle->request || !out_buf) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_output_tensor(output_idx);
        if (out_buf_size != t.get_byte_size()) {
            set_last_error("out_buf_size != tensor.get_byte_size()"); return 1;
        }
        std::memcpy(out_buf, t.data(), out_buf_size);
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in output_copy"); return 1; }
}

// ===================== Core enumeration =====================

int32_t cascadia_core_list_devices(
    char* out_buf, size_t out_cap, size_t* out_len) {
    try {
        ov::Core core;
        std::string joined;
        for (const auto& d : core.get_available_devices()) {
            if (!joined.empty()) joined.push_back('\n');
            joined += d;
        }
        return copy_name_to_buf(joined, out_buf, out_cap, out_len);
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in list_devices"); return 1; }
}

int32_t cascadia_core_get_property(
    const char* device, const char* property,
    char* out_buf, size_t out_cap, size_t* out_len) {
    if (!device || !property) { set_last_error("null arg in get_property"); return 1; }
    try {
        ov::Core core;
        // OV returns ov::Any; serialise via the stream operator into a
        // std::string. Works for all built-in property value types
        // (string, int, enum, vector<*>, ov::PropertyName).
        ov::Any v = core.get_property(std::string(device), std::string(property));
        std::ostringstream oss;
        v.print(oss);
        return copy_name_to_buf(oss.str(), out_buf, out_cap, out_len);
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown exception in get_property"); return 1; }
}

}  // extern "C"
