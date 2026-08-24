// C++ implementation of the C ABI declared in shim.h. Links against
// openvino-genai (libopenvino_genai) and openvino (libopenvino). All C++
// exceptions are caught and translated to int32_t error codes; the most
// recent message is stashed in a thread-local static buffer.

#include "shim.h"

#include "gemv_offload.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/variable_state.hpp>
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/genai/chat_history.hpp>

namespace {

thread_local std::string g_last_error;
// Machine-readable class of the last error, alongside the message: 0 = none,
// 1 = generic, 2 = resource exhaustion (EAGAIN / ENOMEM — e.g. the GPU plugin
// failing thread creation or allocation under memory pressure, which surfaces
// as "resource unavailable try again"). Classified HERE, once, so Rust callers
// don't sniff message strings. thread_local like the message, so reading it
// right after a failed call on the same thread is race-free.
thread_local int32_t g_last_error_code = 0;

void set_last_error(const char* msg) {
    g_last_error = msg ? msg : "(null)";
    g_last_error_code = 1;
}

void set_last_error(const std::exception& e) {
    g_last_error = e.what();
    g_last_error_code = 1;
    // Typed first: EAGAIN/ENOMEM system_errors (thread creation / allocation
    // failure inside a plugin). Message sniff second: OV usually flattens the
    // cause into an ov::Exception string by the time it reaches us.
    if (const auto* se = dynamic_cast<const std::system_error*>(&e)) {
        const auto c = se->code();
        if (c == std::errc::resource_unavailable_try_again ||
            c == std::errc::not_enough_memory) {
            g_last_error_code = 2;
            return;
        }
    }
    if (g_last_error.find("resource unavailable") != std::string::npos ||
        g_last_error.find("not enough memory") != std::string::npos) {
        g_last_error_code = 2;
    }
}

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

int32_t cascadia_last_error_code() {
    return g_last_error_code;
}

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
    // Accumulated token ids. The whole sequence is re-decoded on every read
    // (decoding only the delta ids would split multi-byte codepoints whose
    // tokens land in different reads) and the FULL decode is returned; the
    // caller owns the emitted-prefix bookkeeping. Tracking it here as a byte
    // count that was never checked against the decode meant a re-decode that
    // shrank desynced the stream permanently and silently.
    std::vector<int64_t> acc_ids;
    // Latched from GenerationOutput. The terminal read can return no new
    // output at all (the last tokens were consumed by an earlier read, and an
    // evicted request produces none), so the reason has to be remembered when
    // it appears rather than looked for at the end.
    ov::genai::GenerationFinishReason finish_reason =
        ov::genai::GenerationFinishReason::NONE;
};

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
        // ContinuousBatchingPipeline's string add_request() overload does NOT
        // honour GenerationConfig::apply_chat_template — verified on OV GenAI
        // 2026.2 by an A/B against LLMPipeline, where the same flag through the
        // same GenerationConfig changes the output and here it changed nothing.
        // So when the caller asked for templating we render it ourselves;
        // otherwise the untouched string overload keeps its exact behaviour.
        const bool want_template =
            cfg->cfg.apply_chat_template && !handle->tok->get_chat_template().empty();
        if (want_template) {
            ov::genai::ChatHistory history(std::vector<ov::AnyMap>{
                {{"role", std::string("user")}, {"content", std::string(prompt)}}});
            const std::string rendered =
                handle->tok->apply_chat_template(history, /*add_generation_prompt=*/true);
            // The rendered template already carries BOS. Encoding it again with
            // special tokens would prepend a second one — a silently degraded
            // prompt, invisible in the output unless compared side by side.
            auto enc = handle->tok->encode(
                rendered, ov::AnyMap{{"add_special_tokens", false}});
            h->handle = handle->pipe->add_request(request_id, enc.input_ids, cfg->cfg);
        } else {
            h->handle =
                handle->pipe->add_request(request_id, std::string(prompt), cfg->cfg);
        }
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

int32_t cascadia_cb_handle_read(
    cascadia_cb_pipeline_t* pipeline, cascadia_cb_handle_t* handle,
    char** out_text, size_t* out_text_len, uint32_t* out_new_tokens,
    int32_t* out_status, int32_t* out_finish_reason) {
    if (!pipeline || !pipeline->tok || !handle || !handle->handle || !out_text ||
        !out_text_len || !out_new_tokens || !out_status || !out_finish_reason) {
        set_last_error("null arg in cb_handle_read");
        return 1;
    }
    try {
        const size_t before = handle->acc_ids.size();
        // can_read() first: read() asserts on a stopped/cancelled handle, and
        // pulling from an empty stream would block.
        if (handle->handle->can_read()) {
            // read() surfaces tokens generated since the previous read().
            // Single sequence per request (n / best_of are not exposed), so
            // take the first (only) sequence output.
            ov::genai::GenerationOutputs outs = handle->handle->read();
            if (!outs.empty()) {
                const auto& out = outs.begin()->second;
                handle->acc_ids.insert(
                    handle->acc_ids.end(), out.generated_ids.begin(), out.generated_ids.end());
                if (out.finish_reason != ov::genai::GenerationFinishReason::NONE) {
                    handle->finish_reason = out.finish_reason;
                }
            }
        }

        const auto status = handle->handle->get_status();
        const bool running = status == ov::genai::GenerationStatus::RUNNING;
        // IGNORED is kept distinct from a normal finish: it means the request
        // ran out of KV cache and could not be continued, which the caller must
        // report as a failure rather than as a completed answer. STOP (early
        // stop, output kept) is a normal completion.
        switch (status) {
            case ov::genai::GenerationStatus::RUNNING:
                *out_status = CASCADIA_CB_STATUS_RUNNING;
                break;
            case ov::genai::GenerationStatus::FINISHED:
            case ov::genai::GenerationStatus::STOP:
                *out_status = CASCADIA_CB_STATUS_FINISHED;
                break;
            case ov::genai::GenerationStatus::CANCEL:
                *out_status = CASCADIA_CB_STATUS_CANCELLED;
                break;
            default:
                *out_status = CASCADIA_CB_STATUS_IGNORED;
                break;
        }
        switch (handle->finish_reason) {
            case ov::genai::GenerationFinishReason::STOP:
            // A tool call is a normal stop as far as the wire format goes.
            case ov::genai::GenerationFinishReason::TOOL_CALL:
                *out_finish_reason = CASCADIA_CB_FINISH_STOP;
                break;
            case ov::genai::GenerationFinishReason::LENGTH:
                *out_finish_reason = CASCADIA_CB_FINISH_LENGTH;
                break;
            default:
                *out_finish_reason = CASCADIA_CB_FINISH_NONE;
                break;
        }

        // Return the whole decode, not a delta. The caller diffs it against
        // what it has already emitted, which lets it notice a non-monotonic
        // re-decode instead of silently slicing at a stale offset.
        std::string full = pipeline->tok->decode(handle->acc_ids);
        if (full.size() > MAX_GENERATED_TEXT_BYTES) {
            set_last_error("cb decoded text exceeds MAX_GENERATED_TEXT_BYTES");
            return 1;
        }
        char* buf = static_cast<char*>(std::malloc(full.size() + 1));
        if (!buf) {
            set_last_error("malloc failure for cb text");
            return 1;
        }
        std::memcpy(buf, full.data(), full.size());
        buf[full.size()] = 0;
        // Length is explicit: a byte-level vocabulary can decode a token to
        // 0x00, and a NUL-terminated read would truncate there for the rest of
        // the request.
        *out_text = buf;
        *out_text_len = full.size();
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
        // cancel() vs stop(): both end the request and let the scheduler
        // reclaim its KV blocks. The difference is what is kept — STOP retains
        // the output produced so far, CANCEL discards it (and drops the turn
        // from chat history). Discarding is right for a client that went away.
        // Verified against OV GenAI 2026.2, where GenerationHandle::drop() was
        // replaced by cancel().
        handle->handle->cancel();
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

// Drop the InferRequest and build a fresh one off the retained CompiledModel. `reset_state` only
// calls VariableState::reset(), which is not enough after a `set_state_blob`: a restarted process
// (fresh request) provably clears the residue, this does the same without the restart or a recompile.
int32_t cascadia_runtime_recreate_request(cascadia_runtime_t* handle) {
    if (!handle || !handle->compiled) {
        set_last_error("null runtime handle"); return 1;
    }
    try {
        handle->request = std::make_shared<ov::InferRequest>(handle->compiled->create_infer_request());
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in runtime_recreate_request"); return 1;
    }
}

// ───── Issue-34: KV state export/import for warm-pull (cascadia_engine::KvCoordination on OV) ─────
//
// A stateful OV LLM keeps its KV cache as `VariableState`s on the InferRequest. We serialize them all
// into ONE self-describing, opaque blob (the consumer restores via set_state, never interprets the
// tensors — fingerprint + layout_version upstream guarantee same model). Blob layout (little-endian):
//   u32 count
//   per state: u32 name_len, name[name_len], u8 dtype_code, u8 rank, u64 dims[rank], u64 nbytes, data[nbytes]

static uint8_t ov_type_to_code(const ov::element::Type& t) {
    if (t == ov::element::f32)  return 0;
    if (t == ov::element::f16)  return 1;
    if (t == ov::element::bf16) return 2;
    if (t == ov::element::i64)  return 3;
    if (t == ov::element::i32)  return 4;
    if (t == ov::element::u8)   return 5;
    if (t == ov::element::i8)   return 6;
    if (t == ov::element::f64)  return 7;
    return 255;
}
static ov::element::Type code_to_ov_type(uint8_t c) {
    switch (c) {
        case 0: return ov::element::f32;
        case 1: return ov::element::f16;
        case 2: return ov::element::bf16;
        case 3: return ov::element::i64;
        case 4: return ov::element::i32;
        case 5: return ov::element::u8;
        case 6: return ov::element::i8;
        case 7: return ov::element::f64;
        default: return ov::element::dynamic;
    }
}

// Parse a KV VariableState name of the form "past_key_values.<layer>.<key|value>" into a canonical
// ordinal (layer*2 + [is_value]). get/set_state_blob use this to align a donor blob to the recipient's
// states by KV IDENTITY rather than by OV's query_state() ORDER or by the raw state NAME — neither of
// which is stable across two independently-compiled instances of the same IR (a cross-chain warm-pull:
// the donor engine on chain A and our engine on chain B are separate compilations). Returns false for
// any name that isn't exactly this shape, and the caller then keeps the plain positional path
// (behavior-identical for same-instance round-trips and for models that don't use this KV naming).
static bool kv_canonical_key(const std::string& name, uint64_t* out_key) {
    // Family A — llama/genai stateful KV: "past_key_values.<layer>.<key|value>". A KV VariableState fuses
    // the past_key_values input + present output tensors, and a separate compilation can surface
    // get_name() as the two aliases JOINED ("past_key_values.1.valuepresent.1.value"), so match the kv
    // token as a PREFIX of the suffix, not the whole remainder (exact-suffix rejected the donor form ⇒
    // applied 0 of N ⇒ cold). key = layer*2 + {key:0, value:1}.
    {
        static const char PFX[] = "past_key_values.";
        const size_t L = sizeof(PFX) - 1;
        if (name.size() > L && name.compare(0, L, PFX) == 0) {
            size_t dot = name.find('.', L);
            if (dot != std::string::npos && dot != L) {
                bool digits = true;
                uint64_t layer = 0;
                for (size_t i = L; i < dot; ++i) {
                    char c = name[i];
                    if (c < '0' || c > '9') { digits = false; break; }
                    layer = layer * 10 + static_cast<uint64_t>(c - '0');
                }
                if (digits) {
                    const std::string suffix = name.substr(dot + 1);
                    if (suffix.rfind("value", 0) == 0) { *out_key = layer * 2 + 1; return true; }
                    if (suffix.rfind("key", 0) == 0)   { *out_key = layer * 2 + 0; return true; }
                }
            }
        }
    }
    // Family B — qwen36 hybrid (full-attention + DeltaNet/SSM) stateful KV. The compiled query_state()
    // name is "cache_params.past.<kind>.<idx>" (kind ∈ {key,value,conv,ssm}; idx = the per-kind layer
    // index), typically FUSED with the "cache_params.present.<kind>.<idx>" output alias joined onto the
    // end (get_name() concatenates the ReadValue input + Assign output tensor names). Parse the past.*
    // identity and ignore any trailing alias. key = idx*4 + {key:0, value:1, conv:2, ssm:3}: four disjoint
    // mod-4 lanes so the kinds never collide and each kind's indices stay ordered. A model uses ONE family,
    // so this never overlaps Family A. Without this, qwen36 states parse non-canonical ⇒ positional restore
    // ⇒ cross-instance slot scramble ⇒ DIVERGE.
    {
        static const char PFX[] = "cache_params.past.";
        const size_t L = sizeof(PFX) - 1;
        // SEARCH (not anchor): get_name() fuses the past + present aliases and a separate compilation may
        // order them either way ("cache_params.past.…cache_params.present.…" or present-first), so locate
        // the past.* alias wherever it sits and parse its (kind, idx).
        size_t at = name.find(PFX);
        if (at != std::string::npos) {
            const size_t kstart = at + L;
            size_t dot = name.find('.', kstart);
            if (dot != std::string::npos && dot != kstart) {
                const std::string kind = name.substr(kstart, dot - kstart);
                uint64_t koff;
                if (kind == "key") koff = 0;
                else if (kind == "value") koff = 1;
                else if (kind == "conv") koff = 2;
                else if (kind == "ssm") koff = 3;
                else return false;
                size_t s = dot + 1, e = s;
                while (e < name.size() && name[e] >= '0' && name[e] <= '9') ++e;
                if (e == s) return false;                            // need a numeric index
                uint64_t idx = 0;
                for (size_t i = s; i < e; ++i) idx = idx * 10 + static_cast<uint64_t>(name[i] - '0');
                *out_key = idx * 4 + koff;                            // other alias ignored
                return true;
            }
        }
    }
    return false;
}

// Two-call: pass buf=nullptr/cap=0 to learn the size in *len_out, then call again with a buffer of
// that size. Returns 0 on success (incl. the size-query); 1 on error. `*len_out` is always set.
int32_t cascadia_runtime_get_state_blob(cascadia_runtime_t* handle, uint8_t* buf, size_t cap,
                                        size_t* len_out) {
    if (!handle || !handle->request || !len_out) {
        set_last_error("null runtime handle / len_out"); return 1;
    }
    try {
        auto states = handle->request->query_state();
        std::vector<std::pair<std::string, ov::Tensor>> snaps;
        snaps.reserve(states.size());
        size_t total = 4;
        for (auto& s : states) {
            std::string name = s.get_name();
            ov::Tensor t = s.get_state();
            total += 4 + name.size() + 1 + 1 + 8 * t.get_shape().size() + 8 + t.get_byte_size();
            snaps.emplace_back(std::move(name), t);
        }
        *len_out = total;
        if (!buf || cap < total) return 0; // size query
        // Emit states in canonical (layer, key/value) order parsed from the name, so a consumer on a
        // DIFFERENTLY-compiled instance realigns by identity even when its own query_state() order or
        // raw state names differ. Non-canonical names (other models) keep query_state() order.
        {
            std::vector<uint64_t> keys(snaps.size());
            bool canonical = true;
            for (size_t i = 0; i < snaps.size(); ++i)
                if (!kv_canonical_key(snaps[i].first, &keys[i])) { canonical = false; break; }
            if (canonical) {
                std::vector<size_t> ord(snaps.size());
                for (size_t i = 0; i < snaps.size(); ++i) ord[i] = i;
                std::stable_sort(ord.begin(), ord.end(),
                                 [&](size_t a, size_t b) { return keys[a] < keys[b]; });
                std::vector<std::pair<std::string, ov::Tensor>> sorted;
                sorted.reserve(snaps.size());
                for (size_t i : ord) sorted.push_back(std::move(snaps[i]));
                snaps.swap(sorted);
            }
        }
        uint8_t* p = buf;
        auto put32 = [&](uint32_t v) { std::memcpy(p, &v, 4); p += 4; };
        auto put64 = [&](uint64_t v) { std::memcpy(p, &v, 8); p += 8; };
        put32(static_cast<uint32_t>(snaps.size()));
        for (auto& kv : snaps) {
            const std::string& name = kv.first;
            ov::Tensor& t = kv.second;
            put32(static_cast<uint32_t>(name.size()));
            std::memcpy(p, name.data(), name.size()); p += name.size();
            *p++ = ov_type_to_code(t.get_element_type());
            auto shape = t.get_shape();
            *p++ = static_cast<uint8_t>(shape.size());
            for (size_t d : shape) put64(static_cast<uint64_t>(d));
            size_t nb = t.get_byte_size();
            put64(static_cast<uint64_t>(nb));
            std::memcpy(p, t.data(), nb); p += nb;
        }
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown C++ exception in get_state_blob"); return 1; }
}

int32_t cascadia_runtime_set_state_blob(cascadia_runtime_t* handle, const uint8_t* buf, size_t len) {
    if (!handle || !handle->request) { set_last_error("null runtime handle"); return 1; }
    if (!buf || len < 4) { set_last_error("set_state_blob: short buffer"); return 1; }
    try {
        // Canonical restore keyed by KV IDENTITY (layer, key/value) parsed from each VariableState
        // name. get_state_blob emits states in this same canonical order, so blob slot i maps to the
        // model state that shares its identity — independent of OV's per-INSTANCE query_state() ORDER
        // and of instance-specific state NAMES. A cross-chain warm-pull restores a slice captured on a
        // separately-compiled donor engine: neither its state order (plain positional ⇒ states land in
        // the WRONG slots ⇒ garbage) nor its raw names (name-keyed ⇒ "applied 0 of N" ⇒ always cold)
        // are guaranteed equal to ours. Sorting BOTH ends by the identity parsed from the name (and
        // asserting the identities agree per slot) is the only cross-instance-stable alignment — it is
        // what OvMoe achieves by restoring per layer ordinal. Non-canonical names fall back to
        // positional (same-instance round-trips stay byte-identical; other models unaffected). Byte-size
        // + count + identity guards degrade any genuine mismatch to an all-or-nothing cold reprefill.
        auto states = handle->request->query_state();
        std::vector<size_t> order(states.size());
        for (size_t i = 0; i < states.size(); ++i) order[i] = i;
        std::vector<uint64_t> mkeys(states.size());
        bool canonical = true;
        for (size_t i = 0; i < states.size(); ++i)
            if (!kv_canonical_key(states[i].get_name(), &mkeys[i])) { canonical = false; break; }
        if (canonical) {
            std::stable_sort(order.begin(), order.end(),
                             [&](size_t a, size_t b) { return mkeys[a] < mkeys[b]; });
        }
        const uint8_t* p = buf;
        const uint8_t* end = buf + len;
        auto need = [&](size_t n) { if (static_cast<size_t>(end - p) < n) throw std::runtime_error("set_state_blob: truncated"); };
        auto get32 = [&]() { need(4); uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; };
        auto get64 = [&]() { need(8); uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; };
        uint32_t count = get32();
        if (count != states.size()) {
            std::ostringstream oss;
            oss << "set_state_blob: blob has " << count << " states, model has " << states.size();
            set_last_error(oss.str().c_str());
            return 1;
        }
        uint32_t applied = 0;
        std::string dbg_b0, dbg_blast; // diagnostic: first/last blob state names
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t nl = get32(); need(nl);
            std::string bname(reinterpret_cast<const char*>(p), nl); p += nl;
            if (i == 0) dbg_b0 = bname;
            if (i + 1 == count) dbg_blast = bname;
            need(1); uint8_t dcode = *p++;
            need(1); uint8_t rank = *p++;
            ov::Shape shape; shape.reserve(rank);
            for (uint8_t r = 0; r < rank; ++r) shape.push_back(static_cast<size_t>(get64()));
            uint64_t nb = get64(); need(nb);
            const size_t dst = order[i];
            // In canonical mode the donor's slot identity must equal the destination's — a divergent
            // set (wrong rank/model, or a donor that couldn't emit canonically) leaves applied<count ⇒
            // the all-or-nothing cold fallback below, never a silent mis-map.
            bool identity_ok = true;
            if (canonical) {
                uint64_t bkey;
                identity_ok = kv_canonical_key(bname, &bkey) && bkey == mkeys[dst];
            }
            // Validate the declared dtype + dims against nb BEFORE constructing the tensor:
            // ov::Tensor allocates from the declared shape, so a corrupt/hostile blob declaring
            // dims like [1,1,1,2^50] would drive a multi-TB allocation attempt ahead of the old
            // post-construction byte-size guard (the wire-level frame bound cannot see inside an
            // opaque blob). Overflow-checked product; any disagreement skips the entry, and
            // applied<count fails the whole restore below.
            const ov::element::Type etype = code_to_ov_type(dcode);
            bool size_ok = etype != ov::element::dynamic;
            if (size_ok) {
                const uint64_t esz = etype.size(); // bytes/element; every supported code is byte-aligned
                uint64_t elems = 1;
                for (size_t d : shape) {
                    if (d != 0 && elems > UINT64_MAX / d) { size_ok = false; break; }
                    elems *= static_cast<uint64_t>(d);
                }
                size_ok = size_ok && esz != 0 && elems <= UINT64_MAX / esz && elems * esz == nb;
            }
            if (identity_ok && size_ok) {
                ov::Tensor t(etype, shape);
                std::memcpy(t.data(), p, nb);
                states[dst].set_state(t);
                ++applied;
            }
            p += nb;
        }
        // A skipped state (byte-size mismatch) leaves that variable at its CURRENT value while its
        // siblings carry the donor's — a silent half-restore the caller then treats as a good warm
        // resume. Fail instead so the engine falls back to a cold reprefill.
        if (applied != count) {
            std::ostringstream oss;
            oss << "set_state_blob: applied " << applied << " of " << count << " states"
                << " (canonical=" << (canonical ? 1 : 0)
                << " model0=" << (states.empty() ? "" : states[order[0]].get_name())
                << " modelN=" << (states.empty() ? "" : states[order[states.size() - 1]].get_name())
                << " blob0=" << dbg_b0 << " blobN=" << dbg_blast << ")";
            set_last_error(oss.str().c_str());
            return 1;
        }
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
    catch (...) { set_last_error("unknown C++ exception in set_state_blob"); return 1; }
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
