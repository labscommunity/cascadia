// C++ implementation of the C ABI declared in shim.h.
//
// Links against openvino-genai (libopenvino_genai). All C++ exceptions are
// caught and translated to int32_t error codes; the most recent message is
// stashed in a thread-local-ish static buffer.

#include "shim.h"

#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/tokenizer.hpp>

namespace {

thread_local std::string g_last_error;

void set_last_error(const char* msg) {
    g_last_error = msg ? msg : "(null)";
}

void set_last_error(const std::exception& e) { g_last_error = e.what(); }

ov::AnyMap collect_properties(const char* const* kv, size_t count) {
    ov::AnyMap props;
    if (!kv) return props;
    for (size_t i = 0; i < count; ++i) {
        const char* key = kv[2 * i];
        const char* val = kv[2 * i + 1];
        if (key && val) {
            // The OV plugin C++ API accepts string-encoded values for
            // typed properties; mirrors what the C API does.
            props[std::string(key)] = std::string(val);
        }
    }
    return props;
}

}  // namespace

struct tahoma_pipeline_t {
    std::unique_ptr<ov::genai::LLMPipeline> pipe;
};

struct tahoma_genconfig_t {
    ov::genai::GenerationConfig cfg;
};

struct tahoma_tokenizer_t {
    ov::genai::Tokenizer tok;
};

extern "C" {

const char* tahoma_last_error_message() {
    return g_last_error.c_str();
}

int32_t tahoma_pipeline_create(
    const char* model_path, const char* device,
    const char* const* properties_kv, size_t properties_count,
    tahoma_pipeline_t** out_handle) {
    try {
        auto props = collect_properties(properties_kv, properties_count);
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(
            std::filesystem::path(model_path), std::string(device), props);
        auto* handle = new tahoma_pipeline_t{std::move(pipe)};
        *out_handle = handle;
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_create");
        return 1;
    }
}

int32_t tahoma_pipeline_create_with_draft(
    const char* model_path, const char* device,
    const char* draft_model_path, const char* draft_device,
    const char* const* properties_kv, size_t properties_count,
    tahoma_pipeline_t** out_handle) {
    try {
        auto props = collect_properties(properties_kv, properties_count);
        const std::string draft_dev =
            (draft_device && *draft_device) ? std::string(draft_device)
                                            : std::string(device);
        // ov::genai::draft_model returns std::pair<string, ov::Any>; insert it
        // into the AnyMap as the C++ API expects.
        auto draft_kv = ov::genai::draft_model(
            std::filesystem::path(draft_model_path), draft_dev);
        props.emplace(draft_kv.first, draft_kv.second);
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(
            std::filesystem::path(model_path), std::string(device), props);
        *out_handle = new tahoma_pipeline_t{std::move(pipe)};
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_create_with_draft");
        return 1;
    }
}

int32_t tahoma_pipeline_create_with_prompt_lookup(
    const char* model_path, const char* device,
    const char* const* properties_kv, size_t properties_count,
    tahoma_pipeline_t** out_handle) {
    try {
        auto props = collect_properties(properties_kv, properties_count);
        props[ov::genai::prompt_lookup.name()] = true;
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(
            std::filesystem::path(model_path), std::string(device), props);
        *out_handle = new tahoma_pipeline_t{std::move(pipe)};
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_create_with_prompt_lookup");
        return 1;
    }
}

void tahoma_pipeline_destroy(tahoma_pipeline_t* handle) {
    delete handle;
}

tahoma_genconfig_t* tahoma_genconfig_new() {
    try {
        return new tahoma_genconfig_t{ov::genai::GenerationConfig{}};
    } catch (...) {
        return nullptr;
    }
}

void tahoma_genconfig_destroy(tahoma_genconfig_t* cfg) { delete cfg; }

void tahoma_genconfig_set_max_new_tokens(tahoma_genconfig_t* cfg, uint32_t v) {
    if (cfg) cfg->cfg.max_new_tokens = v;
}
void tahoma_genconfig_set_temperature(tahoma_genconfig_t* cfg, float v) {
    if (cfg) cfg->cfg.temperature = v;
}
void tahoma_genconfig_set_do_sample(tahoma_genconfig_t* cfg, int32_t enabled) {
    if (cfg) cfg->cfg.do_sample = enabled != 0;
}
void tahoma_genconfig_set_num_assistant_tokens(tahoma_genconfig_t* cfg, uint32_t v) {
    if (cfg) cfg->cfg.num_assistant_tokens = v;
}
void tahoma_genconfig_set_max_ngram_size(tahoma_genconfig_t* cfg, uint32_t v) {
    if (cfg) cfg->cfg.max_ngram_size = v;
}

int32_t tahoma_pipeline_generate(
    tahoma_pipeline_t* handle, const char* prompt, const tahoma_genconfig_t* cfg,
    char** out_text, uint32_t* out_token_count) {
    if (!handle || !handle->pipe) {
        set_last_error("null pipeline handle");
        return 1;
    }
    try {
        ov::genai::DecodedResults results = cfg
            ? handle->pipe->generate(std::string(prompt), cfg->cfg)
            : handle->pipe->generate(std::string(prompt));
        std::string text = results;  // implicit conversion to std::string
        char* buf = static_cast<char*>(std::malloc(text.size() + 1));
        if (!buf) {
            set_last_error("malloc failure for output text");
            return 1;
        }
        std::memcpy(buf, text.data(), text.size());
        buf[text.size()] = 0;
        *out_text = buf;
        if (out_token_count) {
            // perf_metrics().get_num_generated_tokens() returns mean +
            // std; we want the actual count. Try the raw counter.
            try {
                *out_token_count = static_cast<uint32_t>(
                    results.perf_metrics.get_num_generated_tokens());
            } catch (...) {
                *out_token_count = 0;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_generate");
        return 1;
    }
}

void tahoma_free_string(char* s) {
    std::free(s);
}

tahoma_tokenizer_t* tahoma_pipeline_get_tokenizer(tahoma_pipeline_t* handle) {
    if (!handle || !handle->pipe) return nullptr;
    try {
        // Returns a copy/handle owned by the pipeline. We wrap it in our own
        // struct so the Rust side can free it independently when needed.
        auto* wrapper = new tahoma_tokenizer_t{handle->pipe->get_tokenizer()};
        return wrapper;
    } catch (const std::exception& e) {
        set_last_error(e);
        return nullptr;
    }
}

int32_t tahoma_tokenizer_count_tokens(
    tahoma_tokenizer_t* tok, const char* text, uint32_t* out_count) {
    if (!tok || !out_count) {
        set_last_error("null tokenizer or out_count");
        return 1;
    }
    try {
        auto enc = tok->tok.encode(std::string(text));
        // enc.input_ids.get_shape() back-row gives token count.
        const auto& shape = enc.input_ids.get_shape();
        *out_count = shape.empty() ? 0 : static_cast<uint32_t>(shape.back());
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e);
        return 1;
    }
}

}  // extern "C"
