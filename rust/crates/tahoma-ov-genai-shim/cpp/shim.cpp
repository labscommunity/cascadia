// C++ implementation of the C ABI declared in shim.h. Links against
// openvino-genai (libopenvino_genai) and openvino (libopenvino). All C++
// exceptions are caught and translated to int32_t error codes; the most
// recent message is stashed in a thread-local static buffer.

#include "shim.h"

#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/variable_state.hpp>
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
            props[std::string(key)] = std::string(val);
        }
    }
    return props;
}

ov::element::Type dtype_from_code(uint32_t code) {
    switch (code) {
        case TAHOMA_DTYPE_F16: return ov::element::f16;
        case TAHOMA_DTYPE_I8:  return ov::element::i8;
        case TAHOMA_DTYPE_I32: return ov::element::i32;
        case TAHOMA_DTYPE_I64: return ov::element::i64;
        case TAHOMA_DTYPE_F32:
        default: return ov::element::f32;
    }
}

uint32_t code_from_dtype(const ov::element::Type& t) {
    if (t == ov::element::f16) return TAHOMA_DTYPE_F16;
    if (t == ov::element::i8)  return TAHOMA_DTYPE_I8;
    if (t == ov::element::i32) return TAHOMA_DTYPE_I32;
    if (t == ov::element::i64) return TAHOMA_DTYPE_I64;
    return TAHOMA_DTYPE_F32;
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

struct tahoma_runtime_t {
    ov::Core core;
    std::shared_ptr<ov::CompiledModel> compiled;
    std::shared_ptr<ov::InferRequest> request;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

extern "C" {

const char* tahoma_last_error_message() {
    return g_last_error.c_str();
}

// ===================== LLMPipeline (genai) =====================

int32_t tahoma_pipeline_create(
    const char* model_path, const char* device,
    const char* const* properties_kv, size_t properties_count,
    tahoma_pipeline_t** out_handle) {
    try {
        auto props = collect_properties(properties_kv, properties_count);
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(
            std::filesystem::path(model_path), std::string(device), props);
        *out_handle = new tahoma_pipeline_t{std::move(pipe)};
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_create"); return 1;
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
        auto draft_kv = ov::genai::draft_model(
            std::filesystem::path(draft_model_path), draft_dev);
        props.emplace(draft_kv.first, draft_kv.second);
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(
            std::filesystem::path(model_path), std::string(device), props);
        *out_handle = new tahoma_pipeline_t{std::move(pipe)};
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
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
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_create_with_prompt_lookup");
        return 1;
    }
}

void tahoma_pipeline_destroy(tahoma_pipeline_t* handle) { delete handle; }

tahoma_genconfig_t* tahoma_genconfig_new() {
    try { return new tahoma_genconfig_t{ov::genai::GenerationConfig{}}; }
    catch (...) { return nullptr; }
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
        set_last_error("null pipeline handle"); return 1;
    }
    try {
        ov::genai::DecodedResults results = cfg
            ? handle->pipe->generate(std::string(prompt), cfg->cfg)
            : handle->pipe->generate(std::string(prompt));
        std::string text = results;
        char* buf = static_cast<char*>(std::malloc(text.size() + 1));
        if (!buf) { set_last_error("malloc failure for output text"); return 1; }
        std::memcpy(buf, text.data(), text.size());
        buf[text.size()] = 0;
        *out_text = buf;
        if (out_token_count) {
            try {
                *out_token_count = static_cast<uint32_t>(
                    results.perf_metrics.get_num_generated_tokens());
            } catch (...) { *out_token_count = 0; }
        }
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    } catch (...) {
        set_last_error("unknown C++ exception in pipeline_generate"); return 1;
    }
}

void tahoma_free_string(char* s) { std::free(s); }

tahoma_tokenizer_t* tahoma_pipeline_get_tokenizer(tahoma_pipeline_t* handle) {
    if (!handle || !handle->pipe) return nullptr;
    try {
        return new tahoma_tokenizer_t{handle->pipe->get_tokenizer()};
    } catch (const std::exception& e) {
        set_last_error(e); return nullptr;
    }
}

int32_t tahoma_tokenizer_count_tokens(
    tahoma_tokenizer_t* tok, const char* text, uint32_t* out_count) {
    if (!tok || !out_count) {
        set_last_error("null tokenizer or out_count"); return 1;
    }
    try {
        auto enc = tok->tok.encode(std::string(text));
        const auto& shape = enc.input_ids.get_shape();
        *out_count = shape.empty() ? 0 : static_cast<uint32_t>(shape.back());
        return 0;
    } catch (const std::exception& e) {
        set_last_error(e); return 1;
    }
}

// ===================== ov-runtime (Core/CompiledModel/InferRequest) =====================

int32_t tahoma_runtime_compile(
    const char* model_xml_path, const char* device,
    const char* const* properties_kv, size_t properties_count,
    tahoma_runtime_t** out_handle) {
    if (!out_handle) { set_last_error("null out_handle"); return 1; }
    try {
        auto handle = std::make_unique<tahoma_runtime_t>();
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

void tahoma_runtime_destroy(tahoma_runtime_t* handle) { delete handle; }

int32_t tahoma_runtime_reset_state(tahoma_runtime_t* handle) {
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

size_t tahoma_runtime_input_count(tahoma_runtime_t* handle) {
    return handle ? handle->input_names.size() : 0;
}
size_t tahoma_runtime_output_count(tahoma_runtime_t* handle) {
    return handle ? handle->output_names.size() : 0;
}

int32_t tahoma_runtime_input_name(
    tahoma_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len) {
    if (!handle || idx >= handle->input_names.size()) {
        set_last_error("input idx out of range"); return 1;
    }
    return copy_name_to_buf(handle->input_names[idx], out_buf, out_cap, out_len);
}

int32_t tahoma_runtime_output_name(
    tahoma_runtime_t* handle, size_t idx,
    char* out_buf, size_t out_cap, size_t* out_len) {
    if (!handle || idx >= handle->output_names.size()) {
        set_last_error("output idx out of range"); return 1;
    }
    return copy_name_to_buf(handle->output_names[idx], out_buf, out_cap, out_len);
}

int32_t tahoma_runtime_set_input(
    tahoma_runtime_t* handle, const char* tensor_name,
    uint32_t dtype, const size_t* shape, size_t rank,
    const void* data, size_t data_size) {
    if (!handle || !handle->request) {
        set_last_error("null runtime"); return 1;
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
    }
}

int32_t tahoma_runtime_infer(tahoma_runtime_t* handle) {
    if (!handle || !handle->request) { set_last_error("null runtime"); return 1; }
    try { handle->request->infer(); return 0; }
    catch (const std::exception& e) { set_last_error(e); return 1; }
}

int32_t tahoma_runtime_output_rank(
    tahoma_runtime_t* handle, size_t output_idx, size_t* out_rank) {
    if (!handle || !handle->request || !out_rank) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_output_tensor(output_idx);
        *out_rank = t.get_shape().size();
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
}

int32_t tahoma_runtime_output_shape(
    tahoma_runtime_t* handle, size_t output_idx,
    size_t* out_shape, size_t shape_cap) {
    if (!handle || !handle->request) { set_last_error("null runtime"); return 1; }
    try {
        auto t = handle->request->get_output_tensor(output_idx);
        auto shp = t.get_shape();
        if (shape_cap < shp.size()) {
            set_last_error("shape_cap too small"); return 1;
        }
        for (size_t i = 0; i < shp.size(); ++i) out_shape[i] = shp[i];
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
}

int32_t tahoma_runtime_output_dtype(
    tahoma_runtime_t* handle, size_t output_idx, uint32_t* out_dtype) {
    if (!handle || !handle->request || !out_dtype) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_output_tensor(output_idx);
        *out_dtype = code_from_dtype(t.get_element_type());
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
}

int32_t tahoma_runtime_output_byte_size(
    tahoma_runtime_t* handle, size_t output_idx, size_t* out_byte_size) {
    if (!handle || !handle->request || !out_byte_size) {
        set_last_error("null arg"); return 1;
    }
    try {
        auto t = handle->request->get_output_tensor(output_idx);
        *out_byte_size = t.get_byte_size();
        return 0;
    } catch (const std::exception& e) { set_last_error(e); return 1; }
}

int32_t tahoma_runtime_output_copy(
    tahoma_runtime_t* handle, size_t output_idx,
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
}

}  // extern "C"
