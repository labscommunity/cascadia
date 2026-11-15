//! Safe Rust wrapper around the C++ openvino-genai shim.
//!
//! Two build modes:
//!
//! * default — compile-time stub. All operations return [`Error::Stub`].
//!   Use this on platforms without OpenVINO installed (Mac dev, CI).
//! * `--features openvino` — link the real C++ shim and call into the
//!   OV GenAI library. Requires `INTEL_OPENVINO_DIR` set at build time.
//!
//! The Rust API surface is identical across both modes, so all higher-level
//! crates can build unconditionally and the engines surface the runtime
//! error to the user when stub-built.

#[allow(unused_imports)]
use std::ffi::{CStr, CString};
#[allow(unused_imports)]
use std::os::raw::{c_char, c_int};
#[allow(unused_imports)]
use std::ptr;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("openvino-genai shim is built without the `openvino` feature; rebuild with `--features openvino`")]
    Stub,

    #[error("invalid utf8 in path/string: {0}")]
    Utf8(String),

    #[error("openvino-genai error: {0}")]
    Native(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Construct a C string, returning an Error on interior NUL.
#[allow(dead_code)]
fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|e| Error::Utf8(e.to_string()))
}

#[cfg(feature = "openvino")]
mod sys {
    use std::os::raw::{c_char, c_int};

    #[repr(C)]
    pub struct cascadia_pipeline_t {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct cascadia_genconfig_t {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct cascadia_tokenizer_t {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct cascadia_runtime_t {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct cascadia_cb_pipeline_t {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct cascadia_cb_handle_t {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn cascadia_last_error_message() -> *const c_char;
        pub fn cascadia_last_error_code() -> i32;

        // Test-only: see cpp/shim.cpp. Reports how collect_properties() stored
        // `key` — 1 = int64 (written to *out_i64), 0 = string, -1 = absent.
        pub fn cascadia_debug_property_int64_kind(
            properties_kv: *const *const c_char,
            properties_count: usize,
            key: *const c_char,
            out_i64: *mut i64,
        ) -> c_int;

        pub fn cascadia_pipeline_create(
            model_path: *const c_char,
            device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut cascadia_pipeline_t,
        ) -> c_int;

        pub fn cascadia_pipeline_create_with_draft(
            model_path: *const c_char,
            device: *const c_char,
            draft_model_path: *const c_char,
            draft_device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut cascadia_pipeline_t,
        ) -> c_int;

        pub fn cascadia_pipeline_create_with_prompt_lookup(
            model_path: *const c_char,
            device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut cascadia_pipeline_t,
        ) -> c_int;

        pub fn cascadia_pipeline_create_vlm(
            model_path: *const c_char,
            device: *const c_char,
            enable_prompt_lookup: i32,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut cascadia_pipeline_t,
        ) -> c_int;

        pub fn cascadia_pipeline_destroy(handle: *mut cascadia_pipeline_t);

        pub fn cascadia_genconfig_new() -> *mut cascadia_genconfig_t;
        pub fn cascadia_genconfig_destroy(cfg: *mut cascadia_genconfig_t);
        pub fn cascadia_genconfig_set_max_new_tokens(cfg: *mut cascadia_genconfig_t, v: u32);
        pub fn cascadia_genconfig_set_temperature(cfg: *mut cascadia_genconfig_t, v: f32);
        pub fn cascadia_genconfig_set_do_sample(cfg: *mut cascadia_genconfig_t, enabled: i32);
        pub fn cascadia_genconfig_set_num_assistant_tokens(cfg: *mut cascadia_genconfig_t, v: u32);
        pub fn cascadia_genconfig_set_max_ngram_size(cfg: *mut cascadia_genconfig_t, v: u32);
        pub fn cascadia_genconfig_set_apply_chat_template(
            cfg: *mut cascadia_genconfig_t,
            enabled: i32,
        );
        pub fn cascadia_genconfig_set_top_p(cfg: *mut cascadia_genconfig_t, v: f32);
        pub fn cascadia_genconfig_set_top_k(cfg: *mut cascadia_genconfig_t, v: u32);
        pub fn cascadia_genconfig_set_frequency_penalty(cfg: *mut cascadia_genconfig_t, v: f32);
        pub fn cascadia_genconfig_set_presence_penalty(cfg: *mut cascadia_genconfig_t, v: f32);
        pub fn cascadia_genconfig_set_rng_seed(cfg: *mut cascadia_genconfig_t, v: u64);

        pub fn cascadia_pipeline_generate(
            handle: *mut cascadia_pipeline_t,
            prompt: *const c_char,
            cfg: *const cascadia_genconfig_t,
            out_text: *mut *mut c_char,
            out_token_count: *mut u32,
        ) -> c_int;

        pub fn cascadia_free_string(s: *mut c_char);

        pub fn cascadia_cb_pipeline_create(
            model_path: *const c_char,
            device: *const c_char,
            cache_size_gb: u64,
            max_num_seqs: u64,
            max_num_batched_tokens: u64,
            dynamic_split_fuse: i32,
            enable_prefix_caching: i32,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut cascadia_cb_pipeline_t,
        ) -> c_int;
        pub fn cascadia_cb_pipeline_destroy(handle: *mut cascadia_cb_pipeline_t);
        pub fn cascadia_cb_add_request(
            handle: *mut cascadia_cb_pipeline_t,
            request_id: u64,
            prompt: *const c_char,
            cfg: *const cascadia_genconfig_t,
            out_handle: *mut *mut cascadia_cb_handle_t,
        ) -> c_int;
        pub fn cascadia_cb_step(handle: *mut cascadia_cb_pipeline_t) -> c_int;
        pub fn cascadia_cb_handle_read(
            pipeline: *mut cascadia_cb_pipeline_t,
            handle: *mut cascadia_cb_handle_t,
            out_text: *mut *mut c_char,
            out_text_len: *mut usize,
            out_new_tokens: *mut u32,
            out_status: *mut i32,
            out_finish_reason: *mut i32,
        ) -> c_int;
        pub fn cascadia_cb_handle_cancel(handle: *mut cascadia_cb_handle_t) -> c_int;
        pub fn cascadia_cb_handle_destroy(handle: *mut cascadia_cb_handle_t);
        pub fn cascadia_cb_count_tokens(
            handle: *mut cascadia_cb_pipeline_t,
            text: *const c_char,
            out_count: *mut u32,
        ) -> c_int;

        pub fn cascadia_pipeline_get_tokenizer(
            handle: *mut cascadia_pipeline_t,
        ) -> *mut cascadia_tokenizer_t;

        pub fn cascadia_tokenizer_destroy(tok: *mut cascadia_tokenizer_t);

        pub fn cascadia_tokenizer_count_tokens(
            tok: *mut cascadia_tokenizer_t,
            text: *const c_char,
            out_count: *mut u32,
        ) -> c_int;

        pub fn cascadia_runtime_compile(
            model_xml_path: *const c_char,
            device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut cascadia_runtime_t,
        ) -> c_int;

        pub fn cascadia_runtime_compile_gemv_offload(
            model_xml_path: *const c_char,
            device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_offloaded: *mut u32,
            out_handle: *mut *mut cascadia_runtime_t,
        ) -> c_int;

        pub fn cascadia_runtime_import_blob(
            blob_path: *const c_char,
            device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut cascadia_runtime_t,
        ) -> c_int;

        pub fn cascadia_runtime_compile_inspected(
            model_xml_path: *const c_char,
            device: *const c_char,
            extension_path: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut cascadia_runtime_t,
        ) -> c_int;
        pub fn cascadia_runtime_has_op_type(
            handle: *mut cascadia_runtime_t,
            type_name: *const c_char,
        ) -> c_int;
        pub fn cascadia_runtime_gpu_memory_bytes(
            handle: *mut cascadia_runtime_t,
            device: *const c_char,
            out_bytes: *mut u64,
        ) -> c_int;
        pub fn cascadia_runtime_set_state_bool(
            handle: *mut cascadia_runtime_t,
            name: *const c_char,
            value: i32,
            out_found: *mut i32,
        ) -> c_int;
        pub fn cascadia_runtime_set_input_strings(
            handle: *mut cascadia_runtime_t,
            tensor_name: *const c_char,
            strs: *const *const c_char,
            lens: *const usize,
            count: usize,
        ) -> c_int;

        pub fn cascadia_runtime_port_info(
            handle: *mut cascadia_runtime_t,
            is_output: i32,
            idx: usize,
            out_dtype: *mut u32,
            out_dims: *mut i64,
            dims_cap: usize,
            out_rank: *mut usize,
        ) -> c_int;
        pub fn cascadia_runtime_destroy(handle: *mut cascadia_runtime_t);
        pub fn cascadia_runtime_reset_state(handle: *mut cascadia_runtime_t) -> c_int;
        pub fn cascadia_runtime_profiling(
            handle: *mut cascadia_runtime_t,
            out_buf: *mut c_char,
            buf_cap: usize,
            out_len: *mut usize,
        ) -> c_int;
        pub fn cascadia_runtime_recreate_request(handle: *mut cascadia_runtime_t) -> c_int;
        // Issue-34: serialize/restore all KV variable-states as one opaque blob (warm-pull).
        pub fn cascadia_runtime_get_state_blob(
            handle: *mut cascadia_runtime_t,
            buf: *mut u8,
            cap: usize,
            len_out: *mut usize,
        ) -> c_int;
        pub fn cascadia_runtime_set_state_blob(
            handle: *mut cascadia_runtime_t,
            buf: *const u8,
            len: usize,
        ) -> c_int;

        pub fn cascadia_runtime_input_count(handle: *mut cascadia_runtime_t) -> usize;
        pub fn cascadia_runtime_output_count(handle: *mut cascadia_runtime_t) -> usize;

        pub fn cascadia_runtime_input_name(
            handle: *mut cascadia_runtime_t,
            idx: usize,
            out_buf: *mut c_char,
            out_cap: usize,
            out_len: *mut usize,
        ) -> c_int;
        pub fn cascadia_runtime_output_name(
            handle: *mut cascadia_runtime_t,
            idx: usize,
            out_buf: *mut c_char,
            out_cap: usize,
            out_len: *mut usize,
        ) -> c_int;

        pub fn cascadia_runtime_input_name_all(
            handle: *mut cascadia_runtime_t,
            idx: usize,
            out_buf: *mut c_char,
            out_cap: usize,
            out_len: *mut usize,
        ) -> c_int;
        pub fn cascadia_runtime_output_name_all(
            handle: *mut cascadia_runtime_t,
            idx: usize,
            out_buf: *mut c_char,
            out_cap: usize,
            out_len: *mut usize,
        ) -> c_int;

        pub fn cascadia_runtime_set_input(
            handle: *mut cascadia_runtime_t,
            tensor_name: *const c_char,
            dtype: u32,
            shape: *const usize,
            rank: usize,
            data: *const u8,
            data_size: usize,
        ) -> c_int;

        pub fn cascadia_runtime_infer(handle: *mut cascadia_runtime_t) -> c_int;

        pub fn cascadia_runtime_output_rank(
            handle: *mut cascadia_runtime_t,
            output_idx: usize,
            out_rank: *mut usize,
        ) -> c_int;
        pub fn cascadia_runtime_output_shape(
            handle: *mut cascadia_runtime_t,
            output_idx: usize,
            out_shape: *mut usize,
            shape_cap: usize,
        ) -> c_int;
        pub fn cascadia_runtime_output_dtype(
            handle: *mut cascadia_runtime_t,
            output_idx: usize,
            out_dtype: *mut u32,
        ) -> c_int;
        pub fn cascadia_runtime_input_rank(
            handle: *mut cascadia_runtime_t,
            input_idx: usize,
            out_rank: *mut usize,
        ) -> c_int;
        pub fn cascadia_runtime_input_shape(
            handle: *mut cascadia_runtime_t,
            input_idx: usize,
            out_shape: *mut usize,
            shape_cap: usize,
        ) -> c_int;
        pub fn cascadia_runtime_input_dtype(
            handle: *mut cascadia_runtime_t,
            input_idx: usize,
            out_dtype: *mut u32,
        ) -> c_int;
        pub fn cascadia_runtime_output_byte_size(
            handle: *mut cascadia_runtime_t,
            output_idx: usize,
            out: *mut usize,
        ) -> c_int;
        pub fn cascadia_runtime_output_copy(
            handle: *mut cascadia_runtime_t,
            output_idx: usize,
            out_buf: *mut u8,
            out_buf_size: usize,
        ) -> c_int;

        pub fn cascadia_core_list_devices(
            out_buf: *mut c_char,
            out_cap: usize,
            out_len: *mut usize,
        ) -> c_int;

        pub fn cascadia_ov_core_library_path(
            out_buf: *mut c_char,
            out_cap: usize,
            out_len: *mut usize,
        ) -> c_int;

        pub fn cascadia_ov_version(
            out_buf: *mut c_char,
            out_cap: usize,
            out_len: *mut usize,
        ) -> c_int;

        pub fn cascadia_core_get_property(
            device: *const c_char,
            property: *const c_char,
            out_buf: *mut c_char,
            out_cap: usize,
            out_len: *mut usize,
        ) -> c_int;
    }
}

/// Build a native genconfig from `cfg`. The caller owns the returned pointer
/// and must release it with `cascadia_genconfig_destroy`.
///
/// # Safety
/// Caller must be on the `openvino` build (native shim linked).
#[cfg(feature = "openvino")]
unsafe fn native_genconfig(cfg: &GenConfig) -> Result<*mut sys::cascadia_genconfig_t> {
    let raw_cfg = sys::cascadia_genconfig_new();
    if raw_cfg.is_null() {
        return Err(Error::Native("genconfig allocation failed".into()));
    }
    sys::cascadia_genconfig_set_max_new_tokens(raw_cfg, cfg.max_new_tokens.max(1));
    sys::cascadia_genconfig_set_do_sample(raw_cfg, if cfg.do_sample { 1 } else { 0 });
    sys::cascadia_genconfig_set_temperature(raw_cfg, cfg.temperature.max(0.0));
    if cfg.num_assistant_tokens > 0 {
        sys::cascadia_genconfig_set_num_assistant_tokens(raw_cfg, cfg.num_assistant_tokens);
    }
    if cfg.max_ngram_size > 0 {
        sys::cascadia_genconfig_set_max_ngram_size(raw_cfg, cfg.max_ngram_size);
    }
    sys::cascadia_genconfig_set_apply_chat_template(
        raw_cfg,
        if cfg.skip_chat_template { 0 } else { 1 },
    );
    // OpenAI-compatible sampling knobs (#14). top_p / penalties take
    // their OV defaults (1.0 / 0.0) when unset, so setting them is a
    // no-op; top_k=0 and seed=None mean "disabled" so only forward
    // when explicitly chosen (OV treats top_k=0 as no truncation).
    sys::cascadia_genconfig_set_top_p(raw_cfg, cfg.top_p);
    if cfg.top_k > 0 {
        sys::cascadia_genconfig_set_top_k(raw_cfg, cfg.top_k);
    }
    sys::cascadia_genconfig_set_frequency_penalty(raw_cfg, cfg.frequency_penalty);
    sys::cascadia_genconfig_set_presence_penalty(raw_cfg, cfg.presence_penalty);
    if let Some(seed) = cfg.seed {
        sys::cascadia_genconfig_set_rng_seed(raw_cfg, seed);
    }
    Ok(raw_cfg)
}

#[cfg(feature = "openvino")]
fn last_native_error() -> String {
    unsafe {
        let p = sys::cascadia_last_error_message();
        if p.is_null() {
            String::from("(no error message)")
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// Was the last shim error on this thread a resource-exhaustion class error
/// (EAGAIN / ENOMEM inside a plugin — code 2 from the C++ classifier)? Read it
/// right after a failed call, before any other shim call on the same thread.
/// Always `false` in the stub build.
pub fn last_error_resource_exhausted() -> bool {
    #[cfg(feature = "openvino")]
    {
        unsafe { sys::cascadia_last_error_code() == 2 }
    }
    #[cfg(not(feature = "openvino"))]
    {
        false
    }
}

/// Configuration for one generate call.
#[derive(Clone, Debug, Default)]
pub struct GenConfig {
    pub max_new_tokens: u32,
    pub do_sample: bool,
    pub temperature: f32,
    pub num_assistant_tokens: u32,
    pub max_ngram_size: u32,
    /// When true, the GenAI pipeline skips its internal chat-template apply —
    /// the caller pre-rendered the template (e.g. to honor enable_thinking).
    /// Default false = apply (matches OV's `apply_chat_template = true`).
    pub skip_chat_template: bool,
    /// OpenAI-compatible sampling knobs (#14). Forwarded to the matching
    /// `ov::genai::GenerationConfig` fields. Defaults (top_p=1.0 via the
    /// builder, penalties=0, top_k=0, seed=None) are no-ops.
    pub top_p: f32,
    pub top_k: u32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
}

/// Optional plugin-config knob (CACHE_DIR, KV_CACHE_PRECISION, etc.).
#[derive(Clone, Debug, Default)]
pub struct PluginConfig {
    /// Plain `(key, value)` string pairs. Both are passed verbatim to the
    /// OV plugin; OV accepts string-encoded values for typed properties
    /// (e.g. `("CACHE_DIR", "/tmp/ov_cache")`).
    pub entries: Vec<(String, String)>,
}

impl PluginConfig {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.entries.push((key.into(), value.into()));
        self
    }
}

/// One generate result.
#[derive(Clone, Debug)]
pub struct GenResult {
    pub text: String,
    pub generated_tokens: u32,
}

/// Owned LLMPipeline handle.
pub struct LlmPipeline {
    #[cfg(feature = "openvino")]
    handle: *mut sys::cascadia_pipeline_t,
}

unsafe impl Send for LlmPipeline {}

impl LlmPipeline {
    pub fn new(model_path: &str, device: &str, plugin: &PluginConfig) -> Result<Self> {
        Self::do_new(model_path, device, plugin, None, None, false)
    }

    pub fn with_draft(
        model_path: &str,
        device: &str,
        draft_model_path: &str,
        draft_device: &str,
        plugin: &PluginConfig,
    ) -> Result<Self> {
        Self::do_new(
            model_path,
            device,
            plugin,
            Some((draft_model_path, draft_device)),
            None,
            false,
        )
    }

    pub fn with_prompt_lookup(
        model_path: &str,
        device: &str,
        plugin: &PluginConfig,
    ) -> Result<Self> {
        Self::do_new(model_path, device, plugin, None, None, true)
    }

    /// VLM-layout export (e.g. Qwen3.5/3.6: `openvino_language_model.xml`
    /// + separate embeddings IRs), served text-only via `VLMPipeline`.
    /// `prompt_lookup` enables prompt-lookup decoding (OV GenAI >= 2026.2
    /// supports it on VLM pipelines).
    pub fn vlm(
        model_path: &str,
        device: &str,
        prompt_lookup: bool,
        plugin: &PluginConfig,
    ) -> Result<Self> {
        Self::do_new(model_path, device, plugin, None, Some(prompt_lookup), false)
    }

    #[cfg(not(feature = "openvino"))]
    fn do_new(
        _model_path: &str,
        _device: &str,
        _plugin: &PluginConfig,
        _draft: Option<(&str, &str)>,
        _vlm_prompt_lookup: Option<bool>,
        _prompt_lookup: bool,
    ) -> Result<Self> {
        Err(Error::Stub)
    }

    #[cfg(feature = "openvino")]
    fn do_new(
        model_path: &str,
        device: &str,
        plugin: &PluginConfig,
        draft: Option<(&str, &str)>,
        vlm_prompt_lookup: Option<bool>,
        prompt_lookup: bool,
    ) -> Result<Self> {
        let model_c = cstr(model_path)?;
        let device_c = cstr(device)?;

        // Build the plugin-config flat array.
        let mut owned: Vec<CString> = Vec::with_capacity(plugin.entries.len() * 2);
        for (k, v) in &plugin.entries {
            owned.push(cstr(k)?);
            owned.push(cstr(v)?);
        }
        let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();

        let mut handle: *mut sys::cascadia_pipeline_t = ptr::null_mut();
        let rc = unsafe {
            if let Some(pl) = vlm_prompt_lookup {
                sys::cascadia_pipeline_create_vlm(
                    model_c.as_ptr(),
                    device_c.as_ptr(),
                    if pl { 1 } else { 0 },
                    ptrs.as_ptr(),
                    plugin.entries.len(),
                    &mut handle,
                )
            } else if let Some((dpath, ddev)) = draft {
                let dpath_c = cstr(dpath)?;
                let ddev_c = cstr(ddev)?;
                sys::cascadia_pipeline_create_with_draft(
                    model_c.as_ptr(),
                    device_c.as_ptr(),
                    dpath_c.as_ptr(),
                    ddev_c.as_ptr(),
                    ptrs.as_ptr(),
                    plugin.entries.len(),
                    &mut handle,
                )
            } else if prompt_lookup {
                sys::cascadia_pipeline_create_with_prompt_lookup(
                    model_c.as_ptr(),
                    device_c.as_ptr(),
                    ptrs.as_ptr(),
                    plugin.entries.len(),
                    &mut handle,
                )
            } else {
                sys::cascadia_pipeline_create(
                    model_c.as_ptr(),
                    device_c.as_ptr(),
                    ptrs.as_ptr(),
                    plugin.entries.len(),
                    &mut handle,
                )
            }
        };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok(Self { handle })
    }

    #[cfg(not(feature = "openvino"))]
    pub fn generate(&self, _prompt: &str, _cfg: &GenConfig) -> Result<GenResult> {
        Err(Error::Stub)
    }

    #[cfg(feature = "openvino")]
    pub fn generate(&self, prompt: &str, cfg: &GenConfig) -> Result<GenResult> {
        let prompt_c = cstr(prompt)?;
        unsafe {
            let raw_cfg = native_genconfig(cfg)?;
            let mut text_p: *mut c_char = ptr::null_mut();
            let mut tok_count: u32 = 0;
            let rc = sys::cascadia_pipeline_generate(
                self.handle,
                prompt_c.as_ptr(),
                raw_cfg,
                &mut text_p,
                &mut tok_count,
            );
            sys::cascadia_genconfig_destroy(raw_cfg);
            if rc != 0 || text_p.is_null() {
                return Err(Error::Native(last_native_error()));
            }
            let text = CStr::from_ptr(text_p).to_string_lossy().into_owned();
            sys::cascadia_free_string(text_p);
            Ok(GenResult {
                text,
                generated_tokens: tok_count,
            })
        }
    }

    /// Best-effort token count via the pipeline's tokenizer. Returns
    /// `None` when the underlying call fails (e.g. tokenizer not available
    /// in stub builds).
    #[cfg(not(feature = "openvino"))]
    pub fn count_tokens(&self, _text: &str) -> Option<u32> {
        None
    }

    #[cfg(feature = "openvino")]
    pub fn count_tokens(&self, text: &str) -> Option<u32> {
        let text_c = cstr(text).ok()?;
        unsafe {
            let tok = sys::cascadia_pipeline_get_tokenizer(self.handle);
            if tok.is_null() {
                return None;
            }
            // Always destroy the borrowed tokenizer handle, even on
            // count failure — otherwise we leak heap on every call.
            let mut out: u32 = 0;
            let rc = sys::cascadia_tokenizer_count_tokens(tok, text_c.as_ptr(), &mut out);
            sys::cascadia_tokenizer_destroy(tok);
            if rc != 0 {
                return None;
            }
            Some(out)
        }
    }
}

#[cfg(feature = "openvino")]
impl Drop for LlmPipeline {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::cascadia_pipeline_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// Scheduler knobs for [`CbPipeline`]. Zero / `None` leaves the field at
/// `ov::genai::SchedulerConfig`'s own default — the shim simply does not
/// assign it.
///
/// Those defaults, verified against OV GenAI 2026.2
/// (`runtime/include/openvino/genai/scheduler_config.hpp`): `max_num_seqs`
/// 256, `max_num_batched_tokens` 256, `dynamic_split_fuse` on,
/// `enable_prefix_caching` off, and `cache_size` 0 — which, together with
/// `num_kv_blocks` 0, turns on dynamic cache allocation. Re-check these on an
/// SDK bump; upstream retunes scheduler defaults between releases.
///
/// Note `dynamic_split_fuse`: with it OFF, a prompt longer than
/// `max_num_batched_tokens` is a hard error upstream, not a slow path.
#[derive(Clone, Debug, Default)]
pub struct CbSchedulerConfig {
    pub cache_size_gb: u64,
    pub max_num_seqs: u64,
    pub max_num_batched_tokens: u64,
    pub dynamic_split_fuse: Option<bool>,
    pub enable_prefix_caching: Option<bool>,
}

/// Per-request generation state reported by [`CbPipeline::read`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CbStatus {
    Running,
    /// Completed normally (EOS, stop sequence, or the token cap).
    Finished,
    /// Aborted via [`CbHandle::cancel`].
    Cancelled,
    /// OpenVINO could not continue the request — it ran out of KV cache and
    /// abandoned it. Distinct from [`CbStatus::Finished`] because the caller
    /// must report it as a failure, not as a completed answer.
    Ignored,
}

/// Why generation stopped, as reported by OpenVINO. `Unknown` means it has not
/// said, and the caller should fall back to its own inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CbFinish {
    Unknown,
    Stop,
    Length,
}

/// One incremental read: the newly generated text suffix plus its token count.
#[derive(Clone, Debug)]
pub struct CbRead {
    /// Newly emittable text. NOT the text of `new_tokens` tokens — see below.
    pub text_delta: String,
    /// Tokens appended to this request's accumulated sequence by this read.
    ///
    /// This is deliberately not the token count of `text_delta`: the UTF-8
    /// hold-back can return an empty delta while reporting tokens, because the
    /// bytes those tokens decoded to are not safe to emit yet. Callers must
    /// count tokens even when the delta is empty, or usage under-reports.
    pub new_tokens: u32,
    pub status: CbStatus,
    /// OpenVINO's own reason for stopping, once it reports one.
    pub finish: CbFinish,
    /// Set once per request when the detokenizer's re-decode stopped extending
    /// what had already been emitted and the handle had to re-anchor. The
    /// shim takes no logging dependency, so the caller reports it — with the
    /// task context the shim does not have.
    pub resynced: bool,
}

/// How many bytes of `full` are safe to hand to the caller.
///
/// While the request is still running this holds back anything that the next
/// token may complete or replace:
///
/// * a trailing incomplete UTF-8 sequence, and
/// * a trailing RUN of U+FFFD. A byte-level detokenizer emits one U+FFFD per
///   undecodable byte, so a 4-byte codepoint arriving over three reads shows up
///   as one, then two replacement chars before resolving. Holding only the last
///   one (as the C++ this replaces did) emitted the earlier ones as real text
///   and then desynced the byte offset permanently.
///
/// Nothing is held on the terminal read, so a U+FFFD the model genuinely
/// produced is delayed, never dropped.
#[allow(dead_code)]
fn emittable_len(full: &[u8], running: bool) -> usize {
    if !running {
        return full.len();
    }
    const REPLACEMENT: [u8; 3] = [0xEF, 0xBF, 0xBD];
    let mut end = match std::str::from_utf8(full) {
        Ok(_) => full.len(),
        Err(e) => e.valid_up_to(),
    };
    while end >= REPLACEMENT.len() && full[end - REPLACEMENT.len()..end] == REPLACEMENT {
        end -= REPLACEMENT.len();
    }
    end
}

/// Advance `emitted` over the latest full decode, returning the newly
/// emittable text and whether the stream had to re-anchor.
///
/// Detokenizers are not guaranteed prefix-stable: appending a token can rewrite
/// earlier bytes, and a decode gets SHORTER when a run of U+FFFD collapses into
/// the codepoint it stood in for. Slicing at a remembered byte offset without
/// checking assumes that never happens; when it does, the offset is wrong for
/// the rest of the request and the client silently receives garbage — which is
/// exactly what the C++ this replaces did.
///
/// On divergence, re-anchor and emit nothing for the divergent region. The
/// bytes already handed out cannot be recalled, and emitting the corrected
/// suffix would duplicate text; re-anchoring at least stops the error
/// compounding and is reportable.
///
/// Public because the ov-runtime static paths need the identical algorithm:
/// they drive their own tokenizer and have the same non-prefix-stable decode.
/// One implementation, so the two cannot drift.
pub fn advance_emitted(emitted: &mut Vec<u8>, full: &[u8], running: bool) -> (String, bool) {
    if !full.starts_with(emitted) {
        emitted.clear();
        emitted.extend_from_slice(full);
        return (String::new(), true);
    }
    // max(): a re-decode may legitimately hold back MORE than last time (a
    // completed codepoint replaced by a shorter U+FFFD run), and the emitted
    // prefix must never go backwards.
    let end = emittable_len(full, running).max(emitted.len());
    let bytes = &full[emitted.len()..end];
    emitted.extend_from_slice(bytes);
    (String::from_utf8_lossy(bytes).into_owned(), false)
}

/// Owned `ContinuousBatchingPipeline` handle (issue #20). One pipeline
/// serves many concurrent requests; [`CbHandle`]s must be dropped before
/// the pipeline they came from.
pub struct CbPipeline {
    #[cfg(feature = "openvino")]
    inner: std::sync::Arc<CbInner>,
}

/// Shared owner of the native pipeline. Both [`CbPipeline`] and every
/// [`CbHandle`] it mints hold an `Arc` of this, so the native
/// `ContinuousBatchingPipeline` outlives its handles by construction.
///
/// The C header requires handles be destroyed before the pipeline; previously
/// that was upheld only by the declaration order of two fields in a struct in
/// another crate, where reordering them — something reviewers read as cosmetic
/// — would have been a use-after-free with no compiler complaint.
#[cfg(feature = "openvino")]
struct CbInner {
    ptr: *mut sys::cascadia_cb_pipeline_t,
}

// SAFETY: `CbInner` owns the native pipeline exclusively and OV GenAI
// pipelines have no thread affinity, so moving one between threads is sound.
// `Sync` is required only because `Arc<T>` demands `T: Send + Sync` to be
// `Send`; nothing hands out `&CbInner`, and every call that reaches the native
// pipeline goes through `&mut`-gated engine methods serialised behind the
// runner's engine mutex. The shim itself is NOT thread-safe (see shim.h).
#[cfg(feature = "openvino")]
unsafe impl Send for CbInner {}
#[cfg(feature = "openvino")]
unsafe impl Sync for CbInner {}

// SAFETY: as above — exclusive ownership, no thread affinity. Deliberately not
// `Sync`: `&CbPipeline` must not cross threads.
unsafe impl Send for CbPipeline {}

/// Owned per-request generation handle from [`CbPipeline::add_request`].
///
/// Holds an `Arc` on the pipeline that minted it, so it can neither outlive it
/// nor be read through a different one.
pub struct CbHandle {
    #[cfg(feature = "openvino")]
    inner: std::sync::Arc<CbInner>,
    #[cfg(feature = "openvino")]
    handle: *mut sys::cascadia_cb_handle_t,
    /// Bytes of this request's decode already handed to the caller.
    #[cfg(feature = "openvino")]
    emitted: Vec<u8>,
    /// Latches the one-per-request divergence warning.
    #[cfg(feature = "openvino")]
    warned_divergence: bool,
}

// SAFETY: as `CbPipeline`. A handle and its pipeline must stay on the same
// thread; both are reached only through the engine, which the runner serialises.
unsafe impl Send for CbHandle {}

impl CbPipeline {
    #[cfg(not(feature = "openvino"))]
    pub fn new(
        _model_path: &str,
        _device: &str,
        _sched: &CbSchedulerConfig,
        _plugin: &PluginConfig,
    ) -> Result<Self> {
        Err(Error::Stub)
    }

    #[cfg(feature = "openvino")]
    pub fn new(
        model_path: &str,
        device: &str,
        sched: &CbSchedulerConfig,
        plugin: &PluginConfig,
    ) -> Result<Self> {
        let model_c = cstr(model_path)?;
        let device_c = cstr(device)?;
        let mut owned: Vec<CString> = Vec::with_capacity(plugin.entries.len() * 2);
        for (k, v) in &plugin.entries {
            owned.push(cstr(k)?);
            owned.push(cstr(v)?);
        }
        let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();
        let tri = |v: Option<bool>| match v {
            None => -1i32,
            Some(false) => 0,
            Some(true) => 1,
        };
        let mut handle: *mut sys::cascadia_cb_pipeline_t = ptr::null_mut();
        let rc = unsafe {
            sys::cascadia_cb_pipeline_create(
                model_c.as_ptr(),
                device_c.as_ptr(),
                sched.cache_size_gb,
                sched.max_num_seqs,
                sched.max_num_batched_tokens,
                tri(sched.dynamic_split_fuse),
                tri(sched.enable_prefix_caching),
                ptrs.as_ptr(),
                plugin.entries.len(),
                &mut handle,
            )
        };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok(Self {
            inner: std::sync::Arc::new(CbInner { ptr: handle }),
        })
    }

    #[cfg(not(feature = "openvino"))]
    pub fn add_request(
        &self,
        _request_id: u64,
        _prompt: &str,
        _cfg: &GenConfig,
    ) -> Result<CbHandle> {
        Err(Error::Stub)
    }

    /// Enroll a prompt into the running batch. `request_id` must be unique
    /// among live requests on this pipeline.
    #[cfg(feature = "openvino")]
    pub fn add_request(&self, request_id: u64, prompt: &str, cfg: &GenConfig) -> Result<CbHandle> {
        let prompt_c = cstr(prompt)?;
        unsafe {
            let raw_cfg = native_genconfig(cfg)?;
            let mut handle: *mut sys::cascadia_cb_handle_t = ptr::null_mut();
            let rc = sys::cascadia_cb_add_request(
                self.inner.ptr,
                request_id,
                prompt_c.as_ptr(),
                raw_cfg,
                &mut handle,
            );
            sys::cascadia_genconfig_destroy(raw_cfg);
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(CbHandle {
                inner: std::sync::Arc::clone(&self.inner),
                handle,
                emitted: Vec::new(),
                warned_divergence: false,
            })
        }
    }

    #[cfg(not(feature = "openvino"))]
    pub fn step(&self) -> Result<()> {
        Err(Error::Stub)
    }

    /// Advance the batch by one scheduler iteration. The scheduler picks which
    /// enrolled requests run, so a given request may not progress. A no-op
    /// when the pipeline has no non-finished requests.
    #[cfg(feature = "openvino")]
    pub fn step(&self) -> Result<()> {
        let rc = unsafe { sys::cascadia_cb_step(self.inner.ptr) };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok(())
    }

    #[cfg(not(feature = "openvino"))]
    pub fn count_tokens(&self, _text: &str) -> Option<u32> {
        None
    }

    /// Best-effort token count via the pipeline's tokenizer.
    #[cfg(feature = "openvino")]
    pub fn count_tokens(&self, text: &str) -> Option<u32> {
        let text_c = cstr(text).ok()?;
        let mut out: u32 = 0;
        let rc =
            unsafe { sys::cascadia_cb_count_tokens(self.inner.ptr, text_c.as_ptr(), &mut out) };
        if rc != 0 {
            None
        } else {
            Some(out)
        }
    }
}

#[cfg(feature = "openvino")]
impl Drop for CbInner {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::cascadia_cb_pipeline_destroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl CbHandle {
    #[cfg(not(feature = "openvino"))]
    pub fn read(&mut self) -> Result<CbRead> {
        Err(Error::Stub)
    }

    /// Drain newly generated text for this request since the previous read.
    ///
    /// Lives on the handle rather than the pipeline so it cannot be called
    /// with a handle minted by a different pipeline — that decoded the text
    /// with the wrong tokenizer and returned success.
    #[cfg(feature = "openvino")]
    pub fn read(&mut self) -> Result<CbRead> {
        unsafe {
            let mut text_p: *mut c_char = ptr::null_mut();
            let mut text_len: usize = 0;
            let mut new_tokens: u32 = 0;
            let mut status: i32 = 0;
            let mut finish: i32 = 0;
            let rc = sys::cascadia_cb_handle_read(
                self.inner.ptr,
                self.handle,
                &mut text_p,
                &mut text_len,
                &mut new_tokens,
                &mut status,
                &mut finish,
            );
            if rc != 0 || text_p.is_null() {
                return Err(Error::Native(last_native_error()));
            }
            let full = std::slice::from_raw_parts(text_p as *const u8, text_len);
            let running = status == 0;
            let (text_delta, resynced) = self.take_delta(full, running);
            sys::cascadia_free_string(text_p);
            // An unrecognised code means the C ABI grew a state this build does
            // not know how to treat. Guessing "finished" would report a failure
            // as an answer, which is the bug this widening exists to fix.
            let status = match status {
                0 => CbStatus::Running,
                1 => CbStatus::Finished,
                2 => CbStatus::Cancelled,
                3 => CbStatus::Ignored,
                other => {
                    return Err(Error::Native(format!("unknown cb status {other}")));
                }
            };
            Ok(CbRead {
                text_delta,
                new_tokens,
                resynced,
                status,
                finish: match finish {
                    1 => CbFinish::Stop,
                    2 => CbFinish::Length,
                    _ => CbFinish::Unknown,
                },
            })
        }
    }

    /// Advance `emitted` over `full` and return the newly emittable text.
    ///
    /// Detokenizers are not guaranteed prefix-stable — appending a token can
    /// rewrite earlier bytes, and a decode can get SHORTER when a run of
    /// U+FFFD collapses into the codepoint it was standing in for. Slicing at
    /// a remembered byte offset without checking assumes it never happens; when
    /// it does the offset is wrong for the rest of the request and the client
    /// silently receives garbage. Re-anchor instead, emitting nothing for the
    /// divergent region: the already-emitted bytes cannot be recalled, and
    /// re-emitting the corrected suffix would duplicate text.
    #[cfg(feature = "openvino")]
    fn take_delta(&mut self, full: &[u8], running: bool) -> (String, bool) {
        let (delta, diverged) = advance_emitted(&mut self.emitted, full, running);
        let first_divergence = diverged && !self.warned_divergence;
        self.warned_divergence |= diverged;
        (delta, first_divergence)
    }

    #[cfg(not(feature = "openvino"))]
    pub fn cancel(&self) -> Result<()> {
        Err(Error::Stub)
    }

    /// Abort this request (client disconnect / cancel). Safe when already
    /// finished.
    #[cfg(feature = "openvino")]
    pub fn cancel(&self) -> Result<()> {
        let rc = unsafe { sys::cascadia_cb_handle_cancel(self.handle) };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok(())
    }
}

#[cfg(feature = "openvino")]
impl Drop for CbHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::cascadia_cb_handle_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// dtype codes; 0..=4 are shared with cascadia-transport's `DType` enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    I32 = 3,
    I64 = 4,
    Bf16 = 5,
    /// OpenVINO `boolean`: one byte per element, 0 or 1.
    Bool = 6,
}

impl DType {
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => Self::F16,
            2 => Self::I8,
            3 => Self::I32,
            4 => Self::I64,
            5 => Self::Bf16,
            6 => Self::Bool,
            _ => Self::F32,
        }
    }
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I8 | Self::Bool => 1,
            Self::I64 => 8,
        }
    }
}

/// A compiled model's input or output port, as declared (no inference needed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortInfo {
    /// Every tensor name of the port.
    pub aliases: Vec<String>,
    /// `None` for an element type without a [`DType`] (e.g. string).
    pub dtype: Option<DType>,
    /// `None` where the dim is dynamic.
    pub dims: Vec<Option<usize>>,
}

/// Safe wrapper around the low-level OV Core/CompiledModel/InferRequest.
/// Used by the ov-runtime + ov-dist-spec engines. The genai LLMPipeline
/// has its own [`LlmPipeline`] type above.
pub struct Runtime {
    #[cfg(feature = "openvino")]
    handle: *mut sys::cascadia_runtime_t,
}

unsafe impl Send for Runtime {}

impl Runtime {
    /// Compile a model from disk and create an InferRequest.
    pub fn compile(model_xml_path: &str, device: &str, plugin: &PluginConfig) -> Result<Self> {
        Self::do_compile(model_xml_path, device, plugin)
    }

    #[cfg(not(feature = "openvino"))]
    fn do_compile(_path: &str, _device: &str, _plugin: &PluginConfig) -> Result<Self> {
        Err(Error::Stub)
    }

    #[cfg(feature = "openvino")]
    fn do_compile(model_xml_path: &str, device: &str, plugin: &PluginConfig) -> Result<Self> {
        let path_c = cstr(model_xml_path)?;
        let device_c = cstr(device)?;
        let mut owned: Vec<CString> = Vec::with_capacity(plugin.entries.len() * 2);
        for (k, v) in &plugin.entries {
            owned.push(cstr(k)?);
            owned.push(cstr(v)?);
        }
        let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();

        let mut handle: *mut sys::cascadia_runtime_t = ptr::null_mut();
        let rc = unsafe {
            sys::cascadia_runtime_compile(
                path_c.as_ptr(),
                device_c.as_ptr(),
                ptrs.as_ptr(),
                plugin.entries.len(),
                &mut handle,
            )
        };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok(Self { handle })
    }

    /// Compile like [`Runtime::compile`], reading the model first so its op
    /// types can be queried with [`Runtime::has_op_type`]. `extension`, when
    /// set, is an OpenVINO extension library (see
    /// [`find_tokenizers_extension`]) added to this handle's Core first.
    pub fn compile_inspected(
        model_xml_path: &str,
        device: &str,
        extension: Option<&str>,
        plugin: &PluginConfig,
    ) -> Result<Self> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = (model_xml_path, device, extension, plugin);
            Err(Error::Stub)
        }
        #[cfg(feature = "openvino")]
        {
            let path_c = cstr(model_xml_path)?;
            let device_c = cstr(device)?;
            let ext_c = extension.map(cstr).transpose()?;
            let mut owned: Vec<CString> = Vec::with_capacity(plugin.entries.len() * 2);
            for (k, v) in &plugin.entries {
                owned.push(cstr(k)?);
                owned.push(cstr(v)?);
            }
            let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();
            let mut handle: *mut sys::cascadia_runtime_t = ptr::null_mut();
            let rc = unsafe {
                sys::cascadia_runtime_compile_inspected(
                    path_c.as_ptr(),
                    device_c.as_ptr(),
                    ext_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
                    ptrs.as_ptr(),
                    plugin.entries.len(),
                    &mut handle,
                )
            };
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(Self { handle })
        }
    }

    /// GPU memory this handle's model holds now: OpenVINO's
    /// `GPU_MEMORY_STATISTICS` for `device` on the handle's own Core, summed over
    /// allocation types. An error off GPU or where the plugin has no such property.
    pub fn gpu_memory_bytes(&self, device: &str) -> Result<u64> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = device;
            Err(Error::Stub)
        }
        #[cfg(feature = "openvino")]
        {
            let dev = cstr(device)?;
            let mut out = 0u64;
            let rc = unsafe {
                sys::cascadia_runtime_gpu_memory_bytes(self.handle, dev.as_ptr(), &mut out)
            };
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(out)
        }
    }

    /// Whether the model had an op of type `type_name` (e.g. `RandomUniform`)
    /// before compiling. Only for a [`Runtime::compile_inspected`] handle.
    pub fn has_op_type(&self, type_name: &str) -> Result<bool> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = type_name;
            Err(Error::Stub)
        }
        #[cfg(feature = "openvino")]
        {
            let ty = cstr(type_name)?;
            match unsafe { sys::cascadia_runtime_has_op_type(self.handle, ty.as_ptr()) } {
                1 => Ok(true),
                0 => Ok(false),
                _ => Err(Error::Native(last_native_error())),
            }
        }
    }

    /// Set the infer request's variable state `name` to a boolean scalar.
    /// `Ok(false)` when the model has no such state (nothing changes).
    pub fn set_state_bool(&mut self, name: &str, value: bool) -> Result<bool> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = (name, value);
            Err(Error::Stub)
        }
        #[cfg(feature = "openvino")]
        {
            let name_c = cstr(name)?;
            let mut found: i32 = 0;
            let rc = unsafe {
                sys::cascadia_runtime_set_state_bool(
                    self.handle,
                    name_c.as_ptr(),
                    i32::from(value),
                    &mut found,
                )
            };
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(found != 0)
        }
    }

    /// Bind a 1-D string tensor (`[values.len()]`) by name, for tokenizer IRs.
    pub fn set_input_strings(&mut self, name: &str, values: &[&str]) -> Result<()> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = (name, values);
            Err(Error::Stub)
        }
        #[cfg(feature = "openvino")]
        {
            let name_c = cstr(name)?;
            let ptrs: Vec<*const c_char> =
                values.iter().map(|v| v.as_ptr() as *const c_char).collect();
            let lens: Vec<usize> = values.iter().map(|v| v.len()).collect();
            let rc = unsafe {
                sys::cascadia_runtime_set_input_strings(
                    self.handle,
                    name_c.as_ptr(),
                    ptrs.as_ptr(),
                    lens.as_ptr(),
                    values.len(),
                )
            };
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(())
        }
    }

    /// Import a precompiled blob (from `ov::CompiledModel::export_model`,
    /// e.g. an AOT cross-compile on a big-RAM host with `NPU_PLATFORM` set)
    /// instead of compiling from IR — the compiler (and its ~5.5x-INT4-bytes
    /// host-RAM transient) never runs on this box.
    pub fn import_blob(blob_path: &str, device: &str, plugin: &PluginConfig) -> Result<Self> {
        Self::do_import_blob(blob_path, device, plugin)
    }

    #[cfg(not(feature = "openvino"))]
    fn do_import_blob(_path: &str, _device: &str, _plugin: &PluginConfig) -> Result<Self> {
        Err(Error::Stub)
    }

    #[cfg(feature = "openvino")]
    fn do_import_blob(blob_path: &str, device: &str, plugin: &PluginConfig) -> Result<Self> {
        let path_c = cstr(blob_path)?;
        let device_c = cstr(device)?;
        let mut owned: Vec<CString> = Vec::with_capacity(plugin.entries.len() * 2);
        for (k, v) in &plugin.entries {
            owned.push(cstr(k)?);
            owned.push(cstr(v)?);
        }
        let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();

        let mut handle: *mut sys::cascadia_runtime_t = ptr::null_mut();
        let rc = unsafe {
            sys::cascadia_runtime_import_blob(
                path_c.as_ptr(),
                device_c.as_ptr(),
                ptrs.as_ptr(),
                plugin.entries.len(),
                &mut handle,
            )
        };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok(Self { handle })
    }

    /// Compile with the CascadiaInt4Gemv offload pass: NNCF sym-INT4
    /// decompress→MatMul chains execute from the read_model mmap through the
    /// extension op instead of a plugin-repacked resident weight copy.
    /// Returns the runtime plus the number of MatMuls offloaded. CPU-class
    /// devices only (the op runs via the evaluate() fallback); do not pass
    /// CACHE_DIR (op member tensors don't survive blob serialization).
    pub fn compile_gemv_offload(
        model_xml_path: &str,
        device: &str,
        plugin: &PluginConfig,
    ) -> Result<(Self, u32)> {
        Self::do_compile_gemv_offload(model_xml_path, device, plugin)
    }

    #[cfg(not(feature = "openvino"))]
    fn do_compile_gemv_offload(
        _path: &str,
        _device: &str,
        _plugin: &PluginConfig,
    ) -> Result<(Self, u32)> {
        Err(Error::Stub)
    }

    #[cfg(feature = "openvino")]
    fn do_compile_gemv_offload(
        model_xml_path: &str,
        device: &str,
        plugin: &PluginConfig,
    ) -> Result<(Self, u32)> {
        let path_c = cstr(model_xml_path)?;
        let device_c = cstr(device)?;
        let mut owned: Vec<CString> = Vec::with_capacity(plugin.entries.len() * 2);
        for (k, v) in &plugin.entries {
            owned.push(cstr(k)?);
            owned.push(cstr(v)?);
        }
        let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();

        let mut handle: *mut sys::cascadia_runtime_t = ptr::null_mut();
        let mut offloaded: u32 = 0;
        let rc = unsafe {
            sys::cascadia_runtime_compile_gemv_offload(
                path_c.as_ptr(),
                device_c.as_ptr(),
                ptrs.as_ptr(),
                plugin.entries.len(),
                &mut offloaded,
                &mut handle,
            )
        };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok((Self { handle }, offloaded))
    }

    /// Per-node profiling of the last inference as TSV lines
    /// `node_name\tnode_type\texec_type\treal_us\tcpu_us` — requires the
    /// model compiled with the `PERF_COUNT=YES` plugin property.
    pub fn profiling(&self) -> Result<String> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        {
            let mut buf = vec![0u8; 1 << 20];
            let mut len: usize = 0;
            let rc = unsafe {
                sys::cascadia_runtime_profiling(
                    self.handle,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len(),
                    &mut len,
                )
            };
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            // The shim truncates at buf_cap, possibly mid-line or mid-UTF-8:
            // decode lossily and flag the cut instead of erroring out or
            // silently dropping tail nodes from the attribution.
            let truncated = len >= buf.len();
            buf.truncate(len);
            let mut s = String::from_utf8_lossy(&buf).into_owned();
            if truncated {
                s.push_str("\n[PROFILING OUTPUT TRUNCATED AT 1 MiB BUFFER CAP]\n");
            }
            Ok(s)
        }
    }

    pub fn reset_state(&mut self) -> Result<()> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        unsafe {
            let rc = sys::cascadia_runtime_reset_state(self.handle);
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(())
        }
    }

    /// Rebuild the InferRequest from the retained CompiledModel, dropping all variable state.
    /// Stronger than [`Runtime::reset_state`], which only calls `VariableState::reset()` — use
    /// after a [`Runtime::set_state_blob`] whose residue `reset_state` does not clear.
    pub fn recreate_request(&mut self) -> Result<()> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        unsafe {
            let rc = sys::cascadia_runtime_recreate_request(self.handle);
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(())
        }
    }

    /// Issue-34: capture all KV variable-states into one opaque blob (two-call: size then fill).
    /// The bytes are self-describing; restore on a peer engine via [`Runtime::set_state_blob`].
    pub fn get_state_blob(&mut self) -> Result<Vec<u8>> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        unsafe {
            let mut needed: usize = 0;
            let rc =
                sys::cascadia_runtime_get_state_blob(self.handle, ptr::null_mut(), 0, &mut needed);
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut buf = vec![0u8; needed];
            let rc = sys::cascadia_runtime_get_state_blob(
                self.handle,
                buf.as_mut_ptr(),
                buf.len(),
                &mut needed,
            );
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            buf.truncate(needed);
            Ok(buf)
        }
    }

    /// Issue-34: restore KV variable-states from a blob produced by [`Runtime::get_state_blob`] on the
    /// same model IR. States are matched by canonical KV IDENTITY (layer/kind ordinal parsed from the
    /// state name) — both ends sort by it, and each slot's identity is asserted — so a blob from a
    /// differently-compiled engine instance restores correctly even though raw `query_state()` order
    /// and state names are not portable across instances. Names that don't parse canonically fall back
    /// to positional matching, accepted only when each blob name equals the destination state's name
    /// verbatim (a same-instance round-trip); plain positional cross-instance restore is exactly what
    /// put states in the wrong slots. If ANY entry fails to apply (identity, byte-size, or element-type
    /// mismatch) the call returns `Err` rather than silently half-restoring. The element-type check
    /// compares against a dtype cached from a materialized read (capture, or a prior probe); a dynamic
    /// destination state that has never materialized cannot be read without throwing, so such entries
    /// apply on the size + identity guards alone (noted on stderr).
    ///
    /// On `Err` the request may already be **partially restored** — entries are applied as the blob
    /// is parsed. Callers must scrub with [`Runtime::recreate_request`] (or `reset_state` where that
    /// suffices) before the next infer.
    pub fn set_state_blob(&mut self, blob: &[u8]) -> Result<()> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = blob;
            return Err(Error::Stub);
        }
        #[cfg(feature = "openvino")]
        unsafe {
            let rc = sys::cascadia_runtime_set_state_blob(self.handle, blob.as_ptr(), blob.len());
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(())
        }
    }

    pub fn input_count(&self) -> usize {
        #[cfg(not(feature = "openvino"))]
        return 0;
        #[cfg(feature = "openvino")]
        unsafe {
            sys::cascadia_runtime_input_count(self.handle)
        }
    }

    pub fn output_count(&self) -> usize {
        #[cfg(not(feature = "openvino"))]
        return 0;
        #[cfg(feature = "openvino")]
        unsafe {
            sys::cascadia_runtime_output_count(self.handle)
        }
    }

    #[cfg(feature = "openvino")]
    fn name_at(
        &self,
        getter: unsafe extern "C" fn(
            *mut sys::cascadia_runtime_t,
            usize,
            *mut c_char,
            usize,
            *mut usize,
        ) -> c_int,
        idx: usize,
    ) -> Result<String> {
        unsafe {
            let mut needed: usize = 0;
            let rc = getter(self.handle, idx, ptr::null_mut(), 0, &mut needed);
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut buf = vec![0u8; needed + 1];
            let rc = getter(
                self.handle,
                idx,
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
                &mut needed,
            );
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            buf.truncate(needed);
            String::from_utf8(buf).map_err(|e| Error::Utf8(e.to_string()))
        }
    }

    pub fn input_name(&self, idx: usize) -> Result<String> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        self.name_at(sys::cascadia_runtime_input_name, idx)
    }
    pub fn output_name(&self, idx: usize) -> Result<String> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        self.name_at(sys::cascadia_runtime_output_name, idx)
    }

    /// All aliases for input port `idx`, as `Vec<String>`. Useful for
    /// matching against canonical names like "hidden_states" or
    /// "attention_mask" where the IR's first/any name is an internal
    /// node ID rather than the canonical name.
    pub fn input_aliases(&self, idx: usize) -> Result<Vec<String>> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        {
            let joined = self.name_at(sys::cascadia_runtime_input_name_all, idx)?;
            Ok(joined.split('\n').map(str::to_string).collect())
        }
    }
    pub fn output_aliases(&self, idx: usize) -> Result<Vec<String>> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        {
            let joined = self.name_at(sys::cascadia_runtime_output_name_all, idx)?;
            Ok(joined.split('\n').map(str::to_string).collect())
        }
    }

    pub fn input_names(&self) -> Result<Vec<String>> {
        let n = self.input_count();
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(self.input_name(i)?);
        }
        Ok(v)
    }

    pub fn output_names(&self) -> Result<Vec<String>> {
        let n = self.output_count();
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(self.output_name(i)?);
        }
        Ok(v)
    }

    /// Input port `idx` as the compiled model declares it.
    pub fn input_port(&self, idx: usize) -> Result<PortInfo> {
        Ok(PortInfo {
            aliases: self.input_aliases(idx)?,
            ..self.port_facts(false, idx)?
        })
    }

    /// Output port `idx` as the compiled model declares it.
    pub fn output_port(&self, idx: usize) -> Result<PortInfo> {
        Ok(PortInfo {
            aliases: self.output_aliases(idx)?,
            ..self.port_facts(true, idx)?
        })
    }

    fn port_facts(&self, is_output: bool, idx: usize) -> Result<PortInfo> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = (is_output, idx);
            Err(Error::Stub)
        }
        #[cfg(feature = "openvino")]
        unsafe {
            let (mut code, mut rank) = (0u32, 0usize);
            let out = i32::from(is_output);
            let rc = sys::cascadia_runtime_port_info(
                self.handle,
                out,
                idx,
                &mut code,
                ptr::null_mut(),
                0,
                &mut rank,
            );
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut dims = vec![0i64; rank];
            let rc = sys::cascadia_runtime_port_info(
                self.handle,
                out,
                idx,
                &mut code,
                dims.as_mut_ptr(),
                rank,
                &mut rank,
            );
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(PortInfo {
                aliases: Vec::new(),
                dtype: (code != 255).then(|| DType::from_code(code)),
                dims: dims.iter().map(|&d| usize::try_from(d).ok()).collect(),
            })
        }
    }

    /// Bind input by name. `data` must be `product(shape) * dtype.bytes_per_element` bytes.
    pub fn set_input(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[usize],
        data: &[u8],
    ) -> Result<()> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = (name, dtype, shape, data);
            return Err(Error::Stub);
        }
        #[cfg(feature = "openvino")]
        unsafe {
            let name_c = cstr(name)?;
            let rc = sys::cascadia_runtime_set_input(
                self.handle,
                name_c.as_ptr(),
                dtype as u32,
                shape.as_ptr(),
                shape.len(),
                data.as_ptr(),
                data.len(),
            );
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(())
        }
    }

    pub fn infer(&mut self) -> Result<()> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        unsafe {
            let rc = sys::cascadia_runtime_infer(self.handle);
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(())
        }
    }

    /// Read output `idx`. Returns (dtype, shape, raw_bytes).
    pub fn output(&self, idx: usize) -> Result<(DType, Vec<usize>, Vec<u8>)> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = idx;
            return Err(Error::Stub);
        }
        #[cfg(feature = "openvino")]
        unsafe {
            let mut rank: usize = 0;
            if sys::cascadia_runtime_output_rank(self.handle, idx, &mut rank) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut shape = vec![0usize; rank];
            if sys::cascadia_runtime_output_shape(self.handle, idx, shape.as_mut_ptr(), rank) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut dtype_code: u32 = 0;
            if sys::cascadia_runtime_output_dtype(self.handle, idx, &mut dtype_code) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut byte_size: usize = 0;
            if sys::cascadia_runtime_output_byte_size(self.handle, idx, &mut byte_size) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut buf = vec![0u8; byte_size];
            if sys::cascadia_runtime_output_copy(self.handle, idx, buf.as_mut_ptr(), byte_size) != 0
            {
                return Err(Error::Native(last_native_error()));
            }
            Ok((DType::from_code(dtype_code), shape, buf))
        }
    }

    /// Input tensor `idx`'s shape — concrete dims for a static (NPU-export)
    /// model. Used by `cascadia profile-stages` to size the zeroed
    /// inputs it feeds when timing a stage.
    pub fn input_shape(&self, idx: usize) -> Result<Vec<usize>> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = idx;
            return Err(Error::Stub);
        }
        #[cfg(feature = "openvino")]
        unsafe {
            let mut rank: usize = 0;
            if sys::cascadia_runtime_input_rank(self.handle, idx, &mut rank) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut shape = vec![0usize; rank];
            if sys::cascadia_runtime_input_shape(self.handle, idx, shape.as_mut_ptr(), rank) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(shape)
        }
    }

    /// Input tensor `idx`'s element type.
    pub fn input_dtype(&self, idx: usize) -> Result<DType> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = idx;
            return Err(Error::Stub);
        }
        #[cfg(feature = "openvino")]
        unsafe {
            let mut code: u32 = 0;
            if sys::cascadia_runtime_input_dtype(self.handle, idx, &mut code) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(DType::from_code(code))
        }
    }
}

#[cfg(feature = "openvino")]
impl Drop for Runtime {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::cascadia_runtime_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

// ============ Core enumeration (no compile) ============
//
// Free functions over a transient ov::Core; used by `cascadia
// profile-devices` to discover plugins on the worker host.

/// List the OV device names available on this host (e.g. `["CPU",
/// "GPU", "NPU"]`). Returns `Err(Stub)` on builds without the
/// `openvino` feature.
#[cfg(not(feature = "openvino"))]
pub fn list_devices() -> Result<Vec<String>> {
    Err(Error::Stub)
}

#[cfg(feature = "openvino")]
pub fn list_devices() -> Result<Vec<String>> {
    unsafe { fetch_buffered_string(|buf, cap, len| sys::cascadia_core_list_devices(buf, cap, len)) }
        .map(|s| {
            if s.is_empty() {
                Vec::new()
            } else {
                s.split('\n').map(str::to_owned).collect()
            }
        })
}

/// Path of the OpenVINO core library this process loaded.
#[cfg(not(feature = "openvino"))]
pub fn ov_core_library_path() -> Result<std::path::PathBuf> {
    Err(Error::Stub)
}

#[cfg(feature = "openvino")]
pub fn ov_core_library_path() -> Result<std::path::PathBuf> {
    unsafe {
        fetch_buffered_string(|buf, cap, len| sys::cascadia_ov_core_library_path(buf, cap, len))
    }
    .map(std::path::PathBuf::from)
}

/// The SDK layout one platform uses, for locating the tokenizers extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdkLayout {
    /// File name of the `openvino_tokenizers` extension library.
    pub extension: &'static str,
    /// Where the SDK keeps it, relative to `INTEL_OPENVINO_DIR`.
    pub sdk_subdir: &'static str,
}

impl SdkLayout {
    /// Windows SDKs keep Release and Debug builds side by side; only the
    /// Release one (`openvino_tokenizers.dll`, not `openvino_tokenizersd.dll`)
    /// matches the Release core the binary links.
    pub const WINDOWS: Self = Self {
        extension: "openvino_tokenizers.dll",
        sdk_subdir: "runtime/bin/intel64/Release",
    };
    pub const LINUX: Self = Self {
        extension: "libopenvino_tokenizers.so",
        sdk_subdir: "runtime/lib/intel64",
    };
    pub const MACOS_ARM64: Self = Self {
        extension: "libopenvino_tokenizers.dylib",
        sdk_subdir: "runtime/lib/arm64/Release",
    };

    pub fn host() -> Self {
        if cfg!(windows) {
            Self::WINDOWS
        } else if cfg!(target_os = "macos") {
            Self::MACOS_ARM64
        } else {
            Self::LINUX
        }
    }
}

/// Where to look for the tokenizers extension, in order: next to the loaded
/// core library (an SDK's own directory, or a release bundle's), then under
/// `INTEL_OPENVINO_DIR`.
pub fn tokenizers_extension_candidates(
    layout: SdkLayout,
    core_library: Option<&std::path::Path>,
    ov_dir: Option<&std::path::Path>,
) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Some(dir) = core_library.and_then(std::path::Path::parent) {
        out.push(dir.join(layout.extension));
    }
    if let Some(root) = ov_dir {
        let p = root.join(layout.sdk_subdir).join(layout.extension);
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// Locate the `openvino_tokenizers` extension the loaded runtime can use,
/// naming every path tried when none exists.
pub fn find_tokenizers_extension() -> Result<std::path::PathBuf> {
    let core = ov_core_library_path().ok();
    let ov_dir = std::env::var_os("INTEL_OPENVINO_DIR")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from);
    let tried =
        tokenizers_extension_candidates(SdkLayout::host(), core.as_deref(), ov_dir.as_deref());
    if let Some(p) = tried.iter().find(|p| p.is_file()) {
        return Ok(p.clone());
    }
    let list: Vec<String> = tried.iter().map(|p| p.display().to_string()).collect();
    Err(Error::Native(format!(
        "the OpenVINO tokenizers extension ({}) was not found; tried: {}",
        SdkLayout::host().extension,
        if list.is_empty() {
            "nothing (no loaded OpenVINO core library and INTEL_OPENVINO_DIR unset)".into()
        } else {
            list.join(", ")
        }
    )))
}

/// The OpenVINO GenAI / core versions this binary was compiled against and
/// the ones its loader actually picked up. `built_*` are the `M.m.p` from
/// the SDK headers at build time; `runtime_*` are the libraries' own build
/// strings (`2026.4.1.0-20466-7bb2e7f0e2b-releases/2026/4`). A release
/// bundle whose `lib/` was swapped for another SDK shows up as a skew
/// between the two; `cascadia doctor` prints both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OvVersion {
    pub built_genai: String,
    pub built_core: String,
    pub runtime_genai: String,
    pub runtime_core: String,
}

impl OvVersion {
    /// `M.m.p` prefix of a runtime build string (`2026.4.1.0-20466-…` →
    /// `2026.4.1`), comparable to the `built_*` fields.
    pub fn runtime_release(build: &str) -> String {
        let head = build.split('-').next().unwrap_or(build);
        let mut parts: Vec<&str> = head.split('.').take(3).collect();
        while parts.len() > 2 && parts.last() == Some(&"0") {
            parts.pop();
        }
        parts.join(".")
    }

    fn built_release(built: &str) -> String {
        Self::runtime_release(built)
    }

    /// True when the loaded GenAI runtime is not the release the binary was
    /// compiled against (major.minor.patch, trailing zeros ignored).
    pub fn genai_skew(&self) -> bool {
        Self::built_release(&self.built_genai) != Self::runtime_release(&self.runtime_genai)
    }
}

#[cfg(not(feature = "openvino"))]
pub fn ov_version() -> Result<OvVersion> {
    Err(Error::Stub)
}

#[cfg(feature = "openvino")]
pub fn ov_version() -> Result<OvVersion> {
    let s =
        unsafe { fetch_buffered_string(|buf, cap, len| sys::cascadia_ov_version(buf, cap, len)) }?;
    let mut it = s.split('\n').map(str::to_owned);
    let mut next = || it.next().unwrap_or_default();
    Ok(OvVersion {
        built_genai: next(),
        built_core: next(),
        runtime_genai: next(),
        runtime_core: next(),
    })
}

/// Query a single OV property (e.g. `FULL_DEVICE_NAME`,
/// `DEVICE_ARCHITECTURE`) on a single device. The returned value is
/// the property's `to_string()` form.
#[cfg(not(feature = "openvino"))]
pub fn device_property(_device: &str, _property: &str) -> Result<String> {
    Err(Error::Stub)
}

#[cfg(feature = "openvino")]
pub fn device_property(device: &str, property: &str) -> Result<String> {
    let dev_c = cstr(device)?;
    let prop_c = cstr(property)?;
    unsafe {
        fetch_buffered_string(|buf, cap, len| {
            sys::cascadia_core_get_property(dev_c.as_ptr(), prop_c.as_ptr(), buf, cap, len)
        })
    }
}

/// Convenience: full human-readable device name (e.g. `Intel(R) Arc(TM)
/// 140V GPU (16GB)`). Equivalent to `device_property(device,
/// "FULL_DEVICE_NAME")` but lives at this top level so callers don't
/// have to remember the property string.
pub fn device_full_name(device: &str) -> Result<String> {
    device_property(device, "FULL_DEVICE_NAME")
}

/// Internal: two-call buffered-string helper. Calls `f(null, 0, &len)`
/// to size-query, then allocates `len + 1` and calls `f(buf, cap, &len)`
/// to fill. Pulls non-zero rc into Error::Native via the last_native_error
/// channel.
#[cfg(feature = "openvino")]
unsafe fn fetch_buffered_string<F>(mut f: F) -> Result<String>
where
    F: FnMut(*mut c_char, usize, *mut usize) -> c_int,
{
    let mut len: usize = 0;
    let rc = f(ptr::null_mut(), 0, &mut len);
    if rc != 0 {
        return Err(Error::Native(last_native_error()));
    }
    if len == 0 {
        return Ok(String::new());
    }
    // Allocate len + 1 for the trailing NUL the shim writes.
    let mut buf: Vec<u8> = vec![0u8; len + 1];
    let rc2 = f(buf.as_mut_ptr() as *mut c_char, len + 1, &mut len);
    if rc2 != 0 {
        return Err(Error::Native(last_native_error()));
    }
    buf.truncate(len);
    String::from_utf8(buf).map_err(|e| Error::Utf8(e.to_string()))
}

#[cfg(test)]
mod resync_tests {
    use super::advance_emitted;

    const FFFD: &[u8] = &[0xEF, 0xBF, 0xBD];
    const GRIN: &[u8] = &[0xF0, 0x9F, 0x98, 0x80];

    /// The ordinary case: each read extends the last, and concatenating the
    /// deltas reproduces the final decode exactly once.
    #[test]
    fn monotonic_reads_emit_each_byte_exactly_once() {
        let mut emitted = Vec::new();
        let mut out = String::new();
        for full in [
            b"He".as_slice(),
            b"Hello".as_slice(),
            b"Hello wo".as_slice(),
        ] {
            let (d, diverged) = advance_emitted(&mut emitted, full, true);
            assert!(!diverged);
            out.push_str(&d);
        }
        let (d, _) = advance_emitted(&mut emitted, b"Hello world", false);
        out.push_str(&d);
        assert_eq!(out, "Hello world");
    }

    /// A decode that gets SHORTER cannot be a prefix extension. Before the
    /// rewrite this silently emitted nothing and left the byte offset stale for
    /// the rest of the request; now it re-anchors and says so.
    #[test]
    fn a_shorter_redecode_reanchors_and_reports() {
        let mut emitted = Vec::new();
        let (d, diverged) = advance_emitted(&mut emitted, b"abcdef", true);
        assert_eq!(d, "abcdef");
        assert!(!diverged);

        let (d, diverged) = advance_emitted(&mut emitted, b"abc", true);
        assert!(diverged, "a shorter decode must be reported");
        assert_eq!(d, "", "must not re-emit or emit a bogus slice");
        assert_eq!(emitted, b"abc", "must re-anchor onto the new decode");
    }

    /// Same length, different content — the case a byte-count check cannot see
    /// at all.
    #[test]
    fn a_rewritten_prefix_of_equal_length_reanchors() {
        let mut emitted = Vec::new();
        advance_emitted(&mut emitted, b"abcdef", true);
        let (d, diverged) = advance_emitted(&mut emitted, b"abcXef", true);
        assert!(diverged);
        assert_eq!(d, "");
        assert_eq!(emitted, b"abcXef");
    }

    /// After re-anchoring, the stream must keep working rather than staying
    /// desynced — this is the compounding-corruption case.
    #[test]
    fn growth_after_a_reanchor_still_emits_correctly() {
        let mut emitted = Vec::new();
        advance_emitted(&mut emitted, b"abcdef", true);
        let (_, diverged) = advance_emitted(&mut emitted, b"abc", true);
        assert!(diverged);
        let (d, diverged) = advance_emitted(&mut emitted, b"abcXYZ", true);
        assert!(!diverged, "growth from the new anchor is not a divergence");
        assert_eq!(d, "XYZ");
    }

    /// The real-world shape: a 4-byte codepoint arrives as a growing run of
    /// U+FFFD, then resolves — which SHRINKS the decode. Nothing partial may
    /// reach the caller, and the codepoint must arrive exactly once.
    #[test]
    fn emoji_resolving_from_a_replacement_run_never_leaks_a_partial() {
        let mut emitted = Vec::new();
        let mut out = String::new();
        let steps: [(Vec<u8>, bool); 3] = [
            ([b"hi ".as_slice(), FFFD].concat(), true),
            ([b"hi ".as_slice(), &FFFD.repeat(2)].concat(), true),
            ([b"hi ".as_slice(), GRIN].concat(), false),
        ];
        for (full, running) in steps {
            let (d, _) = advance_emitted(&mut emitted, &full, running);
            out.push_str(&d);
        }
        assert_eq!(out, "hi \u{1F600}");
    }

    /// Held-back bytes must be released on the terminal read, not lost.
    #[test]
    fn terminal_read_flushes_held_bytes() {
        let mut emitted = Vec::new();
        let partial = [b"ok".as_slice(), &GRIN[..2]].concat();
        let (d, _) = advance_emitted(&mut emitted, &partial, true);
        assert_eq!(d, "ok", "incomplete tail held while running");
        // A tail that is genuinely torn (the request ended mid-codepoint)
        // surfaces as exactly one replacement char rather than being dropped.
        let (d, _) = advance_emitted(&mut emitted, &partial, false);
        assert_eq!(d, "\u{FFFD}");
    }
}

#[cfg(test)]
mod holdback_tests {
    use super::emittable_len;

    const FFFD: &[u8] = &[0xEF, 0xBF, 0xBD];
    const EURO: &[u8] = &[0xE2, 0x82, 0xAC]; // U+20AC, 3 bytes
    const GRIN: &[u8] = &[0xF0, 0x9F, 0x98, 0x80]; // U+1F600, 4 bytes

    fn run(bytes: &[u8]) -> usize {
        emittable_len(bytes, true)
    }

    #[test]
    fn empty_and_ascii_are_fully_emittable() {
        assert_eq!(run(b""), 0);
        assert_eq!(run(b"abc"), 3);
    }

    #[test]
    fn complete_multibyte_codepoints_are_emittable() {
        assert_eq!(run(EURO), EURO.len());
        assert_eq!(run(GRIN), GRIN.len());
        let mixed = [b"hi".as_slice(), GRIN, b"!".as_slice()].concat();
        assert_eq!(run(&mixed), mixed.len());
    }

    #[test]
    fn truncated_sequences_are_held_at_every_offset() {
        // Every proper prefix of a multi-byte codepoint holds back entirely.
        for cp in [EURO, GRIN] {
            for take in 1..cp.len() {
                let buf = [b"ok".as_slice(), &cp[..take]].concat();
                assert_eq!(
                    run(&buf),
                    2,
                    "prefix of {take} bytes should hold back, buf={buf:02x?}"
                );
            }
        }
    }

    /// The bug this replaced: a byte-level detokenizer emits one U+FFFD per
    /// undecodable byte, so a 4-byte codepoint split across reads appears as a
    /// growing RUN of them. Holding only the last one emitted the earlier ones
    /// as real text and desynced every later read.
    #[test]
    fn a_run_of_trailing_replacements_is_held_whole() {
        for n in 1..=3 {
            let buf = [b"ok".as_slice(), &FFFD.repeat(n)].concat();
            assert_eq!(run(&buf), 2, "{n} trailing U+FFFD should all be held");
        }
    }

    /// A U+FFFD the model really produced is delayed, never dropped: once
    /// non-replacement text follows it, it becomes emittable, and the terminal
    /// read releases everything regardless.
    #[test]
    fn a_genuine_replacement_char_is_delayed_not_lost() {
        let followed = [b"a".as_slice(), FFFD, b"b".as_slice()].concat();
        assert_eq!(run(&followed), followed.len());

        let trailing = [b"a".as_slice(), FFFD].concat();
        assert_eq!(run(&trailing), 1);
        assert_eq!(emittable_len(&trailing, false), trailing.len());
    }

    #[test]
    fn terminal_read_releases_even_invalid_bytes() {
        let torn = [b"ok".as_slice(), &GRIN[..2]].concat();
        assert_eq!(emittable_len(&torn, false), torn.len());
    }

    /// Walking one emoji through three reads must never emit a partial or a
    /// replacement char, and must emit the codepoint exactly once.
    #[test]
    fn emoji_split_across_reads_emits_once_and_whole() {
        // Read 1 and 2 decode to a growing run of U+FFFD; read 3 resolves.
        let steps: [Vec<u8>; 3] = [
            [b"hi ".as_slice(), FFFD].concat(),
            [b"hi ".as_slice(), &FFFD.repeat(2)].concat(),
            [b"hi ".as_slice(), GRIN].concat(),
        ];
        let mut emitted = 0usize;
        let mut out: Vec<u8> = Vec::new();
        for (i, full) in steps.iter().enumerate() {
            let running = i < steps.len() - 1;
            let end = emittable_len(full, running).max(emitted);
            out.extend_from_slice(&full[emitted..end]);
            emitted = end;
        }
        assert_eq!(
            String::from_utf8(out).expect("emitted bytes must be valid utf8"),
            "hi \u{1F600}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "openvino"))]
    #[test]
    fn stub_construct_returns_error() {
        let plugin = PluginConfig::new();
        let result = LlmPipeline::new("/fake", "CPU", &plugin);
        assert!(matches!(result, Err(Error::Stub)));
    }

    #[cfg(not(feature = "openvino"))]
    #[test]
    fn stub_generate_returns_error() {
        // Construct via direct field assignment to test generate path.
        // (We can't actually construct LlmPipeline in stub mode since
        // do_new() returns Err, so this is instead an indirect check.)
        assert!(matches!(
            LlmPipeline::with_prompt_lookup("/x", "CPU", &PluginConfig::new()),
            Err(Error::Stub)
        ));
    }

    #[test]
    fn plugin_config_builder() {
        let p = PluginConfig::new()
            .with("CACHE_DIR", "/tmp/x")
            .with("KV_CACHE_PRECISION", "u8");
        assert_eq!(p.entries.len(), 2);
        assert_eq!(p.entries[0].0, "CACHE_DIR");
    }

    #[cfg(not(feature = "openvino"))]
    #[test]
    fn stub_list_devices_returns_error() {
        assert!(matches!(list_devices(), Err(Error::Stub)));
    }

    #[cfg(not(feature = "openvino"))]
    #[test]
    fn stub_device_full_name_returns_error() {
        assert!(matches!(device_full_name("CPU"), Err(Error::Stub)));
    }

    #[cfg(feature = "openvino")]
    #[test]
    fn live_list_devices_includes_cpu() {
        // OV always exposes CPU plugin on any host with the runtime
        // installed. If this fails on a real OV install, the shim
        // construction or the link is broken.
        let devs = list_devices().expect("list_devices");
        assert!(
            devs.iter().any(|d| d == "CPU"),
            "expected CPU in {:?}",
            devs
        );
    }

    #[cfg(feature = "openvino")]
    #[test]
    fn live_cpu_full_name_nonempty() {
        let name = device_full_name("CPU").expect("FULL_DEVICE_NAME");
        assert!(!name.is_empty(), "FULL_DEVICE_NAME for CPU was empty");
    }

    // The NPU static-LLM keys must reach OV as int64 Anys — ov::genai reads
    // them via Any::as<int64_t>() and throws on a string Any. This is the
    // regression guard for the string->int64 coercion in collect_properties
    // and for the exact key set shared with cascadia-cli's ov_perf_properties:
    // a rename on either side of the FFI flips a key to string/absent here.
    #[cfg(feature = "openvino")]
    #[test]
    fn collect_properties_coerces_npu_integer_keys_to_int64() {
        use std::ffi::CString;
        use std::os::raw::c_char;

        let pairs = [
            ("MAX_PROMPT_LEN", "1024"),
            ("MIN_RESPONSE_LEN", "128"),
            ("NPUW_LLM_PREFILL_CHUNK_SIZE", "512"),
            ("CACHE_DIR", "/tmp/ov_cache"),
        ];
        let mut owned: Vec<CString> = Vec::new();
        for &(k, v) in pairs.iter() {
            owned.push(CString::new(k).unwrap());
            owned.push(CString::new(v).unwrap());
        }
        let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();

        // (kind, value): kind 1 = int64 (value set), 0 = string, -1 = absent.
        let kind_of = |key: &str| -> (i32, i64) {
            let key_c = CString::new(key).unwrap();
            let mut out: i64 = 0;
            let k = unsafe {
                super::sys::cascadia_debug_property_int64_kind(
                    ptrs.as_ptr(),
                    pairs.len(),
                    key_c.as_ptr(),
                    &mut out,
                )
            };
            (k, out)
        };

        assert_eq!(kind_of("MAX_PROMPT_LEN"), (1, 1024));
        assert_eq!(kind_of("MIN_RESPONSE_LEN"), (1, 128));
        assert_eq!(kind_of("NPUW_LLM_PREFILL_CHUNK_SIZE"), (1, 512));
        // Plugin-parsed properties stay strings.
        assert_eq!(kind_of("CACHE_DIR").0, 0, "CACHE_DIR must stay a string");
        // Absent key.
        assert_eq!(kind_of("NOPE").0, -1);
    }

    // A value that isn't a clean integer must fall back to a string Any (so OV
    // reports the bad value) rather than being silently truncated by stoll's
    // partial parse ("512abc" -> 512).
    #[cfg(feature = "openvino")]
    #[test]
    fn collect_properties_rejects_partial_integer_to_string() {
        use std::ffi::CString;
        use std::os::raw::c_char;

        let owned = [
            CString::new("MAX_PROMPT_LEN").unwrap(),
            CString::new("512abc").unwrap(),
        ];
        let ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();
        let key_c = CString::new("MAX_PROMPT_LEN").unwrap();
        let mut out: i64 = 0;
        let kind = unsafe {
            super::sys::cascadia_debug_property_int64_kind(
                ptrs.as_ptr(),
                1,
                key_c.as_ptr(),
                &mut out,
            )
        };
        assert_eq!(kind, 0, "partial-numeric value must fall back to string");
    }
}

#[cfg(test)]
mod policy_support_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn bool_dtype_round_trips_as_one_byte() {
        assert_eq!(DType::from_code(DType::Bool as u32), DType::Bool);
        assert_eq!(DType::Bool as u32, 6);
        assert_eq!(DType::Bool.bytes_per_element(), 1);
    }

    #[test]
    fn tokenizers_extension_is_looked_up_next_to_the_core_then_in_the_sdk() {
        let got = tokenizers_extension_candidates(
            SdkLayout::WINDOWS,
            Some(Path::new("C:/sdk/runtime/bin/intel64/Release/openvino.dll")),
            Some(Path::new("C:/other")),
        );
        assert_eq!(
            got,
            vec![
                PathBuf::from("C:/sdk/runtime/bin/intel64/Release/openvino_tokenizers.dll"),
                PathBuf::from("C:/other/runtime/bin/intel64/Release/openvino_tokenizers.dll"),
            ]
        );
        assert!(got.iter().all(|p| !p.to_string_lossy().contains("Debug")));
        // A release bundle: the libs sit in lib/ beside the binary.
        let got = tokenizers_extension_candidates(
            SdkLayout::LINUX,
            Some(Path::new("/opt/cascadia/lib/libopenvino.so.2641")),
            None,
        );
        assert_eq!(
            got,
            vec![PathBuf::from("/opt/cascadia/lib/libopenvino_tokenizers.so")]
        );
        // The SDK path is not listed twice when the core lives in it.
        let got = tokenizers_extension_candidates(
            SdkLayout::LINUX,
            Some(Path::new("/sdk/runtime/lib/intel64/libopenvino.so.2641")),
            Some(Path::new("/sdk")),
        );
        assert_eq!(got.len(), 1);
        assert!(tokenizers_extension_candidates(SdkLayout::LINUX, None, None).is_empty());
    }

    #[cfg(not(feature = "openvino"))]
    #[test]
    fn policy_entry_points_are_clean_errors_in_the_stub_build() {
        assert!(matches!(
            Runtime::compile_inspected("m.xml", "CPU", Some("ext"), &PluginConfig::new()),
            Err(Error::Stub)
        ));
        assert!(matches!(ov_core_library_path(), Err(Error::Stub)));
    }

    #[cfg(feature = "openvino")]
    #[test]
    fn the_loaded_core_and_its_tokenizers_extension_are_found() {
        // Loading a plugin list forces the core library into the process.
        list_devices().unwrap();
        let core = ov_core_library_path().unwrap();
        assert!(core.is_file(), "{}", core.display());
        let ext = find_tokenizers_extension().unwrap();
        assert!(ext.is_file(), "{}", ext.display());
    }

    #[cfg(feature = "openvino")]
    #[test]
    fn op_types_include_ops_inside_a_loop_body() {
        // tests/fixtures/loop_random.xml: RandomUniform only inside a Loop body.
        let xml = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/loop_random.xml"
        );
        let plugin = PluginConfig::new().with("INFERENCE_PRECISION_HINT", "f32");
        let rt = Runtime::compile_inspected(xml, "CPU", None, &plugin).unwrap();
        assert!(rt.has_op_type("Loop").unwrap());
        assert!(rt.has_op_type("RandomUniform").unwrap());
        assert!(!rt.has_op_type("Multiply").unwrap());
    }
}
