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
        pub fn cascadia_cb_has_unfinished(
            handle: *mut cascadia_cb_pipeline_t,
            out_has: *mut i32,
        ) -> c_int;
        pub fn cascadia_cb_handle_read(
            pipeline: *mut cascadia_cb_pipeline_t,
            handle: *mut cascadia_cb_handle_t,
            out_text_delta: *mut *mut c_char,
            out_new_tokens: *mut u32,
            out_status: *mut i32,
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

        pub fn cascadia_runtime_destroy(handle: *mut cascadia_runtime_t);
        pub fn cascadia_runtime_reset_state(handle: *mut cascadia_runtime_t) -> c_int;
        pub fn cascadia_runtime_profiling(
            handle: *mut cascadia_runtime_t,
            out_buf: *mut c_char,
            buf_cap: usize,
            out_len: *mut usize,
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

/// Scheduler knobs for [`CbPipeline`]. Zero / `None` keep ov-genai defaults
/// (cache_size 0 = dynamic KV allocation, max_num_seqs 256,
/// max_num_batched_tokens 256, dynamic_split_fuse on, prefix caching off).
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
    Finished,
    /// Dropped/ignored by the pipeline (cancelled, or evicted by OV).
    Dropped,
}

/// One incremental read: the newly generated text suffix plus its token count.
#[derive(Clone, Debug)]
pub struct CbRead {
    pub text_delta: String,
    pub new_tokens: u32,
    pub status: CbStatus,
}

/// Owned `ContinuousBatchingPipeline` handle (issue #20). One pipeline
/// serves many concurrent requests; [`CbHandle`]s must be dropped before
/// the pipeline they came from.
pub struct CbPipeline {
    #[cfg(feature = "openvino")]
    handle: *mut sys::cascadia_cb_pipeline_t,
}

// Serialised behind the runner's engine mutex, like LlmPipeline.
unsafe impl Send for CbPipeline {}

/// Owned per-request generation handle from [`CbPipeline::add_request`].
pub struct CbHandle {
    #[cfg(feature = "openvino")]
    handle: *mut sys::cascadia_cb_handle_t,
}

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
        Ok(Self { handle })
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
                self.handle,
                request_id,
                prompt_c.as_ptr(),
                raw_cfg,
                &mut handle,
            );
            sys::cascadia_genconfig_destroy(raw_cfg);
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok(CbHandle { handle })
        }
    }

    #[cfg(not(feature = "openvino"))]
    pub fn step(&self) -> Result<()> {
        Err(Error::Stub)
    }

    /// Advance every enrolled request by one scheduler iteration.
    #[cfg(feature = "openvino")]
    pub fn step(&self) -> Result<()> {
        let rc = unsafe { sys::cascadia_cb_step(self.handle) };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok(())
    }

    #[cfg(not(feature = "openvino"))]
    pub fn has_unfinished(&self) -> Result<bool> {
        Err(Error::Stub)
    }

    #[cfg(feature = "openvino")]
    pub fn has_unfinished(&self) -> Result<bool> {
        let mut has: i32 = 0;
        let rc = unsafe { sys::cascadia_cb_has_unfinished(self.handle, &mut has) };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok(has != 0)
    }

    #[cfg(not(feature = "openvino"))]
    pub fn read(&self, _handle: &CbHandle) -> Result<CbRead> {
        Err(Error::Stub)
    }

    /// Drain newly generated text for one request since the previous read.
    #[cfg(feature = "openvino")]
    pub fn read(&self, handle: &CbHandle) -> Result<CbRead> {
        unsafe {
            let mut text_p: *mut c_char = ptr::null_mut();
            let mut new_tokens: u32 = 0;
            let mut status: i32 = 0;
            let rc = sys::cascadia_cb_handle_read(
                self.handle,
                handle.handle,
                &mut text_p,
                &mut new_tokens,
                &mut status,
            );
            if rc != 0 || text_p.is_null() {
                return Err(Error::Native(last_native_error()));
            }
            let text_delta = CStr::from_ptr(text_p).to_string_lossy().into_owned();
            sys::cascadia_free_string(text_p);
            Ok(CbRead {
                text_delta,
                new_tokens,
                status: match status {
                    0 => CbStatus::Running,
                    1 => CbStatus::Finished,
                    _ => CbStatus::Dropped,
                },
            })
        }
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
        let rc = unsafe { sys::cascadia_cb_count_tokens(self.handle, text_c.as_ptr(), &mut out) };
        if rc != 0 {
            None
        } else {
            Some(out)
        }
    }
}

#[cfg(feature = "openvino")]
impl Drop for CbPipeline {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::cascadia_cb_pipeline_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

impl CbHandle {
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

/// dtype codes shared with cascadia-transport's `DType` enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    I32 = 3,
    I64 = 4,
    Bf16 = 5,
}

impl DType {
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => Self::F16,
            2 => Self::I8,
            3 => Self::I32,
            4 => Self::I64,
            5 => Self::Bf16,
            _ => Self::F32,
        }
    }
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I8 => 1,
            Self::I64 => 8,
        }
    }
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
