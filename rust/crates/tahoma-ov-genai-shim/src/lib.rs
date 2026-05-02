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
    pub struct tahoma_pipeline_t {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct tahoma_genconfig_t {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct tahoma_tokenizer_t {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct tahoma_runtime_t {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn tahoma_last_error_message() -> *const c_char;

        pub fn tahoma_pipeline_create(
            model_path: *const c_char,
            device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut tahoma_pipeline_t,
        ) -> c_int;

        pub fn tahoma_pipeline_create_with_draft(
            model_path: *const c_char,
            device: *const c_char,
            draft_model_path: *const c_char,
            draft_device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut tahoma_pipeline_t,
        ) -> c_int;

        pub fn tahoma_pipeline_create_with_prompt_lookup(
            model_path: *const c_char,
            device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut tahoma_pipeline_t,
        ) -> c_int;

        pub fn tahoma_pipeline_destroy(handle: *mut tahoma_pipeline_t);

        pub fn tahoma_genconfig_new() -> *mut tahoma_genconfig_t;
        pub fn tahoma_genconfig_destroy(cfg: *mut tahoma_genconfig_t);
        pub fn tahoma_genconfig_set_max_new_tokens(cfg: *mut tahoma_genconfig_t, v: u32);
        pub fn tahoma_genconfig_set_temperature(cfg: *mut tahoma_genconfig_t, v: f32);
        pub fn tahoma_genconfig_set_do_sample(cfg: *mut tahoma_genconfig_t, enabled: i32);
        pub fn tahoma_genconfig_set_num_assistant_tokens(cfg: *mut tahoma_genconfig_t, v: u32);
        pub fn tahoma_genconfig_set_max_ngram_size(cfg: *mut tahoma_genconfig_t, v: u32);

        pub fn tahoma_pipeline_generate(
            handle: *mut tahoma_pipeline_t,
            prompt: *const c_char,
            cfg: *const tahoma_genconfig_t,
            out_text: *mut *mut c_char,
            out_token_count: *mut u32,
        ) -> c_int;

        pub fn tahoma_free_string(s: *mut c_char);

        pub fn tahoma_pipeline_get_tokenizer(
            handle: *mut tahoma_pipeline_t,
        ) -> *mut tahoma_tokenizer_t;

        pub fn tahoma_tokenizer_count_tokens(
            tok: *mut tahoma_tokenizer_t,
            text: *const c_char,
            out_count: *mut u32,
        ) -> c_int;

        pub fn tahoma_runtime_compile(
            model_xml_path: *const c_char,
            device: *const c_char,
            properties_kv: *const *const c_char,
            properties_count: usize,
            out_handle: *mut *mut tahoma_runtime_t,
        ) -> c_int;

        pub fn tahoma_runtime_destroy(handle: *mut tahoma_runtime_t);
        pub fn tahoma_runtime_reset_state(handle: *mut tahoma_runtime_t) -> c_int;

        pub fn tahoma_runtime_input_count(handle: *mut tahoma_runtime_t) -> usize;
        pub fn tahoma_runtime_output_count(handle: *mut tahoma_runtime_t) -> usize;

        pub fn tahoma_runtime_input_name(
            handle: *mut tahoma_runtime_t, idx: usize,
            out_buf: *mut c_char, out_cap: usize, out_len: *mut usize,
        ) -> c_int;
        pub fn tahoma_runtime_output_name(
            handle: *mut tahoma_runtime_t, idx: usize,
            out_buf: *mut c_char, out_cap: usize, out_len: *mut usize,
        ) -> c_int;

        pub fn tahoma_runtime_set_input(
            handle: *mut tahoma_runtime_t, tensor_name: *const c_char,
            dtype: u32, shape: *const usize, rank: usize,
            data: *const u8, data_size: usize,
        ) -> c_int;

        pub fn tahoma_runtime_infer(handle: *mut tahoma_runtime_t) -> c_int;

        pub fn tahoma_runtime_output_rank(
            handle: *mut tahoma_runtime_t, output_idx: usize, out_rank: *mut usize,
        ) -> c_int;
        pub fn tahoma_runtime_output_shape(
            handle: *mut tahoma_runtime_t, output_idx: usize,
            out_shape: *mut usize, shape_cap: usize,
        ) -> c_int;
        pub fn tahoma_runtime_output_dtype(
            handle: *mut tahoma_runtime_t, output_idx: usize, out_dtype: *mut u32,
        ) -> c_int;
        pub fn tahoma_runtime_output_byte_size(
            handle: *mut tahoma_runtime_t, output_idx: usize, out: *mut usize,
        ) -> c_int;
        pub fn tahoma_runtime_output_copy(
            handle: *mut tahoma_runtime_t, output_idx: usize,
            out_buf: *mut u8, out_buf_size: usize,
        ) -> c_int;
    }
}

#[cfg(feature = "openvino")]
fn last_native_error() -> String {
    unsafe {
        let p = sys::tahoma_last_error_message();
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
    handle: *mut sys::tahoma_pipeline_t,
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

    #[cfg(not(feature = "openvino"))]
    fn do_new(
        _model_path: &str,
        _device: &str,
        _plugin: &PluginConfig,
        _draft: Option<(&str, &str)>,
        _: Option<()>,
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
        _: Option<()>,
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

        let mut handle: *mut sys::tahoma_pipeline_t = ptr::null_mut();
        let rc = unsafe {
            if let Some((dpath, ddev)) = draft {
                let dpath_c = cstr(dpath)?;
                let ddev_c = cstr(ddev)?;
                sys::tahoma_pipeline_create_with_draft(
                    model_c.as_ptr(),
                    device_c.as_ptr(),
                    dpath_c.as_ptr(),
                    ddev_c.as_ptr(),
                    ptrs.as_ptr(),
                    plugin.entries.len(),
                    &mut handle,
                )
            } else if prompt_lookup {
                sys::tahoma_pipeline_create_with_prompt_lookup(
                    model_c.as_ptr(),
                    device_c.as_ptr(),
                    ptrs.as_ptr(),
                    plugin.entries.len(),
                    &mut handle,
                )
            } else {
                sys::tahoma_pipeline_create(
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
            let raw_cfg = sys::tahoma_genconfig_new();
            if raw_cfg.is_null() {
                return Err(Error::Native("genconfig allocation failed".into()));
            }
            sys::tahoma_genconfig_set_max_new_tokens(raw_cfg, cfg.max_new_tokens.max(1));
            sys::tahoma_genconfig_set_do_sample(raw_cfg, if cfg.do_sample { 1 } else { 0 });
            sys::tahoma_genconfig_set_temperature(raw_cfg, cfg.temperature.max(0.0));
            if cfg.num_assistant_tokens > 0 {
                sys::tahoma_genconfig_set_num_assistant_tokens(raw_cfg, cfg.num_assistant_tokens);
            }
            if cfg.max_ngram_size > 0 {
                sys::tahoma_genconfig_set_max_ngram_size(raw_cfg, cfg.max_ngram_size);
            }
            let mut text_p: *mut c_char = ptr::null_mut();
            let mut tok_count: u32 = 0;
            let rc = sys::tahoma_pipeline_generate(
                self.handle,
                prompt_c.as_ptr(),
                raw_cfg,
                &mut text_p,
                &mut tok_count,
            );
            sys::tahoma_genconfig_destroy(raw_cfg);
            if rc != 0 || text_p.is_null() {
                return Err(Error::Native(last_native_error()));
            }
            let text = CStr::from_ptr(text_p).to_string_lossy().into_owned();
            sys::tahoma_free_string(text_p);
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
            let tok = sys::tahoma_pipeline_get_tokenizer(self.handle);
            if tok.is_null() {
                return None;
            }
            let mut out: u32 = 0;
            let rc = sys::tahoma_tokenizer_count_tokens(tok, text_c.as_ptr(), &mut out);
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
            unsafe { sys::tahoma_pipeline_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// dtype codes shared with tahoma-transport's `DType` enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    I32 = 3,
    I64 = 4,
}

impl DType {
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => Self::F16,
            2 => Self::I8,
            3 => Self::I32,
            4 => Self::I64,
            _ => Self::F32,
        }
    }
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
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
    handle: *mut sys::tahoma_runtime_t,
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

        let mut handle: *mut sys::tahoma_runtime_t = ptr::null_mut();
        let rc = unsafe {
            sys::tahoma_runtime_compile(
                path_c.as_ptr(), device_c.as_ptr(),
                ptrs.as_ptr(), plugin.entries.len(), &mut handle,
            )
        };
        if rc != 0 {
            return Err(Error::Native(last_native_error()));
        }
        Ok(Self { handle })
    }

    pub fn reset_state(&mut self) -> Result<()> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        unsafe {
            let rc = sys::tahoma_runtime_reset_state(self.handle);
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
            sys::tahoma_runtime_input_count(self.handle)
        }
    }

    pub fn output_count(&self) -> usize {
        #[cfg(not(feature = "openvino"))]
        return 0;
        #[cfg(feature = "openvino")]
        unsafe {
            sys::tahoma_runtime_output_count(self.handle)
        }
    }

    #[cfg(feature = "openvino")]
    fn name_at(&self, getter: unsafe extern "C" fn(
        *mut sys::tahoma_runtime_t, usize, *mut c_char, usize, *mut usize,
    ) -> c_int, idx: usize) -> Result<String> {
        unsafe {
            let mut needed: usize = 0;
            let rc = getter(self.handle, idx, ptr::null_mut(), 0, &mut needed);
            if rc != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut buf = vec![0u8; needed + 1];
            let rc = getter(self.handle, idx, buf.as_mut_ptr() as *mut c_char,
                            buf.len(), &mut needed);
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
        self.name_at(sys::tahoma_runtime_input_name, idx)
    }
    pub fn output_name(&self, idx: usize) -> Result<String> {
        #[cfg(not(feature = "openvino"))]
        return Err(Error::Stub);
        #[cfg(feature = "openvino")]
        self.name_at(sys::tahoma_runtime_output_name, idx)
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
    pub fn set_input(&mut self, name: &str, dtype: DType, shape: &[usize], data: &[u8]) -> Result<()> {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = (name, dtype, shape, data);
            return Err(Error::Stub);
        }
        #[cfg(feature = "openvino")]
        unsafe {
            let name_c = cstr(name)?;
            let rc = sys::tahoma_runtime_set_input(
                self.handle, name_c.as_ptr(),
                dtype as u32, shape.as_ptr(), shape.len(),
                data.as_ptr(), data.len(),
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
            let rc = sys::tahoma_runtime_infer(self.handle);
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
            if sys::tahoma_runtime_output_rank(self.handle, idx, &mut rank) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut shape = vec![0usize; rank];
            if sys::tahoma_runtime_output_shape(self.handle, idx, shape.as_mut_ptr(), rank) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut dtype_code: u32 = 0;
            if sys::tahoma_runtime_output_dtype(self.handle, idx, &mut dtype_code) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut byte_size: usize = 0;
            if sys::tahoma_runtime_output_byte_size(self.handle, idx, &mut byte_size) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            let mut buf = vec![0u8; byte_size];
            if sys::tahoma_runtime_output_copy(self.handle, idx, buf.as_mut_ptr(), byte_size) != 0 {
                return Err(Error::Native(last_native_error()));
            }
            Ok((DType::from_code(dtype_code), shape, buf))
        }
    }
}

#[cfg(feature = "openvino")]
impl Drop for Runtime {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { sys::tahoma_runtime_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
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
}
