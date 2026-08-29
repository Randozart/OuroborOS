use anyhow::Result;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::ptr;

/// Generated llama bindings (re-exported for verification tests).
pub mod ffi {
    #![allow(dead_code, non_snake_case, non_camel_case_types, non_upper_case_globals, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[allow(unused_imports)]
use ffi as bindings_unused;

mod bindings {
    pub use crate::ffi::*;
}

use bindings::*;

/// Sampling configuration for generation.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    /// Temperature; 0.0 means greedy decoding.
    pub temp: f32,
    /// Top-k cutoff; 0 disables.
    pub top_k: i32,
    /// Top-p nucleus; >=1.0 disables.
    pub top_p: f32,
    /// RNG seed for distribution sampler.
    pub seed: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temp: 0.0, top_k: 0, top_p: 1.0, seed: 42 }
    }
}

impl SamplingParams {
    /// Deterministic greedy decoding.
    pub fn greedy() -> Self {
        Self::default()
    }

    /// Typical creative defaults for small ternary models.
    pub fn creative() -> Self {
        Self { temp: 0.8, top_k: 40, top_p: 0.95, seed: 42 }
    }
}

/// A graph-node output captured from a reference decode (oracle harness).
#[derive(Debug, Clone)]
pub struct CapturedNode {
    pub name: String,
    pub data: Vec<f32>,
}

struct Capture {
    nodes: Vec<CapturedNode>,
}

unsafe extern "C" fn capture_cb(
    t: *mut bindings::ggml_tensor,
    is_add: bool,
    ud: *mut std::os::raw::c_void,
) -> bool {
    if t.is_null() {
        return true;
    }
    if is_add {
        // claim every node: forces per-node compute + post-eval hook
        return true;
    }
    if (*t).type_ != bindings::ggml_type_GGML_TYPE_F32 || !bindings::ggml_is_contiguous(t) {
        return true;
    }
    let nb = bindings::ggml_nbytes(t);
    if nb == 0 || nb > 64 * 1024 * 1024 {
        return true;
    }
    let cap = &mut *(ud as *mut Capture);
    let mut buf = vec![0u8; nb];
    bindings::ggml_backend_tensor_get(
        t,
        buf.as_mut_ptr() as *mut std::os::raw::c_void,
        0,
        nb,
    );
    let name = std::ffi::CStr::from_ptr(bindings::ggml_get_name(t))
        .to_string_lossy()
        .into_owned();
    cap.nodes.push(CapturedNode {
        name,
        data: buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
    });
    true
}

/// Safe wrapper around a loaded BitNet model + inference context.
pub struct BitNetModel {
    model: *mut llama_model,
    ctx: *mut llama_context,
    n_ctx: u32,
}

/// Token id type alias for callers.
pub type LlamaToken = i32;

unsafe impl Send for BitNetModel {}

impl BitNetModel {
    /// Load a BitNet GGUF model and create an inference context.
    pub fn load(model_path: &str, n_ctx: u32, n_threads: u32) -> Result<Self> {
        unsafe {
            llama_backend_init();

            let mut mparams = llama_model_default_params();
            mparams.n_gpu_layers = 0;
            mparams.use_mmap = true;

            let c_path = CString::new(model_path)?;
            let model = llama_model_load_from_file(c_path.as_ptr(), mparams);
            if model.is_null() {
                anyhow::bail!("Failed to load model: {}", model_path);
            }

            let mut cparams = llama_context_default_params();
            cparams.n_ctx = n_ctx;
            cparams.n_batch = 512;
            cparams.n_ubatch = 512;
            cparams.n_threads = n_threads as c_int;
            cparams.n_threads_batch = n_threads as c_int;

            let ctx = llama_init_from_model(model, cparams);
            if ctx.is_null() {
                llama_model_free(model);
                anyhow::bail!("Failed to create context");
            }

            let actual_n_ctx = llama_n_ctx(ctx);

            Ok(Self {
                model,
                ctx,
                n_ctx: actual_n_ctx,
            })
        }
    }

    /// Load tokenizer/vocab only — no weights, no context. For tokenize/detok.
    pub fn load_vocab_only(model_path: &str) -> Result<Self> {
        unsafe {
            llama_backend_init();
            let mut mparams = llama_model_default_params();
            mparams.vocab_only = true;
            let c_path = CString::new(model_path)?;
            let model = llama_model_load_from_file(c_path.as_ptr(), mparams);
            if model.is_null() {
                anyhow::bail!("Failed to load vocab-only model: {}", model_path);
            }
            Ok(Self { model, ctx: std::ptr::null_mut(), n_ctx: 0 })
        }
    }

    /// Decode prompt tokens with the scheduler's per-node eval callback
    /// enabled; returns every f32 node output (name, data) in compute order.
    /// This is the ORACLE HARNESS: differential reference for the Rust forward.
    pub fn decode_capture(&self, tokens: &[llama_token]) -> Result<Vec<CapturedNode>> {
        unsafe {
            if tokens.is_empty() {
                anyhow::bail!("empty tokens");
            }
            let mut cap = Capture { nodes: Vec::new() };

            let mut cparams = llama_context_default_params();
            cparams.n_ctx = (tokens.len() + 8) as u32;
            cparams.n_batch = tokens.len() as u32;
            cparams.n_ubatch = tokens.len() as u32;
            cparams.n_threads = 4;
            cparams.n_threads_batch = 4;
            cparams.cb_eval = Some(capture_cb);
            cparams.cb_eval_user_data = &mut cap as *mut Capture as *mut std::os::raw::c_void;

            let cctx = llama_init_from_model(self.model, cparams);
            if cctx.is_null() {
                anyhow::bail!("capture ctx init failed");
            }

            let mut toks = tokens.to_vec();
            let batch = llama_batch_get_one(toks.as_mut_ptr(), toks.len() as c_int);
            let ret = llama_decode(cctx, batch);
            llama_free(cctx);
            if ret != 0 {
                anyhow::bail!("capture decode failed: {}", ret);
            }
            Ok(cap.nodes)
        }
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Clear KV, prefill tokens, return logits after the last position.
    pub fn logits_for_tokens(&self, tokens: &[llama_token]) -> Result<Vec<f32>> {
        unsafe {
            self.reset_context();
            if tokens.is_empty() {
                anyhow::bail!("empty tokens");
            }
            let mut toks = tokens.to_vec();
            let batch = llama_batch_get_one(toks.as_mut_ptr(), toks.len() as c_int);
            if llama_decode(self.ctx, batch) != 0 {
                anyhow::bail!("decode failed");
            }
            let n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(self.model)) as usize;
            let ptr = llama_get_logits_ith(self.ctx, -1);
            if ptr.is_null() {
                anyhow::bail!("no logits");
            }
            Ok(std::slice::from_raw_parts(ptr, n_vocab).to_vec())
        }
    }

    pub fn model_ptr(&self) -> *mut llama_model {
        self.model
    }

    /// Tokenize text into token IDs.
    pub fn tokenize(&self, text: &str, add_special: bool) -> Vec<llama_token> {
        unsafe {
            let vocab = llama_model_get_vocab(self.model);
            let c_text = match CString::new(text) {
                Ok(c) => c,
                Err(_) => return Vec::new(),
            };
            let text_len = text.len() as c_int;

            let need = llama_tokenize(
                vocab,
                c_text.as_ptr(),
                text_len,
                ptr::null_mut(),
                0,
                add_special,
                false,
            );
            if need == 0 {
                return Vec::new();
            }
            // Negative return = required buffer size (llama.cpp convention)
            let cap = if need < 0 { (-need) as usize } else { need as usize };

            let mut tokens = vec![0 as llama_token; cap];
            let got = llama_tokenize(
                vocab,
                c_text.as_ptr(),
                text_len,
                tokens.as_mut_ptr(),
                cap as c_int,
                add_special,
                false,
            );
            if got < 0 {
                return Vec::new();
            }
            tokens.truncate(got as usize);
            tokens
        }
    }

    /// Detokenize a single token to text.
    pub fn token_to_piece(&self, token: llama_token) -> String {
        unsafe {
            let vocab = llama_model_get_vocab(self.model);
            let mut buf = vec![0u8; 256];
            let n = llama_token_to_piece(vocab, token, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int, 0, false);
            if n <= 0 {
                return String::new();
            }
            buf.truncate(n as usize);
            String::from_utf8_lossy(&buf).into_owned()
        }
    }

    /// Run one decode step on a single token, return sampled token.
    ///
    /// # Safety
    /// `smpl` must be a live sampler chain pointer.
    pub unsafe fn step(&self, smpl: *mut llama_sampler, mut last_token: llama_token) -> Result<llama_token> {
        unsafe {
            let batch = llama_batch_get_one(&mut last_token, 1);
            let ret = llama_decode(self.ctx, batch);
            if ret != 0 {
                anyhow::bail!("llama_decode failed: {}", ret);
            }
            let tok = llama_sampler_sample(smpl, self.ctx, -1);
            Ok(tok)
        }
    }

    /// Build a sampler chain from params: greedy, or top-k/top-p/temp/dist.
    unsafe fn build_sampler(params: &SamplingParams) -> *mut llama_sampler {
        let chain_params = llama_sampler_chain_default_params();
        let smpl = llama_sampler_chain_init(chain_params);
        if params.temp <= 0.0 {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
            return smpl;
        }
        if params.top_k > 0 {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
        }
        if params.top_p < 1.0 {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));
        smpl
    }

    /// Clear the KV cache so consecutive generations stay independent.
    pub fn reset_context(&self) {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            if !mem.is_null() {
                llama_memory_clear(mem, false);
            }
        }
    }

    /// Generate text autoregressively with greedy sampling.
    pub fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String> {
        self.generate_with(prompt, max_tokens, &SamplingParams::greedy())
    }

    /// Generate text autoregressively with explicit sampling.
    pub fn generate_with(
        &self,
        prompt: &str,
        max_tokens: u32,
        params: &SamplingParams,
    ) -> Result<String> {
        unsafe {
            let mut tokens = self.tokenize(prompt, true);
            if tokens.is_empty() {
                anyhow::bail!("Failed to tokenize prompt");
            }

            self.reset_context();
            let smpl = Self::build_sampler(params);

            let vocab = llama_model_get_vocab(self.model);
            let mut out = String::new();

            // Prefill: all prompt tokens at once.
            let prefill = llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as c_int);
            if llama_decode(self.ctx, prefill) != 0 {
                llama_sampler_free(smpl);
                anyhow::bail!("prefill decode failed");
            }
            let mut cur = llama_sampler_sample(smpl, self.ctx, -1);

            for _ in 0..max_tokens {
                if llama_vocab_is_eog(vocab, cur) {
                    break;
                }
                out.push_str(&self.token_to_piece(cur));
                match self.step(smpl, cur) {
                    Ok(next) => cur = next,
                    Err(_) => break,
                }
            }

            llama_sampler_free(smpl);
            Ok(out)
        }
    }

    /// Benchmark prompt processing + generation, return tok/s for both phases.
    pub fn benchmark(&self, prompt: &str, n_gen: u32) -> Result<(f64, f64)> {
        let tokens = self.tokenize(prompt, true);
        if tokens.is_empty() {
            anyhow::bail!("Failed to tokenize prompt");
        }

        let t0 = std::time::Instant::now();
        let out = self.generate(prompt, n_gen)?;
        let elapsed = t0.elapsed().as_secs_f64();

        let pp = tokens.len() as f64 / elapsed;
        let tg = if n_gen > 0 { n_gen as f64 / elapsed } else { 0.0 };
        let _ = out;
        Ok((pp, tg))
    }
}

impl Drop for BitNetModel {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
            llama_backend_free();
        }
    }
}
