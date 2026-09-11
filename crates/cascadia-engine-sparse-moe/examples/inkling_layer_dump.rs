//! Per-layer residual-stream dump for the Inkling real-weight parity harness.
//!
//! Loads the first `K` layers of an `export_inkling.py` export (embed + embed
//! norm always; the head only when `K == num_layers`), runs a token list
//! through them twice — token by token (the decode path,
//! [`Layer::forward_token`]) and, after a reset, as one batched prefill
//! ([`Layer::forward_prefill`]) — and writes every intermediate residual-stream
//! tensor to a safetensors file. `tools/inkling_ref/real_layer_parity.py`
//! compares that file against transformers' own `InklingForCausalLM` built
//! from the SAME (int4-dequantised) weights, so the shell's per-layer math can
//! be proven on the real 975B checkpoint without ever running HF end to end.
//!
//! ```text
//! cargo run -p cascadia-engine-sparse-moe --release --example inkling_layer_dump -- \
//!     --export /data/inkling-int4 --layers 3 --tokens 1,2,3,4 \
//!     --out /data/inkling_dump_k3.safetensors [--experts eager|mmap] [--max-seq N]
//! ```
//!
//! Tensors written (F32 unless noted; `T` tokens, `H` hidden):
//! - `tokens` (I64 `[T]`) — the ids, so the Python side needs no re-typing;
//! - `embed_out` `[T, H]` — `rmsnorm(embed[t], embed_norm)`, layer 0's input;
//! - `layer{L}_out_decode` / `layer{L}_out_prefill` `[T, H]` for `L in 0..K`;
//! - `logits_decode` / `logits_prefill` `[T, unpadded_vocab]` when the head is
//!   loaded (`K == num_layers`).
//!
//! `--experts` defaults to the runner's rule (mmap for > 32 experts, eager
//! otherwise); `--max-seq` sizes the global layers' KV rows (default
//! `max(INKLING_DEFAULT_MAX_SEQ, T)`). Per-layer timings for both paths and
//! the decode-vs-prefill max |Δ| (expected 0 — the batched path is bit-exact)
//! are printed.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cascadia_engine_sparse_moe::dsv4::loader::ExpertsMode;
use cascadia_engine_sparse_moe::inkling::loader::{load_stage, read_manifest, InklingStage};
use cascadia_engine_sparse_moe::inkling::model::argmax;
use cascadia_engine_sparse_moe::inkling::rmsnorm_f32;
use cascadia_engine_sparse_moe::inkling::stage::INKLING_DEFAULT_MAX_SEQ;

const USAGE: &str = "usage: inkling_layer_dump --export DIR --layers K --tokens 1,2,3 \
                     [--out dump.safetensors] [--experts eager|mmap] [--max-seq N]";

struct Args {
    export: PathBuf,
    layers: usize,
    tokens: Vec<u32>,
    out: PathBuf,
    experts: Option<ExpertsMode>,
    max_seq: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut export = None;
    let mut layers = None;
    let mut tokens = None;
    let mut out = PathBuf::from("inkling_dump.safetensors");
    let mut experts = None;
    let mut max_seq = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| -> Result<String, String> {
            it.next()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--export" => export = Some(PathBuf::from(value("--export")?)),
            "--layers" => {
                layers = Some(
                    value("--layers")?
                        .parse::<usize>()
                        .map_err(|e| format!("--layers: {e}"))?,
                )
            }
            "--tokens" => {
                let raw = value("--tokens")?;
                let ids: Result<Vec<u32>, _> = raw
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u32>())
                    .collect();
                tokens = Some(ids.map_err(|e| format!("--tokens {raw:?}: {e}"))?);
            }
            "--out" => out = PathBuf::from(value("--out")?),
            "--experts" => {
                experts = Some(match value("--experts")?.as_str() {
                    "eager" => ExpertsMode::Eager,
                    "mmap" => ExpertsMode::Mmap,
                    other => return Err(format!("--experts {other:?}: expected eager | mmap")),
                })
            }
            "--max-seq" => {
                max_seq = Some(
                    value("--max-seq")?
                        .parse::<usize>()
                        .map_err(|e| format!("--max-seq: {e}"))?,
                )
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    let tokens = tokens.ok_or_else(|| format!("--tokens is required\n{USAGE}"))?;
    if tokens.is_empty() {
        return Err("--tokens: need at least one token id".into());
    }
    Ok(Args {
        export: export.ok_or_else(|| format!("--export is required\n{USAGE}"))?,
        layers: layers.ok_or_else(|| format!("--layers is required\n{USAGE}"))?,
        tokens,
        out,
        experts,
        max_seq,
    })
}

/// `rmsnorm(embed[t], embed_norm)` — what `InklingRunner::embed_token` /
/// `Model::embed_token` compute (the runner cannot be used here: with a single
/// rank it always loads the head, and it does not expose its layers mutably).
fn embed_token(stage: &InklingStage, token: u32) -> Vec<f32> {
    let m = &stage.manifest;
    let (table, norm) = stage
        .embed
        .as_ref()
        .expect("stage owns layer 0, so it carries the embed");
    assert!(
        (token as usize) < m.vocab_size,
        "token {token} >= vocab {}",
        m.vocab_size
    );
    let mut x = table.row(token as usize, m.hidden_size);
    rmsnorm_f32(&mut x, norm, m.rms_norm_eps);
    x
}

/// One tensor of the dump: raw little-endian bytes + safetensors header fields.
struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl Tensor {
    fn f32(name: impl Into<String>, shape: Vec<usize>, v: &[f32]) -> Self {
        assert_eq!(v.len(), shape.iter().product::<usize>(), "tensor shape");
        Self {
            name: name.into(),
            dtype: "F32",
            shape,
            bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    fn i64(name: impl Into<String>, v: &[i64]) -> Self {
        Self {
            name: name.into(),
            dtype: "I64",
            shape: vec![v.len()],
            bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

/// Minimal safetensors writer: `u64 LE header length`, the JSON header (padded
/// with spaces to a multiple of 8, the reference implementation's convention),
/// then the concatenated tensor bytes. Readable by `safetensors.torch.load_file`
/// and by the crate's `dsv4::st::StFile`.
fn write_safetensors(
    path: &Path,
    tensors: &[Tensor],
    metadata: &[(&str, String)],
) -> std::io::Result<()> {
    let mut header = serde_json::Map::new();
    let meta: serde_json::Map<String, serde_json::Value> = metadata
        .iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.clone())))
        .collect();
    header.insert("__metadata__".into(), serde_json::Value::Object(meta));
    let mut off = 0usize;
    for t in tensors {
        let end = off + t.bytes.len();
        header.insert(
            t.name.clone(),
            serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, end]}),
        );
        off = end;
    }
    let mut hdr = serde_json::to_vec(&serde_json::Value::Object(header))?;
    while hdr.len() % 8 != 0 {
        hdr.push(b' ');
    }
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(&(hdr.len() as u64).to_le_bytes())?;
    f.write_all(&hdr)?;
    for t in tensors {
        f.write_all(&t.bytes)?;
    }
    f.flush()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let m = read_manifest(&args.export)?;
    let k = args.layers;
    if k == 0 || k > m.num_layers {
        return Err(format!(
            "--layers {k} out of range: the export has {} layers (need 1..={})",
            m.num_layers, m.num_layers
        )
        .into());
    }
    if let Some(bad) = args.tokens.iter().find(|&&t| t as usize >= m.vocab_size) {
        return Err(format!("token {bad} >= vocab_size {}", m.vocab_size).into());
    }
    let t_len = args.tokens.len();
    let hidden = m.hidden_size;
    let max_seq = args
        .max_seq
        .unwrap_or_else(|| INKLING_DEFAULT_MAX_SEQ.max(t_len));
    if max_seq < t_len {
        return Err(format!("--max-seq {max_seq} < {t_len} tokens").into());
    }
    let mode = args.experts.unwrap_or(if m.num_experts > 32 {
        ExpertsMode::Mmap
    } else {
        ExpertsMode::Eager
    });
    let last = k == m.num_layers;
    println!(
        "[inkling_layer_dump] export={} layers=0..{k} of {} (head: {}) tokens={t_len} hidden={hidden} \
         experts={mode:?} max_seq={max_seq}",
        args.export.display(),
        m.num_layers,
        if last { "yes" } else { "no" }
    );

    let t_load = Instant::now();
    let mut stage = load_stage(&args.export, max_seq, 0, k, true, last, mode)?;
    let cache_bytes: usize = stage.layers.iter().map(|l| l.cache_bytes()).sum();
    println!(
        "[inkling_layer_dump] loaded {} layer(s) in {:.1}s (sequence state {} MiB)",
        stage.layers.len(),
        t_load.elapsed().as_secs_f64(),
        cache_bytes >> 20
    );

    // ---- embeddings (shared by both paths) ----
    let mut embed_out = Vec::with_capacity(t_len * hidden);
    for &t in &args.tokens {
        embed_out.extend(embed_token(&stage, t));
    }

    // ---- decode path: token by token through every layer ----
    let mut dec_out: Vec<Vec<f32>> = vec![Vec::with_capacity(t_len * hidden); k];
    let mut dec_time = vec![Duration::ZERO; k];
    for l in &mut stage.layers {
        l.reset();
    }
    let t_dec = Instant::now();
    for r in 0..t_len {
        let mut x = embed_out[r * hidden..(r + 1) * hidden].to_vec();
        for (li, layer) in stage.layers.iter_mut().enumerate() {
            let t0 = Instant::now();
            x = layer.forward_token(&x);
            dec_time[li] += t0.elapsed();
            dec_out[li].extend_from_slice(&x);
        }
    }
    let dec_total = t_dec.elapsed();
    // The head is `Head::logits` — the same code `Model` and the staged
    // runner use (final norm, `/ mup`, unembed, sliced to `unpadded_vocab`).
    let mut logits_dec = Vec::new();
    if let Some(head) = stage.head.as_ref() {
        let t0 = Instant::now();
        for r in 0..t_len {
            let x = &dec_out[k - 1][r * hidden..(r + 1) * hidden];
            logits_dec.extend(head.logits(x));
        }
        println!(
            "[inkling_layer_dump] head (decode rows) {:.1} ms for {t_len} positions",
            ms(t0.elapsed())
        );
    }

    // ---- prefill path: reset, then one batched pass per layer ----
    for l in &mut stage.layers {
        l.reset();
    }
    let mut pre_out: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut pre_time = Vec::with_capacity(k);
    let t_pre = Instant::now();
    let mut xs = embed_out.clone();
    for layer in stage.layers.iter_mut() {
        let t0 = Instant::now();
        xs = layer.forward_prefill(&xs, t_len);
        pre_time.push(t0.elapsed());
        pre_out.push(xs.clone());
    }
    let pre_total = t_pre.elapsed();
    let mut logits_pre = Vec::new();
    if let Some(head) = stage.head.as_ref() {
        for r in 0..t_len {
            let x = &pre_out[k - 1][r * hidden..(r + 1) * hidden];
            logits_pre.extend(head.logits(x));
        }
    }

    // ---- report ----
    println!(
        "[inkling_layer_dump] {:>5} {:<7} {:<5} {:>12} {:>10} {:>12} {:>14}",
        "layer", "attn", "mlp", "decode ms", "ms/tok", "prefill ms", "dec-vs-pre |Δ|"
    );
    for li in 0..k {
        let attn = if m.is_sliding(li) {
            "sliding"
        } else {
            "global"
        };
        let mlp = if m.dense_layers.contains(&li) {
            "dense"
        } else {
            "moe"
        };
        println!(
            "[inkling_layer_dump] {li:>5} {attn:<7} {mlp:<5} {:>12.2} {:>10.3} {:>12.2} {:>14.3e}",
            ms(dec_time[li]),
            ms(dec_time[li]) / t_len as f64,
            ms(pre_time[li]),
            max_abs_diff(&dec_out[li], &pre_out[li])
        );
    }
    println!(
        "[inkling_layer_dump] totals: decode {:.1} ms ({:.2} ms/tok), prefill {:.1} ms ({:.2} ms/tok)",
        ms(dec_total),
        ms(dec_total) / t_len as f64,
        ms(pre_total),
        ms(pre_total) / t_len as f64
    );
    if !logits_dec.is_empty() {
        let v = m.unpadded_vocab();
        let am: Vec<usize> = (0..t_len)
            .map(|r| argmax(&logits_pre[r * v..(r + 1) * v]))
            .collect();
        println!(
            "[inkling_layer_dump] logits: dec-vs-pre |Δ| {:.3e}; per-position argmax {am:?}",
            max_abs_diff(&logits_dec, &logits_pre)
        );
    }

    // ---- write ----
    let mut tensors = vec![
        Tensor::i64(
            "tokens",
            &args.tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
        ),
        Tensor::f32("embed_out", vec![t_len, hidden], &embed_out),
    ];
    for li in 0..k {
        tensors.push(Tensor::f32(
            format!("layer{li}_out_decode"),
            vec![t_len, hidden],
            &dec_out[li],
        ));
        tensors.push(Tensor::f32(
            format!("layer{li}_out_prefill"),
            vec![t_len, hidden],
            &pre_out[li],
        ));
    }
    if !logits_dec.is_empty() {
        let v = m.unpadded_vocab();
        tensors.push(Tensor::f32("logits_decode", vec![t_len, v], &logits_dec));
        tensors.push(Tensor::f32("logits_prefill", vec![t_len, v], &logits_pre));
    }
    let metadata = [
        ("format", "inkling_layer_dump".to_string()),
        ("export", args.export.display().to_string()),
        ("layers", k.to_string()),
        ("num_layers", m.num_layers.to_string()),
        ("experts", format!("{mode:?}").to_lowercase()),
        ("max_seq", max_seq.to_string()),
    ];
    write_safetensors(&args.out, &tensors, &metadata)?;
    let bytes: usize = tensors.iter().map(|t| t.bytes.len()).sum();
    println!(
        "[inkling_layer_dump] wrote {} tensors ({:.1} MiB) to {}",
        tensors.len(),
        bytes as f64 / (1024.0 * 1024.0),
        args.out.display()
    );
    Ok(())
}
