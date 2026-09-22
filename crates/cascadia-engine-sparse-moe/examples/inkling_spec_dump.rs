//! Hidden-state dump for the speculative-decoding studies (handoff E1, the
//! minimum E2/E4 need): greedy generation over a prompt file, one process for
//! all prompts (one load), one safetensors per prompt.
//!
//! ```text
//! cargo run -p cascadia-engine-sparse-moe --release --example inkling_spec_dump -- \
//!     --export /data/inkling-int4 --prompts prompts.jsonl --out-dir /data/spec-dump \
//!     [--generate 160] [--boundary 5,11,17,23,29,35,41,47,53,59] [--experts eager|mmap] \
//!     [--max-seq N] [--limit K]
//! ```
//!
//! `prompts.jsonl`: one `{"i": 7, "family": 7, "text": "<rendered chat prompt>"}` per line
//! (or `"ids": [..]` instead of `"text"`). `text` is tokenised exactly like the engine does
//! (`tokenizers` crate, `<export>/tokenizer.json`, `encode(text, add_special_tokens = true)`).
//!
//! Per prompt (`P` prompt tokens, `N` generated, `T = P + N`, `H` hidden), `p{i:04}.safetensors`:
//! - `tokens` I64 `[T]`: prompt ids then the greedy ids;
//! - `embed_out` F32 `[T, H]`: `rmsnorm(embed[t], embed_norm)`, layer 0's input;
//! - `final_out` F32 `[T-1, H]`: the LAST layer's output (the residual the head receives,
//!   BEFORE the final norm) at every position that was run (the last generated token is
//!   never fed back);
//! - `argmax` I64 `[T-1]`: the head's greedy token at each of those positions
//!   (`argmax[t] == tokens[t+1]` for `t >= P-1` by construction);
//! - `layer{L}_out` F32 `[N, H]` for each `--boundary` layer L: that layer's output at the
//!   generated positions `P-1 ..= T-2` (the positions whose next token the model chose).
//! Metadata: `prompt_len`, `generated`, `i`, `family`. `gen.jsonl` in the same directory gets
//! one line per prompt with the decoded text and the timings. A prompt whose file already
//! exists is skipped (resume). Generation stops early at an EOS id.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use cascadia_engine_sparse_moe::dsv4::loader::ExpertsMode;
use cascadia_engine_sparse_moe::inkling::loader::{
    load_stage, read_manifest, ExpertSet, InklingStage,
};
use cascadia_engine_sparse_moe::inkling::model::argmax;
use cascadia_engine_sparse_moe::inkling::rmsnorm_f32;
use cascadia_engine_sparse_moe::inkling::stage::INKLING_DEFAULT_MAX_SEQ;

const USAGE: &str = "usage: inkling_spec_dump --export DIR --prompts prompts.jsonl --out-dir DIR \
                     [--generate 160] [--boundary 5,11,...] [--experts eager|mmap] [--max-seq N] [--limit K]";

struct Args {
    export: PathBuf,
    prompts: PathBuf,
    out_dir: PathBuf,
    generate: usize,
    boundary: Vec<usize>,
    experts: Option<ExpertsMode>,
    max_seq: Option<usize>,
    limit: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut export = None;
    let mut prompts = None;
    let mut out_dir = None;
    let mut generate = 160usize;
    let mut boundary = vec![5, 11, 17, 23, 29, 35, 41, 47, 53, 59];
    let mut experts = None;
    let mut max_seq = None;
    let mut limit = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| -> Result<String, String> {
            it.next()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--export" => export = Some(PathBuf::from(value("--export")?)),
            "--prompts" => prompts = Some(PathBuf::from(value("--prompts")?)),
            "--out-dir" => out_dir = Some(PathBuf::from(value("--out-dir")?)),
            "--generate" => {
                generate = value("--generate")?
                    .parse()
                    .map_err(|e| format!("--generate: {e}"))?
            }
            "--boundary" => {
                let raw = value("--boundary")?;
                boundary = raw
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("--boundary {raw:?}: {e}"))?;
            }
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
                        .parse()
                        .map_err(|e| format!("--max-seq: {e}"))?,
                )
            }
            "--limit" => {
                limit = Some(
                    value("--limit")?
                        .parse()
                        .map_err(|e| format!("--limit: {e}"))?,
                )
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    if generate < 2 {
        return Err("--generate: need at least 2".into());
    }
    Ok(Args {
        export: export.ok_or_else(|| format!("--export is required\n{USAGE}"))?,
        prompts: prompts.ok_or_else(|| format!("--prompts is required\n{USAGE}"))?,
        out_dir: out_dir.ok_or_else(|| format!("--out-dir is required\n{USAGE}"))?,
        generate,
        boundary,
        experts,
        max_seq,
        limit,
    })
}

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

/// Same minimal safetensors writer as `inkling_layer_dump`.
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
    let tmp = path.with_extension("safetensors.tmp");
    {
        let mut f = BufWriter::new(File::create(&tmp)?);
        f.write_all(&(hdr.len() as u64).to_le_bytes())?;
        f.write_all(&hdr)?;
        for t in tensors {
            f.write_all(&t.bytes)?;
        }
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}

struct Prompt {
    i: i64,
    family: i64,
    ids: Vec<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cascadia_engine_sparse_moe::init_thread_pool();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let m = read_manifest(&args.export)?;
    let k = m.num_layers;
    let hidden = m.hidden_size;
    let vocab = m.unpadded_vocab();
    if let Some(bad) = args.boundary.iter().find(|&&l| l >= k) {
        return Err(format!("--boundary layer {bad} >= {k}").into());
    }
    std::fs::create_dir_all(&args.out_dir)?;

    // ---- prompts: tokenise like the engine (tokenizers crate, add_special_tokens = true) ----
    let tok = tokenizers::Tokenizer::from_file(args.export.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer.json: {e}"))?;
    let mut prompts = Vec::new();
    for line in BufReader::new(File::open(&args.prompts)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)?;
        let i = v.get("i").and_then(|x| x.as_i64()).unwrap_or(prompts.len() as i64);
        let family = v.get("family").and_then(|x| x.as_i64()).unwrap_or(-1);
        let ids: Vec<u32> = if let Some(arr) = v.get("ids").and_then(|x| x.as_array()) {
            arr.iter()
                .map(|x| x.as_u64().map(|u| u as u32).ok_or("ids: not an integer"))
                .collect::<Result<_, _>>()?
        } else {
            let text = v
                .get("text")
                .and_then(|x| x.as_str())
                .ok_or("prompt line needs \"text\" or \"ids\"")?;
            tok.encode(text, true)
                .map_err(|e| format!("encode: {e}"))?
                .get_ids()
                .to_vec()
        };
        if ids.is_empty() {
            return Err(format!("prompt {i}: no tokens").into());
        }
        if let Some(bad) = ids.iter().find(|&&t| t as usize >= m.vocab_size) {
            return Err(format!("prompt {i}: token {bad} >= vocab_size").into());
        }
        prompts.push(Prompt { i, family, ids });
    }
    if let Some(n) = args.limit {
        prompts.truncate(n);
    }
    let longest = prompts.iter().map(|p| p.ids.len()).max().unwrap_or(0) + args.generate;
    let max_seq = args
        .max_seq
        .unwrap_or_else(|| INKLING_DEFAULT_MAX_SEQ.max(longest));
    if max_seq < longest {
        return Err(format!("--max-seq {max_seq} < longest sequence {longest}").into());
    }
    let mode = args.experts.unwrap_or(if m.num_experts > 32 {
        ExpertsMode::Mmap
    } else {
        ExpertsMode::Eager
    });
    println!(
        "[inkling_spec_dump] export={} prompts={} generate={} boundary={:?} experts={mode:?} max_seq={max_seq}",
        args.export.display(),
        prompts.len(),
        args.generate,
        args.boundary
    );

    let t_load = Instant::now();
    let mut stage = load_stage(&args.export, max_seq, 0, k, true, true, mode, ExpertSet::All)?;
    println!(
        "[inkling_spec_dump] loaded {} layer(s) in {:.1}s",
        stage.layers.len(),
        t_load.elapsed().as_secs_f64()
    );

    let mut gen_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(args.out_dir.join("gen.jsonl"))?;
    let t_all = Instant::now();
    for (pi, p) in prompts.iter().enumerate() {
        let out = args.out_dir.join(format!("p{:04}.safetensors", p.i));
        if out.exists() {
            println!("[inkling_spec_dump] prompt {} exists, skipping", p.i);
            continue;
        }
        let plen = p.ids.len();
        for l in &mut stage.layers {
            l.reset();
        }
        let mut tokens: Vec<u32> = p.ids.clone();
        let mut embed_out: Vec<f32> = Vec::with_capacity((plen + args.generate) * hidden);
        for &t in &p.ids {
            embed_out.extend(embed_token(&stage, t));
        }
        let mut final_out: Vec<f32> = Vec::with_capacity((plen + args.generate) * hidden);
        let mut am: Vec<i64> = Vec::with_capacity(plen + args.generate);
        let mut bnd: Vec<Vec<f32>> = vec![Vec::with_capacity(args.generate * hidden); args.boundary.len()];

        // ---- prefill: all prompt rows, one batched pass per layer ----
        let t_pre = Instant::now();
        let mut xs = embed_out.clone();
        for (li, layer) in stage.layers.iter_mut().enumerate() {
            xs = layer.forward_prefill(&xs, plen);
            if let Some(bi) = args.boundary.iter().position(|&l| l == li) {
                bnd[bi].extend_from_slice(&xs[(plen - 1) * hidden..plen * hidden]);
            }
        }
        final_out.extend_from_slice(&xs);
        let head = stage.head.as_ref().expect("last layer loaded, so the head is");
        for r in 0..plen {
            let lg = head.logits(&xs[r * hidden..(r + 1) * hidden]);
            am.push(argmax(&lg[..vocab]) as i64);
        }
        let pre_s = t_pre.elapsed().as_secs_f64();

        // ---- greedy decode ----
        let t_dec = Instant::now();
        let mut next = *am.last().unwrap() as u32;
        let mut eos = false;
        for n in 0..args.generate {
            tokens.push(next);
            let e = embed_token(&stage, next);
            embed_out.extend_from_slice(&e);
            if m.eos_token_ids.contains(&next) {
                eos = true;
                break;
            }
            if n + 1 == args.generate {
                break; // the last generated token is never fed back
            }
            let mut x = e;
            for (li, layer) in stage.layers.iter_mut().enumerate() {
                x = layer.forward_token(&x);
                if let Some(bi) = args.boundary.iter().position(|&l| l == li) {
                    bnd[bi].extend_from_slice(&x);
                }
            }
            final_out.extend_from_slice(&x);
            let head = stage.head.as_ref().unwrap();
            let lg = head.logits(&x);
            next = argmax(&lg[..vocab]) as u32;
            am.push(next as i64);
        }
        let dec_s = t_dec.elapsed().as_secs_f64();
        let t_len = tokens.len();
        let n_gen = t_len - plen;
        let rows = am.len(); // positions that were run
        assert_eq!(final_out.len(), rows * hidden);
        assert_eq!(embed_out.len(), t_len * hidden);

        let mut tensors = vec![
            Tensor::i64("tokens", &tokens.iter().map(|&t| t as i64).collect::<Vec<_>>()),
            Tensor::f32("embed_out", vec![t_len, hidden], &embed_out),
            Tensor::f32("final_out", vec![rows, hidden], &final_out),
            Tensor::i64("argmax", &am),
        ];
        for (bi, &l) in args.boundary.iter().enumerate() {
            let r = bnd[bi].len() / hidden;
            tensors.push(Tensor::f32(format!("layer{l}_out"), vec![r, hidden], &bnd[bi]));
        }
        let metadata = [
            ("format", "inkling_spec_dump".to_string()),
            ("i", p.i.to_string()),
            ("family", p.family.to_string()),
            ("prompt_len", plen.to_string()),
            ("generated", n_gen.to_string()),
            ("eos", eos.to_string()),
        ];
        write_safetensors(&out, &tensors, &metadata)?;
        let text = tok
            .decode(&tokens[plen..], false)
            .unwrap_or_else(|e| format!("<decode error: {e}>"));
        let line = serde_json::json!({
            "i": p.i, "family": p.family, "prompt_len": plen, "generated": n_gen, "eos": eos,
            "prefill_s": pre_s, "decode_s": dec_s, "gen_ids": &tokens[plen..], "text": text,
        });
        writeln!(gen_log, "{line}")?;
        gen_log.flush()?;
        println!(
            "[inkling_spec_dump] {}/{} prompt {} family {}: P={plen} N={n_gen} prefill {:.1}s decode {:.1}s ({:.2} s/tok) elapsed {:.0}s",
            pi + 1,
            prompts.len(),
            p.i,
            p.family,
            pre_s,
            dec_s,
            dec_s / (n_gen.max(2) - 1) as f64,
            t_all.elapsed().as_secs_f64()
        );
    }
    println!(
        "[inkling_spec_dump] done: {} prompts in {:.0}s",
        prompts.len(),
        t_all.elapsed().as_secs_f64()
    );
    Ok(())
}
