//! Replay the fleet's own outputs through the speculation drafter, offline.
//!
//!     lm_draft_replay TOKENIZER.json CORPUS.jsonl [HOST:PORT [qwen|llama3|plain]] [--docs N] [--wait MS] [--think MS]
//!
//! For every response of the corpus (`autolab/bench/collect.py`): at each
//! position the drafter ([`Draft::propose_one`]: drafter model first when an
//! address is given, n-gram tables otherwise) names the next token; the true
//! token is then appended, as the pipeline would after verifying it. Prints
//! the share of right guesses, by source, and how long a guess took to get.
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cascadia_engine_sparse_moe::lm_draft::{LmConfig, LmTemplate};
use cascadia_engine_sparse_moe::ngram_draft::{Draft, SharedNgrams};
use tokenizers::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let pos: Vec<&String> = {
        let mut skip = false;
        args.iter()
            .filter(|a| {
                if skip {
                    skip = false;
                    return false;
                }
                if a.starts_with("--") {
                    skip = true;
                    return false;
                }
                true
            })
            .collect()
    };
    if pos.len() < 2 {
        eprintln!("usage: lm_draft_replay TOKENIZER.json CORPUS.jsonl [HOST:PORT [qwen|llama3|plain]] [--docs N] [--wait MS] [--think MS]");
        std::process::exit(2);
    }
    let tok = Arc::new(Tokenizer::from_file(pos[0]).expect("tokenizer"));
    let docs: usize = flag("--docs").and_then(|v| v.parse().ok()).unwrap_or(24);
    let wait = Duration::from_millis(flag("--wait").and_then(|v| v.parse().ok()).unwrap_or(40));
    // What rank 0 spends computing a frame between learning a token and wanting the next guess.
    let think = Duration::from_millis(flag("--think").and_then(|v| v.parse().ok()).unwrap_or(0));
    let lm = pos.get(2).map(|addr| LmConfig {
        addr: addr.to_string(),
        template: pos
            .get(3)
            .map(|t| LmTemplate::parse(t))
            .unwrap_or(LmTemplate::Qwen),
        n_predict: 32,
    });
    let corpus: Vec<serde_json::Value> = std::fs::read_to_string(pos[1])
        .expect("corpus")
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let shared = Arc::new(Mutex::new(SharedNgrams::default()));
    let enc = |s: &str| -> Vec<i64> {
        tok.encode(s, false)
            .expect("encode")
            .get_ids()
            .iter()
            .map(|&t| t as i64)
            .collect()
    };
    // The tables learn from the documents that are NOT replayed.
    for d in corpus.iter().skip(docs) {
        let mut seq = enc(d["prompt"].as_str().unwrap_or(""));
        seq.extend(enc(d["text"].as_str().unwrap_or("")));
        shared.lock().expect("lock").learn(&seq);
    }
    let (mut n, mut hits, mut proposed) = (0u64, 0u64, 0u64);
    let (mut lm_n, mut lm_hits, mut ng_n, mut ng_hits) = (0u64, 0u64, 0u64, 0u64);
    let mut ask = Duration::ZERO;
    let started = Instant::now();
    for d in corpus.iter().take(docs) {
        let prompt = enc(d["prompt"].as_str().unwrap_or(""));
        let text = enc(d["text"].as_str().unwrap_or(""));
        let mut draft = Draft::new().with_draft_k(1).with_shared(shared.clone());
        if let Some(cfg) = lm.clone() {
            draft = draft.with_lm(cfg, tok.clone());
        }
        draft.warm_with_prompt(&prompt);
        for (k, &truth) in text.iter().enumerate() {
            if k > 0 {
                let before = draft.sources();
                let t0 = Instant::now();
                let g = draft.propose_one(wait);
                ask += t0.elapsed();
                let after = draft.sources();
                n += 1;
                if let Some(g) = g {
                    proposed += 1;
                    let hit = g == truth;
                    hits += u64::from(hit);
                    if after.0 > before.0 {
                        lm_n += 1;
                        lm_hits += u64::from(hit);
                    } else {
                        ng_n += 1;
                        ng_hits += u64::from(hit);
                    }
                }
            }
            draft.append(truth);
            if !think.is_zero() {
                draft.poke();
                std::thread::sleep(think);
            }
        }
    }
    let pct = |a: u64, b: u64| {
        if b == 0 {
            0.0
        } else {
            100.0 * a as f64 / b as f64
        }
    };
    println!(
        "docs {docs}  positions {n}  proposed {proposed}  right {hits}  a = {:.1} %  (model: {lm_n} guesses, {:.1} % right; tables: {ng_n} guesses, {:.1} % right)",
        pct(hits, n),
        pct(lm_hits, lm_n),
        pct(ng_hits, ng_n)
    );
    println!(
        "asking took {:.1} ms per position on average; whole replay {:.1} s",
        ask.as_secs_f64() * 1e3 / n.max(1) as f64,
        started.elapsed().as_secs_f64()
    );
}
