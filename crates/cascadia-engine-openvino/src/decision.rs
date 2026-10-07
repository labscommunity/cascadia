//! Decision heads run where the final hidden states live (single box or the last pipeline rank).
//! See docs/decisions.md. Pure logic here; qwen36.rs drives it.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use cascadia_types::{DecisionReadout, HiddenNeed, ReadoutHead};
use serde::{Deserialize, Serialize};

pub struct Collector {
    need: HiddenNeed,
    hidden: usize,
    rows: BTreeMap<u32, Vec<f32>>,
    all: Vec<f32>,
}

/// Shape only: the gathered rows can be tens of MB, and a stray `{:?}` must not log them.
impl std::fmt::Debug for Collector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let need = match &self.need {
            HiddenNeed::None => "none".to_string(),
            HiddenNeed::All => "all".to_string(),
            HiddenNeed::Positions(p) => format!("positions({})", p.len()),
        };
        f.debug_struct("Collector")
            .field("need", &need)
            .field("hidden", &self.hidden)
            .field("rows", &self.rows.len())
            .field("all_len", &self.all.len())
            .finish()
    }
}

impl Collector {
    pub fn new(need: HiddenNeed, hidden: usize) -> Self {
        Self {
            need,
            hidden,
            rows: BTreeMap::new(),
            all: Vec::new(),
        }
    }

    pub fn wants_hidden(&self) -> bool {
        self.need != HiddenNeed::None
    }

    pub fn absorb(&mut self, t0: usize, n: usize, rows: &[f32]) -> Result<(), String> {
        if rows.len() < n * self.hidden {
            return Err(format!(
                "final_hidden has {} values for {n} rows of {}",
                rows.len(),
                self.hidden
            ));
        }
        match &self.need {
            HiddenNeed::None => {}
            HiddenNeed::All => {
                if t0 != self.len() {
                    return Err(format!(
                        "hidden rows arrive at {t0} but {} are held (prefix skipped?)",
                        self.len()
                    ));
                }
                self.all.extend_from_slice(&rows[..n * self.hidden])
            }
            HiddenNeed::Positions(want) => {
                for &p in want {
                    let p = p as usize;
                    if p >= t0 && p < t0 + n {
                        let r = (p - t0) * self.hidden;
                        self.rows
                            .insert(p as u32, rows[r..r + self.hidden].to_vec());
                    }
                }
            }
        }
        Ok(())
    }

    pub fn row(&self, p: u32) -> Option<&[f32]> {
        self.rows.get(&p).map(Vec::as_slice)
    }

    pub fn all(&self) -> &[f32] {
        &self.all
    }

    pub fn len(&self) -> usize {
        self.all.len() / self.hidden.max(1)
    }
}

pub enum HeadAssets {
    Verbalizer,
    Pointer {
        dp: usize,
        hidden: usize,
        q_w: Vec<f32>,
        q_b: Vec<f32>,
        k_w: Vec<f32>,
        k_b: Vec<f32>,
    },
    Joint {
        runtime: Runtime,
        lm_head: PathBuf,
        vocab: usize,
        hidden: usize,
    },
}

fn read_f32(path: &Path, expect: usize) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() != expect * 4 {
        return Err(format!(
            "{}: {} bytes, expected {}",
            path.display(),
            b.len(),
            expect * 4
        ));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl HeadAssets {
    pub fn load(model_dir: &Path, hidden: usize) -> Result<HeadAssets, String> {
        let dir = model_dir.join("decision");
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("head.json"))
                .map_err(|e| format!("decision/head.json: {e}"))?,
        )
        .map_err(|e| format!("decision/head.json: {e}"))?;
        match meta["kind"].as_str() {
            Some("verbalizer") => Ok(HeadAssets::Verbalizer),
            Some("pointer") => {
                let dp = meta["dp"].as_u64().ok_or("pointer head.json lacks dp")? as usize;
                Ok(HeadAssets::Pointer {
                    dp,
                    hidden,
                    q_w: read_f32(&dir.join("pointer_q_w.f32"), dp * hidden)?,
                    q_b: read_f32(&dir.join("pointer_q_b.f32"), dp)?,
                    k_w: read_f32(&dir.join("pointer_k_w.f32"), dp * hidden)?,
                    k_b: read_f32(&dir.join("pointer_k_b.f32"), dp)?,
                })
            }
            Some("joint") => {
                let vocab = meta["vocab"]
                    .as_u64()
                    .ok_or("joint head.json lacks vocab")? as usize;
                let xml = dir.join("joint_head.xml");
                let runtime = Runtime::compile(
                    xml.to_str().ok_or("non-utf8 path")?,
                    "CPU",
                    &PluginConfig::default(),
                )
                .map_err(|e| format!("joint head compile: {e}"))?;
                Ok(HeadAssets::Joint {
                    runtime,
                    lm_head: dir.join("lm_head.f16"),
                    vocab,
                    hidden,
                })
            }
            other => Err(format!("decision/head.json: unknown kind {other:?}")),
        }
    }
}

fn affine(w: &[f32], b: &[f32], x: &[f32], dp: usize, hidden: usize) -> Vec<f32> {
    (0..dp)
        .map(|o| {
            b[o] + w[o * hidden..(o + 1) * hidden]
                .iter()
                .zip(x)
                .map(|(a, c)| a * c)
                .sum::<f32>()
        })
        .collect()
}

/// [rows, T] matrix whose row r averages span r (half-open), as Clef's `_mean_span`.
pub fn span_mask(spans: &[[u32; 2]], t: usize) -> Vec<f32> {
    let mut m = vec![0f32; spans.len() * t];
    for (r, s) in spans.iter().enumerate() {
        let (a, b) = (s[0] as usize, s[1] as usize);
        let w = 1.0 / (b - a) as f32;
        for p in a..b {
            m[r * t + p] = w;
        }
    }
    m
}

fn lm_head_mean(path: &Path, hidden: usize, vocab: usize, ids: &[u32]) -> Result<Vec<f32>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut acc = vec![0f32; hidden];
    let mut buf = vec![0u8; hidden * 2];
    for &id in ids {
        if id as usize >= vocab {
            return Err(format!("lexical id {id} >= vocab {vocab}"));
        }
        f.seek(SeekFrom::Start(id as u64 * hidden as u64 * 2))
            .map_err(|e| e.to_string())?;
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        for (a, c) in acc.iter_mut().zip(buf.chunks_exact(2)) {
            *a += half::f16::from_le_bytes([c[0], c[1]]).to_f32();
        }
    }
    let n = ids.len().max(1) as f32;
    Ok(acc.into_iter().map(|x| x / n).collect())
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn run_head(
    head: &ReadoutHead,
    assets: &mut HeadAssets,
    prompt_ids: &[u32],
    last_logits: &[f32],
    col: &Collector,
) -> Result<Vec<f32>, String> {
    match (head, assets) {
        (ReadoutHead::Verbalizer { token_ids }, HeadAssets::Verbalizer) => token_ids
            .iter()
            .map(|&t| {
                last_logits.get(t as usize).copied().ok_or_else(|| {
                    format!("verbalizer id {t} outside logits ({})", last_logits.len())
                })
            })
            .collect(),
        (
            ReadoutHead::Pointer {
                decide_pos,
                option_pos,
            },
            HeadAssets::Pointer {
                dp,
                hidden,
                q_w,
                q_b,
                k_w,
                k_b,
            },
        ) => {
            let hd = col
                .row(*decide_pos)
                .ok_or("pointer: <decide> row was not collected")?;
            let q = affine(q_w, q_b, hd, *dp, *hidden);
            let scale = 1.0 / (*dp as f32).sqrt();
            option_pos
                .iter()
                .map(|&p| {
                    let ho = col
                        .row(p)
                        .ok_or_else(|| format!("pointer: option row {p} was not collected"))?;
                    let k = affine(k_w, k_b, ho, *dp, *hidden);
                    Ok(k.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() * scale)
                })
                .collect()
        }
        (
            ReadoutHead::Joint { questions },
            HeadAssets::Joint {
                runtime,
                lm_head,
                vocab,
                hidden,
            },
        ) => {
            let t = col.len();
            if t != prompt_ids.len() {
                return Err(format!(
                    "joint: collected {t} rows for a {}-token prompt",
                    prompt_ids.len()
                ));
            }
            let mut spans: Vec<[u32; 2]> = questions.iter().map(|q| q.span).collect();
            let mut lexical = Vec::new();
            let mut option_q = Vec::new();
            for (qi, q) in questions.iter().enumerate() {
                for o in &q.options {
                    spans.push(o.span);
                    lexical.extend(lm_head_mean(
                        lm_head,
                        *hidden,
                        *vocab,
                        &prompt_ids[o.span[0] as usize..o.span[1] as usize],
                    )?);
                    option_q.push(qi as i64);
                }
            }
            let type_ids: Vec<i64> = questions.iter().map(|q| q.qtype as i64).collect();
            let n_opt = option_q.len();
            let mask = span_mask(&spans, t);
            let set = |rt: &mut Runtime, name: &str, dt: DType, shape: &[usize], data: &[u8]| {
                rt.set_input(name, dt, shape, data)
                    .map_err(|e| format!("joint input {name}: {e}"))
            };
            set(
                runtime,
                "hidden",
                DType::F32,
                &[t, *hidden],
                &le_f32(col.all()),
            )?;
            set(
                runtime,
                "span_mask",
                DType::F32,
                &[spans.len(), t],
                &le_f32(&mask),
            )?;
            set(
                runtime,
                "lexical",
                DType::F32,
                &[n_opt, *hidden],
                &le_f32(&lexical),
            )?;
            set(
                runtime,
                "option_q",
                DType::I64,
                &[n_opt],
                &le_i64(&option_q),
            )?;
            set(
                runtime,
                "type_ids",
                DType::I64,
                &[type_ids.len()],
                &le_i64(&type_ids),
            )?;
            runtime.infer().map_err(|e| format!("joint infer: {e}"))?;
            let (_, _, bytes) = runtime
                .output(0)
                .map_err(|e| format!("joint output: {e}"))?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        (h, _) => Err(format!(
            "decision head {h:?} does not match this model's decision/ assets"
        )),
    }
}

/// Ceiling for a FRAME_DECIDE / FRAME_DECISION body. A 73,728-token Kev readout is ~600 KB of JSON.
pub const MAX_DECISION_BODY_BYTES: usize = 16 << 20;

fn framed(json: Vec<u8>) -> Vec<u8> {
    let mut b = (json.len() as u32).to_be_bytes().to_vec();
    b.extend(json);
    b
}

fn unframe(b: &[u8]) -> Result<&[u8], String> {
    let n =
        u32::from_be_bytes(b.get(..4).ok_or("short decision body")?.try_into().unwrap()) as usize;
    if n > MAX_DECISION_BODY_BYTES {
        return Err(format!(
            "decision body {n} bytes exceeds {MAX_DECISION_BODY_BYTES}"
        ));
    }
    b.get(4..4usize.checked_add(n).ok_or("overflow")?)
        .ok_or_else(|| "truncated decision body".to_string())
}

/// FRAME_DECIDE body: u32 BE length + JSON(DecisionReadout). The receiver still runs
/// `DecisionReadout::validate` before using it.
pub fn encode_readout(r: &DecisionReadout) -> Vec<u8> {
    framed(serde_json::to_vec(r).expect("readout serializes"))
}

pub fn decode_readout(b: &[u8]) -> Result<DecisionReadout, String> {
    serde_json::from_slice(unframe(b)?).map_err(|e| e.to_string())
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Ok(Vec<f32>),
    Err(String),
}

/// FRAME_DECISION body: u32 BE length + JSON `{"ok":[logits]}` or `{"err":"why"}`; the last rank
/// reports a head failure instead of hanging rank 0.
pub fn encode_result(r: &Result<Vec<f32>, String>) -> Vec<u8> {
    let o = match r {
        Ok(v) => Outcome::Ok(v.clone()),
        Err(e) => Outcome::Err(e.clone()),
    };
    framed(serde_json::to_vec(&o).expect("outcome serializes"))
}

pub fn decode_result(b: &[u8]) -> Result<Result<Vec<f32>, String>, String> {
    match serde_json::from_slice::<Outcome>(unframe(b)?).map_err(|e| e.to_string())? {
        Outcome::Ok(v) => Ok(Ok(v)),
        Outcome::Err(e) => Ok(Err(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::{HiddenNeed, ReadoutHead};

    #[test]
    fn collector_keeps_requested_rows_across_chunks() {
        let mut c = Collector::new(HiddenNeed::Positions(vec![1, 4]), 2);
        c.absorb(0, 3, &[0.0, 0.0, 1.0, 1.5, 2.0, 2.0]).unwrap();
        c.absorb(3, 2, &[3.0, 3.0, 4.0, 4.5]).unwrap();
        assert_eq!(c.row(1).unwrap(), &[1.0, 1.5]);
        assert_eq!(c.row(4).unwrap(), &[4.0, 4.5]);
        assert!(c.row(2).is_none());
    }

    #[test]
    fn collector_all_keeps_every_row_in_order() {
        let mut c = Collector::new(HiddenNeed::All, 1);
        c.absorb(0, 2, &[0.0, 1.0]).unwrap();
        c.absorb(2, 1, &[2.0]).unwrap();
        assert_eq!(c.all(), &[0.0, 1.0, 2.0]);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn collector_all_refuses_a_gap() {
        let mut c = Collector::new(HiddenNeed::All, 1);
        assert!(
            c.absorb(5, 1, &[0.0]).is_err(),
            "rows 0..5 were never seen (warm restore)"
        );
    }

    #[test]
    fn verbalizer_picks_logit_rows() {
        let mut a = HeadAssets::Verbalizer;
        let head = ReadoutHead::Verbalizer {
            token_ids: vec![2, 0],
        };
        let c = Collector::new(HiddenNeed::None, 4);
        assert_eq!(
            run_head(&head, &mut a, &[], &[0.5, 1.5, 2.5], &c).unwrap(),
            vec![2.5, 0.5]
        );
        assert!(run_head(
            &ReadoutHead::Verbalizer { token_ids: vec![9] },
            &mut a,
            &[],
            &[0.0],
            &c
        )
        .is_err());
    }

    #[test]
    fn pointer_head_matches_hand_computation() {
        // hidden 2, dp 1: q = [1,0]·h + 0 ; k = [0,1]·h + 1 ; z = k*q / sqrt(1)
        let mut a = HeadAssets::Pointer {
            dp: 1,
            hidden: 2,
            q_w: vec![1.0, 0.0],
            q_b: vec![0.0],
            k_w: vec![0.0, 1.0],
            k_b: vec![1.0],
        };
        let mut c = Collector::new(HiddenNeed::Positions(vec![0, 1, 2]), 2);
        c.absorb(0, 3, &[0.0, 2.0, 0.0, 3.0, 2.0, 0.0]).unwrap(); // pos0=[0,2] pos1=[0,3] decide=pos2=[2,0]
        let head = ReadoutHead::Pointer {
            decide_pos: 2,
            option_pos: vec![0, 1],
        };
        // q = 2 ; k0 = 3 ; k1 = 4 ⇒ z = [6, 8]
        assert_eq!(
            run_head(&head, &mut a, &[], &[], &c).unwrap(),
            vec![6.0, 8.0]
        );
    }

    #[test]
    fn joint_span_mask_rows_average_their_span() {
        let m = span_mask(&[[0, 2], [2, 3]], 3);
        assert_eq!(m, vec![0.5, 0.5, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn frame_bodies_round_trip() {
        let r = cascadia_types::DecisionReadout {
            prompt_ids: vec![1, 2],
            head: ReadoutHead::Pointer {
                decide_pos: 1,
                option_pos: vec![0],
            },
            cache_prefix: None,
        };
        assert_eq!(decode_readout(&encode_readout(&r)).unwrap(), r);
        assert_eq!(
            decode_result(&encode_result(&Ok(vec![1.5, -2.0]))).unwrap(),
            Ok(vec![1.5, -2.0])
        );
        assert_eq!(
            decode_result(&encode_result(&Err("boom".into()))).unwrap(),
            Err("boom".to_string())
        );
        assert!(decode_result(&[0, 0, 0, 9]).is_err(), "truncated");
        assert!(
            decode_readout(&[0xff, 0xff, 0xff, 0xff]).is_err(),
            "length beyond the buffer"
        );
    }
}
