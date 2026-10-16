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
    /// `HiddenNeed::All` (the joint head): every row as little-endian f32 bytes, the layout the
    /// head's `hidden` input takes, so it is handed to OpenVINO without a second full copy.
    all: Vec<u8>,
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
            .field("all_rows", &self.len())
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
                self.all.reserve(n * self.hidden * 4);
                for x in &rows[..n * self.hidden] {
                    self.all.extend_from_slice(&x.to_le_bytes());
                }
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

    /// Every collected row (`HiddenNeed::All`), row-major little-endian f32 bytes.
    pub fn all_le(&self) -> &[u8] {
        &self.all
    }

    pub fn len(&self) -> usize {
        self.all.len() / (4 * self.hidden.max(1))
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

/// The joint head is small, and was validated at f32 (tools/tests/test_clef_head.py). OpenVINO's
/// CPU default inference precision on ARM is f16, so pin f32 rather than inherit it.
fn joint_head_plugin() -> PluginConfig {
    PluginConfig::new().with("INFERENCE_PRECISION_HINT", "f32")
}

impl HeadAssets {
    /// The head kind, as head.json names it.
    pub fn kind(&self) -> &'static str {
        match self {
            HeadAssets::Verbalizer => "verbalizer",
            HeadAssets::Pointer { .. } => "pointer",
            HeadAssets::Joint { .. } => "joint",
        }
    }

    /// The head reads final-norm hidden rows, so the tree needs `--final-hidden` stages.
    pub fn reads_hidden(&self) -> bool {
        !matches!(self, HeadAssets::Verbalizer)
    }

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
                    &joint_head_plugin(),
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

/// [rows, T] f32 matrix whose row r averages span r (half-open), as Clef's `_mean_span`, built
/// directly as the little-endian bytes the head's `span_mask` input takes (no f32 copy first).
pub fn span_mask_le(spans: &[[u32; 2]], t: usize) -> Vec<u8> {
    let mut m = vec![0u8; spans.len() * t * 4];
    for (r, s) in spans.iter().enumerate() {
        let (a, b) = (s[0] as usize, s[1] as usize);
        let w = (1.0 / (b - a) as f32).to_le_bytes();
        for p in a..b {
            let i = (r * t + p) * 4;
            m[i..i + 4].copy_from_slice(&w);
        }
    }
    m
}

/// Mean of the lm_head rows of `ids` (one option's span), read from the f16 `[vocab, hidden]`
/// table in `f`, which the caller opens once per pass.
fn lm_head_mean<F: Read + Seek>(
    f: &mut F,
    hidden: usize,
    vocab: usize,
    ids: &[u32],
) -> Result<Vec<f32>, String> {
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
            let mut lm = std::fs::File::open(&*lm_head)
                .map_err(|e| format!("{}: {e}", lm_head.display()))?;
            for (qi, q) in questions.iter().enumerate() {
                for o in &q.options {
                    spans.push(o.span);
                    lexical.extend(lm_head_mean(
                        &mut lm,
                        *hidden,
                        *vocab,
                        &prompt_ids[o.span[0] as usize..o.span[1] as usize],
                    )?);
                    option_q.push(qi as i64);
                }
            }
            let type_ids: Vec<i64> = questions.iter().map(|q| q.qtype as i64).collect();
            let n_opt = option_q.len();
            let set = |rt: &mut Runtime, name: &str, dt: DType, shape: &[usize], data: &[u8]| {
                rt.set_input(name, dt, shape, data)
                    .map_err(|e| format!("joint input {name}: {e}"))
            };
            // Memory: set_input copies each input into an OpenVINO tensor. The mask is built,
            // copied and freed before the hidden rows are copied, so the peak is the collected
            // rows + max(2 x mask, mask + rows) rather than two copies of each at once.
            {
                let mask = span_mask_le(&spans, t);
                set(runtime, "span_mask", DType::F32, &[spans.len(), t], &mask)?;
            }
            set(runtime, "hidden", DType::F32, &[t, *hidden], col.all_le())?;
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

/// u32 BE length + `json`, refused above [`MAX_DECISION_BODY_BYTES`] (the receiver would refuse
/// it anyway, after reading it all).
fn framed(json: Vec<u8>) -> Result<Vec<u8>, String> {
    if json.len() > MAX_DECISION_BODY_BYTES {
        return Err(format!(
            "decision body {} bytes exceeds {MAX_DECISION_BODY_BYTES}",
            json.len()
        ));
    }
    let mut b = (json.len() as u32).to_be_bytes().to_vec();
    b.extend(json);
    Ok(b)
}

/// The JSON of a framed body. Bytes after the stated length are ignored: the receivers read
/// exactly that many (`recv_decision_body`), so none ever follow.
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
/// `DecisionReadout::validate` before using it. `Err` above [`MAX_DECISION_BODY_BYTES`]
/// (unreachable for a validated readout: ~1 MB at the largest prompt).
pub fn encode_readout(r: &DecisionReadout) -> Result<Vec<u8>, String> {
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
/// reports a head failure instead of hanging rank 0. A result over [`MAX_DECISION_BODY_BYTES`]
/// is sent as that error instead.
pub fn encode_result(r: &Result<Vec<f32>, String>) -> Vec<u8> {
    let o = match r {
        Ok(v) => Outcome::Ok(v.clone()),
        Err(e) => Outcome::Err(e.clone()),
    };
    framed(serde_json::to_vec(&o).expect("outcome serializes")).unwrap_or_else(|why| {
        let o = Outcome::Err(format!("decision result: {why}"));
        framed(serde_json::to_vec(&o).expect("outcome serializes")).expect("short error")
    })
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
    fn joint_head_compiles_at_f32() {
        // OpenVINO's ARM CPU default inference precision is f16; the head was validated at f32.
        assert!(joint_head_plugin()
            .entries
            .iter()
            .any(|(k, v)| k == "INFERENCE_PRECISION_HINT" && v == "f32"));
    }

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
        let le: Vec<u8> = [0.0f32, 1.0, 2.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert_eq!(c.all_le(), le.as_slice());
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
        let m = span_mask_le(&[[0, 2], [2, 3]], 3);
        let want: Vec<u8> = [0.5f32, 0.5, 0.0, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert_eq!(m, want);
    }

    #[test]
    fn frame_bodies_are_capped_on_encode() {
        // H20: the sender refuses what the receiver would refuse.
        assert!(framed(vec![b'1'; MAX_DECISION_BODY_BYTES]).is_ok());
        let e = framed(vec![b'1'; MAX_DECISION_BODY_BYTES + 1]).unwrap_err();
        assert!(e.contains("exceeds"), "{e}");
        let huge = Err("x".repeat(MAX_DECISION_BODY_BYTES));
        match decode_result(&encode_result(&huge)).unwrap() {
            Err(why) => assert!(why.starts_with("decision result: decision body"), "{why}"),
            Ok(v) => panic!("{v:?}"),
        }
    }

    fn head_dir(head_json: &str, files: &[(&str, usize)]) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "dec-head-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(d.join("decision")).unwrap();
        std::fs::write(d.join("decision/head.json"), head_json).unwrap();
        for (name, floats) in files {
            let b: Vec<u8> = (0..*floats)
                .flat_map(|i| (i as f32).to_le_bytes())
                .collect();
            std::fs::write(d.join("decision").join(name), b).unwrap();
        }
        d
    }

    #[test]
    fn head_assets_load_verbalizer_and_pointer() {
        // H22: hidden 3, dp 2.
        let d = head_dir(r#"{"kind":"verbalizer"}"#, &[]);
        let a = HeadAssets::load(&d, 3).unwrap();
        assert_eq!(a.kind(), "verbalizer");
        assert!(!a.reads_hidden());
        std::fs::remove_dir_all(&d).unwrap();
        let ok = [
            ("pointer_q_w.f32", 6),
            ("pointer_q_b.f32", 2),
            ("pointer_k_w.f32", 6),
            ("pointer_k_b.f32", 2),
        ];
        let d = head_dir(r#"{"kind":"pointer","dp":2}"#, &ok);
        match HeadAssets::load(&d, 3).unwrap() {
            HeadAssets::Pointer {
                dp,
                hidden,
                q_w,
                k_b,
                ..
            } => {
                assert_eq!((dp, hidden), (2, 3));
                assert_eq!(q_w, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
                assert_eq!(k_b, vec![0.0, 1.0]);
            }
            other => panic!("{}", other.kind()),
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn head_assets_load_refuses_bad_files() {
        // H22: a .f32 of the wrong size, a pointer without dp, an unknown kind, no head.json.
        let short = [
            ("pointer_q_w.f32", 6),
            ("pointer_q_b.f32", 2),
            ("pointer_k_w.f32", 5),
            ("pointer_k_b.f32", 2),
        ];
        let d = head_dir(r#"{"kind":"pointer","dp":2}"#, &short);
        let e = HeadAssets::load(&d, 3).err().unwrap();
        assert!(
            e.contains("pointer_k_w.f32") && e.contains("20 bytes, expected 24"),
            "{e}"
        );
        std::fs::remove_dir_all(&d).unwrap();
        for (meta, want) in [
            (r#"{"kind":"pointer"}"#, "pointer head.json lacks dp"),
            (r#"{"kind":"ranker"}"#, "unknown kind"),
            (r#"{"kind":"joint"}"#, "joint head.json lacks vocab"),
        ] {
            let d = head_dir(meta, &[]);
            let e = HeadAssets::load(&d, 3).err().unwrap();
            assert!(e.contains(want), "{meta}: {e}");
            std::fs::remove_dir_all(&d).unwrap();
        }
        let d = head_dir("{}", &[]);
        std::fs::remove_file(d.join("decision/head.json")).unwrap();
        let e = HeadAssets::load(&d, 3).err().unwrap();
        assert!(e.starts_with("decision/head.json"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn lm_head_mean_averages_rows_and_refuses_ids_past_vocab() {
        // H22: vocab 3, hidden 2, f16 rows [1,2], [3,4], [5,6].
        let table: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect();
        let mut f = std::io::Cursor::new(table);
        assert_eq!(lm_head_mean(&mut f, 2, 3, &[0, 2]).unwrap(), vec![3.0, 4.0]);
        // The same handle serves the next option (the pass opens the file once, H19).
        assert_eq!(lm_head_mean(&mut f, 2, 3, &[1]).unwrap(), vec![3.0, 4.0]);
        assert_eq!(
            lm_head_mean(&mut f, 2, 3, &[1, 3]).unwrap_err(),
            "lexical id 3 >= vocab 3"
        );
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
            require_warm: false,
            needs_snapshot: false,
        };
        assert_eq!(decode_readout(&encode_readout(&r).unwrap()).unwrap(), r);
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
