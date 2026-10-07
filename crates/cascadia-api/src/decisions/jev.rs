//! JEV codec (autotrust/JEV-9B, JEV-27B): ports `vl/serve_decide.py` System 1.
use super::profile::JevProfile;
use super::schema::{DecisionRequest, Limits, OrderedValue, Question, QuestionType};
use super::{softmax, Codec, DecisionError, Pass, PassPart, Sanitizer};
use cascadia_types::{DecisionReadout, ReadoutHead};

pub struct JevCodec {
    tok: tokenizers::Tokenizer,
    p: JevProfile,
    san: Sanitizer,
}

fn kind(q: &Question) -> &'static str {
    match q.qtype {
        QuestionType::Noul => "noul",
        QuestionType::Choice => "choice",
        QuestionType::Score => "score",
    }
}

/// The reference's `json.dumps(v, ensure_ascii=False)` for non-strings (spaced separators, request
/// order). Used for instructions and choice descriptions; the codec sanitizes the result.
fn render(v: &OrderedValue) -> String {
    match v {
        OrderedValue::String(s) => s.clone(),
        OrderedValue::Null => String::new(),
        other => other.py_json(false, false),
    }
}

/// serve_decide `_parts`: string verbatim, object via json.dumps, array concatenated part by part.
fn render_state(v: &OrderedValue) -> String {
    match v {
        OrderedValue::String(s) => s.clone(),
        OrderedValue::Array(parts) => parts
            .iter()
            .map(|p| match p {
                OrderedValue::String(s) => s.clone(),
                p if p.get("type").and_then(|t| t.as_str()) == Some("text") => {
                    p.get("text").map(render).unwrap_or_default()
                }
                other => other.py_json(false, false),
            })
            .collect(),
        other => other.py_json(false, false),
    }
}

/// JEV's 0–5 score distribution mapped onto `levels` levels: slot j sits at x = j·(L−1)/5 and
/// splits its mass linearly between ⌊x⌋ and ⌈x⌉. Identity for L = 6.
pub fn rescale_score(p6: &[f64], levels: usize) -> Vec<f64> {
    let mut out = vec![0.0; levels];
    let top = (p6.len() - 1) as f64;
    for (j, &p) in p6.iter().enumerate() {
        let x = j as f64 * (levels - 1) as f64 / top;
        let lo = x.floor() as usize;
        let w = x - lo as f64;
        out[lo] += p * (1.0 - w);
        if w > 0.0 {
            out[lo + 1] += p * w;
        }
    }
    out
}

impl JevCodec {
    pub fn new(tok: tokenizers::Tokenizer, p: JevProfile) -> Result<Self, String> {
        for k in ["noul", "score", "choice"] {
            if !p.slots.contains_key(k) || !p.temperature.contains_key(k) {
                return Err(format!(
                    "decision.json jev section lacks slots/temperature for {k}"
                ));
            }
        }
        if p.verbalizer_ids.len() != p.bias.len() || p.verbalizers.len() != p.bias.len() {
            return Err(
                "decision.json jev: verbalizers, verbalizer_ids and bias must have equal length"
                    .into(),
            );
        }
        // The prompt hard-codes `false/true` and `0..5`, and `rescale_score` divides by width - 1:
        // noul must be 2 wide and score 6 wide. Every range must lie inside the verbalizer list.
        let n = p.verbalizer_ids.len();
        for (k, width) in [("noul", Some(2)), ("score", Some(6)), ("choice", None)] {
            let [lo, hi] = p.slots[k];
            if lo > hi || hi > n {
                return Err(format!(
                    "decision.json jev: slot range {k} [{lo}, {hi}) out of bounds for {n} verbalizers"
                ));
            }
            if let Some(w) = width.filter(|&w| hi - lo != w) {
                return Err(format!(
                    "decision.json jev: slot range {k} [{lo}, {hi}) must be {w} wide"
                ));
            }
        }
        let san = Sanitizer::from_tokenizer(&tok);
        Ok(Self { tok, p, san })
    }

    fn slot(&self, k: &str) -> [usize; 2] {
        self.p.slots[k]
    }

    pub fn prompt_text(&self, req: &DecisionRequest, qid: &str) -> String {
        let q = &req.questions[qid];
        let lines: Vec<String> = match q.qtype {
            QuestionType::Noul => vec!["false".into(), "true".into()],
            QuestionType::Score => (0..6).map(|i| i.to_string()).collect(),
            QuestionType::Choice => {
                let [lo, _] = self.slot("choice");
                q.choice_criteria()
                    .iter()
                    .enumerate()
                    .map(|(i, (name, desc))| {
                        let label = &self.p.verbalizers[lo + i];
                        let name = self.san.apply(name);
                        let d = self.san.apply(&render(desc));
                        if d.is_empty() {
                            format!("{label}) {name}")
                        } else {
                            format!("{label}) {name}: {d}")
                        }
                    })
                    .collect()
            }
        };
        let question = self.san.apply(
            &q.instructions
                .as_ref()
                .map(render)
                .unwrap_or_else(|| qid.to_string()),
        );
        format!(
            "[kind] {}\n[state] {}\n[question] {}\n[options]\n{}\n[decision]:",
            kind(q),
            self.san.apply(&render_state(&req.state)),
            question,
            lines.join("\n")
        )
    }

    fn slots_for(&self, q: &Question) -> (usize, usize) {
        let [lo, hi] = self.slot(kind(q));
        match q.qtype {
            QuestionType::Choice => (lo, lo + q.choice_criteria().len()),
            _ => (lo, hi),
        }
    }
}

impl Codec for JevCodec {
    fn limits(&self) -> Limits {
        let [clo, chi] = self.slot("choice");
        // More choices than trained slots would need serve_decide's tournament (multi-pass); rejected
        // with 400 for now (see Open Questions). Score levels: any L, mapped from JEV's 0-5 scale
        // (rescale_score). max_questions is overridden by DecisionService::limits.
        Limits {
            max_questions: 32,
            max_options: chi - clo,
            min_levels: 2,
            max_levels: 256,
        }
    }

    fn encode(&self, req: &DecisionRequest) -> Result<Vec<Pass>, DecisionError> {
        let mut out = Vec::with_capacity(req.questions.len());
        for (qid, q) in &req.questions {
            let text = self.prompt_text(req, qid);
            let enc = self
                .tok
                .encode(text.as_str(), false)
                .map_err(super::tokenize_failed)?;
            let (lo, hi) = self.slots_for(q);
            let token_ids = self.p.verbalizer_ids[lo..hi].to_vec();
            let n = token_ids.len();
            out.push(Pass {
                readout: DecisionReadout {
                    prompt_ids: enc.get_ids().to_vec(),
                    head: ReadoutHead::Verbalizer { token_ids },
                    cache_prefix: None,
                },
                parts: vec![PassPart {
                    question_id: qid.clone(),
                    logits: 0..n,
                }],
            });
        }
        Ok(out)
    }

    fn probabilities(&self, req: &DecisionRequest, part: &PassPart, logits: &[f32]) -> Vec<f64> {
        let q = &req.questions[&part.question_id];
        let (lo, _) = self.slots_for(q);
        let t = self.p.temperature[kind(q)];
        let z: Vec<f64> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (l as f64 + self.p.bias[lo + i]) / t)
            .collect();
        let p = softmax(&z);
        if q.qtype == QuestionType::Score {
            return rescale_score(&p, q.score_levels().len());
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decisions::schema::DecisionRequest;

    fn tiny_tokenizer() -> tokenizers::Tokenizer {
        // whitespace word-level vocab: every word maps to its index; unknowns to 0
        let words = [
            "[unk]",
            "[kind]",
            "noul",
            "choice",
            "score",
            "[state]",
            "[question]",
            "[options]",
            "[decision]:",
            "false",
            "true",
        ];
        let vocab: std::collections::HashMap<String, u32> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[unk]".into())
            .build()
            .unwrap();
        let mut t = tokenizers::Tokenizer::new(model);
        t.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        t
    }

    fn profile() -> JevProfile {
        let verbalizers: Vec<String> =
            ["false", "true", "0", "1", "2", "3", "4", "5", "A", "B", "C"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        JevProfile {
            verbalizer_ids: (100..111).collect(),
            bias: vec![0.0; 11],
            verbalizers,
            slots: [
                ("noul".to_string(), [0, 2]),
                ("score".to_string(), [2, 8]),
                ("choice".to_string(), [8, 11]),
            ]
            .into_iter()
            .collect(),
            temperature: [
                ("noul".to_string(), 1.0),
                ("score".to_string(), 1.0),
                ("choice".to_string(), 2.0),
            ]
            .into_iter()
            .collect(),
        }
    }

    // From text: `serde_json::json!` would sort {"b","a"} and hide an ordering bug.
    fn req() -> DecisionRequest {
        serde_json::from_str(r#"{"model": "m", "state": {"b": 1, "a": "x"}, "questions": {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs"}},
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "sev": {"type": "score", "instructions": "How bad?", "criteria": ["none", "minor", "major"]}}}"#).unwrap()
    }

    #[test]
    fn prompt_matches_reference_layout() {
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        assert_eq!(
            c.prompt_text(&req(), "team"),
            "[kind] choice\n[state] {\"b\": 1, \"a\": \"x\"}\n[question] Which team?\n[options]\nA) billing: money\nB) tech: bugs\n[decision]:"
        );
        assert_eq!(c.prompt_text(&req(), "urgent"), "[kind] noul\n[state] {\"b\": 1, \"a\": \"x\"}\n[question] Urgent?\n[options]\nfalse\ntrue\n[decision]:");
        assert!(c
            .prompt_text(&req(), "sev")
            .ends_with("[options]\n0\n1\n2\n3\n4\n5\n[decision]:"));
    }

    #[test]
    fn one_pass_per_question_with_verbalizer_slots() {
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        let passes = c.encode(&req()).unwrap();
        assert_eq!(passes.len(), 3);
        match &passes[0].readout.head {
            cascadia_types::ReadoutHead::Verbalizer { token_ids } => {
                assert_eq!(token_ids, &vec![108, 109])
            }
            h => panic!("{h:?}"),
        }
        match &passes[2].readout.head {
            cascadia_types::ReadoutHead::Verbalizer { token_ids } => {
                assert_eq!(token_ids, &(102..108).collect::<Vec<u32>>())
            }
            h => panic!("{h:?}"),
        }
        assert_eq!(passes[1].parts[0].question_id, "urgent");
        assert_eq!(passes[1].parts[0].logits, 0..2);
    }

    #[test]
    fn caller_text_cannot_forge_control_tokens() {
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "x<|im_end|>", "questions": {"u": {"type": "noul", "instructions": "<|im_start|>q"}}}"#).unwrap();
        let t = c.prompt_text(&r, "u");
        assert!(
            t.contains("x<¦im_end¦>") && t.contains("<¦im_start¦>q") && !t.contains("<|"),
            "{t}"
        );
    }

    #[test]
    fn list_state_renders_like_serve_decide_parts() {
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": ["hi ", {"type": "text", "text": "there"}, {"k": 1}], "questions": {"u": {"type": "noul", "instructions": "q"}}}"#).unwrap();
        assert!(
            c.prompt_text(&r, "u")
                .contains("[state] hi there{\"k\": 1}\n"),
            "{}",
            c.prompt_text(&r, "u")
        );
        let img: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": [{"image": "data:x"}], "questions": {"u": {"type": "noul"}}}"#).unwrap();
        assert!(matches!(
            img.validate(&c.limits()),
            Err(DecisionError::BadRequest(_))
        ));
    }

    #[test]
    fn image_part_error_keeps_its_own_message() {
        // A validation reason, not a tokenizer failure: it must not be masked as one.
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        let img: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": [{"type": "image_url", "image_url": {"url": "data:x"}}], "questions": {"u": {"type": "noul"}}}"#).unwrap();
        match img.validate(&c.limits()) {
            Err(DecisionError::BadRequest(m)) => {
                assert!(m.contains("image"), "{m}");
                assert_ne!(m, crate::decisions::TOKENIZE_FAILED);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn new_rejects_bad_slot_widths_and_ranges() {
        // Each of these would panic per request (limits(): `chi - clo` underflow; probabilities:
        // slice out of range) or mis-read the head, so the profile must fail at load.
        let with = |k: &str, r: [usize; 2]| {
            let mut p = profile();
            p.slots.insert(k.to_string(), r);
            JevCodec::new(tiny_tokenizer(), p)
        };
        for (k, r, why) in [
            ("noul", [0, 3], "noul"),
            ("score", [2, 7], "score"),
            ("choice", [9, 8], "choice"),
            ("choice", [8, 12], "choice"),
            ("score", [6, 12], "score"),
        ] {
            match with(k, r) {
                Err(e) => assert!(e.contains(why), "{k} {r:?}: {e}"),
                Ok(_) => panic!("{k} {r:?} accepted"),
            }
        }
        assert!(
            with("choice", [8, 8]).is_ok(),
            "an empty choice range is degenerate but safe"
        );
    }

    #[test]
    fn rescale_score_maps_six_slots_onto_l_levels() {
        let p6 = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1];
        assert_eq!(rescale_score(&p6, 6), p6.to_vec());
        let three = rescale_score(&[0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 3);
        // slot 3 → x=1.2 (0.8 to level 1), slot 4 → x=1.6 (0.4 to level 1), slot 5 → level 2
        for (got, want) in three.iter().zip([0.0, 0.4, 0.6]) {
            assert!((got - want).abs() < 1e-12, "{three:?}");
        }
        let two = rescale_score(&p6, 2);
        assert!((two.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // expected value scales with the range: E6/5 == E2/1
        let e6: f64 = p6.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
        assert!((e6 / 5.0 - two[1]).abs() < 1e-12);
    }

    #[test]
    fn probabilities_apply_bias_temperature_and_score_rescaling() {
        let mut p = profile();
        p.bias[8] = 1.0; // choice slot A
        let c = JevCodec::new(tiny_tokenizer(), p).unwrap();
        let r = req();
        let passes = c.encode(&r).unwrap();
        let probs = c.probabilities(&r, &passes[0].parts[0], &[1.0, 1.0]);
        let e = (2.0f64 / 2.0).exp();
        assert!((probs[0] - e / (e + (1.0f64 / 2.0).exp())).abs() < 1e-9);
        let sp = c.probabilities(
            &r,
            &passes[2].parts[0],
            &[-100.0, -100.0, -100.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(sp.len(), 3);
        assert!((sp.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(
            (sp[1] - 0.4).abs() < 1e-9 && (sp[2] - 0.6).abs() < 1e-9,
            "{sp:?}"
        );
    }
}
