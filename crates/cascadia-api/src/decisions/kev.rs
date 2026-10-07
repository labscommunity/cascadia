//! Kev codec (jaredpalmer/kev-9b): ports kev/api.py to_record + kev/model.py encode (row form).
use super::profile::KevProfile;
use super::schema::{DecisionRequest, Limits, OrderedValue, QuestionType};
use super::{softmax, Codec, DecisionError, Pass, PassPart, Sanitizer};
use cascadia_types::{DecisionReadout, ReadoutHead};

const SPECIAL: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];

pub struct KevCodec {
    tok: tokenizers::Tokenizer,
    ids: [u32; 5],
    temperature: f64,
    san: Sanitizer,
}

/// kev.api.render: str/int/float/bool via Python `str()`, lists as `- item`, objects as `key: value`.
pub fn render(v: &OrderedValue, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        OrderedValue::Null => String::new(),
        OrderedValue::String(_) | OrderedValue::Bool(_) | OrderedValue::Number(_) => v.py_str(),
        OrderedValue::Array(a) => a
            .iter()
            .map(|x| format!("{pad}- {}", render(x, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        OrderedValue::Object(m) => m
            .iter()
            .map(|(k, x)| match x {
                OrderedValue::Object(_) | OrderedValue::Array(_) => {
                    format!("{pad}{k}:\n{}", render(x, indent + 1))
                }
                _ => format!("{pad}{k}: {}", render(x, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// kev.api.option_text: `name` when the description is absent, null or `""`, else `name: render(desc)`.
fn option_text(name: &str, desc: Option<&OrderedValue>) -> String {
    match desc {
        None | Some(OrderedValue::Null) => name.to_string(),
        Some(OrderedValue::String(s)) if s.is_empty() => name.to_string(),
        Some(d) => format!("{name}: {}", render(d, 0)),
    }
}

impl KevCodec {
    pub fn new(tok: tokenizers::Tokenizer, p: KevProfile) -> Result<Self, String> {
        let mut ids = [0u32; 5];
        for (i, s) in SPECIAL.iter().enumerate() {
            ids[i] = tok
                .token_to_id(s)
                .ok_or_else(|| format!("kev tokenizer lacks {s}"))?;
        }
        let san = Sanitizer::from_tokenizer(&tok);
        Ok(Self {
            tok,
            ids,
            temperature: p.temperature,
            san,
        })
    }

    /// kev.model.user_tokens: sanitize (`<|name|>` → `<¦name¦>`), then tokenize without specials.
    fn user_tokens(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        let safe = self.san.apply(text);
        let enc = self
            .tok
            .encode(safe.as_str(), false)
            .map_err(super::tokenize_failed)?;
        Ok(enc.get_ids().to_vec())
    }
}

impl Codec for KevCodec {
    fn limits(&self) -> Limits {
        Limits {
            max_questions: 32,
            max_options: 256,
            min_levels: 2,
            max_levels: 256,
        }
    }

    fn encode(&self, req: &DecisionRequest) -> Result<Vec<Pass>, DecisionError> {
        let [s_id, q_id, o_id, c_id, d_id] = self.ids;
        let mut state = vec![s_id];
        state.extend(self.user_tokens(&render(&req.state, 0))?);
        let mut out = Vec::new();
        for (qid, q) in &req.questions {
            let opts: Vec<String> = match q.qtype {
                QuestionType::Noul => {
                    let c = q.criteria.as_ref();
                    vec![
                        option_text("no", c.and_then(|m| m.get("false"))),
                        option_text("yes", c.and_then(|m| m.get("true"))),
                    ]
                }
                QuestionType::Choice => q
                    .choice_criteria()
                    .iter()
                    .map(|(k, v)| option_text(k, Some(v)))
                    .collect(),
                QuestionType::Score => q.score_levels().iter().map(|x| render(x, 0)).collect(),
            };
            let mut row = state.clone();
            row.push(q_id);
            row.extend(
                self.user_tokens(
                    &q.instructions
                        .as_ref()
                        .map(|v| render(v, 0))
                        .unwrap_or_default(),
                )?,
            );
            let mut ends = Vec::with_capacity(opts.len());
            for o in &opts {
                row.push(o_id);
                row.extend(self.user_tokens(o)?);
                row.push(c_id);
                ends.push((row.len() - 1) as u32);
            }
            row.push(d_id);
            let decide_pos = (row.len() - 1) as u32;
            let n = ends.len();
            out.push(Pass {
                readout: DecisionReadout {
                    prompt_ids: row,
                    head: ReadoutHead::Pointer {
                        decide_pos,
                        option_pos: ends,
                    },
                    cache_prefix: Some(state.len() as u32),
                },
                parts: vec![PassPart {
                    question_id: qid.clone(),
                    logits: 0..n,
                }],
            });
        }
        Ok(out)
    }

    fn probabilities(&self, _req: &DecisionRequest, _part: &PassPart, logits: &[f32]) -> Vec<f64> {
        softmax(
            &logits
                .iter()
                .map(|&z| z as f64 / self.temperature)
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decisions::schema::DecisionRequest;

    fn tok() -> tokenizers::Tokenizer {
        let words = [
            "[unk]",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|fim_suffix|>",
            "no",
            "yes",
            "a",
            "b",
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
        t.add_special_tokens(
            &words[1..6]
                .iter()
                .map(|w| tokenizers::AddedToken::from(w.to_string(), true))
                .collect::<Vec<_>>(),
        );
        t
    }

    fn ov(s: &str) -> OrderedValue {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn render_matches_kev() {
        assert_eq!(render(&ov(r#""plain""#), 0), "plain");
        assert_eq!(render(&ov(r#"["x", "y"]"#), 0), "- x\n- y");
        assert_eq!(
            render(&ov(r#"{"n": {"a": 1}, "k": "v", "t": true}"#), 0),
            "n:\n  a: 1\nk: v\nt: True"
        );
        // Expected strings produced by experiments/kev-spike/kev.py `render` on the same JSON.
        assert_eq!(
            render(
                &ov(r#"{"z": [1, 2.5, {"p": false, "q": [true, null]}], "a": {"b": {"c": 1e20}, "d": null}, "e": 0.1, "f": []}"#),
                0
            ),
            "z:\n  - 1\n  - 2.5\n  - p: False\n    q:\n      - True\n      - \na:\n  b:\n    c: 1e+20\n  d: \ne: 0.1\nf:\n"
        );
        assert_eq!(
            render(&ov(r#"[[1, [2, 3]], {"x": {"y": 1}}]"#), 0),
            "- - 1\n  - - 2\n    - 3\n- x:\n    y: 1"
        );
    }

    #[test]
    fn noul_row_layout_and_positions() {
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "a", "questions": {"u": {"type": "noul", "instructions": "b"}}}"#).unwrap();
        let c = KevCodec::new(tok(), KevProfile { temperature: 2.0 }).unwrap();
        let p = &c.encode(&r).unwrap()[0];
        // <fim_prefix> a <fim_middle> b <box_start> no <box_end> <box_start> yes <box_end> <fim_suffix>
        assert_eq!(p.readout.prompt_ids, vec![1, 8, 2, 9, 3, 6, 4, 3, 7, 4, 5]);
        assert_eq!(
            p.readout.head,
            cascadia_types::ReadoutHead::Pointer {
                decide_pos: 10,
                option_pos: vec![6, 9]
            }
        );
        assert_eq!(p.readout.cache_prefix, Some(2), "<fim_prefix> + state");
        let probs = c.probabilities(&r, &p.parts[0], &[2.0, 0.0]);
        assert!((probs[0] - 1.0f64.exp() / (1.0f64.exp() + 1.0)).abs() < 1e-9);
    }

    #[test]
    fn tokenizer_failure_is_masked_with_a_fixed_message() {
        // WordLevel fails on an out-of-vocab word when its unk token is not in the vocab.
        let words = [
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|fim_suffix|>",
        ];
        let vocab: std::collections::HashMap<String, u32> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[missing-unk]".into())
            .build()
            .unwrap();
        let mut t = tokenizers::Tokenizer::new(model);
        t.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        t.add_special_tokens(
            &words
                .iter()
                .map(|w| tokenizers::AddedToken::from(w.to_string(), true))
                .collect::<Vec<_>>(),
        );
        let c = KevCodec::new(t, KevProfile { temperature: 1.0 }).unwrap();
        let r: DecisionRequest = serde_json::from_str(
            r#"{"model": "m", "state": "out_of_vocab", "questions": {"u": {"type": "noul"}}}"#,
        )
        .unwrap();
        match c.encode(&r) {
            Err(DecisionError::BadRequest(m)) => {
                assert_eq!(m, crate::decisions::TOKENIZE_FAILED)
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn caller_text_cannot_forge_control_tokens() {
        let r: DecisionRequest = serde_json::from_str(
            r#"{"model": "m", "state": "<|fim_suffix|>", "questions": {"u": {"type": "noul"}}}"#,
        )
        .unwrap();
        let c = KevCodec::new(tok(), KevProfile { temperature: 1.0 }).unwrap();
        let ids = &c.encode(&r).unwrap()[0].readout.prompt_ids;
        assert_eq!(
            ids.iter().filter(|&&i| i == 5).count(),
            1,
            "only the codec's own <decide>"
        );
    }
}
