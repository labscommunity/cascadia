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
    /// The per-pass token cap (`DecisionService::max_tokens`).
    max_tokens: usize,
    /// The request's prefill budget (`DecisionService::max_total_tokens`); the render budget is
    /// 8 bytes per token of it, the same bound as the handler's byte pre-check.
    max_total_tokens: usize,
}

/// Running byte budget for all of one request's rendered text (state, instructions, options).
struct Budget {
    left: usize,
    total: usize,
    tokens: usize,
}

impl Budget {
    fn render(&mut self, v: &OrderedValue) -> Result<String, DecisionError> {
        let s = render_bounded(v, 0, self.left).map_err(|_| self.too_long())?;
        self.left -= s.len();
        Ok(s)
    }

    fn take(&mut self, s: &str) -> Result<(), DecisionError> {
        self.left = self
            .left
            .checked_sub(s.len())
            .ok_or_else(|| self.too_long())?;
        Ok(())
    }

    fn too_long(&self) -> DecisionError {
        DecisionError::TooLong(format!(
            "the rendered request text exceeds {} bytes; the limit is 8 bytes per token of the {}-token request budget",
            self.total, self.tokens
        ))
    }
}

/// kev.api.render: str/int/float/bool via Python `str()`, lists as `- item`, objects as `key: value`.
/// Unbounded; the codec uses [`render_bounded`].
pub fn render(v: &OrderedValue, indent: usize) -> String {
    let mut r = Renderer::new(usize::MAX);
    r.value(v, indent)
        .expect("an unbounded render cannot overflow");
    r.out
}

/// [`render`], but it stops as soon as the output would pass `budget` bytes, so a deeply nested
/// value (whose indentation grows with depth) never builds its full expansion. `Err` carries the
/// budget.
pub fn render_bounded(v: &OrderedValue, indent: usize, budget: usize) -> Result<String, usize> {
    let mut r = Renderer::new(budget);
    r.value(v, indent)?;
    Ok(r.out)
}

/// Streams kev.api.render into one buffer. The Python builds each nested item as a string and
/// `.lstrip()`s it; here `trim` drops whitespace until the item's first non-whitespace char, which
/// yields the same bytes. Every item's `- ` (and every key's `:`) is non-whitespace and comes
/// before any nested item, so one flag (never stacked) is enough.
struct Renderer {
    out: String,
    budget: usize,
    trim: bool,
}

impl Renderer {
    fn new(budget: usize) -> Self {
        Self {
            out: String::new(),
            budget,
            trim: false,
        }
    }

    fn push(&mut self, mut s: &str) -> Result<(), usize> {
        if self.trim {
            s = s.trim_start();
            if s.is_empty() {
                return Ok(());
            }
            self.trim = false;
        }
        if s.len() > self.budget - self.out.len() {
            return Err(self.budget);
        }
        self.out.push_str(s);
        Ok(())
    }

    fn pad(&mut self, indent: usize) -> Result<(), usize> {
        for _ in 0..indent {
            self.push("  ")?;
        }
        Ok(())
    }

    fn value(&mut self, v: &OrderedValue, indent: usize) -> Result<(), usize> {
        match v {
            OrderedValue::Null => Ok(()),
            OrderedValue::String(s) => self.push(s),
            OrderedValue::Bool(_) | OrderedValue::Number(_) => self.push(&v.py_str()),
            OrderedValue::Array(a) => {
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        self.push("\n")?;
                    }
                    self.pad(indent)?;
                    self.push("- ")?;
                    self.trim = true;
                    let r = self.value(x, indent + 1);
                    self.trim = false;
                    r?;
                }
                Ok(())
            }
            OrderedValue::Object(m) => {
                for (i, (k, x)) in m.iter().enumerate() {
                    if i > 0 {
                        self.push("\n")?;
                    }
                    self.pad(indent)?;
                    self.push(k)?;
                    match x {
                        OrderedValue::Object(_) | OrderedValue::Array(_) => {
                            self.push(":\n")?;
                            self.value(x, indent + 1)?;
                        }
                        _ => {
                            self.push(": ")?;
                            self.value(x, 0)?;
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

/// kev.api.option_text: `name` when the description is absent, null or `""`, else `name: render(desc)`.
fn option_text(
    name: &str,
    desc: Option<&OrderedValue>,
    b: &mut Budget,
) -> Result<String, DecisionError> {
    b.take(name)?;
    Ok(match desc {
        None | Some(OrderedValue::Null) => name.to_string(),
        Some(OrderedValue::String(s)) if s.is_empty() => name.to_string(),
        Some(d) => {
            b.take(": ")?;
            format!("{name}: {}", b.render(d)?)
        }
    })
}

impl KevCodec {
    pub fn new(
        tok: tokenizers::Tokenizer,
        p: KevProfile,
        max_tokens: usize,
        max_total_tokens: usize,
    ) -> Result<Self, String> {
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
            max_tokens,
            max_total_tokens,
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
        let total = self.max_total_tokens.saturating_mul(8);
        let mut b = Budget {
            left: total,
            total,
            tokens: self.max_total_tokens,
        };
        let mut state = vec![s_id];
        state.extend(self.user_tokens(&b.render(&req.state)?)?);
        // Every row is the state plus at least <fim_middle> and <fim_suffix>: refuse before the
        // per-question clones.
        if state.len() >= self.max_tokens {
            return Err(DecisionError::TooLong(format!(
                "the state is {} tokens; a decision pass is limited to {}",
                state.len(),
                self.max_tokens
            )));
        }
        let mut out = Vec::new();
        for (qid, q) in &req.questions {
            let opts: Vec<String> = match q.qtype {
                QuestionType::Noul => {
                    let c = q.criteria.as_ref();
                    vec![
                        option_text("no", c.and_then(|m| m.get("false")), &mut b)?,
                        option_text("yes", c.and_then(|m| m.get("true")), &mut b)?,
                    ]
                }
                QuestionType::Choice => q
                    .choice_criteria()
                    .iter()
                    .map(|(k, v)| option_text(k, Some(v), &mut b))
                    .collect::<Result<_, _>>()?,
                QuestionType::Score => q
                    .score_levels()
                    .iter()
                    .map(|x| b.render(x))
                    .collect::<Result<_, _>>()?,
            };
            let mut row = state.clone();
            row.push(q_id);
            let instructions = match &q.instructions {
                Some(v) => b.render(v)?,
                None => String::new(),
            };
            row.extend(self.user_tokens(&instructions)?);
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
    fn bounded_render_matches_render_within_budget() {
        let v = ov(
            r#"{"z": [1, 2.5, {"p": false, "q": [true, null]}], "a": {"b": {"c": 1e20}, "d": null}, "e": 0.1, "f": [], "g": [" lead", [[], "x"], {" ": {"  k": 1}}]}"#,
        );
        let full = render(&v, 0);
        assert_eq!(
            render_bounded(&v, 0, full.len()).as_deref(),
            Ok(full.as_str())
        );
        assert_eq!(render_bounded(&v, 0, full.len() - 1), Err(full.len() - 1));
    }

    #[test]
    fn bounded_render_stops_at_the_budget() {
        // 120 nested arrays around 15,000 leaves: each leaf line is indented by up to 238 spaces, so the
        // full text would be ~3.6 MB. The renderer must stop at the budget, not build it all.
        let s = format!(
            "{}{}{}",
            "[".repeat(120),
            vec!["1"; 15_000].join(","),
            "]".repeat(120)
        );
        let v = ov(&s);
        let mut r = Renderer::new(64 * 1024);
        assert_eq!(r.value(&v, 0), Err(64 * 1024));
        assert!(r.out.len() <= 64 * 1024, "{}", r.out.len());
        assert!(r.out.capacity() <= 128 * 1024, "{}", r.out.capacity());
    }

    #[test]
    fn deeply_nested_state_is_too_long_and_option_text_counts() {
        let s = format!(
            r#"{{"model": "m", "state": {}{}{}, "questions": {{"u": {{"type": "noul"}}}}}}"#,
            "[".repeat(120),
            vec!["1"; 15_000].join(","),
            "]".repeat(120)
        );
        let r: DecisionRequest = serde_json::from_str(&s).unwrap();
        let c = KevCodec::new(tok(), KevProfile { temperature: 1.0 }, 100_000, 400_000).unwrap();
        assert!(matches!(c.encode(&r), Err(DecisionError::TooLong(_))));
        // The budget also covers instructions and option descriptions.
        let r: DecisionRequest = serde_json::from_str(
            r#"{"model": "m", "state": "a", "questions": {"u": {"type": "noul", "instructions": "b b b b b b b b"}}}"#,
        )
        .unwrap();
        let c = KevCodec::new(tok(), KevProfile { temperature: 1.0 }, 1000, 2).unwrap();
        assert!(matches!(c.encode(&r), Err(DecisionError::TooLong(_))));
        // Option text "name: desc" is charged in full, separator included: state "a" (1) +
        // "no: x" (5) is 6 bytes, so a 5-byte budget fits only if ": " were free.
        let r: DecisionRequest = serde_json::from_str(
            r#"{"model": "m", "state": "a", "questions": {"u": {"type": "noul", "criteria": {"false": "x"}}}}"#,
        )
        .unwrap();
        let mut b = Budget {
            left: 5,
            total: 5,
            tokens: 0,
        };
        b.render(&r.state).unwrap();
        let crit = r.questions["u"].criteria.as_ref().unwrap();
        assert!(option_text("no", crit.get("false"), &mut b).is_err());
        let mut b = Budget {
            left: 6,
            total: 6,
            tokens: 0,
        };
        assert_eq!(
            option_text("no", crit.get("false"), &mut b).unwrap(),
            "no: x"
        );
        assert_eq!(b.left, 1, "6 - len(\"no: x\")");
    }

    #[test]
    fn noul_row_layout_and_positions() {
        let r: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": "a", "questions": {"u": {"type": "noul", "instructions": "b"}}}"#).unwrap();
        let c = KevCodec::new(tok(), KevProfile { temperature: 2.0 }, 1000, 4000).unwrap();
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
        let c = KevCodec::new(t, KevProfile { temperature: 1.0 }, 1000, 4000).unwrap();
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
        let c = KevCodec::new(tok(), KevProfile { temperature: 1.0 }, 1000, 4000).unwrap();
        let ids = &c.encode(&r).unwrap()[0].readout.prompt_ids;
        assert_eq!(
            ids.iter().filter(|&&i| i == 5).count(),
            1,
            "only the codec's own <decide>"
        );
    }
}
