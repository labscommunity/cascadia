//! JEV codec (autotrust/JEV-9B, JEV-27B): ports `vl/serve_decide.py` System 1.
use super::profile::JevProfile;
pub use super::profile::JevStrategy;
use super::pyrandom::{crc32, PyRandom};
use super::schema::{DecisionRequest, Limits, OrderedValue, Question, QuestionType};
use super::{softmax, Codec, DecisionError, Pass, PassPart, Sanitizer};
use cascadia_types::{DecisionReadout, ReadoutHead};
use std::sync::atomic::AtomicBool;

pub struct JevCodec {
    tok: tokenizers::Tokenizer,
    p: JevProfile,
    san: Sanitizer,
    /// The per-pass token cap (`DecisionService::max_tokens`): every pass carries the whole
    /// state, so the first pass over it ends the encode instead of tokenizing every question.
    pass_cap: usize,
    /// The request's prefill budget (`max_total_tokens`): JEV prefills every pass in full, so
    /// the encode stops once the passes so far exceed it (P2-SEC-4).
    total_cap: usize,
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
                // `validate` refuses a text part without a string `text` (H8).
                p if p.get("type").and_then(|t| t.as_str()) == Some("text") => {
                    p.get("text").map(render).unwrap_or_default()
                }
                other => other.py_json(false, false),
            })
            .collect(),
        other => other.py_json(false, false),
    }
}

/// Width of a tournament group and of its final (serve_decide `_groups(n, k=16)` and
/// `16 - len(chosen)`): fixed at 16 whatever the trained slot count.
pub const TOURNAMENT_WIDTH: usize = 16;

/// The passes one choice question needs (serve_decide `s1_dist`). Every pass over m options
/// labels them by position (`JevCodec::pass_labels(m)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChoicePlan {
    /// n ≤ trained slots: today's single pass on the trained head, whatever the strategy.
    Native,
    /// One pass over all n options.
    Single,
    /// Four passes, option `orders[k][i]` at position i of pass k; see [`combine_permute`].
    Permute { orders: [Vec<usize>; 4] },
    /// One pass per group, then a final over [`tournament_final`]; see [`combine_tournament`].
    Tournament { groups: Vec<Vec<usize>> },
}

/// serve_decide `_groups(n, k)`: g = ⌈n/k⌉ contiguous groups, the first `n mod g` one larger.
pub fn groups(n: usize, k: usize) -> Vec<Vec<usize>> {
    let g = n.div_ceil(k);
    let (base, extra) = (n / g, n % g);
    let mut i = 0;
    (0..g)
        .map(|j| {
            let size = base + usize::from(j < extra);
            i += size;
            (i - size..i).collect()
        })
        .collect()
}

/// serve_decide's permute mean: `p[o[k]] += dist[k] / len(orders)`, in order, no renormalisation.
pub fn combine_permute(orders: &[Vec<usize>], dists: &[Vec<f64>]) -> Vec<f64> {
    let n = orders.first().map_or(0, Vec::len);
    let mut p = vec![0.0; n];
    for (o, d) in orders.iter().zip(dists) {
        for (&i, &v) in o.iter().zip(d) {
            p[i] += v / orders.len() as f64;
        }
    }
    p
}

/// The tournament's finalists, sorted: each group's winner by `(in_group, -index)`, topped up
/// to TOURNAMENT_WIDTH with the best losers by `(-in_group, index)`. `in_group[g]` is group g's
/// pass, aligned with `groups[g]`.
pub fn tournament_final(groups: &[Vec<usize>], in_group: &[Vec<f64>]) -> Vec<usize> {
    let cmp = |a: &(usize, f64), b: &(usize, f64)| {
        // probability descending, then index ascending; passes are finite softmax outputs
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };
    let mut fin = Vec::with_capacity(TOURNAMENT_WIDTH);
    let mut rest = Vec::new();
    for (g, d) in groups.iter().zip(in_group) {
        let mut members: Vec<(usize, f64)> = g.iter().copied().zip(d.iter().copied()).collect();
        members.sort_by(cmp);
        fin.push(members[0].0);
        rest.extend_from_slice(&members[1..]);
    }
    rest.sort_by(cmp);
    let top_up = TOURNAMENT_WIDTH.saturating_sub(fin.len());
    fin.extend(rest.iter().take(top_up).map(|&(o, _)| o));
    fin.sort_unstable();
    fin
}

/// serve_decide's tournament combination, in f64 and in its summation order: a finalist gets
/// `final · Σ_g share[g]·cap[g]`, anyone else `share[group] · in_group`, then the whole is
/// normalised; `share[g]` / `cap[g]` sum the final / in-group mass of group g's finalists.
pub fn combine_tournament(
    groups: &[Vec<usize>],
    in_group: &[Vec<f64>],
    fin: &[usize],
    final_p: &[f64],
) -> Vec<f64> {
    let n: usize = groups.iter().map(Vec::len).sum();
    let mut group_of = vec![0usize; n];
    let mut ig = vec![0.0f64; n];
    for (gi, (g, d)) in groups.iter().zip(in_group).enumerate() {
        for (&o, &v) in g.iter().zip(d) {
            group_of[o] = gi;
            ig[o] = v;
        }
    }
    let mut fp = vec![None; n];
    let (mut share, mut cap) = (vec![0.0f64; groups.len()], vec![0.0f64; groups.len()]);
    for (&f, &v) in fin.iter().zip(final_p) {
        fp[f] = Some(v);
        share[group_of[f]] += v;
        cap[group_of[f]] += ig[f];
    }
    let among: f64 = share.iter().zip(&cap).map(|(a, b)| a * b).sum();
    let p: Vec<f64> = (0..n)
        .map(|o| match fp[o] {
            Some(v) => v * among,
            None => share[group_of[o]] * ig[o],
        })
        .collect();
    let s: f64 = p.iter().sum();
    p.into_iter().map(|x| x / s).collect()
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
        p.check_calibration()
            .and_then(|()| p.check_choice_labels())
            .map_err(|e| format!("decision.json jev: {e}"))?;
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
            // A zero-width choice range would refuse every choice question with a confusing
            // "1..=0 options" 400 (EDGE-6).
            if width.is_none() && hi == lo {
                return Err(format!(
                    "decision.json jev: slot range {k} [{lo}, {hi}) must be at least 1 wide"
                ));
            }
            if let Some(w) = width.filter(|&w| hi - lo != w) {
                return Err(format!(
                    "decision.json jev: slot range {k} [{lo}, {hi}) must be {w} wide"
                ));
            }
        }
        let san = Sanitizer::from_tokenizer(&tok);
        Ok(Self {
            tok,
            p,
            san,
            pass_cap: usize::MAX,
            total_cap: usize::MAX,
        })
    }

    /// Stop encoding at the first pass longer than `max_tokens` (a 413 either way).
    pub fn with_pass_cap(mut self, max_tokens: usize) -> Self {
        self.pass_cap = max_tokens;
        self
    }

    /// Stop encoding once the passes so far exceed `max_total_tokens` (a 413 either way: no pass
    /// shares a prefix, so the handler's budget counts every one in full).
    pub fn with_total_cap(mut self, max_total_tokens: usize) -> Self {
        self.total_cap = max_total_tokens;
        self
    }

    fn slot(&self, k: &str) -> [usize; 2] {
        self.p.slots[k]
    }

    /// Trained choice slots (serve_decide `_native()`): 16 for the shipped JEV heads.
    fn trained_choices(&self) -> usize {
        let [lo, hi] = self.slot("choice");
        hi - lo
    }

    /// The passes a choice question over `n` options needs under `strategy`. `question` is the
    /// text serve_decide seeds the permute orders from (`zlib.crc32(question.encode())`).
    pub fn plan_choice(&self, n: usize, strategy: JevStrategy, question: &str) -> ChoicePlan {
        if n <= self.trained_choices() {
            return ChoicePlan::Native;
        }
        match strategy {
            JevStrategy::Single => ChoicePlan::Single,
            JevStrategy::Permute => {
                // One generator for the three samples, as `rng.sample(...) for _ in range(3)`.
                let mut rng = PyRandom::new(crc32(question.as_bytes()));
                ChoicePlan::Permute {
                    orders: [
                        (0..n).collect(),
                        rng.sample_all(n),
                        rng.sample_all(n),
                        rng.sample_all(n),
                    ],
                }
            }
            JevStrategy::Tournament => ChoicePlan::Tournament {
                groups: groups(n, TOURNAMENT_WIDTH),
            },
        }
    }

    /// Labels and token ids of a pass over `m` options, by position: `choice_labels[..m]`, or
    /// the trained choice verbalizers in a profile without labels (then m ≤ trained slots).
    pub fn pass_labels(&self, m: usize) -> (&[String], &[u32]) {
        if self.p.choice_labels.is_empty() {
            let [lo, _] = self.slot("choice");
            (
                &self.p.verbalizers[lo..lo + m],
                &self.p.verbalizer_ids[lo..lo + m],
            )
        } else {
            (&self.p.choice_labels[..m], &self.p.choice_label_ids[..m])
        }
    }

    /// One choice pass over `logits.len()` options: `softmax((logit + bias) / T["choice"])` with
    /// the trained bias for the first `hi - lo` slots and 0 beyond (serve_decide `s1_pass`).
    pub fn choice_pass_probabilities(&self, logits: &[f32]) -> Vec<f64> {
        let [lo, hi] = self.slot("choice");
        let t = self.p.temperature["choice"];
        let z: Vec<f64> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let b = if lo + i < hi {
                    self.p.bias[lo + i]
                } else {
                    0.0
                };
                (l as f64 + b) / t
            })
            .collect();
        softmax(&z)
    }

    pub fn prompt_text(&self, req: &DecisionRequest, qid: &str) -> String {
        self.prompt_with_state(req, qid, &self.san.apply(&render_state(&req.state)))
    }

    /// [`Self::prompt_text`] with the state already rendered and sanitized.
    fn prompt_with_state(&self, req: &DecisionRequest, qid: &str, state: &str) -> String {
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
            state,
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
        // Choices: as many as the profile has labels for (serve_decide's MAX_OPTIONS, 256), the
        // ones past the trained slots answered by `plan_choice`; a profile from before Task 35
        // has none and stays at the trained count. Score levels: any L, mapped from JEV's 0-5
        // scale (rescale_score). max_questions is overridden by DecisionService::limits.
        let max_options = match self.p.choice_labels.len() {
            0 => self.trained_choices(),
            n => n,
        };
        Limits {
            max_questions: 32,
            max_options,
            min_levels: 2,
            max_levels: 256,
            image_parts: true,
        }
    }

    fn encode_until(
        &self,
        req: &DecisionRequest,
        cancel: &AtomicBool,
    ) -> Result<Vec<Pass>, DecisionError> {
        let mut out = Vec::with_capacity(req.questions.len());
        let mut running = 0usize;
        // Every pass carries the whole state: render and sanitize it once, not per question.
        let state = self.san.apply(&render_state(&req.state));
        for (qid, q) in &req.questions {
            super::check_cancel(cancel)?;
            // Multi-pass plans are not wired into the passes yet (Task 37): refuse a choice past
            // the trained slots rather than index past the trained head.
            if q.qtype == QuestionType::Choice {
                let n = q.choice_criteria().len();
                if n > self.trained_choices() {
                    return Err(DecisionError::BadRequest(format!(
                        "{qid}: choice questions with {n} options need a multi-pass plan, which \
                         this server does not run yet (at most {} options)",
                        self.trained_choices()
                    )));
                }
            }
            let text = self.prompt_with_state(req, qid, &state);
            // No BOS (`add_special_tokens = false`, none in the text): serve_decide does the same
            // for the Qwen-based JEV models, whose tokenizers add none. A JEV on a base that
            // expects `<bos>` (Gemma) would need it prepended here; none ships (H6).
            let enc = self
                .tok
                .encode(text.as_str(), false)
                .map_err(super::tokenize_failed)?;
            if enc.len() > self.pass_cap {
                return Err(super::pass_too_long(enc.len(), self.pass_cap));
            }
            running += enc.len();
            if running > self.total_cap {
                return Err(DecisionError::TooLong(format!(
                    "this request needs at least {running} prefill tokens; the per-request limit \
                     is {}",
                    self.total_cap
                )));
            }
            let (lo, hi) = self.slots_for(q);
            let token_ids = self.p.verbalizer_ids[lo..hi].to_vec();
            let n = token_ids.len();
            out.push(Pass {
                readout: DecisionReadout {
                    prompt_ids: enc.get_ids().to_vec(),
                    head: ReadoutHead::Verbalizer { token_ids },
                    cache_prefix: None,
                    require_warm: false,
                    needs_snapshot: false,
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
            choice_labels: vec![],
            choice_label_ids: vec![],
            strategy: JevStrategy::Single,
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
    fn empty_descriptions_and_missing_instructions_render_like_serve_decide() {
        // H7: an empty (or null) description drops the `: ` (`A) name`), and a question without
        // instructions asks its own id.
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        let r: DecisionRequest = serde_json::from_str(
            r#"{"model": "m", "state": "s", "questions": {
            "pick_one": {"type": "choice", "criteria": {"billing": "", "tech": null, "ops": "on call"}}}}"#,
        )
        .unwrap();
        assert_eq!(
            c.prompt_text(&r, "pick_one"),
            "[kind] choice\n[state] s\n[question] pick_one\n[options]\nA) billing\nB) tech\nC) ops: on call\n[decision]:"
        );
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
    fn encode_stops_at_the_first_long_pass_and_on_cancel() {
        let c = JevCodec::new(tiny_tokenizer(), profile()).unwrap();
        let full = c.encode(&req()).unwrap();
        let longest = full
            .iter()
            .map(|p| p.readout.prompt_ids.len())
            .max()
            .unwrap();
        let first = full[0].readout.prompt_ids.len();
        let capped = JevCodec::new(tiny_tokenizer(), profile())
            .unwrap()
            .with_pass_cap(first - 1);
        match capped.encode(&req()) {
            Err(DecisionError::TooLong(m)) => assert_eq!(
                m,
                format!(
                    "a decision pass is {first} tokens; the limit is {}",
                    first - 1
                )
            ),
            other => panic!("{other:?}"),
        }
        let roomy = JevCodec::new(tiny_tokenizer(), profile())
            .unwrap()
            .with_pass_cap(longest);
        assert_eq!(roomy.encode(&req()).unwrap().len(), 3);
        let gone = std::sync::atomic::AtomicBool::new(true);
        assert!(c.encode_until(&req(), &gone).is_err());
    }

    #[test]
    fn encode_stops_once_the_running_total_passes_the_request_budget() {
        // P2-SEC-4: JEV prefills every pass in full, so once the passes tokenized so far exceed
        // max_total_tokens the request is a 413 whatever follows: stop tokenizing there.
        let full = JevCodec::new(tiny_tokenizer(), profile())
            .unwrap()
            .encode(&req())
            .unwrap();
        let len = |i: usize| full[i].readout.prompt_ids.len();
        let total: usize = (0..3).map(len).sum();
        let capped = JevCodec::new(tiny_tokenizer(), profile())
            .unwrap()
            .with_total_cap(len(0) + len(1) - 1);
        match capped.encode(&req()) {
            Err(DecisionError::TooLong(m)) => assert_eq!(
                m,
                format!(
                    "this request needs at least {} prefill tokens; the per-request limit is {}",
                    len(0) + len(1),
                    len(0) + len(1) - 1
                ),
                "stopped after the second pass"
            ),
            other => panic!("{other:?}"),
        }
        let exact = JevCodec::new(tiny_tokenizer(), profile())
            .unwrap()
            .with_total_cap(total);
        assert_eq!(exact.encode(&req()).unwrap().len(), 3);
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
        // serve_decide `_parts` reads an `image` element as an image part: JEV's limits refuse it
        // (Kev and Clef read the same element as data; EDGE-1).
        assert!(c.limits().image_parts);
        let img: DecisionRequest = serde_json::from_str(r#"{"model": "m", "state": [{"image": "data:x"}], "questions": {"u": {"type": "noul"}}}"#).unwrap();
        match img.validate(&c.limits()) {
            Err(DecisionError::BadRequest(m)) => {
                assert_eq!(
                    m,
                    "image parts in state are not supported (text-only server)"
                )
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn new_rechecks_the_calibration_bounds() {
        // Task 24 M1: a profile built without `DecisionProfile::load` gets the same refusal.
        let mut p = profile();
        p.temperature.insert("score".into(), 0.0);
        let e = JevCodec::new(tiny_tokenizer(), p).err().unwrap();
        assert!(e.contains("temperature must be in"), "{e}");
        let mut p = profile();
        p.bias[3] = 1e308;
        let e = JevCodec::new(tiny_tokenizer(), p).err().unwrap();
        assert!(e.contains("bias must be in"), "{e}");
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
        match with("choice", [8, 8]) {
            Err(e) => assert!(e.contains("at least 1 wide"), "{e}"),
            Ok(_) => panic!("a zero-width choice range refuses every choice question"),
        }
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

    /// A JEV with `trained` choice slots after noul and score, and `labels` choice labels (0 for
    /// a decision.json from before Task 35). Choice slot i has bias 0.1·(i+1).
    fn wide_profile(trained: usize, labels: usize) -> JevProfile {
        let mut p = profile();
        let n = 8 + trained;
        p.verbalizers = (0..n).map(|i| format!("v{i}")).collect();
        p.verbalizer_ids = (100..100 + n as u32).collect();
        p.bias = (0..n)
            .map(|i| if i < 8 { 0.0 } else { 0.1 * (i - 7) as f64 })
            .collect();
        p.slots.insert("choice".into(), [8, n]);
        p.choice_labels = (0..labels).map(|i| format!("L{i}")).collect();
        p.choice_label_ids = (0..labels as u32)
            .map(|i| {
                if (i as usize) < trained {
                    108 + i
                } else {
                    1000 + i
                }
            })
            .collect();
        p
    }

    fn choice_req(n: usize) -> DecisionRequest {
        let criteria: Vec<String> = (0..n).map(|i| format!("\"o{i}\": \"\"")).collect();
        serde_json::from_str(&format!(
            r#"{{"model": "m", "state": "s", "questions": {{"q": {{"type": "choice", "criteria": {{{}}}}}}}}}"#,
            criteria.join(", ")
        ))
        .unwrap()
    }

    #[test]
    fn max_options_is_the_label_count_else_the_trained_slots() {
        let old = JevCodec::new(tiny_tokenizer(), wide_profile(16, 0)).unwrap();
        assert_eq!(old.limits().max_options, 16);
        assert!(choice_req(16).validate(&old.limits()).is_ok());
        match choice_req(17).validate(&old.limits()) {
            Err(DecisionError::BadRequest(m)) => assert!(m.contains("1..=16"), "{m}"),
            other => panic!("{other:?}"),
        }
        let wide = JevCodec::new(tiny_tokenizer(), wide_profile(16, 40)).unwrap();
        assert_eq!(wide.limits().max_options, 40);
        assert!(choice_req(40).validate(&wide.limits()).is_ok());
        match choice_req(41).validate(&wide.limits()) {
            Err(DecisionError::BadRequest(m)) => assert!(m.contains("1..=40"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn new_refuses_labels_that_do_not_extend_the_trained_head() {
        let mut p = wide_profile(16, 40);
        p.choice_label_ids[3] = 7;
        let e = JevCodec::new(tiny_tokenizer(), p).err().unwrap();
        assert!(e.contains("must equal the trained choice"), "{e}");
        let mut p = wide_profile(16, 40);
        p.choice_labels.pop();
        let e = JevCodec::new(tiny_tokenizer(), p).err().unwrap();
        assert!(e.contains("equal length"), "{e}");
    }

    #[test]
    fn encode_refuses_wide_choices_until_the_plans_are_wired() {
        // limits() admits up to choice_labels.len() options, but encode still builds one native
        // pass: anything past the trained slots is a 400, not an index past verbalizer_ids.
        let c = JevCodec::new(tiny_tokenizer(), wide_profile(16, 40)).unwrap();
        assert_eq!(c.encode(&choice_req(16)).unwrap().len(), 1);
        match c.encode(&choice_req(17)) {
            Err(DecisionError::BadRequest(m)) => assert!(m.contains("17 options"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn plan_is_native_up_to_the_trained_slots_whatever_the_strategy() {
        let c = JevCodec::new(tiny_tokenizer(), wide_profile(16, 256)).unwrap();
        for s in [
            JevStrategy::Single,
            JevStrategy::Tournament,
            JevStrategy::Permute,
        ] {
            for n in [1, 2, 16] {
                assert_eq!(c.plan_choice(n, s, "q"), ChoicePlan::Native, "{s:?} n={n}");
            }
        }
        assert_eq!(
            c.plan_choice(17, JevStrategy::Single, "q"),
            ChoicePlan::Single
        );
        assert_eq!(
            c.plan_choice(256, JevStrategy::Single, "q"),
            ChoicePlan::Single
        );
    }

    #[test]
    fn tournament_groups_match_serve_decide() {
        // _groups(n, 16): g = ceil(n/16) contiguous groups, the first n % g one larger.
        let c = JevCodec::new(tiny_tokenizer(), wide_profile(16, 256)).unwrap();
        let sizes = |n: usize| match c.plan_choice(n, JevStrategy::Tournament, "q") {
            ChoicePlan::Tournament { groups } => {
                let flat: Vec<usize> = groups.iter().flatten().copied().collect();
                assert_eq!(flat, (0..n).collect::<Vec<_>>(), "contiguous, in order");
                groups.iter().map(Vec::len).collect::<Vec<_>>()
            }
            other => panic!("{other:?}"),
        };
        assert_eq!(sizes(17), [9, 8]);
        assert_eq!(sizes(20), [10, 10]);
        assert_eq!(sizes(33), [11, 11, 11]);
        assert_eq!(sizes(40), [14, 13, 13]);
        assert_eq!(sizes(256), [16; 16]);
    }

    #[test]
    fn permute_orders_are_identity_then_three_cpython_samples() {
        // The second order is pyrandom's fixture for this question at n = 17.
        let c = JevCodec::new(tiny_tokenizer(), wide_profile(16, 256)).unwrap();
        let q = "Which team should handle this ticket?";
        let ChoicePlan::Permute { orders } = c.plan_choice(17, JevStrategy::Permute, q) else {
            panic!()
        };
        assert_eq!(orders[0], (0..17).collect::<Vec<_>>());
        assert_eq!(
            orders[1],
            [11, 14, 8, 16, 2, 13, 3, 5, 15, 1, 4, 9, 6, 7, 0, 10, 12]
        );
        let mut rng = crate::decisions::pyrandom::PyRandom::new(2887938576);
        for o in &orders[1..] {
            assert_eq!(o, &rng.sample_all(17));
        }
    }

    #[test]
    fn permute_combines_like_serve_decide() {
        // python3 -c 'import random,zlib
        // q="Which team should handle this ticket?"; n=17
        // rng=random.Random(zlib.crc32(q.encode()))
        // orders=[list(range(n))]+[rng.sample(range(n),n) for _ in range(3)]
        // dists=[]
        // for k in range(4):
        //     raw=[(i*7+k*3)%17+1 for i in range(n)]; s=sum(raw); dists.append([r/s for r in raw])
        // p=[0.0]*n
        // for o,r in zip(orders,dists):
        //     for i,v in zip(o,r): p[i]+=v/len(orders)
        // print([repr(x) for x in p])'
        let c = JevCodec::new(tiny_tokenizer(), wide_profile(16, 256)).unwrap();
        let ChoicePlan::Permute { orders } = c.plan_choice(
            17,
            JevStrategy::Permute,
            "Which team should handle this ticket?",
        ) else {
            panic!()
        };
        let dists: Vec<Vec<f64>> = (0..4)
            .map(|k| {
                let raw: Vec<u32> = (0..17).map(|i| (i * 7 + k * 3) % 17 + 1).collect();
                let s: u32 = raw.iter().sum();
                raw.iter().map(|&r| r as f64 / s as f64).collect()
            })
            .collect();
        let want = [
            0.06372549019607843,
            0.08986928104575163,
            0.0718954248366013,
            0.04084967320261438,
            0.06535947712418301,
            0.027777777777777776,
            0.049019607843137254,
            0.08496732026143791,
            0.042483660130718956,
            0.07352941176470588,
            0.04901960784313725,
            0.029411764705882356,
            0.07026143790849673,
            0.053921568627450976,
            0.06862745098039215,
            0.06862745098039216,
            0.05065359477124183,
        ];
        assert_eq!(combine_permute(&orders, &dists), want, "bit for bit");
    }

    #[test]
    fn tournament_final_and_combination_by_hand() {
        // n = 20: groups 0..10 and 10..20. Group 0 ties options 2 and 5 at 0.3 (the winner is 2,
        // the lower index); group 1's winner is 13 (0.55). The 14 best losers by (-p, index):
        // 5, then the 0.05s by index, 0..15 without 2 and 13; so the final is 0..16 and 16..20
        // stay out.
        let groups = vec![(0..10).collect::<Vec<_>>(), (10..20).collect::<Vec<_>>()];
        let mut g0 = vec![0.05; 10];
        g0[2] = 0.3;
        g0[5] = 0.3;
        let mut g1 = vec![0.05; 10];
        g1[3] = 0.55;
        let in_group = vec![g0, g1];
        let fin = tournament_final(&groups, &in_group);
        assert_eq!(fin, (0..16).collect::<Vec<_>>());
        // final: 2 → 0.5, 13 → 0.25, 5 and 0 → 0.125, the rest 0.
        let mut final_p = vec![0.0; 16];
        final_p[2] = 0.5;
        final_p[13] = 0.25;
        final_p[5] = 0.125;
        final_p[0] = 0.125;
        // share = [0.75, 0.25]; cap = [1.0 (all of 0..10), 0.8 (10..16)]; among = 0.75 + 0.2 =
        // 0.95. Finalists get final·0.95; 16..20 get share[1]·0.05 = 0.0125; the total is 1.
        let p = combine_tournament(&groups, &in_group, &fin, &final_p);
        let mut want = vec![0.0; 20];
        want[2] = 0.475;
        want[13] = 0.2375;
        want[5] = 0.11875;
        want[0] = 0.11875;
        for w in &mut want[16..] {
            *w = 0.0125;
        }
        for (o, (got, w)) in p.iter().zip(&want).enumerate() {
            assert!((got - w).abs() < 1e-15, "option {o}: {got} vs {w}");
        }
    }

    #[test]
    fn tournament_final_keeps_every_winner_and_tops_up_to_sixteen() {
        // 256 options in 16 groups of 16: the final is exactly the 16 group winners.
        let groups: Vec<Vec<usize>> = (0..16).map(|g| (g * 16..g * 16 + 16).collect()).collect();
        let in_group: Vec<Vec<f64>> = (0..16)
            .map(|g| {
                let mut d = vec![0.5 / 15.0; 16];
                d[g % 16] = 0.5;
                d
            })
            .collect();
        let fin = tournament_final(&groups, &in_group);
        assert_eq!(fin, (0..16).map(|g| g * 16 + g).collect::<Vec<_>>());
    }

    #[test]
    fn pass_probabilities_use_trained_bias_then_zero() {
        // serve_decide s1_pass: bias = head.bias[lo+i] if lo+i < hi else 0.0.
        let c = JevCodec::new(tiny_tokenizer(), wide_profile(16, 40)).unwrap();
        let logits: Vec<f32> = (0..20).map(|i| (i % 3) as f32).collect();
        let t = 2.0;
        let z: Vec<f64> = (0..20)
            .map(|i| (logits[i] as f64 + if i < 16 { 0.1 * (i + 1) as f64 } else { 0.0 }) / t)
            .collect();
        assert_eq!(c.choice_pass_probabilities(&logits), softmax(&z));
        // Up to the trained slots it is exactly today's per-question probabilities.
        let r = choice_req(5);
        let passes = c.encode(&r).unwrap();
        assert_eq!(
            c.choice_pass_probabilities(&logits[..5]),
            c.probabilities(&r, &passes[0].parts[0], &logits[..5])
        );
    }

    #[test]
    fn pass_labels_are_positional() {
        let c = JevCodec::new(tiny_tokenizer(), wide_profile(16, 40)).unwrap();
        let (labels, ids) = c.pass_labels(20);
        assert_eq!(labels.len(), 20);
        assert_eq!((labels[0].as_str(), labels[19].as_str()), ("L0", "L19"));
        assert_eq!(&ids[..16], &(108..124).collect::<Vec<u32>>()[..]);
        assert_eq!(ids[16], 1016);
        // Without labels: the trained verbalizers.
        let old = JevCodec::new(tiny_tokenizer(), wide_profile(16, 0)).unwrap();
        let (labels, ids) = old.pass_labels(3);
        assert_eq!(labels, ["v8", "v9", "v10"]);
        assert_eq!(ids, [108, 109, 110]);
    }
}
