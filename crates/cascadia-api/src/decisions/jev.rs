//! JEV codec (autotrust/JEV-9B, JEV-27B): ports `vl/serve_decide.py` System 1.
pub use super::profile::JevStrategy;
use super::profile::{JevProfile, ThinkProfile};
use super::pyrandom::{crc32, PyRandom};
use super::schema::{
    DecisionRequest, Limits, OrderedValue, Question, QuestionType, RequestStrategy,
};
use super::{softmax, Codec, DecisionError, Pass, PassPart, Sanitizer};
use cascadia_types::{DecisionReadout, ReadoutHead};
use indexmap::IndexMap;
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
    /// The model's window (config.json `max_position_embeddings`), when known: the engine caps
    /// decision prompts at it, which bounds System 2's read-back.
    window: Option<usize>,
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
            .and_then(|()| p.check_system2())
            .map_err(|e| format!("decision.json jev: {e}"))?;
        // System 2 stops its generation at this id and writes `end_think` before the answer: they
        // must be the same token, or the read-back would follow a channel the model never closed.
        if let Some(id) = p.end_think_id {
            let enc = tok
                .encode(p.system2.end_think.as_str(), false)
                .map_err(|e| format!("decision.json jev: end_think: {e}"))?;
            if enc.get_ids() != [id] {
                return Err(format!(
                    "decision.json jev: end_think_id {id} is not the tokenizer's {:?} (ids {:?})",
                    p.system2.end_think,
                    enc.get_ids()
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
            window: None,
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

    /// The model's window (`max_position_embeddings`), `None` when the tree does not state it.
    pub fn with_window(mut self, window: Option<usize>) -> Self {
        self.window = window;
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
                self.option_texts(q)
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("{}) {t}", self.p.verbalizers[lo + i]))
                    .collect()
            }
        };
        self.layout(q, qid, state, &lines)
    }

    /// Each choice option's line after its label, sanitized: `name`, or `name: description`
    /// (an empty or null description drops the `: `, H7).
    fn option_texts(&self, q: &Question) -> Vec<String> {
        q.choice_criteria()
            .iter()
            .map(|(name, desc)| {
                let name = self.san.apply(name);
                let d = self.san.apply(&render(desc));
                if d.is_empty() {
                    name
                } else {
                    format!("{name}: {d}")
                }
            })
            .collect()
    }

    /// serve_decide `s1_pass`'s content around the option lines.
    fn layout(&self, q: &Question, qid: &str, state: &str, lines: &[String]) -> String {
        format!(
            "[kind] {}\n[state] {}\n[question] {}\n[options]\n{}\n[decision]:",
            kind(q),
            state,
            self.san.apply(&question_raw(q, qid)),
            lines.join("\n")
        )
    }

    /// One pass's prompt ids, refused over the per-pass cap. No BOS (`add_special_tokens =
    /// false`, none in the text): serve_decide does the same for the Qwen-based JEV models, whose
    /// tokenizers add none. A JEV on a base that expects `<bos>` (Gemma) would need it prepended
    /// here; none ships (H6).
    fn tokenize_pass(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(super::tokenize_failed)?;
        if enc.len() > self.pass_cap {
            return Err(super::pass_too_long(enc.len(), self.pass_cap));
        }
        Ok(enc.get_ids().to_vec())
    }

    /// The plan for question `qid`: Native unless it is a choice past the trained slots, then
    /// the request's `strategy` (the profile's for `auto`), seeded by the question as received.
    fn plan_for(&self, req: &DecisionRequest, qid: &str) -> ChoicePlan {
        let q = &req.questions[qid];
        if q.qtype != QuestionType::Choice {
            return ChoicePlan::Native;
        }
        let strategy = req
            .strategy
            .and_then(RequestStrategy::fixed)
            .unwrap_or(self.p.strategy);
        self.plan_choice(q.choice_criteria().len(), strategy, &question_raw(q, qid))
    }

    /// Today's one pass per question, on the trained slots.
    fn native_pass(
        &self,
        req: &DecisionRequest,
        qid: &str,
        state: &str,
    ) -> Result<Pass, DecisionError> {
        let q = &req.questions[qid];
        let prompt_ids = self.tokenize_pass(&self.prompt_with_state(req, qid, state))?;
        let (lo, hi) = self.slots_for(q);
        let token_ids = self.p.verbalizer_ids[lo..hi].to_vec();
        let n = token_ids.len();
        Ok(Self::pass(qid, prompt_ids, token_ids, n))
    }

    /// One pass of a multi-pass plan over the options `idx` (indices into `texts`), labelled by
    /// position from A: serve_decide `s1_pass` over `[opts[i] for i in idx]`.
    fn plan_pass(
        &self,
        q: &Question,
        qid: &str,
        state: &str,
        texts: &[String],
        idx: &[usize],
    ) -> Result<Pass, DecisionError> {
        let (labels, ids) = self.pass_labels(idx.len());
        let lines: Vec<String> = idx
            .iter()
            .zip(labels)
            .map(|(&o, l)| format!("{l}) {}", texts[o]))
            .collect();
        let prompt_ids = self.tokenize_pass(&self.layout(q, qid, state, &lines))?;
        Ok(Self::pass(qid, prompt_ids, ids.to_vec(), idx.len()))
    }

    fn pass(qid: &str, prompt_ids: Vec<u32>, token_ids: Vec<u32>, n: usize) -> Pass {
        Pass {
            readout: DecisionReadout {
                prompt_ids,
                head: ReadoutHead::Verbalizer { token_ids },
                cache_prefix: None,
                require_warm: false,
                needs_snapshot: false,
            },
            parts: vec![PassPart {
                question_id: qid.to_string(),
                logits: 0..n,
            }],
        }
    }

    /// The longest final a tournament over `texts` can need: its TOURNAMENT_WIDTH costliest option
    /// lines (ties to the lower index), in index order and labelled from A. A line's cost is its
    /// text as the prompt holds it, between `") "` and the newline: `" {t}\n"`. Not the text
    /// alone: with Qwen's pre-tokenizer the leading space joins the first word (`"Merkel"` is 3
    /// tokens alone, `" Merkel"` 1) or splits off before a digit or CJK text, and trailing
    /// punctuation joins the newline. Where these costs add up, as they do for Qwen (the label
    /// and `)` are pretokens of their own), no final the groups pick is longer; the handler
    /// still checks the real one for tokenizers where they do not.
    fn worst_final(
        &self,
        q: &Question,
        qid: &str,
        state: &str,
        texts: &[String],
    ) -> Result<usize, DecisionError> {
        let mut by_len = texts
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let enc = self
                    .tok
                    .encode(format!(" {t}\n"), false)
                    .map_err(super::tokenize_failed)?;
                Ok((enc.len(), i))
            })
            .collect::<Result<Vec<_>, DecisionError>>()?;
        by_len.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut idx: Vec<usize> = by_len
            .iter()
            .take(TOURNAMENT_WIDTH)
            .map(|&(_, i)| i)
            .collect();
        idx.sort_unstable();
        Ok(self
            .plan_pass(q, qid, state, texts, &idx)?
            .readout
            .prompt_ids
            .len())
    }

    /// Count `len` more prefill tokens, with `reserve` still owed to second rounds: past the
    /// request's budget the request is a 413 whatever follows, so the encode stops (P2-SEC-4).
    fn charge(&self, running: &mut usize, reserve: usize, len: usize) -> Result<(), DecisionError> {
        *running += len;
        let need = *running + reserve;
        if need > self.total_cap {
            return Err(DecisionError::TooLong(format!(
                "this request needs at least {need} prefill tokens; the per-request limit is {}",
                self.total_cap
            )));
        }
        Ok(())
    }

    fn slots_for(&self, q: &Question) -> (usize, usize) {
        let [lo, hi] = self.slot(kind(q));
        match q.qtype {
            QuestionType::Choice => (lo, lo + q.choice_criteria().len()),
            _ => (lo, hi),
        }
    }
}

/// What System 2's read-back appends after the end of thinking (serve_decide `s2_dist`).
pub const ANSWER_PREFIX: &str = "Answer: (";

/// The options a System 2 noul shows, as (A) and (B) (serve_decide `s2_dist`).
const THINK_NOUL: [&str; 2] = ["Yes (true)", "No (false)"];

/// System 2's answer distribution from the read-back's label logits, in `Question::option_keys`
/// order: a plain softmax (no bias, no temperature: serve_decide `_softmax`), a noul's
/// (A) Yes / (B) No remapped to [P(false), P(true)].
pub fn system2_probabilities(qtype: QuestionType, logits: &[f32]) -> Vec<f64> {
    let z: Vec<f64> = logits.iter().map(|&l| l as f64).collect();
    let p = softmax(&z);
    match qtype {
        QuestionType::Noul => vec![p[1], p[0]],
        _ => p,
    }
}

/// serve_decide `_fold`: `(1 - w) * p1 + w * p2`.
pub fn blend(p1: &[f64], p2: &[f64], w: f64) -> Vec<f64> {
    p1.iter()
        .zip(p2)
        .map(|(x, y)| (1.0 - w) * x + w * y)
        .collect()
}

/// System 2 (Task 43): the prompts and token ids `handler::think` runs. The base model reasons
/// in the tree's chat template, then one Verbalizer readout scores the answer labels.
impl JevCodec {
    pub fn system2(&self) -> &ThinkProfile {
        &self.p.system2
    }

    /// Why this tree cannot think (`JevProfile::system2_refusal`), or `None`.
    pub fn system2_refusal(&self) -> Option<String> {
        self.p.system2_refusal()
    }

    /// The most options one System 2 pass can label.
    pub fn answer_label_count(&self) -> usize {
        self.p.answer_label_ids.len()
    }

    pub fn end_think_id(&self) -> Option<u32> {
        self.p.end_think_id
    }

    /// The longest read-back the engine admits: the model's window, at most
    /// `MAX_DECISION_PROMPT` (the engine's `decision_prompt_cap`).
    pub fn read_back_cap(&self) -> usize {
        self.window
            .map_or(cascadia_types::MAX_DECISION_PROMPT, |w| {
                w.min(cascadia_types::MAX_DECISION_PROMPT)
            })
    }

    /// The ids the read-back scores for `n` options.
    pub fn answer_ids(&self, n: usize) -> Vec<u32> {
        self.p.answer_label_ids[..n].to_vec()
    }

    /// The request's state as every pass shows it: rendered and sanitized.
    pub fn rendered_state(&self, req: &DecisionRequest) -> String {
        self.san.apply(&render_state(&req.state))
    }

    /// serve_decide `s2_dist`'s user turn inside the chat template's frame (`pre`, `post`):
    /// the state, the question, the options labelled `(A)`, `(B)`, ... and the instruction to
    /// answer as `Answer: (X)`. A choice shows System 1's option text; a noul Yes / No.
    pub fn think_prompt(
        &self,
        req: &DecisionRequest,
        qid: &str,
        state: &str,
        pre: &str,
        post: &str,
    ) -> String {
        let q = &req.questions[qid];
        let shown: Vec<String> = match q.qtype {
            QuestionType::Noul => THINK_NOUL.map(String::from).to_vec(),
            _ => self.option_texts(q),
        };
        let body: Vec<String> = shown
            .iter()
            .zip(&self.p.answer_labels)
            .map(|(o, l)| format!("({l}) {o}"))
            .collect();
        format!(
            "{pre}{state}\n\nQuestion: {}\n\nOptions:\n{}\n\nThink it through carefully, then give \
             your final answer on the last line in the form: Answer: (X){post}",
            self.san.apply(&question_raw(q, qid)),
            body.join("\n")
        )
    }

    /// What the read-back appends after the thought: `end_think`, the template's after-think
    /// text and `"Answer: ("`.
    pub fn read_back_tail(&self) -> String {
        format!(
            "{}{}{ANSWER_PREFIX}",
            self.p.system2.end_think, self.p.system2.after_think
        )
    }

    /// The read-back prompt: the thinking prompt, the thought, then [`Self::read_back_tail`].
    pub fn read_back_text(&self, prompt: &str, thought: &str) -> String {
        format!("{prompt}{thought}{}", self.read_back_tail())
    }

    /// Token ids of `text` (no BOS, as every JEV pass), without the per-pass cap: the read-back
    /// holds up to `think_budget` tokens of thought on top of a capped prompt.
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        self.tok
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(super::tokenize_failed)
    }

    /// The generated thought's text, special tokens kept (serve_decide's
    /// `skip_special_tokens=False`).
    pub fn decode(&self, ids: &[u32]) -> Result<String, DecisionError> {
        self.tok.decode(ids, false).map_err(super::tokenize_failed)
    }
}

/// The question as the caller sent it (serve_decide `req.question`): the instructions, or the
/// question id when there are none. The prompt shows it sanitized; the permute seed takes it raw.
fn question_raw(q: &Question, qid: &str) -> String {
    q.instructions
        .as_ref()
        .map(render)
        .unwrap_or_else(|| qid.to_string())
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
        self.encode_rounds(req, cancel).map(|(passes, _)| passes)
    }

    /// Round 1 of every question in request order: one native pass, or its plan's first passes
    /// (the single pass, the four permute passes, the tournament groups). A tournament's final
    /// is counted as its worst case ([`Self::worst_final`]) and returned as the reserve.
    fn encode_rounds(
        &self,
        req: &DecisionRequest,
        cancel: &AtomicBool,
    ) -> Result<(Vec<Pass>, usize), DecisionError> {
        let mut out = Vec::with_capacity(req.questions.len());
        let (mut running, mut reserve) = (0usize, 0usize);
        // Every pass carries the whole state: render and sanitize it once, not per question.
        let state = self.san.apply(&render_state(&req.state));
        for (qid, q) in &req.questions {
            super::check_cancel(cancel)?;
            let (rounds, with_final): (Vec<Vec<usize>>, bool) = match self.plan_for(req, qid) {
                ChoicePlan::Native => {
                    let pass = self.native_pass(req, qid, &state)?;
                    self.charge(&mut running, reserve, pass.readout.prompt_ids.len())?;
                    out.push(pass);
                    continue;
                }
                ChoicePlan::Single => (vec![(0..q.choice_criteria().len()).collect()], false),
                ChoicePlan::Permute { orders } => (orders.into(), false),
                ChoicePlan::Tournament { groups } => (groups, true),
            };
            let texts = self.option_texts(q);
            if with_final {
                let worst = self.worst_final(q, qid, &state, &texts)?;
                reserve += worst;
                self.charge(&mut running, reserve, 0)?;
            }
            for idx in &rounds {
                super::check_cancel(cancel)?;
                let pass = self.plan_pass(q, qid, &state, &texts, idx)?;
                self.charge(&mut running, reserve, pass.readout.prompt_ids.len())?;
                out.push(pass);
            }
        }
        Ok((out, reserve))
    }

    /// Each tournament's final, in request order: the finalists of its groups' passes
    /// ([`tournament_final`]) in index order, labelled from A.
    fn second_round(
        &self,
        req: &DecisionRequest,
        first: &IndexMap<String, Vec<Vec<f64>>>,
        cancel: &AtomicBool,
    ) -> Result<Vec<Pass>, DecisionError> {
        let mut out = Vec::new();
        let mut state = None;
        for (qid, q) in &req.questions {
            let ChoicePlan::Tournament { groups } = self.plan_for(req, qid) else {
                continue;
            };
            super::check_cancel(cancel)?;
            let Some(in_group) = first.get(qid).filter(|d| d.len() == groups.len()) else {
                return Err(DecisionError::Engine("decision engine error".into()));
            };
            let fin = tournament_final(&groups, in_group);
            let state: &String =
                state.get_or_insert_with(|| self.san.apply(&render_state(&req.state)));
            out.push(self.plan_pass(q, qid, state, &self.option_texts(q), &fin)?);
        }
        Ok(out)
    }

    fn probabilities(&self, req: &DecisionRequest, part: &PassPart, logits: &[f32]) -> Vec<f64> {
        let q = &req.questions[&part.question_id];
        if q.qtype == QuestionType::Choice && q.choice_criteria().len() > self.trained_choices() {
            // One pass of a multi-pass plan, over its own options by position.
            return self.choice_pass_probabilities(logits);
        }
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

    /// serve_decide `s1_dist`'s combination for the question's plan.
    fn combine(&self, req: &DecisionRequest, qid: &str, passes: &[Vec<f64>]) -> Option<Vec<f64>> {
        match self.plan_for(req, qid) {
            ChoicePlan::Native | ChoicePlan::Single => match passes {
                [p] => Some(p.clone()),
                _ => None,
            },
            ChoicePlan::Permute { orders } => {
                (passes.len() == orders.len()).then(|| combine_permute(&orders, passes))
            }
            ChoicePlan::Tournament { groups } => {
                let (in_group, last) = passes.split_at(groups.len().min(passes.len()));
                let [final_p] = last else {
                    return None;
                };
                let fin = tournament_final(&groups, in_group);
                Some(combine_tournament(&groups, in_group, &fin, final_p))
            }
        }
    }

    fn as_jev(&self) -> Option<&JevCodec> {
        Some(self)
    }

    fn dump_strategy(&self, req: &DecisionRequest, qid: &str) -> Option<&'static str> {
        if req.questions[qid].qtype != QuestionType::Choice {
            return None;
        }
        Some(match self.plan_for(req, qid) {
            ChoicePlan::Native => "native",
            ChoicePlan::Single => "single",
            ChoicePlan::Permute { .. } => "permute",
            ChoicePlan::Tournament { .. } => "tournament",
        })
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
            ..Default::default()
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

    /// One token per character, id = the byte, so a pass's text decodes from its prompt ids.
    fn char_tokenizer() -> tokenizers::Tokenizer {
        let mut vocab: std::collections::HashMap<String, u32> = (32u8..127)
            .map(|b| ((b as char).to_string(), b as u32))
            .collect();
        vocab.insert("\n".into(), 10);
        vocab.insert("[unk]".into(), 0);
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[unk]".into())
            .build()
            .unwrap();
        let mut t = tokenizers::Tokenizer::new(model);
        t.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::split::Split::new(
                tokenizers::pre_tokenizers::split::SplitPattern::Regex("".into()),
                tokenizers::SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap(),
        ));
        t
    }

    const TICKET: &str = "Which team should handle this ticket?";

    /// One choice question `q` over o0..o{n-1} asking `instructions`, with a request `strategy`.
    fn wide_req(n: usize, strategy: &str, instructions: &str) -> DecisionRequest {
        let criteria: Vec<String> = (0..n).map(|i| format!("\"o{i}\": \"\"")).collect();
        let strategy = match strategy {
            "" => String::new(),
            s => format!(r#""strategy": "{s}", "#),
        };
        serde_json::from_str(&format!(
            r#"{{"model": "m", "state": "s", {strategy}"questions": {{"q": {{"type": "choice", "instructions": {}, "criteria": {{{}}}}}}}}}"#,
            serde_json::to_string(instructions).unwrap(),
            criteria.join(", ")
        ))
        .unwrap()
    }

    fn head_ids(p: &Pass) -> Vec<u32> {
        match &p.readout.head {
            ReadoutHead::Verbalizer { token_ids } => token_ids.clone(),
            h => panic!("{h:?}"),
        }
    }

    fn decoded(p: &Pass) -> String {
        p.readout
            .prompt_ids
            .iter()
            .map(|&i| i as u8 as char)
            .collect()
    }

    /// The prompt of a pass over options `idx`, labelled by position from L0.
    fn sub_prompt(idx: &[usize], instructions: &str) -> String {
        let lines: Vec<String> = idx
            .iter()
            .enumerate()
            .map(|(i, o)| format!("L{i}) o{o}"))
            .collect();
        format!(
            "[kind] choice\n[state] s\n[question] {instructions}\n[options]\n{}\n[decision]:",
            lines.join("\n")
        )
    }

    fn label_ids(m: usize) -> Vec<u32> {
        wide_profile(16, 256).choice_label_ids[..m].to_vec()
    }

    #[test]
    fn native_choices_are_unchanged_by_labels_and_strategy() {
        // Up to the trained slots the pass is today's: the trained verbalizers (v8.. here, not the
        // L0.. labels), their ids and the per-question probabilities, whatever the strategy.
        let old = JevCodec::new(char_tokenizer(), wide_profile(16, 0)).unwrap();
        let wide = JevCodec::new(char_tokenizer(), wide_profile(16, 256)).unwrap();
        let logits: Vec<f32> = (0..16).map(|i| (i % 5) as f32 * 0.7).collect();
        for n in [1, 2, 16] {
            let base = wide_req(n, "", TICKET);
            let want = old.encode(&base).unwrap();
            assert_eq!(want.len(), 1);
            assert_eq!(decoded(&want[0]), old.prompt_text(&base, "q"));
            assert!(
                decoded(&want[0]).contains("\nv8) o0\n"),
                "{}",
                decoded(&want[0])
            );
            let want_p = old.probabilities(&base, &want[0].parts[0], &logits[..n]);
            for s in ["", "auto", "single", "tournament", "permute"] {
                let r = wide_req(n, s, TICKET);
                let (got, reserve) = wide.encode_rounds(&r, &AtomicBool::new(false)).unwrap();
                assert_eq!(got.len(), 1, "{s} n={n}");
                assert_eq!(reserve, 0);
                assert_eq!(got[0].readout.prompt_ids, want[0].readout.prompt_ids, "{s}");
                assert_eq!(head_ids(&got[0]), head_ids(&want[0]), "{s}");
                assert_eq!(got[0].parts[0].logits, 0..n);
                assert_eq!(
                    wide.probabilities(&r, &got[0].parts[0], &logits[..n]),
                    want_p,
                    "bit for bit"
                );
                assert!(wide
                    .second_round(&r, &IndexMap::new(), &AtomicBool::new(false))
                    .unwrap()
                    .is_empty());
                assert_eq!(wide.dump_strategy(&r, "q"), Some("native"));
            }
        }
    }

    #[test]
    fn single_plan_is_one_pass_over_every_label() {
        let c = JevCodec::new(char_tokenizer(), wide_profile(16, 256)).unwrap();
        let r = wide_req(20, "single", TICKET);
        let (passes, reserve) = c.encode_rounds(&r, &AtomicBool::new(false)).unwrap();
        assert_eq!((passes.len(), reserve), (1, 0));
        let all: Vec<usize> = (0..20).collect();
        assert_eq!(decoded(&passes[0]), sub_prompt(&all, TICKET));
        assert_eq!(head_ids(&passes[0]), label_ids(20));
        assert_eq!(passes[0].parts[0].logits, 0..20);
        let logits: Vec<f32> = (0..20).map(|i| (i % 3) as f32).collect();
        let p = c.probabilities(&r, &passes[0].parts[0], &logits);
        assert_eq!(p, c.choice_pass_probabilities(&logits));
        assert_eq!(c.combine(&r, "q", std::slice::from_ref(&p)), Some(p));
        assert_eq!(c.dump_strategy(&r, "q"), Some("single"));
    }

    #[test]
    fn permute_plan_is_four_passes_relabelled_from_a() {
        let c = JevCodec::new(char_tokenizer(), wide_profile(16, 256)).unwrap();
        let r = wide_req(20, "permute", TICKET);
        let (passes, reserve) = c.encode_rounds(&r, &AtomicBool::new(false)).unwrap();
        assert_eq!((passes.len(), reserve), (4, 0));
        let ChoicePlan::Permute { orders } = c.plan_choice(20, JevStrategy::Permute, TICKET) else {
            panic!()
        };
        for (p, o) in passes.iter().zip(&orders) {
            assert_eq!(decoded(p), sub_prompt(o, TICKET));
            assert_eq!(head_ids(p), label_ids(20));
            assert_eq!(p.parts[0].logits, 0..20);
        }
        let dists: Vec<Vec<f64>> = (0..4)
            .map(|k| {
                let logits: Vec<f32> = (0..20).map(|i| ((i * 7 + k * 3) % 11) as f32).collect();
                c.choice_pass_probabilities(&logits)
            })
            .collect();
        assert_eq!(
            c.combine(&r, "q", &dists),
            Some(combine_permute(&orders, &dists))
        );
        assert_eq!(c.combine(&r, "q", &dists[..3]), None);
        assert_eq!(c.dump_strategy(&r, "q"), Some("permute"));
    }

    /// Task 36's hand-computed n = 20 tournament: group passes, final pass and the combination.
    fn tournament_by_hand() -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let mut g0 = vec![0.05; 10];
        g0[2] = 0.3;
        g0[5] = 0.3;
        let mut g1 = vec![0.05; 10];
        g1[3] = 0.55;
        let mut final_p = vec![0.0; 16];
        final_p[2] = 0.5;
        final_p[13] = 0.25;
        final_p[5] = 0.125;
        final_p[0] = 0.125;
        let mut want = vec![0.0; 20];
        want[2] = 0.475;
        want[13] = 0.2375;
        want[5] = 0.11875;
        want[0] = 0.11875;
        for w in &mut want[16..] {
            *w = 0.0125;
        }
        (vec![g0, g1], final_p, want)
    }

    #[test]
    fn tournament_plan_is_the_groups_then_a_final_over_the_finalists() {
        let c = JevCodec::new(char_tokenizer(), wide_profile(16, 256)).unwrap();
        let r = wide_req(20, "tournament", TICKET);
        let (passes, reserve) = c.encode_rounds(&r, &AtomicBool::new(false)).unwrap();
        assert_eq!(passes.len(), 2);
        let g: Vec<Vec<usize>> = vec![(0..10).collect(), (10..20).collect()];
        for (p, idx) in passes.iter().zip(&g) {
            // Each group is labelled from A, so it reads the trained slots and their bias.
            assert_eq!(decoded(p), sub_prompt(idx, TICKET));
            assert_eq!(head_ids(p), (108..118).collect::<Vec<u32>>());
        }
        // The final is budgeted as its worst case: the 16 longest option lines (o10..o19, then
        // o0..o5 by index), in index order and labelled from A.
        let worst: Vec<usize> = (0..6).chain(10..20).collect();
        assert_eq!(reserve, sub_prompt(&worst, TICKET).len());
        let (in_group, final_p, want) = tournament_by_hand();
        let first: IndexMap<String, Vec<Vec<f64>>> = [("q".to_string(), in_group.clone())].into();
        let fin = c.second_round(&r, &first, &AtomicBool::new(false)).unwrap();
        assert_eq!(fin.len(), 1);
        assert_eq!(
            decoded(&fin[0]),
            sub_prompt(&(0..16).collect::<Vec<_>>(), TICKET)
        );
        assert_eq!(head_ids(&fin[0]), (108..124).collect::<Vec<u32>>());
        assert_eq!(fin[0].parts[0].logits, 0..16);
        assert!(fin[0].readout.prompt_ids.len() <= reserve);
        let mut all = in_group.clone();
        all.push(final_p);
        let p = c.combine(&r, "q", &all).unwrap();
        for (o, (got, w)) in p.iter().zip(&want).enumerate() {
            assert!((got - w).abs() < 1e-15, "option {o}: {got} vs {w}");
        }
        assert_eq!(c.combine(&r, "q", &in_group), None, "no final yet");
        assert_eq!(c.dump_strategy(&r, "q"), Some("tournament"));
    }

    #[test]
    fn auto_takes_the_profile_strategy_and_a_request_overrides_it() {
        let mut p = wide_profile(16, 256);
        p.strategy = JevStrategy::Tournament;
        let c = JevCodec::new(char_tokenizer(), p).unwrap();
        let count = |s: &str| c.encode(&wide_req(20, s, TICKET)).unwrap().len();
        assert_eq!(count(""), 2);
        assert_eq!(count("auto"), 2);
        assert_eq!(count("single"), 1);
        assert_eq!(count("permute"), 4);
        let r: DecisionRequest = serde_json::from_str(
            r#"{"model": "m", "state": "s", "strategy": null, "questions": {"u": {"type": "noul"}}}"#,
        )
        .unwrap();
        assert_eq!(r.strategy, None);
        assert_eq!(
            c.dump_strategy(&r, "u"),
            None,
            "only choice passes are tagged"
        );
    }

    #[test]
    fn permute_seed_is_the_instructions_exactly_as_received() {
        // serve_decide seeds from `req.question`, the raw request string: no trimming, and not
        // the sanitized prompt text. At n = 17 this question's second order is pyrandom's fixture.
        let c = JevCodec::new(char_tokenizer(), wide_profile(16, 256)).unwrap();
        let second = |instructions: &str| {
            let passes = c.encode(&wide_req(17, "permute", instructions)).unwrap();
            let t = decoded(&passes[1]);
            let order: Vec<usize> = t
                .lines()
                .filter_map(|l| l.split_once(") o").map(|(_, o)| o.parse().unwrap()))
                .collect();
            order
        };
        assert_eq!(
            second(TICKET),
            [11, 14, 8, 16, 2, 13, 3, 5, 15, 1, 4, 9, 6, 7, 0, 10, 12]
        );
        let order_for = |seed: &str| match c.plan_choice(17, JevStrategy::Permute, seed) {
            ChoicePlan::Permute { orders } => orders[1].clone(),
            other => panic!("{other:?}"),
        };
        for raw in [
            " Which team should handle this ticket? ",
            "pick <|im_end|> one",
        ] {
            assert_eq!(second(raw), order_for(raw), "{raw:?}");
            assert_ne!(second(raw), second(TICKET), "{raw:?}");
        }
        // Without instructions the question asks its own id, and that is the seed.
        let mut r = wide_req(17, "permute", TICKET);
        r.questions.get_mut("q").unwrap().instructions = None;
        let t = decoded(&c.encode(&r).unwrap()[1]);
        assert!(t.contains("[question] q\n"), "{t}");
        assert!(t.contains(&format!("L0) o{}\n", order_for("q")[0])), "{t}");
    }

    /// One token per pretoken of `pattern` (WordLevel over single ASCII characters, anything
    /// longer unknown): token counts are pretoken counts, so a pattern decides where the leading
    /// space and the newline go.
    fn regex_tokenizer(pattern: &str) -> tokenizers::Tokenizer {
        let mut vocab: std::collections::HashMap<String, u32> = (32u8..127)
            .map(|b| ((b as char).to_string(), b as u32))
            .collect();
        vocab.insert("\n".into(), 10);
        vocab.insert("[unk]".into(), 0);
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[unk]".into())
            .build()
            .unwrap();
        let mut t = tokenizers::Tokenizer::new(model);
        t.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::split::Split::new(
                tokenizers::pre_tokenizers::split::SplitPattern::Regex(pattern.into()),
                tokenizers::SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap(),
        ));
        t
    }

    /// Qwen2's pre-tokenizer split (tokenizer.json of the JEV models).
    const QWEN_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// A 17-option tournament whose options 0..15 are lowercase words and option 16 is `last`,
    /// played so that option 16 wins group 2 and option 0 is the weakest loser: the final is
    /// 1..=16. Returns (reserve, real final length).
    fn reserve_and_real_final(tok: tokenizers::Tokenizer, last: &str) -> (usize, usize) {
        let mut names: Vec<String> = (0..16)
            .map(|i| format!("w{}", ((b'b' + i) as char).to_string().repeat(5)))
            .collect();
        names.push(last.to_string());
        let criteria: Vec<String> = names.iter().map(|n| format!("\"{n}\": \"\"")).collect();
        let r: DecisionRequest = serde_json::from_str(&format!(
            r#"{{"model": "m", "state": "s", "strategy": "tournament", "questions": {{"q": {{"type": "choice", "instructions": "Pick", "criteria": {{{}}}}}}}}}"#,
            criteria.join(", ")
        ))
        .unwrap();
        let c = JevCodec::new(tok, wide_profile(16, 256)).unwrap();
        let (passes, reserve) = c.encode_rounds(&r, &AtomicBool::new(false)).unwrap();
        assert_eq!(passes.len(), 2, "groups 0..9 and 9..17");
        let mut g0 = vec![0.99 / 8.0; 9];
        g0[0] = 0.01;
        let mut g1 = vec![0.5 / 7.0; 8];
        g1[7] = 0.5;
        let first: IndexMap<String, Vec<Vec<f64>>> = [("q".to_string(), vec![g0, g1])].into();
        let fin = c.second_round(&r, &first, &AtomicBool::new(false)).unwrap();
        let text: String = fin[0]
            .readout
            .prompt_ids
            .iter()
            .map(|&i| i as u8 as char)
            .collect();
        assert!(
            !text.contains("wbbbbb") && text.contains(last),
            "final is 1..=16: {text}"
        );
        (reserve, fin[0].readout.prompt_ids.len())
    }

    #[test]
    fn worst_final_counts_each_line_as_the_prompt_holds_it() {
        // I1: a tokenizer that joins a leading space to a lowercase word. Alone, "wbbbbb" is 6
        // tokens and "XYZ" 3; in the prompt " wbbbbb" is 1 and " XYZ" 4. Ranking the texts alone
        // reserved a final of 16 words, and the real final with XYZ in it was 3 tokens longer.
        let (reserve, real) = reserve_and_real_final(regex_tokenizer(r" [a-z]+|[\s\S]"), "XYZ");
        assert!(real <= reserve, "real final {real} > reserve {reserve}");
        assert_eq!(
            real, reserve,
            "the line costs add up here, so the bound is tight"
        );
    }

    #[test]
    fn worst_final_covers_the_real_final_with_qwens_split() {
        // Qwen's split: "wbbbbb" and "7" are one pretoken each alone, but in the prompt the line
        // " wbbbbb\n" is 2 (" wbbbbb", "\n") and " 7\n" is 3 (" ", "7", "\n").
        let (reserve, real) = reserve_and_real_final(regex_tokenizer(QWEN_SPLIT), "7");
        assert!(real <= reserve, "real final {real} > reserve {reserve}");
        assert_eq!(real, reserve);
    }

    #[test]
    fn every_pass_and_the_worst_final_count_against_the_budgets_up_front() {
        let r = wide_req(20, "tournament", TICKET);
        let (passes, reserve) = JevCodec::new(char_tokenizer(), wide_profile(16, 256))
            .unwrap()
            .encode_rounds(&r, &AtomicBool::new(false))
            .unwrap();
        let round1: usize = passes.iter().map(|p| p.readout.prompt_ids.len()).sum();
        let need = round1 + reserve;
        let capped = JevCodec::new(char_tokenizer(), wide_profile(16, 256))
            .unwrap()
            .with_total_cap(need - 1);
        match capped.encode_rounds(&r, &AtomicBool::new(false)) {
            Err(DecisionError::TooLong(m)) => assert_eq!(
                m,
                format!(
                    "this request needs at least {need} prefill tokens; the per-request limit is {}",
                    need - 1
                )
            ),
            other => panic!("{other:?}"),
        }
        let exact = JevCodec::new(char_tokenizer(), wide_profile(16, 256))
            .unwrap()
            .with_total_cap(need);
        assert!(exact.encode_rounds(&r, &AtomicBool::new(false)).is_ok());
        // The worst final is a pass too: over the per-pass cap it is refused before any runs.
        let longest = passes
            .iter()
            .map(|p| p.readout.prompt_ids.len())
            .max()
            .unwrap();
        assert!(reserve > longest);
        let short = JevCodec::new(char_tokenizer(), wide_profile(16, 256))
            .unwrap()
            .with_pass_cap(reserve - 1);
        match short.encode_rounds(&r, &AtomicBool::new(false)) {
            Err(DecisionError::TooLong(m)) => assert_eq!(
                m,
                format!(
                    "a decision pass is {reserve} tokens; the limit is {}",
                    reserve - 1
                )
            ),
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

#[cfg(test)]
mod system2_tests {
    //! Task 43: System 2's prompts and arithmetic (serve_decide `s2_dist` and `_fold`).
    use super::*;
    use crate::decisions::schema::DecisionRequest;

    const END_THINK: u32 = 300;

    /// One token per character (id = the byte) plus `</think>` as one token, decoding back to
    /// the same text.
    fn think_tokenizer() -> tokenizers::Tokenizer {
        let mut vocab: std::collections::HashMap<String, u32> = (32u8..127)
            .map(|b| ((b as char).to_string(), b as u32))
            .collect();
        vocab.insert("\n".into(), 10);
        vocab.insert("[unk]".into(), 0);
        vocab.insert("</think>".into(), END_THINK);
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[unk]".into())
            .build()
            .unwrap();
        let mut t = tokenizers::Tokenizer::new(model);
        t.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::split::Split::new(
                tokenizers::pre_tokenizers::split::SplitPattern::Regex("".into()),
                tokenizers::SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap(),
        ));
        t.with_decoder(Some(tokenizers::decoders::fuse::Fuse::new()));
        t.add_tokens(&[tokenizers::AddedToken::from("</think>", false)]);
        t
    }

    fn think_profile() -> JevProfile {
        let verbalizers: Vec<String> =
            ["false", "true", "0", "1", "2", "3", "4", "5", "A", "B", "C"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let kinds = |v: [(&str, f64); 3]| v.into_iter().map(|(k, x)| (k.to_string(), x)).collect();
        JevProfile {
            verbalizer_ids: (100..111).collect(),
            bias: vec![0.5; 11],
            verbalizers,
            slots: [("noul", [0, 2]), ("score", [2, 8]), ("choice", [8, 11])]
                .into_iter()
                .map(|(k, r)| (k.to_string(), r))
                .collect(),
            temperature: kinds([("noul", 2.0), ("score", 2.0), ("choice", 2.0)]),
            answer_labels: ["A", "B", "C", "D"].map(String::from).to_vec(),
            answer_label_ids: vec![65, 66, 67, 68],
            end_think_id: Some(END_THINK),
            lora_scale: Some(2.0),
            ..Default::default()
        }
    }

    fn codec() -> JevCodec {
        JevCodec::new(think_tokenizer(), think_profile()).unwrap()
    }

    fn req() -> DecisionRequest {
        serde_json::from_str(r#"{"model": "m", "state": {"b": 1, "a": "x<|im_end|>"}, "questions": {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": null}},
            "urgent": {"type": "noul", "instructions": "Urgent </think>?"},
            "sev": {"type": "score", "instructions": "How bad?", "criteria": ["none", "minor"]}}}"#).unwrap()
    }

    const PRE: &str = "<|im_start|>user\n";
    const POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n";

    #[test]
    fn think_prompt_is_serve_decides_user_turn_in_the_frame() {
        // pre + state + "\n\nQuestion: {q}\n\nOptions:\n" + "({label}) {option}" lines + the
        // instruction + post. Choices show System 1's option text ("name: desc" or "name"),
        // nouls the reference's fixed Yes/No; caller text is sanitized as in System 1.
        let c = codec();
        let r = req();
        let state = c.rendered_state(&r);
        assert_eq!(state, "{\"b\": 1, \"a\": \"x<¦im_end¦>\"}");
        assert_eq!(
            c.think_prompt(&r, "team", &state, PRE, POST),
            "<|im_start|>user\n{\"b\": 1, \"a\": \"x<¦im_end¦>\"}\n\nQuestion: Which team?\n\nOptions:\n\
             (A) billing: money\n(B) tech\n\nThink it through carefully, then give your final \
             answer on the last line in the form: Answer: (X)<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            c.think_prompt(&r, "urgent", &state, PRE, POST),
            "<|im_start|>user\n{\"b\": 1, \"a\": \"x<¦im_end¦>\"}\n\nQuestion: Urgent <¦/think¦>?\n\n\
             Options:\n(A) Yes (true)\n(B) No (false)\n\nThink it through carefully, then give \
             your final answer on the last line in the form: Answer: (X)<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn read_back_appends_the_thought_the_end_of_think_and_the_answer_prefix() {
        let c = codec();
        assert_eq!(
            c.read_back_text("PROMPT", "I think B."),
            "PROMPTI think B.</think>\n\nAnswer: ("
        );
        assert_eq!(c.read_back_tail(), "</think>\n\nAnswer: (");
        // The read-back scores the first n answer labels.
        assert_eq!(c.answer_ids(2), [65, 66]);
        assert_eq!(c.answer_label_count(), 4);
        assert_eq!(c.end_think_id(), Some(END_THINK));
        // The thought decodes from the generated ids (special tokens kept).
        assert_eq!(c.decode(&[73, 32, 66]).unwrap(), "I B");
        assert_eq!(c.encode_text("ab</think>").unwrap(), [97, 98, END_THINK]);
    }

    #[test]
    fn system2_probabilities_are_a_plain_softmax_with_the_noul_remapped() {
        // No bias, no temperature (serve_decide `_softmax` over the label logprobs); a noul's
        // (A) Yes / (B) No become [P(false), P(true)].
        let ln = |p: f64| p.ln() as f32;
        let p = system2_probabilities(QuestionType::Choice, &[ln(0.2), ln(0.5), ln(0.3)]);
        for (a, b) in p.iter().zip([0.2, 0.5, 0.3]) {
            assert!((a - b).abs() < 1e-6, "{p:?}");
        }
        let p = system2_probabilities(QuestionType::Noul, &[ln(0.9), ln(0.1)]);
        assert!(
            (p[0] - 0.1).abs() < 1e-6 && (p[1] - 0.9).abs() < 1e-6,
            "{p:?}"
        );
    }

    #[test]
    fn blend_weighs_system_two_by_mix() {
        assert_eq!(blend(&[0.6, 0.4], &[0.0, 1.0], 0.5), vec![0.3, 0.7]);
        assert_eq!(blend(&[0.6, 0.4], &[0.0, 1.0], 0.0), vec![0.6, 0.4]);
        let p = blend(&[0.2, 0.3, 0.5], &[1.0, 0.0, 0.0], 0.25);
        for (a, b) in p.iter().zip([0.4, 0.225, 0.375]) {
            assert!((a - b).abs() < 1e-12, "{p:?}");
        }
    }

    #[test]
    fn new_checks_the_end_of_think_token_and_the_system2_fields() {
        let mut p = think_profile();
        p.end_think_id = Some(301);
        let e = JevCodec::new(think_tokenizer(), p).err().unwrap();
        assert!(e.contains("end_think_id"), "{e}");
        let mut p = think_profile();
        p.system2.mix = 2.0;
        assert!(JevCodec::new(think_tokenizer(), p)
            .err()
            .unwrap()
            .contains("mix"));
        let c = codec();
        assert_eq!(c.system2_refusal(), None);
        let mut p = think_profile();
        p.lora_scale = None;
        let c = JevCodec::new(think_tokenizer(), p).unwrap();
        assert!(c.system2_refusal().unwrap().contains("switchable"));
    }
}
