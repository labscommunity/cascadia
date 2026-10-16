use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Calibration bounds checked at load. Shipped profiles use temperatures near 1 (JEV ~0.98,
/// Kev ~2.19) and biases of a few tenths; with these bounds `(logit + bias) / temperature` is at
/// most ~3.4e41 for any finite f32 logit, far inside f64.
pub const MIN_TEMPERATURE: f64 = 1e-3;
pub const MAX_TEMPERATURE: f64 = 1e3;
pub const MAX_ABS_BIAS: f64 = 1e4;
/// Most choice labels a JEV profile carries: serve_decide's MAX_OPTIONS, mirrored as
/// `MAX_CHOICE_LABELS` in tools/decision_export.py (pinned by its test).
pub const MAX_CHOICE_LABELS: usize = 256;

/// How JEV answers a choice question with more options than trained choice slots
/// (serve_decide `s1_dist`): one pass over all of them, a tournament of groups of at most 16 and
/// a final of 16, or the mean of four passes over permuted orders. Default `single`, the
/// reference's default for the Qwen-based JEV models.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JevStrategy {
    #[default]
    Single,
    Tournament,
    Permute,
}

impl FromStr for JevStrategy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "single" => Ok(Self::Single),
            "tournament" => Ok(Self::Tournament),
            "permute" => Ok(Self::Permute),
            other => Err(format!(
                "unknown JEV strategy {other:?}: expected single, tournament or permute"
            )),
        }
    }
}

/// System 2 (adaptive thinking, serve_decide `thinking`): never, when System 1's leading option
/// is below the threshold, or always. Score questions never think.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    #[default]
    Off,
    Auto,
    On,
}

/// serve_decide `PROFILES["qwen"]`'s adaptive-thinking settings, the defaults for every field:
/// thinking off, threshold 0.8, mix 0.5, and Qwen's thinking channel (`</think>`, then `"\n\n"`
/// before the answer). decision.json may override them under `jev.system2`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ThinkProfile {
    /// The mode a request without `thinking` runs.
    #[serde(default)]
    pub thinking: ThinkingMode,
    /// `auto` thinks when System 1's leading probability is below this.
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    /// System 2's weight in the blend `(1 - mix) * p1 + mix * p2`.
    #[serde(default = "default_mix")]
    pub mix: f64,
    /// The text that closes the thinking channel (its token is `jev.end_think_id`).
    #[serde(default = "default_end_think")]
    pub end_think: String,
    /// What the template puts between `end_think` and the answer.
    #[serde(default = "default_after_think")]
    pub after_think: String,
}

fn default_threshold() -> f64 {
    0.8
}

fn default_mix() -> f64 {
    0.5
}

fn default_end_think() -> String {
    "</think>".into()
}

fn default_after_think() -> String {
    "\n\n".into()
}

impl Default for ThinkProfile {
    fn default() -> Self {
        Self {
            thinking: ThinkingMode::Off,
            threshold: default_threshold(),
            mix: default_mix(),
            end_think: default_end_think(),
            after_think: default_after_think(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    Jev,
    Kev,
    Clef,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct JevProfile {
    pub verbalizers: Vec<String>,
    pub verbalizer_ids: Vec<u32>,
    pub bias: Vec<f64>,
    /// kind → [lo, hi) slot range into verbalizer_ids/bias.
    pub slots: HashMap<String, [usize; 2]>,
    /// kind → calibration temperature.
    pub temperature: HashMap<String, f64>,
    /// Single-token choice labels (A..Z, AA..) and their ids, at most MAX_CHOICE_LABELS, the
    /// first `hi - lo` being the trained choice slots (Task 35's export). A pass over m options
    /// labels them `[..m]`. Empty in a decision.json from before Task 35: max 16 options then.
    #[serde(default)]
    pub choice_labels: Vec<String>,
    #[serde(default)]
    pub choice_label_ids: Vec<u32>,
    /// The plan for choices beyond the trained slots.
    #[serde(default)]
    pub strategy: JevStrategy,
    /// System 2's answer labels (A..Z, AA.., at most MAX_CHOICE_LABELS) and their ids: the
    /// tokens after `"Answer: ("` the read-back scores (serve_decide's `ans_labels`, Task 43's
    /// export). Empty in an older decision.json: no thinking then.
    #[serde(default)]
    pub answer_labels: Vec<String>,
    #[serde(default)]
    pub answer_label_ids: Vec<u32>,
    /// The token that closes the thinking channel (`</think>` for Qwen); System 2's generation
    /// stops there. Absent: no thinking.
    #[serde(default)]
    pub end_think_id: Option<u32>,
    /// The trained scale of a switchable tree (`--switchable-lora`, Task 41). Present means the
    /// engine can run the base model (`lora_scale = 0`), which System 2 needs; absent means the
    /// adapter is baked in and thinking is refused.
    #[serde(default)]
    pub lora_scale: Option<f64>,
    /// Adaptive-thinking settings (serve_decide `PROFILES`).
    #[serde(default)]
    pub system2: ThinkProfile,
}

#[derive(Clone, Debug, Deserialize)]
pub struct KevProfile {
    pub temperature: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ClefProfile {}

#[derive(Clone, Debug, Deserialize)]
pub struct DecisionProfile {
    pub family: Family,
    /// Longest admitted pass in tokens.
    pub max_tokens: usize,
    /// Tokens one request may prefill, shared prefix counted once (`prefill_work`; default 4 × max_tokens).
    #[serde(default)]
    pub max_total_tokens: Option<usize>,
    /// Questions per request (default 32). JEV and Kev run one pass per question.
    #[serde(default)]
    pub max_questions: Option<usize>,
    #[serde(default)]
    pub jev: Option<JevProfile>,
    #[serde(default)]
    pub kev: Option<KevProfile>,
    #[serde(default)]
    pub clef: Option<ClefProfile>,
}

impl DecisionProfile {
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let p: DecisionProfile =
            serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
        let ok = match p.family {
            Family::Jev => p.jev.is_some(),
            Family::Kev => p.kev.is_some(),
            Family::Clef => true,
        };
        if !ok {
            return Err(format!(
                "{}: family {:?} needs its `{}` section",
                path.display(),
                p.family,
                format!("{:?}", p.family).to_lowercase()
            ));
        }
        let checked = p.jev.as_ref().map_or(Ok(()), |j| {
            j.check_calibration()
                .and_then(|()| j.check_choice_labels())
                .and_then(|()| j.check_system2())
        });
        checked
            .and(p.kev.as_ref().map_or(Ok(()), KevProfile::check_calibration))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(p)
    }
}

/// A zero, negative or non-finite temperature turns every probability into NaN, and so does
/// one small enough (or a bias large enough) for `(logit + bias) / temperature` to overflow
/// (H17). Within these bounds it stays finite for every finite f32 logit.
fn check_calibration<'a>(
    temps: impl IntoIterator<Item = &'a f64>,
    biases: impl IntoIterator<Item = &'a f64>,
) -> Result<(), String> {
    if let Some(t) = temps
        .into_iter()
        .copied()
        .find(|t| !(MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(t))
    {
        return Err(format!(
            "temperature must be in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}], got {t}"
        ));
    }
    if let Some(b) = biases
        .into_iter()
        .copied()
        .find(|b| b.is_nan() || b.abs() > MAX_ABS_BIAS)
    {
        return Err(format!(
            "bias must be in [-{MAX_ABS_BIAS}, {MAX_ABS_BIAS}], got {b}"
        ));
    }
    Ok(())
}

impl JevProfile {
    /// The calibration bounds, checked at load and again by `JevCodec::new`.
    pub fn check_calibration(&self) -> Result<(), String> {
        check_calibration(self.temperature.values(), &self.bias)
    }

    /// Choice labels, when present, pair up, fit MAX_CHOICE_LABELS and start with the trained
    /// choice slots (the reference asserts the same, serve_decide.py:128). Checked at load and
    /// again by `JevCodec::new`; a profile without a choice range is refused there instead.
    pub fn check_choice_labels(&self) -> Result<(), String> {
        let (labels, ids) = (&self.choice_labels, &self.choice_label_ids);
        if labels.len() != ids.len() {
            return Err(format!(
                "jev.choice_labels and choice_label_ids must have equal length, got {} and {}",
                labels.len(),
                ids.len()
            ));
        }
        if labels.len() > MAX_CHOICE_LABELS {
            return Err(format!(
                "jev.choice_labels must have at most {MAX_CHOICE_LABELS} entries, got {}",
                labels.len()
            ));
        }
        let Some(&[lo, hi]) = self.slots.get("choice") else {
            return Ok(());
        };
        if ids.is_empty() || hi > self.verbalizer_ids.len() || lo > hi {
            return Ok(());
        }
        if ids.len() < hi - lo {
            return Err(format!(
                "jev.choice_labels has {} entries, fewer than the {} trained choice slots",
                ids.len(),
                hi - lo
            ));
        }
        if ids[..hi - lo] != self.verbalizer_ids[lo..hi] {
            return Err(format!(
                "jev.choice_label_ids[..{}] {:?} must equal the trained choice verbalizer_ids[{lo}..{hi}] {:?}",
                hi - lo,
                &ids[..hi - lo],
                &self.verbalizer_ids[lo..hi]
            ));
        }
        Ok(())
    }
}

impl JevProfile {
    /// System 2's fields, when present, are well formed: answer labels pair up and fit
    /// MAX_CHOICE_LABELS, threshold and mix lie in [0, 1], a trained scale is finite and not
    /// negative, and a profile whose default mode thinks can (Task 43). Checked at load and again
    /// by `JevCodec::new`.
    pub fn check_system2(&self) -> Result<(), String> {
        let (labels, ids) = (&self.answer_labels, &self.answer_label_ids);
        if labels.len() != ids.len() {
            return Err(format!(
                "jev.answer_labels and answer_label_ids must have equal length, got {} and {}",
                labels.len(),
                ids.len()
            ));
        }
        if labels.len() > MAX_CHOICE_LABELS {
            return Err(format!(
                "jev.answer_labels must have at most {MAX_CHOICE_LABELS} entries, got {}",
                labels.len()
            ));
        }
        let s = &self.system2;
        for (name, v) in [("threshold", s.threshold), ("mix", s.mix)] {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("jev.system2.{name} must be in [0, 1], got {v}"));
            }
        }
        if let Some(x) = self.lora_scale.filter(|x| !x.is_finite() || *x < 0.0) {
            return Err(format!(
                "jev.lora_scale must be finite and at least 0, got {x}"
            ));
        }
        if s.thinking != ThinkingMode::Off {
            if let Some(why) = self.system2_refusal() {
                return Err(format!(
                    "jev.system2.thinking defaults to {:?}, but {why}",
                    s.thinking
                ));
            }
        }
        Ok(())
    }

    /// Why this tree cannot think, or `None` when it can: System 2 runs the base model
    /// (`lora_scale = 0`, a switchable tree only) and needs the exported answer labels and
    /// end-of-think token.
    pub fn system2_refusal(&self) -> Option<String> {
        if self.lora_scale.is_none() {
            return Some(
                "thinking needs a switchable tree (decision.json jev.lora_scale, exported with \
                 --switchable-lora); this tree has its adapter baked in"
                    .into(),
            );
        }
        if self.answer_label_ids.is_empty() || self.end_think_id.is_none() {
            return Some(
                "thinking needs System 2's export (decision.json jev.answer_labels and \
                 jev.end_think_id); re-export the tree with tools/decision_export.py"
                    .into(),
            );
        }
        None
    }
}

impl KevProfile {
    /// The temperature bound, checked at load and again by `KevCodec::new`.
    pub fn check_calibration(&self) -> Result<(), String> {
        check_calibration([&self.temperature], [])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_each_family() {
        let d = std::env::temp_dir().join(format!("dprof-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("decision.json");
        std::fs::write(
            &p,
            r#"{"family":"kev","max_tokens":100,"max_questions":8,"kev":{"temperature":2.5}}"#,
        )
        .unwrap();
        let prof = DecisionProfile::load(&p).unwrap();
        assert_eq!(prof.family, Family::Kev);
        assert_eq!(prof.max_questions, Some(8));
        assert_eq!(prof.kev.unwrap().temperature, 2.5);
        std::fs::write(&p, r#"{"family":"jev","max_tokens":10,"jev":{"verbalizers":["false","true"],"verbalizer_ids":[1,2],"bias":[0,0],"slots":{"noul":[0,2],"score":[2,2],"choice":[2,2]},"temperature":{"noul":1,"choice":1,"score":1}}}"#).unwrap();
        assert_eq!(
            DecisionProfile::load(&p)
                .unwrap()
                .jev
                .unwrap()
                .verbalizer_ids,
            vec![1, 2]
        );
        std::fs::write(&p, r#"{"family":"jev","max_tokens":10}"#).unwrap();
        assert!(DecisionProfile::load(&p).unwrap_err().contains("jev"));
        std::fs::write(
            &p,
            r#"{"family":"kev","max_tokens":100,"kev":{"temperature":0}}"#,
        )
        .unwrap();
        assert!(DecisionProfile::load(&p)
            .unwrap_err()
            .contains("temperature"));
        std::fs::write(&p, r#"{"family":"jev","max_tokens":10,"jev":{"verbalizers":["false","true"],"verbalizer_ids":[1,2],"bias":[0,0],"slots":{"noul":[0,2],"score":[2,2],"choice":[2,2]},"temperature":{"noul":1,"choice":-1,"score":1}}}"#).unwrap();
        assert!(DecisionProfile::load(&p)
            .unwrap_err()
            .contains("temperature"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn calibration_that_can_overflow_is_refused() {
        // H17: `(logit + bias) / temperature` must stay finite for every finite f32 logit, or
        // softmax turns it into NaN probabilities.
        let d = std::env::temp_dir().join(format!("dprof-h17-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("decision.json");
        let jev = |bias: &str, t: &str| {
            format!(
                r#"{{"family":"jev","max_tokens":10,"jev":{{"verbalizers":["false","true"],"verbalizer_ids":[1,2],"bias":{bias},"slots":{{"noul":[0,2],"score":[2,2],"choice":[2,2]}},"temperature":{{"noul":{t},"choice":1,"score":1}}}}}}"#
            )
        };
        let kev =
            |t: &str| format!(r#"{{"family":"kev","max_tokens":10,"kev":{{"temperature":{t}}}}}"#);
        // shipped calibrations (JEV ~0.98 and small biases, Kev 2.19) and the bounds load
        for ok in [
            jev("[0.1,-0.3]", "0.98"),
            jev("[10000,-10000]", "0.001"),
            jev("[0,0]", "1000"),
            kev("2.193649959389252"),
            kev("0.001"),
            kev("1000"),
        ] {
            std::fs::write(&p, &ok).unwrap();
            assert!(DecisionProfile::load(&p).is_ok(), "{ok}");
        }
        for (bad, what) in [
            (jev("[0,0]", "1e-309"), "temperature"),
            (jev("[0,0]", "0.0009"), "temperature"),
            (jev("[0,0]", "1001"), "temperature"),
            (jev("[1e308,0]", "0.5"), "bias"),
            (jev("[0,-10000.5]", "1"), "bias"),
            (kev("1e-309"), "temperature"),
            (kev("1e300"), "temperature"),
        ] {
            std::fs::write(&p, &bad).unwrap();
            let e = DecisionProfile::load(&p).unwrap_err();
            assert!(e.contains(what), "{bad}: {e}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    fn jev_json(extra: &str) -> String {
        format!(
            r#"{{"family":"jev","max_tokens":10,"jev":{{"verbalizers":["false","true","0","1","2","3","4","5","A","B"],"verbalizer_ids":[1,2,3,4,5,6,7,8,9,10],"bias":[0,0,0,0,0,0,0,0,0,0],"slots":{{"noul":[0,2],"score":[2,8],"choice":[8,10]}},"temperature":{{"noul":1,"choice":1,"score":1}}{extra}}}}}"#
        )
    }

    #[test]
    fn choice_labels_and_strategy_are_optional() {
        // Task 36: a decision.json from before Task 35 has neither field and keeps loading.
        let d = std::env::temp_dir().join(format!("dprof-t36-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("decision.json");
        std::fs::write(&p, jev_json("")).unwrap();
        let j = DecisionProfile::load(&p).unwrap().jev.unwrap();
        assert!(j.choice_labels.is_empty() && j.choice_label_ids.is_empty());
        assert_eq!(j.strategy, JevStrategy::Single);
        std::fs::write(
            &p,
            jev_json(r#","choice_labels":["A","B","C"],"choice_label_ids":[9,10,77],"strategy":"tournament""#),
        )
        .unwrap();
        let j = DecisionProfile::load(&p).unwrap().jev.unwrap();
        assert_eq!(j.choice_labels, ["A", "B", "C"]);
        assert_eq!(j.choice_label_ids, [9, 10, 77]);
        assert_eq!(j.strategy, JevStrategy::Tournament);
        std::fs::write(&p, jev_json(r#","strategy":"vote""#)).unwrap();
        assert!(DecisionProfile::load(&p).unwrap_err().contains("vote"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn choice_labels_must_extend_the_trained_choice_head() {
        let d = std::env::temp_dir().join(format!("dprof-t36b-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("decision.json");
        let many = |n: usize| {
            let labels: Vec<String> = (0..n).map(|i| format!("\"L{i}\"")).collect();
            let ids: Vec<String> = (0..n).map(|i| (9 + i).to_string()).collect();
            format!(
                r#","choice_labels":[{}],"choice_label_ids":[{}]"#,
                labels.join(","),
                ids.join(",")
            )
        };
        std::fs::write(&p, jev_json(&many(MAX_CHOICE_LABELS))).unwrap();
        assert_eq!(
            DecisionProfile::load(&p)
                .unwrap()
                .jev
                .unwrap()
                .choice_labels
                .len(),
            256
        );
        for (extra, why) in [
            (many(MAX_CHOICE_LABELS + 1), "at most 256"),
            (
                r#","choice_labels":["A","B"],"choice_label_ids":[9]"#.into(),
                "equal length",
            ),
            (
                r#","choice_labels":["A"],"choice_label_ids":[9]"#.into(),
                "the 2 trained",
            ),
            (
                r#","choice_labels":["A","B"],"choice_label_ids":[9,11]"#.into(),
                "verbalizer_ids[8..10]",
            ),
        ] {
            std::fs::write(&p, jev_json(&extra)).unwrap();
            let e = DecisionProfile::load(&p).unwrap_err();
            assert!(e.contains(why), "{extra}: {e}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn system2_fields_are_optional_and_default_to_the_qwen_profile() {
        // Task 43: an older decision.json has none of them and keeps loading, without thinking.
        let d = std::env::temp_dir().join(format!("dprof-t43-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("decision.json");
        std::fs::write(&p, jev_json("")).unwrap();
        let j = DecisionProfile::load(&p).unwrap().jev.unwrap();
        assert!(j.answer_labels.is_empty() && j.end_think_id.is_none() && j.lora_scale.is_none());
        assert_eq!(j.system2, ThinkProfile::default());
        assert_eq!(j.system2.thinking, ThinkingMode::Off);
        assert_eq!((j.system2.threshold, j.system2.mix), (0.8, 0.5));
        assert_eq!(
            (j.system2.end_think.as_str(), j.system2.after_think.as_str()),
            ("</think>", "\n\n")
        );
        assert!(j.system2_refusal().unwrap().contains("switchable"));
        let full = r#","answer_labels":["A","B"],"answer_label_ids":[9,10],"end_think_id":77,"lora_scale":2.0,"system2":{"thinking":"auto","mix":0.25}"#;
        std::fs::write(&p, jev_json(full)).unwrap();
        let j = DecisionProfile::load(&p).unwrap().jev.unwrap();
        assert_eq!(j.answer_label_ids, [9, 10]);
        assert_eq!((j.end_think_id, j.lora_scale), (Some(77), Some(2.0)));
        assert_eq!(j.system2.thinking, ThinkingMode::Auto);
        assert_eq!((j.system2.threshold, j.system2.mix), (0.8, 0.25));
        assert_eq!(j.system2_refusal(), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn malformed_system2_fields_are_refused_at_load() {
        let d = std::env::temp_dir().join(format!("dprof-t43b-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("decision.json");
        for (extra, why) in [
            (
                r#","answer_labels":["A","B"],"answer_label_ids":[9]"#,
                "equal length",
            ),
            (
                r#","system2":{"threshold":1.5}"#,
                "threshold must be in [0, 1]",
            ),
            (r#","system2":{"mix":-0.1}"#, "mix must be in [0, 1]"),
            (r#","lora_scale":-1"#, "lora_scale must be finite"),
            (r#","system2":{"thinking":"sometimes"}"#, "sometimes"),
            // A default that thinks on a tree that cannot: refused at load, not per request.
            (r#","system2":{"thinking":"on"}"#, "switchable"),
            (
                r#","lora_scale":1.0,"system2":{"thinking":"auto"}"#,
                "answer_labels",
            ),
        ] {
            std::fs::write(&p, jev_json(extra)).unwrap();
            let e = DecisionProfile::load(&p).unwrap_err();
            assert!(e.contains(why), "{extra}: {e}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn strategy_parses_from_its_name() {
        assert_eq!("single".parse(), Ok(JevStrategy::Single));
        assert_eq!("tournament".parse(), Ok(JevStrategy::Tournament));
        assert_eq!("permute".parse(), Ok(JevStrategy::Permute));
        assert!("Permute".parse::<JevStrategy>().is_err());
    }

    #[test]
    fn softmax_is_stable() {
        let p = crate::decisions::softmax(&[1000.0, 1000.0]);
        assert!((p[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sanitizer_also_neutralizes_non_pipe_added_tokens() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[unk]".to_string(), 0u32)].into_iter().collect())
            .unk_token("[unk]".into())
            .build()
            .unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.add_tokens(&[
            tokenizers::AddedToken::from("<think>", false),
            tokenizers::AddedToken::from("</think>", false),
        ]);
        tok.add_special_tokens(&[tokenizers::AddedToken::from("<|im_end|>", true)]);
        let san = crate::decisions::Sanitizer::from_tokenizer(&tok);
        assert_eq!(
            san.apply("a <think>x</think> <|im_end|> <b>"),
            "a <¦think¦>x<¦/think¦> <¦im_end¦> <b>"
        );
    }

    #[test]
    fn sanitizer_covers_pipe_tokens_outside_the_name_charset() {
        // `sanitize_user_text` only rewrites `<|name|>` with name in [A-Za-z0-9_]; an added token
        // like `<|box-start|>` must still be neutralized, via the added-token list.
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[unk]".to_string(), 0u32)].into_iter().collect())
            .unk_token("[unk]".into())
            .build()
            .unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.add_special_tokens(&[
            tokenizers::AddedToken::from("<|box-start|>", true),
            tokenizers::AddedToken::from("<|a.b|>", true),
            tokenizers::AddedToken::from("<|im_end|>", true),
        ]);
        let san = crate::decisions::Sanitizer::from_tokenizer(&tok);
        let out = san.apply("x <|box-start|> <|a.b|> <|im_end|>");
        for t in ["<|box-start|>", "<|a.b|>", "<|im_end|>"] {
            assert!(!out.contains(t), "{t} survived: {out}");
        }
        assert_eq!(out, "x <¦|box-start|¦> <¦|a.b|¦> <¦im_end¦>");
    }

    #[test]
    fn sanitize_rewrites_only_control_token_shapes() {
        use crate::decisions::sanitize_user_text as s;
        assert_eq!(s("a <|im_end|> b"), "a <¦im_end¦> b");
        assert_eq!(s("<|fim_suffix|><|x_1|>"), "<¦fim_suffix¦><¦x_1¦>");
        assert_eq!(s("x <| not a token |> y"), "x <| not a token |> y");
        assert_eq!(s("<||> and <|"), "<||> and <|");
    }
}
