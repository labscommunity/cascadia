use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

/// Calibration bounds checked at load. Shipped profiles use temperatures near 1 (JEV ~0.98,
/// Kev ~2.19) and biases of a few units; with these bounds `(logit + bias) / temperature` is at
/// most ~3.4e41 for any finite f32 logit, far inside f64.
pub const MIN_TEMPERATURE: f64 = 1e-3;
pub const MAX_TEMPERATURE: f64 = 1e3;
pub const MAX_ABS_BIAS: f64 = 1e4;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    Jev,
    Kev,
    Clef,
}

#[derive(Clone, Debug, Deserialize)]
pub struct JevProfile {
    pub verbalizers: Vec<String>,
    pub verbalizer_ids: Vec<u32>,
    pub bias: Vec<f64>,
    /// kind → [lo, hi) slot range into verbalizer_ids/bias.
    pub slots: HashMap<String, [usize; 2]>,
    /// kind → calibration temperature.
    pub temperature: HashMap<String, f64>,
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
        // A zero, negative or non-finite temperature turns every probability into NaN, and so
        // does one small enough (or a bias large enough) for `(logit + bias) / temperature` to
        // overflow (H17). Within these bounds it stays finite for every finite f32 logit.
        let temps = p
            .jev
            .iter()
            .flat_map(|j| j.temperature.values())
            .chain(p.kev.iter().map(|k| &k.temperature));
        if let Some(t) = temps
            .copied()
            .find(|t| !(MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(t))
        {
            return Err(format!(
                "{}: temperature must be in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}], got {t}",
                path.display()
            ));
        }
        let biases = p.jev.iter().flat_map(|j| j.bias.iter());
        if let Some(b) = biases.copied().find(|b| !(b.abs() <= MAX_ABS_BIAS)) {
            return Err(format!(
                "{}: bias must be in [-{MAX_ABS_BIAS}, {MAX_ABS_BIAS}], got {b}",
                path.display()
            ));
        }
        Ok(p)
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
