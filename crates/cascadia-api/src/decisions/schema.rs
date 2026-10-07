use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::DecisionError;

/// JSON value that keeps object keys in REQUEST order. `serde_json::Value`'s map sorts keys unless the
/// workspace-wide `preserve_order` feature is on, and turning that on would reorder every chat response.
/// Codecs must render exactly what their reference implementations saw, so request JSON is parsed into this.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OrderedValue>),
    Object(Vec<(String, OrderedValue)>),
}

impl<'de> Deserialize<'de> for OrderedValue {
    /// Parses the value's exact source text (serde_json `RawValue`) with `parse_ordered`:
    /// serde_json's default float parser is not correctly rounded for every 17-digit literal,
    /// and the workspace must not enable `float_roundtrip`/`arbitrary_precision`.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = Box::<serde_json::value::RawValue>::deserialize(d)?;
        parse_ordered(raw.get()).map_err(serde::de::Error::custom)
    }
}

/// Nesting bound for `parse_ordered` (untrusted input; serde_json's own limit is also 128).
const MAX_DEPTH: usize = 128;

/// Strict JSON reader into `OrderedValue`: objects keep request order with Python dict semantics
/// (a repeated key keeps its first position and takes the last value), strings are decoded by
/// serde_json (exact escape rules), integers stay integers (`-0` → `0`, as Python), and other
/// numbers go through `str::parse::<f64>` (correctly rounded). Non-finite numbers are rejected.
fn parse_ordered(text: &str) -> Result<OrderedValue, String> {
    let mut p = Reader {
        s: text.as_bytes(),
        text,
        i: 0,
    };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("trailing characters at byte {}", p.i));
    }
    Ok(v)
}

struct Reader<'a> {
    s: &'a [u8],
    text: &'a str,
    i: usize,
}

impl Reader<'_> {
    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.s.get(self.i) {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.s.get(self.i) == Some(&b) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn literal(&mut self, word: &str, v: OrderedValue) -> Result<OrderedValue, String> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            self.err("invalid literal")
        }
    }

    fn value(&mut self, depth: usize) -> Result<OrderedValue, String> {
        self.ws();
        match self.s.get(self.i) {
            None => self.err("unexpected end of input"),
            Some(b'n') => self.literal("null", OrderedValue::Null),
            Some(b't') => self.literal("true", OrderedValue::Bool(true)),
            Some(b'f') => self.literal("false", OrderedValue::Bool(false)),
            Some(b'"') => self.string().map(OrderedValue::String),
            Some(b'[') => self.array(depth + 1),
            Some(b'{') => self.object(depth + 1),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => self.err("expected a JSON value"),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        let start = self.i;
        self.i += 1; // opening quote
        loop {
            match self.s.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => break,
                Some(b'\\') => self.i += 2,
                Some(_) => self.i += 1,
            }
        }
        self.i += 1; // closing quote
                     // The slice ends on an ASCII quote, so it is a char boundary.
        serde_json::from_str::<String>(&self.text[start..self.i])
            .map_err(|e| format!("invalid string at byte {start}: {e}"))
    }

    fn digits(&mut self) -> usize {
        let from = self.i;
        while let Some(b'0'..=b'9') = self.s.get(self.i) {
            self.i += 1;
        }
        self.i - from
    }

    fn number(&mut self) -> Result<OrderedValue, String> {
        let start = self.i;
        self.eat(b'-');
        if self.eat(b'0') {
            if let Some(b'0'..=b'9') = self.s.get(self.i) {
                return self.err("leading zero in number");
            }
        } else if self.digits() == 0 {
            return self.err("invalid number");
        }
        let mut integer = true;
        if self.eat(b'.') {
            integer = false;
            if self.digits() == 0 {
                return self.err("invalid number");
            }
        }
        if self.eat(b'e') || self.eat(b'E') {
            integer = false;
            if !self.eat(b'+') {
                self.eat(b'-');
            }
            if self.digits() == 0 {
                return self.err("invalid number");
            }
        }
        let lit = &self.text[start..self.i];
        if integer {
            if let Ok(n) = lit.parse::<i64>() {
                return Ok(OrderedValue::Number(n.into()));
            }
            if let Ok(n) = lit.parse::<u64>() {
                return Ok(OrderedValue::Number(n.into()));
            }
        }
        let x: f64 = lit
            .parse()
            .map_err(|e| format!("invalid number {lit:?}: {e}"))?;
        serde_json::Number::from_f64(x)
            .map(OrderedValue::Number)
            .ok_or_else(|| format!("number out of range: {lit}"))
    }

    fn array(&mut self, depth: usize) -> Result<OrderedValue, String> {
        if depth > MAX_DEPTH {
            return self.err("nesting too deep");
        }
        self.i += 1; // [
        let mut v = Vec::new();
        self.ws();
        if self.eat(b']') {
            return Ok(OrderedValue::Array(v));
        }
        loop {
            v.push(self.value(depth)?);
            self.ws();
            if self.eat(b']') {
                return Ok(OrderedValue::Array(v));
            }
            if !self.eat(b',') {
                return self.err("expected ',' or ']'");
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<OrderedValue, String> {
        if depth > MAX_DEPTH {
            return self.err("nesting too deep");
        }
        self.i += 1; // {
        let mut v: Vec<(String, OrderedValue)> = Vec::new();
        // key → index in `v`: O(n) even for objects with many keys (this runs before any limit).
        let mut at: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        self.ws();
        if self.eat(b'}') {
            return Ok(OrderedValue::Object(v));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return self.err("expected an object key");
            }
            let k = self.string()?;
            self.ws();
            if !self.eat(b':') {
                return self.err("expected ':'");
            }
            let x = self.value(depth)?;
            // Python dict semantics: a repeated key keeps its first position, takes the last value.
            match at.get(&k) {
                Some(&i) => v[i].1 = x,
                None => {
                    at.insert(k.clone(), v.len());
                    v.push((k, x));
                }
            }
            self.ws();
            if self.eat(b'}') {
                return Ok(OrderedValue::Object(v));
            }
            if !self.eat(b',') {
                return self.err("expected ',' or '}'");
            }
        }
    }
}

impl Serialize for OrderedValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self {
            OrderedValue::Null => s.serialize_unit(),
            OrderedValue::Bool(b) => s.serialize_bool(*b),
            OrderedValue::Number(n) => n.serialize(s),
            OrderedValue::String(x) => s.serialize_str(x),
            OrderedValue::Array(a) => {
                let mut q = s.serialize_seq(Some(a.len()))?;
                for x in a {
                    q.serialize_element(x)?;
                }
                q.end()
            }
            OrderedValue::Object(m) => {
                let mut q = s.serialize_map(Some(m.len()))?;
                for (k, x) in m {
                    q.serialize_entry(k, x)?;
                }
                q.end()
            }
        }
    }
}

impl OrderedValue {
    pub fn is_null(&self) -> bool {
        matches!(self, OrderedValue::Null)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            OrderedValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&OrderedValue> {
        match self {
            OrderedValue::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Python `json.dumps(v, ensure_ascii=False, sort_keys=sort_keys,
    /// separators=(",", ":") if compact else (", ", ": "))`. String escaping is serde_json's, which
    /// matches Python with ensure_ascii=False (escapes `"` `\` and control chars as `\n`/`\u001f`,
    /// keeps non-ASCII). Floats use `py_float`, Python's repr. Known gap: integers beyond u64 parse
    /// as f64 (serde_json has no arbitrary precision here), where Python keeps an int.
    pub fn py_json(&self, sort_keys: bool, compact: bool) -> String {
        let mut out = String::new();
        self.write_py(&mut out, sort_keys, compact);
        out
    }

    fn write_py(&self, out: &mut String, sort: bool, compact: bool) {
        let (item, kv) = if compact { (",", ":") } else { (", ", ": ") };
        match self {
            OrderedValue::Null => out.push_str("null"),
            OrderedValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            OrderedValue::Number(n) => out.push_str(&py_number(n)),
            OrderedValue::String(s) => {
                out.push_str(&serde_json::to_string(s).expect("string serializes"))
            }
            OrderedValue::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    x.write_py(out, sort, compact);
                }
                out.push(']');
            }
            OrderedValue::Object(m) => {
                let mut entries: Vec<&(String, OrderedValue)> = m.iter().collect();
                if sort {
                    // Python sorts str keys by code point; UTF-8 byte order is the same order.
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                }
                out.push('{');
                for (i, (k, x)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    out.push_str(&serde_json::to_string(k).expect("key serializes"));
                    out.push_str(kv);
                    x.write_py(out, sort, compact);
                }
                out.push('}');
            }
        }
    }

    /// Python `str(v)` for scalars (Kev's render): `True`/`False`, `None`, numbers as written.
    pub fn py_str(&self) -> String {
        match self {
            OrderedValue::Null => "None".into(),
            OrderedValue::Bool(true) => "True".into(),
            OrderedValue::Bool(false) => "False".into(),
            OrderedValue::Number(n) => py_number(n),
            OrderedValue::String(s) => s.clone(),
            other => other.py_json(false, false),
        }
    }
}

/// Python's `float.__repr__` (what `json.dumps` and `str()` print): the shortest round-trip digits,
/// fixed notation for exponents −4..16 (always with a fractional part, `1.0`), scientific otherwise
/// with a signed two-digit-minimum exponent (`1e+16`, `1e-05`). The digits come from `ryu`, which,
/// like Python's dtoa, picks the shortest candidate closest to the exact value (ties to even);
/// Rust's `{:e}` does not break those ties the same way. Only the layout is re-done here.
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    let mut buf = ryu::Buffer::new();
    let r = buf.format_finite(x);
    let (sign, r) = match r.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", r),
    };
    // ryu prints `d.ddd`, `0.000ddd`, `ddd.0` or `d.ddde±N`: recover (digits, decimal exponent).
    let (mant, e10) = match r.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().expect("ryu exponent")),
        None => (r, 0),
    };
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let all: String = format!("{int}{frac}");
    let lead = all.len() - all.trim_start_matches('0').len();
    let digits = all[lead..].trim_end_matches('0');
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let exp = int.len() as i32 - lead as i32 - 1 + e10;
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() == 1 {
            digits.to_string()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!(
            "{sign}{m}e{}{:02}",
            if exp < 0 { "-" } else { "+" },
            exp.abs()
        )
    }
}

fn py_number(n: &serde_json::Number) -> String {
    if n.is_f64() {
        py_float(n.as_f64().expect("f64 number"))
    } else {
        n.to_string()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: String,
    /// Any JSON value, in request order; codecs render it.
    pub state: OrderedValue,
    pub questions: IndexMap<String, Question>,
    #[serde(default)]
    pub images: Vec<serde_json::Value>,
    #[serde(default)]
    pub videos: Vec<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Noul,
    Choice,
    Score,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: QuestionType,
    #[serde(default)]
    pub instructions: Option<OrderedValue>,
    /// choice: object name → description (request order); score: array of level descriptions;
    /// noul: optional {"true": ..., "false": ...} overrides.
    #[serde(default)]
    pub criteria: Option<OrderedValue>,
}

/// Per-profile limits (the codec decides them: JEV's head has 16 choice slots). `max_questions`
/// bounds engine work per request: JEV and Kev run one prefill pass per question.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_questions: usize,
    pub max_options: usize,
    pub min_levels: usize,
    pub max_levels: usize,
}

impl Question {
    /// (name, description) in REQUEST order.
    pub fn choice_criteria(&self) -> Vec<(String, OrderedValue)> {
        match &self.criteria {
            Some(OrderedValue::Object(m)) => m.clone(),
            _ => Vec::new(),
        }
    }

    pub fn score_levels(&self) -> Vec<OrderedValue> {
        match &self.criteria {
            Some(OrderedValue::Array(a)) => a.clone(),
            _ => Vec::new(),
        }
    }

    pub fn option_keys(&self) -> Vec<String> {
        match self.qtype {
            QuestionType::Noul => vec!["false".into(), "true".into()],
            QuestionType::Choice => self.choice_criteria().into_iter().map(|(k, _)| k).collect(),
            QuestionType::Score => (0..self.score_levels().len())
                .map(|i| i.to_string())
                .collect(),
        }
    }
}

/// An image part in a list `state` (serve_decide `_parts`): `{"image": ...}` or `{"type": "image_url", ...}`.
pub fn is_image_part(v: &OrderedValue) -> bool {
    v.get("image").is_some() || v.get("type").and_then(|t| t.as_str()) == Some("image_url")
}

impl DecisionRequest {
    pub fn validate(&self, limits: &Limits) -> Result<(), DecisionError> {
        let bad = |m: String| Err(DecisionError::BadRequest(m));
        if !self.images.is_empty() || !self.videos.is_empty() {
            return bad(
                "images and videos are not supported by this server (text-only decision models)"
                    .into(),
            );
        }
        // Every family is text-only: an image part in a list state is a 400 for all of them, not
        // just JEV (Kev and Clef would otherwise render the part into the prompt as text).
        if let OrderedValue::Array(parts) = &self.state {
            if parts.iter().any(is_image_part) {
                return bad("image parts in state are not supported (text-only server)".into());
            }
        }
        if self.questions.is_empty() || self.questions.len() > limits.max_questions {
            return bad(format!(
                "between 1 and {} questions are allowed, got {}",
                limits.max_questions,
                self.questions.len()
            ));
        }
        for (id, q) in &self.questions {
            if id.is_empty() {
                return bad("question ids must be non-empty".into());
            }
            match q.qtype {
                QuestionType::Noul => {}
                QuestionType::Choice => {
                    let n = q.choice_criteria().len();
                    if n == 0 || n > limits.max_options {
                        return bad(format!(
                            "{id}: choice criteria must have 1..={} options, got {n}",
                            limits.max_options
                        ));
                    }
                }
                QuestionType::Score => {
                    let n = q.score_levels().len();
                    if n < limits.min_levels || n > limits.max_levels {
                        return bad(format!(
                            "{id}: score criteria must have {}..={} levels, got {n}",
                            limits.min_levels, limits.max_levels
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        confidence: f64,
        probabilities: IndexMap<String, f64>,
    },
    Score {
        score: f64,
        confidence: f64,
        legend: IndexMap<String, OrderedValue>,
        probabilities: IndexMap<String, f64>,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

/// Python `round(x, 4)`: both round the EXACT binary value half-to-even, and Rust's `{:.4}`
/// formatting is exact, so the 4th decimal matches Kev's and Clef's own servers.
pub fn round_prob(x: f64) -> f64 {
    format!("{x:.4}").parse().expect("formatted float parses")
}

fn normalize(p: &[f64]) -> Vec<f64> {
    let t: f64 = p.iter().sum();
    if t == 0.0 {
        vec![1.0 / p.len() as f64; p.len()]
    } else {
        p.iter().map(|x| x / t).collect()
    }
}

/// TypeSafe system-one-adapter 0.2.1 (mirrored in kev/api.py): (p_max - 1/K) / (1 - 1/K);
/// 0 at uniform, 1 at certainty, 1 for a single option.
pub fn choice_confidence(p: &[f64]) -> f64 {
    let k = p.len() as f64;
    if p.len() == 1 {
        return 1.0;
    }
    let m = normalize(p).into_iter().fold(f64::NEG_INFINITY, f64::max);
    (m - 1.0 / k) / (1.0 - 1.0 / k)
}

/// TypeSafe: max(0, 1 - E|level - mode| / D), D = mean absolute deviation of a uniform
/// distribution over the L levels around (L-1)/2; mode = first most likely level.
pub fn score_confidence(p: &[f64]) -> f64 {
    let l = p.len();
    if l == 1 {
        return 1.0;
    }
    let p = normalize(p);
    let mode = p
        .iter()
        .enumerate()
        .fold(0, |b, (i, &x)| if x > p[b] { i } else { b });
    let mid = (l as f64 - 1.0) / 2.0;
    let d = (0..l).map(|i| (i as f64 - mid).abs()).sum::<f64>() / l as f64;
    let e: f64 = p
        .iter()
        .enumerate()
        .map(|(i, x)| x * (i as f64 - mode as f64).abs())
        .sum();
    (1.0 - e / d).max(0.0)
}

/// Jev/SystemOne answers with TypeSafe's semantics (the wire format's owner): choice = argmax
/// (first on ties) with normalized confidence; score = probability-weighted level with the MAD
/// confidence; noul = p(true). `probs` is in `Question::option_keys` order.
pub fn answer_from_probs(q: &Question, probs: &[f64]) -> Answer {
    let keys = q.option_keys();
    debug_assert_eq!(probs.len(), keys.len());
    let dist: IndexMap<String, f64> = keys
        .iter()
        .cloned()
        .zip(probs.iter().map(|&p| round_prob(p)))
        .collect();
    let argmax = probs
        .iter()
        .enumerate()
        .fold(0, |b, (i, &p)| if p > probs[b] { i } else { b });
    match q.qtype {
        QuestionType::Noul => Answer::Noul {
            noul: round_prob(probs[1]),
        },
        QuestionType::Choice => Answer::Choice {
            choice: keys[argmax].clone(),
            confidence: round_prob(choice_confidence(probs)),
            probabilities: dist,
        },
        QuestionType::Score => Answer::Score {
            score: round_prob(probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum()),
            confidence: round_prob(score_confidence(probs)),
            legend: keys.into_iter().zip(q.score_levels()).collect(),
            probabilities: dist,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse from TEXT: `serde_json::json!` builds a sorted Map (no preserve_order), so going
    /// through a `Value` would lose the order these tests check.
    fn req(v: serde_json::Value) -> DecisionRequest {
        serde_json::from_str(&v.to_string()).unwrap()
    }
    fn req_str(s: &str) -> DecisionRequest {
        serde_json::from_str(s).unwrap()
    }

    const LIMITS: Limits = Limits {
        max_questions: 3,
        max_options: 16,
        min_levels: 2,
        max_levels: 10,
    };

    #[test]
    fn parses_jev_body_preserving_question_order() {
        let r = req_str(
            r#"{
            "model": "m",
            "state": {"msg": "charged twice"},
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs"}},
                "urgent": {"type": "noul", "instructions": "Urgent?"},
                "sev": {"type": "score", "instructions": "How bad?", "criteria": ["none", "minor", "major"]}
            }
        }"#,
        );
        let ids: Vec<&str> = r.questions.keys().map(String::as_str).collect();
        assert_eq!(ids, ["team", "urgent", "sev"]);
        assert_eq!(r.questions["team"].option_keys(), ["billing", "tech"]);
        assert_eq!(r.questions["urgent"].option_keys(), ["false", "true"]);
        assert_eq!(r.questions["sev"].option_keys(), ["0", "1", "2"]);
        r.validate(&LIMITS).unwrap();
    }

    #[test]
    fn rejects_empty_questions_images_and_bad_criteria() {
        let base = |q: serde_json::Value| {
            req(serde_json::json!({"model": "m", "state": "s", "questions": q}))
        };
        assert!(base(serde_json::json!({})).validate(&LIMITS).is_err());
        assert!(
            base(serde_json::json!({"c": {"type": "choice", "criteria": {}}}))
                .validate(&LIMITS)
                .is_err()
        );
        assert!(
            base(serde_json::json!({"s": {"type": "score", "criteria": ["only"]}}))
                .validate(&LIMITS)
                .is_err()
        );
        let mut many = serde_json::Map::new();
        for i in 0..17 {
            many.insert(format!("o{i}"), serde_json::json!("x"));
        }
        assert!(
            base(serde_json::json!({"c": {"type": "choice", "criteria": many}}))
                .validate(&LIMITS)
                .is_err()
        );
        assert!(base(serde_json::json!({"a": {"type": "noul"}, "b": {"type": "noul"}, "c": {"type": "noul"}, "d": {"type": "noul"}})).validate(&LIMITS).is_err(), "over max_questions");
        assert!(
            base(serde_json::json!({"": {"type": "noul"}}))
                .validate(&LIMITS)
                .is_err(),
            "empty question id"
        );
        let img = req(
            serde_json::json!({"model": "m", "state": "s", "images": ["data:x"], "questions": {"u": {"type": "noul"}}}),
        );
        assert!(
            matches!(img.validate(&LIMITS), Err(DecisionError::BadRequest(m)) if m.contains("images"))
        );
    }

    #[test]
    fn image_parts_in_state_are_rejected_for_every_family() {
        for state in [
            r#"[{"image": "data:x"}]"#,
            r#"["hi", {"type": "image_url", "image_url": {"url": "data:x"}}]"#,
        ] {
            let r = req_str(&format!(
                r#"{{"model": "m", "state": {state}, "questions": {{"u": {{"type": "noul"}}}}}}"#
            ));
            match r.validate(&LIMITS) {
                Err(DecisionError::BadRequest(m)) => {
                    assert_eq!(
                        m,
                        "image parts in state are not supported (text-only server)"
                    )
                }
                other => panic!("{state}: {other:?}"),
            }
        }
        let ok = req_str(
            r#"{"model": "m", "state": ["hi", {"type": "text", "text": "x"}], "questions": {"u": {"type": "noul"}}}"#,
        );
        assert!(ok.validate(&LIMITS).is_ok());
    }

    #[test]
    fn ordered_value_keeps_request_order_and_renders_like_python() {
        let v: OrderedValue =
            serde_json::from_str(r#"{"b": 1, "a": {"y": true, "x": null}, "é": [1.5, "q\"t"]}"#)
                .unwrap();
        assert_eq!(
            v.py_json(false, false),
            r#"{"b": 1, "a": {"y": true, "x": null}, "é": [1.5, "q\"t"]}"#
        );
        assert_eq!(
            v.py_json(true, true),
            r#"{"a":{"x":null,"y":true},"b":1,"é":[1.5,"q\"t"]}"#
        );
        assert_eq!(OrderedValue::Bool(true).py_str(), "True");
        assert_eq!(OrderedValue::Null.py_str(), "None");
        // duplicate keys: last value wins, first position kept (Python dict semantics)
        let d: OrderedValue = serde_json::from_str(r#"{"k": 1, "z": 2, "k": 3}"#).unwrap();
        assert_eq!(d.py_json(false, true), r#"{"k":3,"z":2}"#);
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#"{"b":1,"a":{"y":true,"x":null},"é":[1.5,"q\"t"]}"#
        );
    }

    #[test]
    fn many_keys_parse_in_linear_time() {
        let body = format!(
            "{{{}}}",
            (0..200_000)
                .map(|i| format!("\"k{i}\":{i}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let t = std::time::Instant::now();
        let v: OrderedValue = serde_json::from_str(&body).unwrap();
        assert!(matches!(v, OrderedValue::Object(ref m) if m.len() == 200_000));
        assert!(
            t.elapsed() < std::time::Duration::from_secs(5),
            "{:?}",
            t.elapsed()
        );
    }

    #[test]
    fn floats_render_like_python_repr() {
        // expected values are Python's repr() output
        for (x, want) in [
            (1e16, "1e+16"),
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (1.0, "1.0"),
            (1.5, "1.5"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (1e22, "1e+22"),
            (5e-324, "5e-324"),
            (2.5e-05, "2.5e-05"),
            (100.0, "100.0"),
            (1e15, "1000000000000000.0"),
            // shortest-digit ties: Python's repr breaks them half-even (Rust's `{:e}` rounds up)
            (1059438285926254.2, "1059438285926254.2"),
            (26363981746409.312, "26363981746409.312"),
        ] {
            assert_eq!(py_float(x), want, "{x:e}");
        }
        let v: OrderedValue = serde_json::from_str(r#"{"a": 1e16, "b": 3, "c": 2.50}"#).unwrap();
        assert_eq!(v.py_json(false, false), r#"{"a": 1e+16, "b": 3, "c": 2.5}"#);
    }

    #[test]
    fn floats_parse_exactly_and_bad_json_is_rejected() {
        // serde_json's default parser misrounds these 17-digit literals; Python round-trips them.
        for lit in [
            "0.18466034385487662",
            "-9.667831824381321e+24",
            "1059438285926254.2",
        ] {
            let v: OrderedValue = serde_json::from_str(&format!("[{lit}]")).unwrap();
            assert_eq!(v.py_json(false, true), format!("[{lit}]"));
        }
        // Python: json.dumps(json.loads("-0")) == "0"
        let z: OrderedValue = serde_json::from_str("-0").unwrap();
        assert_eq!(z.py_json(false, false), "0");
        assert_eq!(
            serde_json::from_str::<OrderedValue>("18446744073709551615")
                .unwrap()
                .py_str(),
            "18446744073709551615"
        );
        assert!(
            serde_json::from_str::<OrderedValue>("1e400").is_err(),
            "non-finite"
        );
        assert!(serde_json::from_str::<OrderedValue>(r#"{"a":}"#).is_err());
        // our own reader's errors, independent of serde_json's pre-scan
        assert!(parse_ordered("1e400").is_err());
        assert!(parse_ordered(r#"{"a":}"#).is_err());
        assert!(parse_ordered("[1] x").is_err(), "trailing garbage");
        assert!(parse_ordered("01").is_err());
        assert!(parse_ordered(r#""\x""#).is_err(), "bad escape");
        assert_eq!(
            parse_ordered(r#" {"s": "a\"\u00e9\n"} "#)
                .unwrap()
                .get("s")
                .unwrap()
                .as_str(),
            Some("a\"é\n")
        );
        let deep = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
        assert!(parse_ordered(&deep(MAX_DEPTH)).is_ok());
        assert!(
            parse_ordered(&deep(MAX_DEPTH + 1)).is_err(),
            "nesting over the limit"
        );
        assert!(serde_json::from_str::<OrderedValue>(&deep(1000)).is_err());
    }

    #[test]
    fn choice_criteria_keep_request_order_not_alphabetical() {
        let r = req_str(
            r#"{"model": "m", "state": "s", "questions": {"c": {"type": "choice", "criteria": {"zeta": "z", "alpha": "a"}}}}"#,
        );
        assert_eq!(r.questions["c"].option_keys(), ["zeta", "alpha"]);
    }

    #[test]
    fn answers_follow_systemone_shape() {
        let r = req_str(
            r#"{"model": "m", "state": "s", "questions": {
            "c": {"type": "choice", "criteria": {"a": "x", "b": "y"}},
            "n": {"type": "noul"},
            "s": {"type": "score", "criteria": ["lo", "mid", "hi"]}}}"#,
        );
        // TypeSafe choice confidence: (p_max - 1/K) / (1 - 1/K) = (0.75 - 0.5) / 0.5
        let c = serde_json::to_value(answer_from_probs(&r.questions["c"], &[0.25, 0.75])).unwrap();
        assert_eq!(
            c,
            serde_json::json!({"type": "choice", "choice": "b", "confidence": 0.5, "probabilities": {"a": 0.25, "b": 0.75}})
        );
        let n = serde_json::to_value(answer_from_probs(&r.questions["n"], &[0.1, 0.9])).unwrap();
        assert_eq!(n, serde_json::json!({"type": "noul", "noul": 0.9}));
        let s =
            serde_json::to_value(answer_from_probs(&r.questions["s"], &[0.2, 0.3, 0.5])).unwrap();
        assert_eq!(s["score"], serde_json::json!(1.3));
        // TypeSafe score confidence: max(0, 1 - E|i - mode| / D); mode 2, E = 0.7, D = 2/3 → clipped to 0
        assert_eq!(s["confidence"], serde_json::json!(0.0));
        assert_eq!(
            s["legend"],
            serde_json::json!({"0": "lo", "1": "mid", "2": "hi"})
        );
    }

    #[test]
    fn typesafe_confidence_and_python_rounding() {
        assert_eq!(choice_confidence(&[0.25, 0.25, 0.25, 0.25]), 0.0);
        assert_eq!(choice_confidence(&[1.0]), 1.0);
        assert!((score_confidence(&[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert_eq!(score_confidence(&[1.0 / 3.0; 3]), 0.0);
        // Python rounds the EXACT binary value: 0.00125 is stored as 0.0012500000000000000260…, so it rounds up,
        // and 0.00015 is stored as 0.000149999999999999986…, so it rounds down (Python: 0.0013 and 0.0001).
        assert_eq!(round_prob(0.00125), 0.0013);
        assert_eq!(round_prob(0.00015), 0.0001);
        assert_eq!(round_prob(0.123456), 0.1235);
    }
}
