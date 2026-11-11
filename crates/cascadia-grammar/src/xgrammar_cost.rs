//! How long xgrammar takes to compile a grammar for a JSON Schema,
//! estimated without running it.
//!
//! OpenVINO GenAI 2026.4 builds structured output with xgrammar 0.1.31: it
//! converts a request's JSON Schema to an EBNF grammar
//! (`cpp/json_schema_converter.cc`) and compiles that grammar against the
//! model's vocabulary, inside the request, with no time limit. The compile
//! works out a token mask for each position of each rule by scanning the
//! vocabulary from there, so its time grows with the positions a schema
//! makes and how far each can reach. Some shapes make that grow much faster
//! than the schema (times with Qwen2.5-1.5B-Instruct's 151,665-token
//! vocabulary, one thread, on an Apple M-series Mac):
//!
//! - an object's run of properties not in `required`: each rule of the run
//!   can skip to every later property of it, so the time is quadratic in
//!   the run's length (100 optional properties 0.20 s, 300 take 1.7 s, 1000
//!   take 18 s; a 6000-property schema held a worker for over 15 minutes);
//! - `minProperties`/`maxProperties`, which make one rule per property per
//!   count of properties already seen: cubic (70 optional properties 1.5 s,
//!   200 take 45 s);
//! - string `minLength`/`maxLength`, unrolled into one position per
//!   character up to 128 (`maxLength` 32 0.49 s, 64 1.1 s, 128 or more
//!   2.4–4 s, once per distinct pair of bounds), and
//!   repetitions in a `pattern`, unrolled the same way (`[^"]{1,100}` 1.9 s);
//! - a `type` list, which compiles the node once per entry, so nested lists
//!   of one type twice multiply (`["object","object"]` 12 deep 1.9 s, 14
//!   deep 26 s; OpenVINO GenAI itself took 12.8 s on 12 deep, and never
//!   finished 20 deep);
//! - rule names: a rule nested inside others is named after all of them,
//!   and its positions cost with the square of that name's length (objects
//!   nested 100 deep 0.13 s, 300 deep 3.5 s; through `$ref`s, which name a
//!   rule after its pointer, 300 deep 0.03 s);
//! - `number` ranges with a fractional bound, quadratic in their count in
//!   one object (60 of them 6.9 s).
//!
//! [`xgrammar_cost`] walks the schema the way the converter does and adds
//! up a cost in microseconds of compile time for each of these;
//! [`xgrammar_cost_error`] refuses a schema whose cost is over
//! [`MAX_XGRAMMAR_COST`], about 10 s. Each price is set at or about what was
//! measured. The limit leaves the seconds-long compiles of ordinary bounded
//! strings alone (OpenVINO GenAI caches a compiled schema) and stops the
//! minutes-long ones. The walk fails closed: a schema the converter would
//! reject, or one the walk can't price, is refused too.

use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

/// Largest accepted [`xgrammar_cost`], in microseconds of compile time: ten
/// seconds. Checked against xgrammar 0.1.31 on about 200 schemas (real
/// client schemas, a 68 KB extraction schema, strings bounded by
/// `maxLength`, each shape above at sizes either side of the limit): every
/// one it took 10 s or more on is over this limit, as are nested `type`
/// lists it took 2.7 s or more on (OpenVINO GenAI takes about three times
/// as long on those); three strings with `maxLength` 100, 254 and 1000
/// (9.6 s) are under it.
pub const MAX_XGRAMMAR_COST: u64 = 10_000_000;

/// Each rule the converter makes.
const RULE_COST: u64 = 3;
/// A rule's positions cost this many thousandths of a microsecond per
/// position per square of the rule name's length. An object's rule counts
/// [`OBJECT_POSITIONS`] positions, others one.
const NAME_SQUARE_MILLI_COST: u64 = 2;
const OBJECT_POSITIONS: u64 = 5;
/// Each property of an object, plus [`NAME_BYTE_COST`] per byte of the
/// first four bytes of its name (the positions most tokens can enter) and
/// [`NAME_TAIL_BYTE_COST`] per later byte.
const PROPERTY_COST: u64 = 60;
const NAME_BYTE_COST: u64 = 30;
const NAME_TAIL_BYTE_COST: u64 = 5;
/// Each (rule, later property it can skip to) pair of an object's rules,
/// per rule variant (`minProperties`/`maxProperties` make several).
const SKIP_COST: u64 = 32;
/// The same for the alternatives an object can start with.
const START_SKIP_COST: u64 = 6;
/// Each byte of an `enum`/`const` value.
const VALUE_BYTE_COST: u64 = 1;
/// One unrolled copy of a string character, mandatory or optional.
const STRING_COPY_COST: u64 = 10_000;
const STRING_OPTIONAL_COPY_COST: u64 = 20_000;
/// One unrolled copy of an array's item, mandatory (or starred) or optional.
const ITEM_COPY_COST: u64 = 200;
const ITEM_OPTIONAL_COPY_COST: u64 = 160;
/// Each digit of an `integer` range's bounds.
const INTEGER_DIGIT_COST: u64 = 50;
/// A `number` range with whole bounds.
const NUMBER_RANGE_COST: u64 = 500;
/// `number` ranges with a fractional bound: this each, and this times the
/// square of their count among one object's properties.
const FRACTION_RANGE_COST: u64 = 2_400;
/// Each byte of a node's JSON text the converter serialises: it keys its
/// rule cache by the text of each schema it makes a rule for, and a `type`
/// list makes one per entry. In thousandths.
const INDEX_BYTE_MILLI_COST: u64 = 30;

/// How many times over a node under a `type` list that names one type more
/// than once is priced. OpenVINO GenAI 2026.4.1 (OVMS, Qwen2.5-1.5B on an
/// Intel GPU) took 12.8 s on `["object","object"]` nested 12 deep, 3.2 times
/// what xgrammar's Python package takes on the same schema here, though it
/// is about twice as fast on optional properties; no real schema lists a
/// type twice, so the margin costs nothing.
const DUPLICATE_TYPE_FACTOR: u64 = 4;

/// Each nested `type` list that names a type more than once multiplies
/// the price below it by this many tenths on top of its count: xgrammar's
/// time grows 3.1 to 3.2 times a level on `["object","object"]` and
/// `["array","array"]` nested 8 to 14 deep, faster than the walk's own
/// price, which grows with the copies and their rule names.
const DUPLICATE_LEVEL_TENTHS: u64 = 11;

/// Deepest the walk follows schemas into schemas; deeper is refused.
const MAX_WALK_DEPTH: usize = 256;

/// Keys the converter leaves out of its rule-cache key.
const SKIPPED_KEYS: &[&str] = &[
    "title",
    "default",
    "description",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
    "$schema",
];

/// The types the converter has a shared rule for.
const BASIC_TYPES: &[&str] = &[
    "integer", "number", "string", "boolean", "null", "array", "object",
];

/// The cost of `schema` for xgrammar 0.1.31's converter and compiler, in
/// microseconds of compile time (with a 151,665-token vocabulary, one
/// thread, on an Apple M-series Mac), the sum of:
/// - for each object: [`PROPERTY_COST`] per property plus a price per byte
///   of its name, and [`SKIP_COST`] for each pair of a rule and a later
///   property it can skip to (every later property up to and including the
///   next required one), per variant of that rule (`minProperties`/
///   `maxProperties` make one per count of properties already seen);
/// - for each distinct string `minLength`/`maxLength` pair, the characters
///   the compiler unrolls (up to 128 mandatory and 127 optional),
///   [`STRING_COPY_COST`] or [`STRING_OPTIONAL_COPY_COST`] each; for each
///   distinct `pattern`, its regex's unrolled repetitions, priced by how
///   many byte values each repeated atom accepts; a measured price per
///   `format`;
/// - for each rule the converter makes, [`RULE_COST`] and its positions
///   times the square of its name's length (a rule is named after the rules
///   it is nested in, a `$ref` target after its pointer);
/// - array items unrolled for `minItems`/`maxItems`, `integer` and `number`
///   range rules, `enum`/`const` text, and the text the converter serialises
///   for its rule cache.
///
/// It follows the converter's own walk: `$ref` targets compiled once each,
/// `anyOf`/`oneOf` branches each, a one-member `allOf` as its member and a
/// longer one as any value (the converter fuses it to `{}`), and a node
/// once per entry of its `type` list (entries of one type priced once,
/// times their count, so nesting multiplies without the walk doing so).
///
/// `u64::MAX` when the walk stops early: the cost is past
/// [`MAX_XGRAMMAR_COST`] (so the walk's work is linear in the schema's size
/// and ends there), the converter would reject the schema, or the walk
/// can't price it (schemas nested more than 256 deep, a count past what the
/// converter keeps in an `int`).
pub fn xgrammar_cost(schema: &Value) -> u64 {
    Walk::new(schema).run()
}

/// `None` when [`xgrammar_cost`] of `schema` is at most
/// [`MAX_XGRAMMAR_COST`]; otherwise why it is refused: xgrammar would take
/// too long to compile it (naming the largest part of the cost), would
/// reject it, or the walk can't estimate it.
pub fn xgrammar_cost_error(schema: &Value) -> Option<String> {
    let mut walk = Walk::new(schema);
    if walk.run() <= MAX_XGRAMMAR_COST {
        return None;
    }
    match walk.stop {
        Some(Stop::Rejected(why)) => {
            return Some(format!("xgrammar would reject this schema: {why}"))
        }
        Some(Stop::Unpriced(why)) => {
            return Some(format!(
                "xgrammar's compile time for this schema can't be estimated ({why}), so it \
                 is refused"
            ))
        }
        None => {}
    }
    let part = match &walk.worst {
        Some((_, what)) => format!("; the largest part is {what}"),
        None => String::new(),
    };
    Some(format!(
        "xgrammar would take too long to compile a grammar for this schema: its estimated \
         compile time is over {} s (it grows with the square of each run of properties not \
         in `required`, the cube under `minProperties`/`maxProperties`, each character of a \
         string's `minLength`/`maxLength` up to 128, each repetition in a `pattern`, each \
         entry of nested `type` lists, and nesting){part}",
        MAX_XGRAMMAR_COST / 1_000_000
    ))
}

/// A large single part of the cost, described for the error.
enum Part {
    Run(usize),
    Counted(usize),
    Length(i64, i64),
    Pattern,
    Nesting(u64),
    Ranges,
}

impl std::fmt::Display for Part {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Part::Run(k) => write!(
                f,
                "an object with a run of {k} properties not in `required`"
            ),
            Part::Counted(k) => write!(
                f,
                "an object with `minProperties`/`maxProperties` over {k} properties"
            ),
            Part::Length(lo, hi) if *hi < 0 => write!(f, "a string with `minLength` {lo}"),
            Part::Length(lo, hi) => {
                write!(f, "a string with `minLength` {lo} and `maxLength` {hi}")
            }
            Part::Pattern => write!(f, "a `pattern`'s repetitions"),
            Part::Nesting(n) => write!(f, "{n} copies of a nested schema"),
            Part::Ranges => write!(f, "`number` ranges with fractional bounds"),
        }
    }
}

struct Walk<'a> {
    root: &'a Value,
    cost: u64,
    /// `$ref`s already given a rule: the converter compiles each once.
    refs: HashSet<&'a str>,
    /// String rules already made: the converter shares one per distinct
    /// format, pattern, or length bounds.
    strings: HashSet<String>,
    /// JSON text length of each object and array (address) asked for.
    sizes: HashMap<usize, u64>,
    /// The largest single part so far, and its description.
    worst: Option<(u64, String)>,
    /// Inside a `type` list entry whose type the list names more than once.
    duplicated: bool,
    walk_depth: usize,
    /// The walk stops once the cost is past this.
    cap: u64,
    over: bool,
    /// Why the walk stopped without a cost, when it did.
    stop: Option<Stop>,
}

/// Why the walk stopped without pricing the whole schema. Either way the
/// schema is refused: fail closed.
enum Stop {
    /// xgrammar's converter rejects the schema here.
    Rejected(&'static str),
    /// Something the walk does not price.
    Unpriced(String),
}

/// Count of decimal digits of `i`.
fn index_len(i: usize) -> u64 {
    i.checked_ilog10().map_or(1, |d| u64::from(d) + 1)
}

/// The rule name the converter gives a `$ref` target: the pointer's parts
/// joined by `_`, keeping only letters, `_`, `-` and `.`.
fn ref_name_len(pointer: &str) -> u64 {
    let mut n: u64 = 0;
    let mut first = true;
    for part in pointer.split('/') {
        if !first {
            n += 1;
        }
        first = false;
        n += part
            .chars()
            .filter(|c| c.is_ascii_alphabetic() || matches!(c, '_' | '-' | '.'))
            .count() as u64;
    }
    n
}

/// `a * b`, saturating.
fn mul(a: u64, b: u64) -> u64 {
    a.saturating_mul(b)
}

/// How many copies of an element repeated `lo` to `hi` times (`hi` < 0:
/// unbounded) the grammar parser unrolls: mandatory, optional, and whether
/// a starred copy follows. Up to 128 it writes each copy out; past that, 128
/// mandatory copies (after up to 127 optional ones below 128) and a counted
/// repeat of one more rule.
fn unrolled(lo: i64, hi: i64) -> (u64, u64, bool) {
    let lo = lo.max(0);
    if (0..=128).contains(&hi) || (hi < 0 && lo <= 128) {
        let opt = if hi < 0 { 0 } else { (hi - lo).max(0) };
        return (lo as u64, opt as u64, hi < 0);
    }
    let low_opt = if lo < 128 { 127 - lo } else { 0 };
    let low_req = lo.min(127);
    (low_req as u64 + 128, low_opt as u64, hi < 0)
}

/// How many of the 256 byte values a regex atom accepts, from the pattern
/// text starting at an escape's letter.
fn escape_breadth(c: char) -> u64 {
    match c {
        'd' => 10,
        'w' => 63,
        's' => 6,
        'D' => 246,
        'W' => 193,
        'S' => 250,
        'p' | 'P' => 128,
        _ => 1,
    }
}

/// Cost of one compile of a regex `pattern`: each atom's copies (as
/// [`unrolled`]) priced by how many byte values it accepts, a group's
/// copies at its contents' price. `Err` with the reason when xgrammar's
/// regex converter or grammar parser would reject it, or the walk can't
/// price it.
fn regex_cost(pattern: &str) -> Result<u64, &'static str> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0;
    let (cost, _) = regex_seq(&chars, &mut i, 0)?;
    Ok(cost)
}

/// Largest integer xgrammar's grammar parser accepts in a repetition.
const MAX_GRAMMAR_INTEGER: i64 = 1_000_000_000_000_000;

/// Digits from `c[*i]` on, read the way xgrammar's regex converter copies
/// them and its grammar parser reads them (leading zeros are fine, more
/// than 10^15 is an error). `None` when there are none.
fn regex_int(c: &[char], i: &mut usize) -> Result<Option<i64>, &'static str> {
    let start = *i;
    let mut n: i64 = 0;
    while let Some(d) = c.get(*i).and_then(|ch| ch.to_digit(10)) {
        n = n * 10 + i64::from(d);
        if n > MAX_GRAMMAR_INTEGER {
            return Err("a `pattern` repetition count over 10^15");
        }
        *i += 1;
    }
    Ok((*i > start).then_some(n))
}

/// The `{hex}` of a `\u{hex}` escape starting at `c[*i]` (the `{`): one
/// to six hex digits, as xgrammar's `HandleCharEscape` reads it.
fn unicode_escape(c: &[char], i: &mut usize) -> Result<(), &'static str> {
    let start = *i + 1;
    let mut j = start;
    while j < c.len() && j - start <= 6 && c[j].is_ascii_hexdigit() {
        j += 1;
    }
    if j == start || j - start > 6 || c.get(j) != Some(&'}') {
        return Err("a `pattern` with an invalid `\\u{…}` escape");
    }
    *i = j + 1;
    Ok(())
}

/// A `{x}`, `{x,}` or `{x,y}` starting at `c[*i]` (the `{`), parsed as
/// xgrammar's `HandleRepetitionRange` does: anything else is an error.
fn regex_bounds(c: &[char], i: &mut usize) -> Result<(i64, i64), &'static str> {
    const BAD: &str = "a `pattern` with an invalid `{` repetition";
    *i += 1;
    let lo = regex_int(c, i)?.ok_or(BAD)?;
    match c.get(*i) {
        Some('}') => {
            *i += 1;
            Ok((lo, lo))
        }
        Some(',') => {
            *i += 1;
            let hi = regex_int(c, i)?;
            if c.get(*i) != Some(&'}') {
                return Err(BAD);
            }
            *i += 1;
            match hi {
                None => Ok((lo, -1)),
                Some(hi) if hi < lo => {
                    Err("a `pattern` repetition whose upper bound is below its lower")
                }
                Some(hi) => Ok((lo, hi)),
            }
        }
        _ => Err(BAD),
    }
}

/// Price of a mandatory copy of an atom accepting `breadth` byte values,
/// and of an optional one.
fn atom_price(breadth: u64) -> (u64, u64) {
    let b = breadth.clamp(1, 256) - 1;
    (
        2 + STRING_COPY_COST * b / 255,
        2 + STRING_OPTIONAL_COPY_COST * b / 255,
    )
}

/// Parses a sequence (with `|` alternatives) up to `)` or the end; returns
/// its cost and whether it ended at `)`.
fn regex_seq(c: &[char], i: &mut usize, depth: usize) -> Result<(u64, bool), &'static str> {
    let mut total: u64 = 0;
    while *i < c.len() {
        let ch = c[*i];
        *i += 1;
        // One atom: its mandatory price and optional price.
        let (req, opt) = match ch {
            ')' if depth > 0 => return Ok((total, true)),
            ')' => return Err("a `pattern` with an unmatched `)`"),
            '|' | '^' | '$' => continue,
            '{' | '*' | '+' | '?' => return Err("a `pattern` repetition with nothing to repeat"),
            '(' => {
                if depth >= 64 {
                    return Err("a `pattern` with groups nested more than 64 deep");
                }
                // Skip a group's `?:`-style prefix.
                if c.get(*i) == Some(&'?') {
                    while *i < c.len() && c[*i] != ':' && c[*i] != '>' && c[*i] != ')' {
                        *i += 1;
                    }
                    if matches!(c.get(*i), Some(':' | '>')) {
                        *i += 1;
                    }
                }
                let (g, closed) = regex_seq(c, i, depth + 1)?;
                if !closed {
                    return Err("a `pattern` with an unclosed `(`");
                }
                (g, g)
            }
            '[' => {
                let mut neg = false;
                if c.get(*i) == Some(&'^') {
                    neg = true;
                    *i += 1;
                }
                let mut n: u64 = 0;
                let mut first = true;
                while *i < c.len() && (first || c[*i] != ']') {
                    first = false;
                    let a = c[*i];
                    *i += 1;
                    let (lo, width) = if a == '\\' && *i < c.len() {
                        let e = c[*i];
                        *i += 1;
                        if e == 'u' && c.get(*i) == Some(&'{') {
                            unicode_escape(c, i)?;
                            n += 1;
                            continue;
                        }
                        let b = escape_breadth(e);
                        if b > 1 {
                            n += b;
                            continue;
                        }
                        (e as u32, 1)
                    } else {
                        (a as u32, 1u64)
                    };
                    if c.get(*i) == Some(&'-') && c.get(*i + 1).is_some_and(|x| *x != ']') {
                        let hi = c[*i + 1] as u32;
                        *i += 2;
                        n += u64::from(hi.saturating_sub(lo)) + 1;
                    } else {
                        n += width;
                    }
                }
                *i += 1; // the `]`
                let n = n.min(256);
                atom_price(if neg { 256 - n.min(255) } else { n })
            }
            '.' => atom_price(256),
            '\\' => {
                let e = c.get(*i).copied().unwrap_or('\\');
                *i += 1;
                if e == 'u' && c.get(*i) == Some(&'{') {
                    unicode_escape(c, i)?;
                    atom_price(1)
                } else {
                    atom_price(escape_breadth(e))
                }
            }
            _ => atom_price(1),
        };
        // Its quantifier.
        let quantified = matches!(c.get(*i), Some('*' | '+' | '?' | '{'));
        let (lo, hi) = match c.get(*i) {
            Some('*') => {
                *i += 1;
                (0, -1)
            }
            Some('+') => {
                *i += 1;
                (1, -1)
            }
            Some('?') => {
                *i += 1;
                (0, 1)
            }
            Some('{') => regex_bounds(c, i)?,
            _ => (1, 1),
        };
        if quantified {
            if c.get(*i) == Some(&'?') {
                *i += 1; // lazy
            }
            if matches!(c.get(*i), Some('{' | '*' | '+' | '?')) {
                return Err("a `pattern` with two repetitions in a row");
            }
        }
        let (m, o, star) = unrolled(lo, hi);
        let piece = mul(req, m)
            .saturating_add(mul(opt, o))
            .saturating_add(if star { req } else { 0 });
        total = total.saturating_add(piece);
    }
    Ok((total, false))
}

/// Fixed price of a string `format`, measured.
fn format_cost(format: &str) -> Option<u64> {
    Some(match format {
        "email" => 40_000,
        "hostname" => 12_000,
        "ipv6" => 12_000,
        "json-pointer" | "relative-json-pointer" => 9_000,
        "uri" => 26_000,
        "uri-reference" => 30_000,
        "uri-template" => 85_000,
        "date" | "time" | "date-time" | "duration" | "ipv4" | "uuid" => 2_000,
        _ => return None,
    })
}

impl<'a> Walk<'a> {
    fn new(root: &'a Value) -> Self {
        Self::capped(root, MAX_XGRAMMAR_COST)
    }

    /// A walk that stops once its cost is past `cap`.
    fn capped(root: &'a Value, cap: u64) -> Self {
        Walk {
            root,
            cost: 0,
            refs: HashSet::new(),
            strings: HashSet::new(),
            sizes: HashMap::new(),
            worst: None,
            duplicated: false,
            walk_depth: 0,
            cap,
            over: false,
            stop: None,
        }
    }

    fn run(&mut self) -> u64 {
        let root = self.root;
        self.size(root);
        self.add(root_index_cost(self.sizes_of(root)));
        self.new_rule(4, 1, 1);
        self.visit(root, 4, 1, None);
        if self.over || self.stop.is_some() {
            u64::MAX
        } else {
            self.cost
        }
    }

    /// Stops the walk: xgrammar's converter rejects the schema here.
    fn reject(&mut self, why: &'static str) {
        self.stop.get_or_insert(Stop::Rejected(why));
    }

    /// Stops the walk at something it does not price.
    fn unpriced(&mut self, why: String) {
        self.stop.get_or_insert(Stop::Unpriced(why));
    }

    fn sizes_of(&self, v: &Value) -> u64 {
        self.sizes
            .get(&(v as *const Value as usize))
            .copied()
            .unwrap_or(0)
    }

    /// Whether to keep walking.
    fn going(&self) -> bool {
        !self.over && self.stop.is_none()
    }

    fn add(&mut self, c: u64) {
        self.cost = self.cost.saturating_add(c);
        if self.cost > self.cap {
            self.over = true;
        }
    }

    /// Adds `c`, remembering it when it is the largest single part so far.
    fn add_part(&mut self, c: u64, part: impl FnOnce() -> Part) {
        if c > MAX_XGRAMMAR_COST / 20 && self.worst.as_ref().is_none_or(|(w, _)| c > *w) {
            self.worst = Some((c, part().to_string()));
        }
        self.add(c);
    }

    /// JSON text length of `v`, as the converter's cache key writes it.
    fn size(&mut self, v: &Value) -> u64 {
        let key = v as *const Value as usize;
        match v {
            Value::Object(_) | Value::Array(_) => {
                if let Some(s) = self.sizes.get(&key) {
                    return *s;
                }
            }
            Value::String(t) => return t.len() as u64 + 2,
            Value::Number(n) => return n.to_string().len() as u64,
            _ => return 5,
        }
        let mut s: u64 = 2;
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    s = s.saturating_add(k.len() as u64 + 4 + self.size(x));
                }
            }
            Value::Array(a) => {
                for x in a {
                    s = s.saturating_add(1 + self.size(x));
                }
            }
            _ => {}
        }
        self.sizes.insert(key, s);
        s
    }

    /// Whether the converter's rule cache already holds `schema` (with its
    /// `type` read as `ty` when given): `true`, `{}`, or `{"type": T}` for a
    /// basic type T, ignoring annotation keys.
    fn cached(schema: &Value, ty: Option<&str>) -> bool {
        match schema {
            Value::Bool(true) => ty.is_none(),
            Value::Object(m) => {
                let mut t = ty;
                for (k, v) in m {
                    if SKIPPED_KEYS.contains(&k.as_str()) {
                        continue;
                    }
                    if k == "type" {
                        if ty.is_none() {
                            t = v.as_str();
                            if t.is_none() {
                                return false;
                            }
                        }
                        continue;
                    }
                    return false;
                }
                t.is_none_or(|t| BASIC_TYPES.contains(&t))
            }
            _ => false,
        }
    }

    /// The converter's `CreateRuleFromSchema`: a cached rule, or a new rule
    /// named `name` bytes long whose body visits `schema`.
    fn create_rule(&mut self, schema: &'a Value, name: u64, mult: u64) {
        if !self.going() {
            return;
        }
        let size = self.size(schema);
        self.add(mul(mult, size * INDEX_BYTE_MILLI_COST / 1000));
        if Self::cached(schema, None) {
            return;
        }
        self.new_rule(name, mult, 1);
        self.visit(schema, name, mult, None);
    }

    /// A new rule named `name` bytes long, with `positions` positions,
    /// made `mult` times.
    fn new_rule(&mut self, name: u64, mult: u64, positions: u64) {
        let c = mul(mult, rule_cost(name, positions));
        if mult > 1 {
            self.add_part(c, || Part::Nesting(mult));
        } else {
            self.add(c);
        }
    }

    /// The converter's `VisitSchema` for the body of a rule named `name`
    /// bytes long, with `type` read as `ty` for a `type` list entry.
    fn visit(&mut self, schema: &'a Value, name: u64, mult: u64, ty: Option<&'a str>) {
        if !self.going() {
            return;
        }
        if self.walk_depth >= MAX_WALK_DEPTH {
            self.unpriced(format!("schemas nested more than {MAX_WALK_DEPTH} deep"));
            return;
        }
        self.walk_depth += 1;
        self.visit_inner(schema, name, mult, ty);
        self.walk_depth -= 1;
    }

    fn visit_inner(&mut self, schema: &'a Value, name: u64, mult: u64, ty: Option<&'a str>) {
        let o = match schema {
            Value::Bool(true) => return,
            Value::Object(o) => o,
            _ => {
                self.reject("a schema that is neither an object nor `true`");
                return;
            }
        };
        // `VisitSchema` looks at these before `type`, also on a node whose
        // `type` the converter has set (`propertyNames` is read as a string
        // only when none of them is there).
        if let Some(r) = o.get("$ref") {
            return self.visit_ref(r);
        }
        if let Some(v) = o.get("const") {
            let n = self.size(v);
            return self.add(mul(mult, n * VALUE_BYTE_COST));
        }
        if let Some(v) = o.get("enum") {
            if !v.is_array() {
                return self.reject("an `enum` that is not an array");
            }
            let n = self.size(v);
            return self.add(mul(mult, n * VALUE_BYTE_COST));
        }
        if let Some(v) = o.get("anyOf").or_else(|| o.get("oneOf")) {
            let Value::Array(branches) = v else {
                self.reject("an `anyOf`/`oneOf` that is not an array");
                return;
            };
            for (i, b) in branches.iter().enumerate() {
                self.create_rule(b, name + 6 + index_len(i), mult);
            }
            return;
        }
        if let Some(v) = o.get("allOf") {
            match v {
                // One member is visited as the body, under `<name>_case_0`.
                Value::Array(m) if m.len() == 1 => return self.visit(&m[0], name + 7, mult, None),
                // Two or more are fused to `{}`: any value.
                Value::Array(_) => return,
                _ => {
                    self.reject("an `allOf` that is not an array");
                    return;
                }
            }
        }
        let ty = match (ty, o.get("type")) {
            (Some(t), _) => Some(t),
            (None, Some(Value::Array(list))) if !list.is_empty() => {
                return self.visit_type_list(schema, o, list, name, mult)
            }
            (None, Some(Value::Array(_))) => None,
            (None, Some(Value::String(t))) => Some(t.as_str()),
            (None, Some(_)) => {
                self.reject("a `type` that is neither a string nor an array");
                return;
            }
            (None, None) => None,
        };
        match ty {
            Some("object") => self.visit_object(o, name, mult),
            Some("array") => self.visit_array(o, name, mult),
            Some("string") => self.visit_string(o),
            Some("integer") => self.visit_integer(o, mult),
            Some("number") => self.visit_number(o, mult),
            Some("boolean" | "null") => {}
            Some(_) => self.reject("an unknown `type`"),
            None => {
                if o.contains_key("properties")
                    || o.contains_key("additionalProperties")
                    || o.contains_key("unevaluatedProperties")
                {
                    self.visit_object(o, name, mult)
                } else if o.contains_key("items")
                    || o.contains_key("prefixItems")
                    || o.contains_key("unevaluatedItems")
                {
                    self.visit_array(o, name, mult)
                }
            }
        }
    }

    /// `URIToRule`: a local `$ref` gets one rule, named after its pointer
    /// and compiled the first time. A chain of `$ref`s to `$ref`s is
    /// followed in a loop.
    fn visit_ref(&mut self, mut r: &'a Value) {
        loop {
            let Value::String(uri) = r else {
                self.reject("a `$ref` that is not a string");
                return;
            };
            if uri == "#" || !uri.starts_with("#/") || self.refs.contains(uri.as_str()) {
                return;
            }
            let mut cur = self.root;
            for part in uri[2..].split('/').filter(|p| !p.is_empty()) {
                match cur {
                    Value::Object(m) if m.contains_key(part) => cur = &m[part],
                    _ => {
                        self.reject("a `$ref` that does not resolve");
                        return;
                    }
                }
            }
            self.refs.insert(uri.as_str());
            // The suffix that makes the name unique.
            let name = ref_name_len(&uri[2..]) + 1 + index_len(self.refs.len());
            self.new_rule(name, 1, 1);
            match cur.get("$ref") {
                Some(next) if cur.is_object() => r = next,
                _ => return self.visit(cur, name, 1, None),
            }
        }
    }

    /// `VisitTypeArray`: a rule per entry for the node read as that type;
    /// entries of one type are walked once, times their count.
    fn visit_type_list(
        &mut self,
        schema: &'a Value,
        o: &'a Map<String, Value>,
        list: &'a [Value],
        name: u64,
        mult: u64,
    ) {
        let mut counts: Vec<(&'a str, u64)> = Vec::new();
        for t in list {
            // Anything but a basic type name fails the conversion.
            let Some(t) = t.as_str().filter(|t| BASIC_TYPES.contains(t)) else {
                self.reject("a `type` list entry that is not a known type");
                return;
            };
            match counts.iter_mut().find(|(s, _)| *s == t) {
                Some((_, n)) => *n += 1,
                None => counts.push((t, 1)),
            }
        }
        let size = self.size(schema).saturating_sub(self.size(&o["type"]));
        for (t, n) in counts {
            if !self.going() {
                return;
            }
            let m = mul(mult, n);
            self.add(mul(m, size * INDEX_BYTE_MILLI_COST / 1000));
            if Self::cached(schema, Some(t)) {
                continue;
            }
            // A type listed more than once: the walk prices the copies once,
            // times their count, and everything below them
            // [`DUPLICATE_TYPE_FACTOR`] times over (once, however deep such
            // lists nest), and each such list a tenth more again
            // ([`DUPLICATE_LEVEL_TENTHS`]).
            let scale = n > 1 && !self.duplicated;
            let m = if n > 1 {
                mul(m, DUPLICATE_LEVEL_TENTHS) / 10
            } else {
                m
            };
            let m = if scale {
                mul(m, DUPLICATE_TYPE_FACTOR)
            } else {
                m
            };
            let child = name + 1 + t.len() as u64;
            self.duplicated |= scale;
            self.new_rule(child, m, 1);
            self.visit(schema, child, m, Some(t));
            if scale {
                self.duplicated = false;
            }
        }
    }

    fn visit_object(&mut self, o: &'a Map<String, Value>, name: u64, mult: u64) {
        let props = match o.get("properties") {
            None => None,
            Some(Value::Object(p)) => Some(p),
            Some(_) => {
                self.reject("`properties` that is not an object");
                return;
            }
        };
        let mut required: HashSet<&str> = HashSet::new();
        match o.get("required") {
            None => {}
            Some(Value::Array(r)) => {
                for x in r {
                    let Some(s) = x.as_str() else {
                        self.reject("a `required` entry that is not a string");
                        return;
                    };
                    required.insert(s);
                }
            }
            Some(_) => {
                self.reject("`required` that is not an array");
                return;
            }
        }
        let (Ok(min_p), Ok(max_p)) = (
            int_keyword(o, "minProperties"),
            int_keyword(o, "maxProperties"),
        ) else {
            return self.reject("a `minProperties`/`maxProperties` that is not an integer");
        };
        let (min_p, max_p) = (min_p.unwrap_or(0), max_p.unwrap_or(-1));
        if min_p < 0 || max_p < -1 || (max_p >= 0 && min_p > max_p) {
            return self.reject("negative or crossed `minProperties`/`maxProperties`");
        }
        // The converter keeps these in an `int`.
        if min_p > i64::from(i32::MAX) || max_p > i64::from(i32::MAX) {
            return self.unpriced("a `minProperties`/`maxProperties` past 2^31".to_owned());
        }
        if max_p >= 0 && required.len() as i64 > max_p {
            return self.reject("more `required` properties than `maxProperties`");
        }
        // Extra properties: `additionalProperties` unless `false`, else
        // `unevaluatedProperties` when `additionalProperties` is absent.
        let (extra, extra_name): (Option<&'a Value>, u64) = match o.get("additionalProperties") {
            Some(Value::Bool(false)) => (None, 0),
            Some(a) => (Some(a), 5),
            None => match o.get("unevaluatedProperties") {
                Some(Value::Bool(false)) | None => (None, 0),
                Some(u) => (Some(u), 7),
            },
        };
        // The object's rule: its braces and separators.
        self.new_rule(name, mult, OBJECT_POSITIONS - 1);
        let patterns = match o.get("patternProperties") {
            None => None,
            Some(Value::Object(p)) if p.is_empty() => None,
            Some(Value::Object(p)) => Some(p),
            Some(_) => {
                self.reject("`patternProperties` that is not an object");
                return;
            }
        };
        let names = o.get("propertyNames");
        if patterns.is_some() || names.is_some() {
            if max_p == 0 {
                return;
            }
            if let Some(patterns) = patterns {
                for (i, (p, v)) in patterns.iter().enumerate() {
                    self.create_rule(v, name + 6 + index_len(i), mult);
                    match regex_cost(p) {
                        Ok(c) => self.add(mul(mult, c.saturating_add(PROPERTY_COST))),
                        Err(why) => return self.reject(why),
                    }
                }
            } else if let Some(n) = names {
                if !n.is_object() {
                    self.reject("`propertyNames` that is not an object");
                    return;
                }
                // The converter sets its `type` to "string" and compiles it
                // like any schema: a `$ref`, `anyOf`, `allOf`, `const` or
                // `enum` in it is followed, not read as a string.
                self.new_rule(name + 5, mult, 1);
                self.visit(n, name + 5, mult, Some("string"));
            }
            return;
        }
        let props: Vec<(&'a String, &'a Value)> =
            props.map(|p| p.iter().collect()).unwrap_or_default();
        if props.is_empty() {
            if let Some(x) = extra {
                if max_p != 0 {
                    self.create_rule(x, name + 1 + extra_name, mult);
                }
            }
            return;
        }
        if max_p == 0 {
            return;
        }
        // The object's own rules first: a wide run passes the limit there,
        // before the walk visits its properties one by one.
        let flags: Vec<bool> = props
            .iter()
            .map(|(n, _)| required.contains(n.as_str()))
            .collect();
        let k = props.len();
        let skips = skip_pairs(&flags, extra.is_some(), min_p, max_p);
        // The `<name>_part_<i>` rules, one per variant.
        let part_name = name + 6 + index_len(k);
        self.add(mul(mul(mult, skips.rules), rule_cost(part_name, 1)));
        let c = mul(
            mult,
            skips
                .pairs
                .saturating_mul(SKIP_COST)
                .saturating_add(skips.start.saturating_mul(START_SKIP_COST)),
        );
        let run = longest_run(&flags);
        self.add_part(c, || {
            if skips.counted {
                Part::Counted(k)
            } else {
                Part::Run(run)
            }
        });
        // Fractional `number` ranges among the properties cost the square
        // of their count.
        let f = props.iter().filter(|(_, v)| fraction_range(v)).count() as u64;
        if f > 1 {
            let c = mul(mult, FRACTION_RANGE_COST.saturating_mul(f * f - f));
            self.add_part(c, || Part::Ranges);
        }
        for (i, (prop, v)) in props.iter().enumerate() {
            if !self.going() {
                return;
            }
            let n = prop.len() as u64;
            let c =
                PROPERTY_COST + NAME_BYTE_COST * n.min(4) + NAME_TAIL_BYTE_COST * (n.max(4) - 4);
            self.add(mul(mult, c));
            self.create_rule(v, name + 6 + index_len(i), mult);
        }
        if let Some(x) = extra {
            self.create_rule(x, name + 1 + extra_name, mult);
        }
    }

    fn visit_array(&mut self, o: &'a Map<String, Value>, name: u64, mult: u64) {
        let prefix: &'a [Value] = match o.get("prefixItems") {
            None => &[],
            Some(Value::Array(p)) => p,
            Some(_) => {
                self.reject("`prefixItems` that is not an array");
                return;
            }
        };
        let items = o.get("items").or_else(|| o.get("unevaluatedItems"));
        let extra = match items {
            None | Some(Value::Bool(false)) => None,
            Some(v @ (Value::Bool(true) | Value::Object(_))) => Some(v),
            Some(_) => {
                self.reject("`items` that is neither a boolean nor an object");
                return;
            }
        };
        if prefix
            .iter()
            .any(|p| !matches!(p, Value::Bool(true) | Value::Object(_)))
        {
            return self.reject("a `prefixItems` entry that is neither `true` nor an object");
        }
        let (Ok(min), Ok(contains), Ok(max)) = (
            int_keyword(o, "minItems"),
            int_keyword(o, "minContains"),
            int_keyword(o, "maxItems"),
        ) else {
            return self.reject("a `minItems`/`maxItems`/`minContains` that is not an integer");
        };
        let min = min.unwrap_or(0).max(0).max(contains.unwrap_or(0));
        let max = max.unwrap_or(-1);
        if max < -1 || o.get("maxItems").is_some() && max < 0 {
            return self.reject("a negative `maxItems`");
        }
        let n = prefix.len() as i64;
        if (max >= 0 && (min > max || max < n)) || (extra.is_none() && (n < min)) {
            self.reject("unsatisfiable `minItems`/`maxItems`");
            return;
        }
        for (i, p) in prefix.iter().enumerate() {
            self.create_rule(p, name + 6 + index_len(i), mult);
        }
        let Some(x) = extra else {
            return;
        };
        self.create_rule(x, name + 11, mult);
        let n = prefix.len() as i64;
        let (lo, hi) = if n == 0 {
            if max == 0 {
                return;
            }
            ((min - 1).max(0), if max < 0 { -1 } else { max - 1 })
        } else {
            ((min - n).max(0), if max < 0 { -1 } else { max - n })
        };
        if lo > MAX_GRAMMAR_INTEGER || hi > MAX_GRAMMAR_INTEGER {
            return self.reject("a `minItems`/`maxItems` over 10^15");
        }
        let (req, opt, star) = unrolled(lo, hi);
        let c = (req + u64::from(star)) * ITEM_COPY_COST + opt * ITEM_OPTIONAL_COPY_COST;
        self.add(mul(mult, c));
    }

    /// `VisitString`: one rule per distinct format, pattern or length bounds.
    fn visit_string(&mut self, o: &'a Map<String, Value>) {
        if let Some(f) = o.get("format") {
            let Some(f) = f.as_str() else {
                self.reject("a `format` that is not a string");
                return;
            };
            if let Some(c) = format_cost(f) {
                if self.strings.insert(format!("f{f}")) {
                    self.add(c);
                }
                return;
            }
        }
        if let Some(p) = o.get("pattern") {
            let Some(p) = p.as_str() else {
                self.reject("a `pattern` that is not a string");
                return;
            };
            if self.strings.insert(format!("p{p}")) {
                let c = match regex_cost(p) {
                    Ok(c) => c,
                    Err(why) => return self.reject(why),
                };
                self.add_part(c, || Part::Pattern);
            }
            return;
        }
        let lo = o.get("minLength").map(Value::as_i64);
        let hi = o.get("maxLength").map(Value::as_i64);
        if lo.is_none() && hi.is_none() {
            return;
        }
        let (Some(lo), Some(hi)) = (lo.unwrap_or(Some(0)), hi.unwrap_or(Some(-1))) else {
            self.reject("a `minLength`/`maxLength` that is not an integer");
            return;
        };
        if hi >= 0 && lo > hi {
            self.reject("`minLength` over `maxLength`");
            return;
        }
        // The bounds are written into the grammar as `{lo,hi}`, whose
        // parser takes neither a negative lower bound nor more than 10^15.
        if !(0..=MAX_GRAMMAR_INTEGER).contains(&lo) || hi > MAX_GRAMMAR_INTEGER {
            self.reject("a `minLength`/`maxLength` that is negative or over 10^15");
            return;
        }
        if self.strings.insert(format!("l{lo},{hi}")) {
            let (req, opt, star) = unrolled(lo, hi);
            let c = mul(req + u64::from(star), STRING_COPY_COST)
                .saturating_add(mul(opt, STRING_OPTIONAL_COPY_COST));
            self.add_part(c, || Part::Length(lo, hi));
        }
    }

    /// `VisitInteger`: whole bounds, as `int64` or whole doubles, become a
    /// range regex.
    fn visit_integer(&mut self, o: &'a Map<String, Value>, mult: u64) {
        let mut n: u64 = 0;
        let mut bounds = [None, None];
        for (i, k) in RANGE_KEYS.iter().enumerate() {
            let Some(v) = o.get(*k) else { continue };
            let b = if let Some(b) = v.as_i64() {
                b
            } else if let Some(f) = v.as_f64() {
                // A whole double inside int64's range (but not its two ends,
                // which the converter rejects for precision loss).
                if f.fract() != 0.0 || !(-9.2e18..=9.2e18).contains(&f) {
                    return self.reject("an `integer` bound that is not a whole 64-bit number");
                }
                f as i64
            } else {
                return self.reject("an `integer` bound that is not a number");
            };
            let b = match i {
                2 => b.checked_add(1),
                3 => b.checked_sub(1),
                _ => Some(b),
            };
            let Some(b) = b else {
                return self.reject("an exclusive `integer` bound that overflows");
            };
            bounds[i % 2] = Some(b);
            n += b.to_string().len() as u64;
        }
        if let [Some(lo), Some(hi)] = bounds {
            if lo > hi {
                return self.reject("an `integer` range whose minimum is over its maximum");
            }
        }
        self.add(mul(mult, n * INTEGER_DIGIT_COST));
    }

    /// `VisitNumber`: bounds read as doubles become a range regex.
    fn visit_number(&mut self, o: &'a Map<String, Value>, mult: u64) {
        let mut bounds = [None, None];
        let mut fractional = false;
        let mut any = false;
        for (i, k) in RANGE_KEYS.iter().enumerate() {
            let Some(v) = o.get(*k) else { continue };
            let Some(f) = v.as_f64() else {
                return self.reject("a `number` bound that is not a number");
            };
            any = true;
            fractional |= f.fract() != 0.0;
            bounds[i % 2] = Some(f);
        }
        if !any {
            return;
        }
        if let [Some(lo), Some(hi)] = bounds {
            if lo > hi {
                return self.reject("a `number` range whose minimum is over its maximum");
            }
        }
        let c = if fractional {
            FRACTION_RANGE_COST
        } else {
            NUMBER_RANGE_COST
        };
        self.add(mul(mult, c));
    }
}

/// What a rule named `name` bytes long with `positions` positions costs.
fn rule_cost(name: u64, positions: u64) -> u64 {
    RULE_COST.saturating_add(
        positions
            .saturating_mul(name.saturating_mul(name))
            .saturating_mul(NAME_SQUARE_MILLI_COST)
            / 1000,
    )
}

/// An integer keyword as xgrammar reads it (picojson's `is<int64_t>`):
/// `Ok(None)` when absent, `Err` when present but not a JSON integer that
/// fits `int64` (`5000.0` and `5e3` are doubles to picojson, and the
/// converter rejects them).
fn int_keyword(o: &Map<String, Value>, key: &str) -> Result<Option<i64>, ()> {
    match o.get(key) {
        None => Ok(None),
        Some(v) => v.as_i64().map(Some).ok_or(()),
    }
}

/// The keywords that give a number a range.
const RANGE_KEYS: [&str; 4] = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"];

/// The root rule's cache key: the converter serialises the whole schema
/// once for it.
fn root_index_cost(size: u64) -> u64 {
    size * INDEX_BYTE_MILLI_COST / 1000
}

/// Whether `v` is a `number` schema with a fractional range bound.
fn fraction_range(v: &Value) -> bool {
    v.get("type").and_then(Value::as_str) == Some("number")
        && RANGE_KEYS
            .iter()
            .filter_map(|k| v.get(*k))
            .any(|b| b.as_f64().is_some_and(|f| f.fract() != 0.0))
}

/// The longest run of properties not in `required` (`flags`).
fn longest_run(flags: &[bool]) -> usize {
    let (mut best, mut cur) = (0, 0);
    for &r in flags {
        cur = if r { 0 } else { cur + 1 };
        best = best.max(cur);
    }
    best
}

/// For an object's properties (`flags`: which are required), the pairs of
/// (rule, later property it can skip to) the converter's
/// `GetPartialRuleForProperties` builds, summed over rule variants; the
/// same for the alternatives it can start with; and whether
/// `minProperties`/`maxProperties` made more than one variant per rule.
///
/// Rule `i` reaches every later property up to and including the next
/// required one (to the end when none; extra properties count as one more
/// optional one at the end). With no `minProperties`/`maxProperties` there
/// is one variant per rule; otherwise one per count of properties matched
/// so far that the converter's `key_matched_min`/`key_matched_max` allow.
fn skip_pairs(flags: &[bool], extra: bool, min_p: i64, max_p: i64) -> Skips {
    let k = flags.len();
    // reach[i]: properties after i up to and including the next required.
    let mut reach = vec![0u64; k];
    let mut next = if extra { 1u64 } else { 0 };
    for i in (0..k).rev() {
        reach[i] = next;
        next = if flags[i] { 1 } else { next + 1 };
    }
    let first_required = flags.iter().position(|r| *r).unwrap_or(k.saturating_sub(1));
    let start: u64 = reach[..=first_required.min(k - 1)].iter().sum();
    if min_p == 0 && max_p < 0 {
        return Skips {
            pairs: reach.iter().sum(),
            start,
            rules: k as u64,
            counted: false,
        };
    }
    // key_matched_min / key_matched_max as the converter computes them.
    let n = k as i64;
    let mut lo = vec![0i64; k];
    let mut hi = vec![n; k];
    lo[0] = 1;
    hi[0] = 1;
    let mut got_first = flags[0];
    for i in 1..k {
        lo[i] = if flags[i] { lo[i - 1] + 1 } else { lo[i - 1] };
        if !got_first {
            lo[i] = 1;
        }
        hi[i] = hi[i - 1] + 1;
        if flags[i] {
            got_first = true;
        }
    }
    let bounded = max_p >= 0;
    if extra {
        lo[k - 1] = lo[k - 1].max(1);
    } else {
        lo[k - 1] = lo[k - 1].max(min_p);
    }
    if bounded {
        hi[k - 1] = hi[k - 1].min(max_p);
    }
    for i in (0..k.saturating_sub(1)).rev() {
        lo[i] = lo[i].max(lo[i + 1] - 1);
        if bounded {
            hi[i] = if flags[i + 1] {
                hi[i].min(hi[i + 1] - 1)
            } else {
                hi[i].min(hi[i + 1])
            };
        }
    }
    let mut pairs: u64 = 0;
    let mut rules: u64 = 0;
    for i in 0..k {
        // Unbounded: variants from lo[i] to i + 1.
        let top = if bounded { hi[i] } else { i as i64 + 1 };
        let variants = (top - lo[i] + 1).max(0) as u64;
        rules = rules.saturating_add(variants);
        pairs = pairs.saturating_add(variants.saturating_mul(reach[i].max(1)));
    }
    Skips {
        pairs,
        start,
        rules,
        counted: true,
    }
}

/// What [`skip_pairs`] finds in an object's rules.
struct Skips {
    /// (rule variant, later property it can skip to) pairs.
    pairs: u64,
    /// The same for the alternatives the object can start with.
    start: u64,
    /// Rule variants.
    rules: u64,
    /// Whether `minProperties`/`maxProperties` made several variants.
    counted: bool,
}

#[cfg(test)]
#[path = "xgrammar_cost_tests.rs"]
mod tests;
