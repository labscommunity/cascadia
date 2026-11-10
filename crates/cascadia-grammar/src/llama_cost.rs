//! How long llama-server's JSON Schema to grammar converter takes on a
//! schema, estimated without running it.
//!
//! llama-server builds a GBNF grammar from every `response_format` schema
//! and every tool's `parameters` with `common/json-schema-to-grammar.cpp`,
//! inside the request, with no time limit. Its object rule calls
//! `get_recursive_refs` once per suffix of the optional properties, and each
//! call copies the rest of the list at every level, so one object costs time
//! cubic in its count of properties not in `required`. Measured at the pinned
//! llama.cpp (1692f9e, -O3, Apple M-series): 1000 optional properties 0.73 s,
//! 2000 take 5.3 s, 6000 take 176 s (a 64,922-byte schema); 6000 required
//! properties take 0.02 s.
//!
//! It also copies JSON values as it goes, which a schema can multiply: the
//! whole schema once per distinct `$ref`, the whole schema node once per
//! entry of a `type` list, and each property's schema each time an object
//! or `allOf` merge is built (measured 6–74 ns per value copied: a `type`
//! list of 30,000 entries took 5.8 s, 1,600 `allOf` merges of a property
//! with a 16,000-value `enum` took 6.7 s and 434 MB).
//!
//! [`llama_grammar_cost`] walks the schema the way that converter does and
//! adds up a cost in units of about a nanosecond of converter time;
//! [`llama_grammar_cost_error`] refuses a schema whose cost is over
//! [`MAX_LLAMA_GRAMMAR_COST`].

use serde_json::{Map, Value};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

/// Largest accepted [`llama_grammar_cost`], in units of about a nanosecond
/// of converter time: one object of about 430 optional properties with short
/// names, fewer with long names. Checked against the converter at the pin on
/// 24 schemas (real client schemas, a 68 KB extraction schema, and each slow
/// shape above): every one it took 0.36 s or more on is over this limit, and
/// the slowest one under it took 0.18 s (300 optional properties with
/// 24-byte names).
pub const MAX_LLAMA_GRAMMAR_COST: u64 = 512 * 512 * 512;

/// What each schema the converter visits adds to [`llama_grammar_cost`] on
/// top of its optional properties' cube, so that a schema making it visit
/// the same objects over and over (an `allOf` merges a `$ref` target's
/// properties and visits them again each time) is counted too.
pub const LLAMA_VISIT_COST: u64 = 1000;

/// What each JSON value the converter copies adds.
pub const LLAMA_COPY_COST: u64 = 100;

/// What each `allOf` member, and each `$ref` it follows to reach a
/// member's properties, adds.
const STEP_COST: u64 = 100;

/// What each byte of text the converter runs a regex over adds: a rule's
/// name each time a rule is added (every name below a property carries the
/// whole chain of names above it), and an `enum` or `const` value's JSON text
/// each time it is printed (measured 28-117 ns per byte of the value; a
/// 16 KB property name over 3,000 required properties took 5.1 s).
pub const LLAMA_TEXT_BYTE_COST: u64 = 60;

/// What each byte of an optional property's rule names adds per optional
/// property of its object: the converter builds them into each suffix of
/// the optional list (a 58 KB property name over 500 optional properties
/// took 418 s and 2 GB).
const SUFFIX_BYTE_COST: u64 = 30;

/// What each pair of merged `allOf` properties with the same name adds: the
/// converter probes `name0`, `name1`, … for a free rule name (2,500 members
/// each defining `x` took 2.2 s).
const COLLISION_COST: u64 = 1000;

/// Deepest nesting of visits (through properties, items, branches and
/// `$ref`s) the walk follows; deeper is refused.
const MAX_VISIT_DEPTH: usize = 256;

/// The cost of `schema` for llama-server's converter, the sum of:
/// - for each schema it visits, [`LLAMA_VISIT_COST`], plus for an object
///   rule the cube of its optional properties (those not in `required`, and
///   one more for `additionalProperties` that is `true` or a schema);
/// - [`LLAMA_COPY_COST`] for each JSON value it copies: the whole schema per
///   distinct `$ref`, a `$ref` target when first visited, the schema node
///   per `type` list entry, the `anyOf`/`oneOf` list, `items`,
///   `additionalProperties`, each property's schema per object or merge
///   built, and an `enum` or `const` value;
/// - [`LLAMA_TEXT_BYTE_COST`] for each byte of each rule name it adds (a
///   name below a property carries the names of every property above it),
///   and of an `enum` or `const` value's JSON text per visit; and, for an
///   object's optional part, each optional rule name's bytes once per
///   optional property;
/// - a cost per pair of same-named properties an `allOf` merges, per
///   `allOf` member, and per `$ref` a member goes through.
///
/// It follows the converter's own walk: `$ref`s resolve as the converter
/// resolves them and their targets are visited once; `anyOf`/`oneOf`
/// branches and each entry of a `type` list once each; an `allOf` builds one
/// object from its members' properties (through `$ref` members, members
/// under `anyOf` as optional) and visits them all again. A `properties` or
/// `allOf` that is not an object or array is iterated the way the
/// converter's JSON library does (an array's indexes are property names).
///
/// `u64::MAX` when the converter would loop on a `$ref` cycle in an `allOf`
/// member, or the visits nest deeper than 256. The walk stops as soon as the
/// cost is sure to be over [`MAX_LLAMA_GRAMMAR_COST`], so its work is linear
/// in the schema's size plus a number of steps bounded by that limit.
pub fn llama_grammar_cost(schema: &Value) -> u64 {
    Walk::new(schema).run()
}

/// `None` when [`llama_grammar_cost`] of `schema` is at most
/// [`MAX_LLAMA_GRAMMAR_COST`]; otherwise why llama-server would take too
/// long (or loop) building its grammar, naming the object with the most
/// optional properties when there is a large one.
pub fn llama_grammar_cost_error(schema: &Value) -> Option<String> {
    let mut walk = Walk::new(schema);
    let cost = walk.run();
    if cost <= MAX_LLAMA_GRAMMAR_COST {
        return None;
    }
    if let Some(why) = walk.stop {
        return Some(why.to_owned());
    }
    let worst = match walk.worst {
        Some((k, at)) if k >= 64 => format!("; the most optional properties are {k} at #{at}"),
        _ => String::new(),
    };
    Some(format!(
        "llama-server would take too long to build a grammar for this schema: its cost is \
         over the limit of {MAX_LLAMA_GRAMMAR_COST} (each object the converter builds costs \
         the cube of its properties not in `required`, each schema it visits \
         {LLAMA_VISIT_COST}, and each JSON value it copies {LLAMA_COPY_COST}, the whole \
         schema for each distinct `$ref`){worst}"
    ))
}

/// One step of a JSON pointer, kept unformatted until a pointer is reported.
#[derive(Clone)]
enum Seg<'a> {
    /// A keyword such as `properties`.
    Key(&'static str),
    /// A property name or object key.
    Name(Cow<'a, str>),
    Index(usize),
    /// Restart at a `$ref`'s pointer (its text after `#`).
    Root(&'a str),
}

/// Where a merged `allOf` property comes from: a `$ref` target (its pointer
/// after `#`), or member `i` of the holder (under its `anyOf` entry `j`).
#[derive(Clone, Copy)]
enum Owner<'a> {
    Ref(&'a str),
    Member(usize, Option<usize>),
}

/// One merged `allOf` property: name, schema, where it comes from.
type Merged<'a> = (Cow<'a, str>, &'a Value, Owner<'a>);

/// Units of [`Walk::work`]: each is [`STEP_COST`] of the result.
const VISIT_WORK: u64 = LLAMA_VISIT_COST / STEP_COST;

struct Walk<'a> {
    root: &'a Value,
    /// Count of JSON values in each object and array of the schema
    /// (address), itself included.
    sizes: HashMap<usize, u64>,
    /// The converter's `_refs`: each distinct local `$ref` its first pass
    /// finds, with the value it resolves to (`None`: it fails to resolve).
    refs: HashMap<&'a str, Option<&'a Value>>,
    /// The target of each `$ref` an `allOf` member goes through, by the
    /// address of the object holding it (so a long `$ref` is hashed once).
    member_refs: HashMap<usize, Option<&'a Value>>,
    /// `$ref`s already visited: the converter keeps one rule per target.
    visited_refs: HashSet<&'a str>,
    /// Cost of visiting a schema (address), with a `type` list entry
    /// (address, or 0), under a rule name of a given length: a schema
    /// visited again costs the same again.
    memo: HashMap<(usize, usize, u64), u64>,
    /// Length of each `enum`/`const` value's JSON text (address).
    texts: HashMap<usize, u64>,
    /// The JSON pointer of the schema being visited.
    path: Vec<Seg<'a>>,
    /// The object rule with the most optional properties: count, pointer.
    worst: Option<(u64, String)>,
    /// Steps taken, in units of [`STEP_COST`] of the result: past
    /// `MAX / STEP_COST` the result is over the limit and the walk stops.
    work: u64,
    depth: usize,
    /// Set when the walk stopped early (the result is `u64::MAX`).
    over: bool,
    /// The message, when the converter would fail rather than be slow.
    stop: Option<&'static str>,
}

/// A container's entries the way nlohmann's `items()` and range-for walk
/// them: an object's keys and values, an array's indexes and elements, one
/// entry with an empty key for any other non-null value, none for null.
fn entries(v: &Value) -> Vec<(Cow<'_, str>, &Value)> {
    match v {
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| (Cow::Borrowed(k.as_str()), v))
            .collect(),
        Value::Array(a) => a
            .iter()
            .enumerate()
            .map(|(i, v)| (Cow::Owned(i.to_string()), v))
            .collect(),
        Value::Null => Vec::new(),
        other => vec![(Cow::Borrowed(""), other)],
    }
}

/// The converter's array index for a `$ref` token: `std::stoull` reads
/// leading digits after optional spaces and `+`; no digits, or a `-`,
/// fails to resolve.
fn stoull(token: &str) -> Option<usize> {
    let t = token.trim_start();
    let t = t.strip_prefix('+').unwrap_or(t);
    let digits = t.len() - t.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return None;
    }
    t[..digits].parse().ok()
}

fn addr(v: &Value) -> usize {
    v as *const Value as usize
}

/// [`Walk::sizes`] for every object and array under `root`, without
/// recursion.
fn sizes(root: &Value) -> HashMap<usize, u64> {
    let mut sizes = HashMap::new();
    // (value, children pushed)
    let mut stack: Vec<(&Value, bool)> = vec![(root, false)];
    while let Some((v, done)) = stack.pop() {
        let children: Box<dyn Iterator<Item = &Value>> = match v {
            Value::Object(m) => Box::new(m.values()),
            Value::Array(a) => Box::new(a.iter()),
            _ => continue,
        };
        if done {
            let n = children
                .map(|c| sizes.get(&addr(c)).copied().unwrap_or(1))
                .sum::<u64>();
            sizes.insert(addr(v), n + 1);
        } else {
            stack.push((v, true));
            stack.extend(children.map(|c| (c, false)));
        }
    }
    sizes
}

/// Length of `<name>-<sub>`, or `<sub>` under the root.
fn sub_name(name: u64, sub: u64) -> u64 {
    if name == 0 {
        sub
    } else {
        name + 1 + sub
    }
}

/// Length of the converter's name for branch `i` of a union under `name`:
/// `<name>-<i>`, or `alternative-<i>` under the root.
fn alt_name(name: u64, i: usize) -> u64 {
    if name == 0 {
        12 + digits(i)
    } else {
        name + 1 + digits(i)
    }
}

fn digits(i: usize) -> u64 {
    i.to_string().len() as u64
}

/// The `-kv` rule a property `child` bytes long adds to its object.
fn prop_rules(child: u64) -> u64 {
    (child + 3).saturating_mul(LLAMA_TEXT_BYTE_COST)
}

/// An object rule's optional part: `k` optional properties whose rule names
/// add up to `names` bytes. The converter builds every suffix of the list,
/// each step copying the rest of it: about `k³`, plus each name once per
/// optional property.
fn optional_cost(k: u64, names: u64) -> u64 {
    k.saturating_mul(k)
        .saturating_mul(k)
        .saturating_add(k.saturating_mul(names).saturating_mul(SUFFIX_BYTE_COST))
}

fn escape(name: &str) -> String {
    name.replace('~', "~0").replace('/', "~1")
}

impl<'a> Walk<'a> {
    fn new(root: &'a Value) -> Self {
        Self {
            root,
            sizes: sizes(root),
            refs: HashMap::new(),
            member_refs: HashMap::new(),
            visited_refs: HashSet::new(),
            memo: HashMap::new(),
            texts: HashMap::new(),
            path: Vec::new(),
            worst: None,
            work: 0,
            depth: 0,
            over: false,
            stop: None,
        }
    }

    fn run(&mut self) -> u64 {
        let ref_cost = self.resolve_refs();
        let visits = self.visit(self.root, None, 0);
        if self.over {
            return u64::MAX;
        }
        ref_cost.saturating_add(visits)
    }

    /// [`LLAMA_COPY_COST`] for each JSON value in `v`.
    fn copy(&self, v: &Value) -> u64 {
        self.sizes.get(&addr(v)).copied().unwrap_or(1) * LLAMA_COPY_COST
    }

    /// [`LLAMA_TEXT_BYTE_COST`] for each byte of `v`'s JSON text.
    fn text(&mut self, v: &Value) -> u64 {
        let len = *self
            .texts
            .entry(addr(v))
            .or_insert_with(|| serde_json::to_string(v).map_or(0, |t| t.len() as u64));
        len.saturating_mul(LLAMA_TEXT_BYTE_COST)
    }

    /// The converter's `resolve_refs`: every object with a `$ref` (not
    /// looking inside it), anywhere in the schema, data included; each
    /// distinct `#/` ref is resolved from a copy of the whole schema.
    fn resolve_refs(&mut self) -> u64 {
        let mut stack = vec![self.root];
        while let Some(v) = stack.pop() {
            match v {
                Value::Array(a) => stack.extend(a.iter()),
                Value::Object(m) => match m.get("$ref") {
                    Some(Value::String(r)) => {
                        if r.starts_with("#/") && !self.refs.contains_key(r.as_str()) {
                            let target = self.resolve(r);
                            self.refs.insert(r.as_str(), target);
                        }
                    }
                    Some(_) => {}
                    None => stack.extend(m.values()),
                },
                _ => {}
            }
        }
        (self.refs.len() as u64).saturating_mul(self.copy(self.root))
    }

    /// A `#/a/b` ref resolved token by token as the converter does: object
    /// keys as written (no `~0`/`~1` decoding), array indexes by `stoull`.
    fn resolve(&self, r: &str) -> Option<&'a Value> {
        let mut target = self.root;
        for token in r[1..].split('/').skip(1) {
            target = match target {
                Value::Object(m) => m.get(token)?,
                Value::Array(a) => a.get(stoull(token)?)?,
                _ => return None,
            };
        }
        Some(target)
    }

    /// Counts `units` of work; false (and the walk stops) once the result
    /// is sure to be over the limit.
    fn spend(&mut self, units: u64) -> bool {
        self.work += units;
        if self.work > MAX_LLAMA_GRAMMAR_COST / STEP_COST {
            self.over = true;
        }
        !self.over
    }

    fn fail(&mut self, why: &'static str) {
        self.over = true;
        self.stop.get_or_insert(why);
    }

    /// The current JSON pointer, after `#`.
    fn pointer(&self) -> String {
        let start = self
            .path
            .iter()
            .rposition(|s| matches!(s, Seg::Root(_)))
            .unwrap_or(0);
        let mut at = String::new();
        for s in &self.path[start..] {
            match s {
                Seg::Root(r) => at.push_str(r),
                Seg::Key(k) => {
                    at.push('/');
                    at.push_str(k);
                }
                Seg::Name(n) => {
                    at.push('/');
                    at.push_str(&escape(n));
                }
                Seg::Index(i) => {
                    at.push('/');
                    at.push_str(&i.to_string());
                }
            }
        }
        at
    }

    fn note(&mut self, k: u64) {
        if self.worst.as_ref().is_none_or(|(w, _)| k > *w) {
            self.worst = Some((k, self.pointer()));
        }
    }

    /// Visits `node` under a rule name `name` bytes long, with `segs`
    /// pushed onto the pointer.
    fn child(
        &mut self,
        segs: &[Seg<'a>],
        node: &'a Value,
        ty: Option<&'a Value>,
        name: u64,
    ) -> u64 {
        let len = self.path.len();
        self.path.extend(segs.iter().cloned());
        let cost = self.visit(node, ty, name);
        self.path.truncate(len);
        cost
    }

    /// One visit of `node`, as the converter's `visit`, under a rule name
    /// `name` bytes long (0: the root, whose rule is `root`), with `ty`
    /// standing for `type` when a `type` list is visited one entry at a time.
    fn visit(&mut self, node: &'a Value, ty: Option<&'a Value>, name: u64) -> u64 {
        if !self.spend(VISIT_WORK) {
            return u64::MAX;
        }
        // The rule this visit adds, under its name.
        let own = LLAMA_VISIT_COST.saturating_add(name.max(4).saturating_mul(LLAMA_TEXT_BYTE_COST));
        let Some(obj) = node.as_object() else {
            return own;
        };
        let key = (addr(node), ty.map_or(0, addr), name);
        if let Some(&c) = self.memo.get(&key) {
            return c;
        }
        if self.depth >= MAX_VISIT_DEPTH {
            self.fail(
                "this schema nests deeper than 256 levels (through properties, items, \
                 branches or `$ref`s), deeper than the llama-server grammar check follows",
            );
            return u64::MAX;
        }
        self.depth += 1;
        let body = self.visit_body(node, obj, ty, name);
        self.depth -= 1;
        if self.over {
            return u64::MAX;
        }
        let cost = own.saturating_add(body);
        self.memo.insert(key, cost);
        cost
    }

    fn visit_body(
        &mut self,
        node: &'a Value,
        obj: &'a Map<String, Value>,
        ty: Option<&'a Value>,
        name: u64,
    ) -> u64 {
        let ty = ty.or_else(|| obj.get("type")).filter(|t| !t.is_null());
        let is = |name: &str| ty.is_none_or(|t| t == name);
        if let Some(r) = obj.get("$ref") {
            let Some(r) = r.as_str() else { return 0 };
            if !self.visited_refs.insert(r) {
                return 0;
            }
            // Its rule is named `ref` and the pointer, non-alphanumerics as `-`.
            let ref_name = 3 + r.len() as u64 - 1;
            return match self.refs.get(r).copied().flatten() {
                Some(t) => self.copy(t).saturating_add(self.child(
                    &[Seg::Root(&r[1..])],
                    t,
                    None,
                    ref_name,
                )),
                None => 0,
            };
        }
        let union = obj
            .get("oneOf")
            .map(|v| (v, "oneOf"))
            .or_else(|| obj.get("anyOf").map(|v| (v, "anyOf")));
        if let Some((alts, kw)) = union {
            // The converter needs an array here (anything else fails fast).
            let Some(list) = alts.as_array() else {
                return 0;
            };
            let mut cost = self.copy(alts);
            for (i, a) in list.iter().enumerate() {
                let alt = alt_name(name, i);
                cost =
                    cost.saturating_add(self.child(&[Seg::Key(kw), Seg::Index(i)], a, None, alt));
            }
            return cost;
        }
        if let Some(types) = ty.and_then(Value::as_array) {
            let mut cost = 0u64;
            for t in types {
                // Each entry visits its own copy of the whole node.
                cost = cost
                    .saturating_add(self.copy(node))
                    .saturating_add(self.visit(node, Some(t), alt_name(name, 0)));
            }
            return cost;
        }
        if let Some(v) = obj.get("const") {
            return self.copy(v).saturating_add(self.text(v));
        }
        if let Some(v) = obj.get("enum") {
            return self.copy(v).saturating_add(self.text(v));
        }
        let additional = obj.get("additionalProperties");
        if is("object")
            && (obj.contains_key("properties")
                || additional.is_some_and(|a| a != &Value::Bool(true)))
        {
            return self.object(obj, additional, name);
        }
        if (is("object") || is("string")) && obj.contains_key("allOf") {
            return self.all_of(obj, name);
        }
        if is("array") {
            let items = obj
                .get("items")
                .map(|v| (v, "items"))
                .or_else(|| obj.get("prefixItems").map(|v| (v, "prefixItems")));
            if let Some((items, kw)) = items {
                let mut cost = self.copy(items);
                match items.as_array() {
                    Some(list) => {
                        for (i, s) in list.iter().enumerate() {
                            // `<name>-tuple-<i>`
                            let tuple = sub_name(name, 6 + digits(i));
                            cost = cost.saturating_add(self.child(
                                &[Seg::Key(kw), Seg::Index(i)],
                                s,
                                None,
                                tuple,
                            ));
                        }
                    }
                    None => {
                        let item = sub_name(name, 4);
                        cost = cost.saturating_add(self.child(&[Seg::Key(kw)], items, None, item))
                    }
                }
                return cost;
            }
        }
        0
    }

    fn object(
        &mut self,
        obj: &'a Map<String, Value>,
        additional: Option<&'a Value>,
        name: u64,
    ) -> u64 {
        let required: HashSet<&str> = obj
            .get("required")
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let props = obj.get("properties").map(entries).unwrap_or_default();
        let mut k = 0u64;
        // Bytes of the optional properties' rule names.
        let mut optional_names = 0u64;
        let mut cost = 0u64;
        for (p, s) in props {
            let child = sub_name(name, p.len() as u64);
            cost = cost.saturating_add(prop_rules(child));
            if !required.contains(p.as_ref()) {
                k += 1;
                optional_names = optional_names.saturating_add(child + 5);
            }
            let segs = [Seg::Key("properties"), Seg::Name(p)];
            cost = cost
                .saturating_add(self.copy(s))
                .saturating_add(self.child(&segs, s, None, child));
            if self.over {
                return u64::MAX;
            }
        }
        if let Some(a) = additional {
            cost = cost.saturating_add(self.copy(a));
            // `<name>-additional`, its value `<name>-additional-value`.
            let sub = sub_name(name, 10);
            if matches!(a, Value::Bool(true) | Value::Object(_)) {
                k += 1;
                optional_names = optional_names.saturating_add(sub + 5);
                cost = cost.saturating_add(prop_rules(sub));
            }
            if a.is_object() {
                let segs = [Seg::Key("additionalProperties")];
                cost = cost.saturating_add(self.child(&segs, a, None, sub + 6));
            }
        }
        self.note(k);
        optional_cost(k, optional_names).saturating_add(cost)
    }

    fn all_of(&mut self, obj: &'a Map<String, Value>, name: u64) -> u64 {
        let Some(members) = obj.get("allOf") else {
            return 0;
        };
        let mut props: Vec<Merged<'a>> = Vec::new();
        let mut required: HashSet<Cow<'a, str>> = HashSet::new();
        let mut cost = 0u64;
        for (i, (_, m)) in entries(members).into_iter().enumerate() {
            match m.as_object().and_then(|o| o.get("anyOf")) {
                Some(alts) => {
                    for (j, (_, alt)) in entries(alts).into_iter().enumerate() {
                        let at = Owner::Member(i, Some(j));
                        cost = cost.saturating_add(self.component(
                            alt,
                            false,
                            at,
                            &mut props,
                            &mut required,
                        ));
                    }
                }
                None => {
                    let at = Owner::Member(i, None);
                    cost =
                        cost.saturating_add(self.component(m, true, at, &mut props, &mut required))
                }
            }
            if self.over {
                return u64::MAX;
            }
        }
        // Same-named properties: the converter probes for a free rule name
        // past each earlier one.
        let mut same: HashMap<&str, u64> = HashMap::new();
        for (p, _, _) in &props {
            let n = same.entry(p.as_ref()).or_insert(0);
            cost = cost.saturating_add(n.saturating_mul(COLLISION_COST));
            *n += 1;
        }
        let mut k = 0u64;
        let mut optional_names = 0u64;
        for (p, _, _) in &props {
            let child = sub_name(name, p.len() as u64);
            cost = cost.saturating_add(prop_rules(child));
            if !required.contains(p) {
                k += 1;
                optional_names = optional_names.saturating_add(child + 5);
            }
        }
        for (p, s, owner) in props {
            let child = sub_name(name, p.len() as u64);
            let len = self.path.len();
            match owner {
                Owner::Ref(r) => self.path.push(Seg::Root(r)),
                Owner::Member(i, j) => {
                    self.path.extend([Seg::Key("allOf"), Seg::Index(i)]);
                    if let Some(j) = j {
                        self.path.extend([Seg::Key("anyOf"), Seg::Index(j)]);
                    }
                }
            }
            self.path.push(Seg::Key("properties"));
            self.path.push(Seg::Name(p));
            cost = cost
                .saturating_add(self.copy(s))
                .saturating_add(self.visit(s, None, child));
            self.path.truncate(len);
            if self.over {
                return u64::MAX;
            }
        }
        self.note(k);
        optional_cost(k, optional_names).saturating_add(cost)
    }

    /// The converter's `add_component`: a member's properties, through its
    /// `$ref`s (which it follows with no limit and no cycle check), required
    /// when `req`. Returns the steps' cost.
    fn component(
        &mut self,
        mut c: &'a Value,
        req: bool,
        mut at: Owner<'a>,
        props: &mut Vec<Merged<'a>>,
        required: &mut HashSet<Cow<'a, str>>,
    ) -> u64 {
        let mut cost = STEP_COST;
        if !self.spend(1) {
            return cost;
        }
        let mut seen: HashSet<usize> = HashSet::new();
        while let Some(r) = c.as_object().and_then(|o| o.get("$ref")) {
            let Some(r) = r.as_str() else { return cost };
            if !seen.insert(addr(c)) {
                self.fail(
                    "llama-server can't build a grammar for this schema: an `allOf` member's \
                     `$ref`s form a cycle, which its converter follows until its stack \
                     overflows",
                );
                return cost;
            }
            cost += STEP_COST;
            if !self.spend(1) {
                return cost;
            }
            let target = match self.member_refs.get(&addr(c)) {
                Some(&t) => t,
                None => {
                    let t = self.refs.get(r).copied().flatten();
                    self.member_refs.insert(addr(c), t);
                    t
                }
            };
            match target {
                Some(t) => {
                    at = Owner::Ref(&r[1..]);
                    c = t;
                }
                None => return cost,
            }
        }
        let Some(p) = c.as_object().and_then(|o| o.get("properties")) else {
            return cost;
        };
        for (name, s) in entries(p) {
            // Its copy, below; its visit spends on its own.
            if !self.spend(1) {
                return cost;
            }
            if req {
                required.insert(name.clone());
            }
            props.push((name, s, at));
        }
        cost
    }
}

#[cfg(test)]
#[path = "llama_cost_tests.rs"]
mod tests;
