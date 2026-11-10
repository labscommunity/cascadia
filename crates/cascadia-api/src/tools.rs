//! Request-side tool handling: the `tools` entries a client may send
//! (function and custom tools), and [`ToolPolicy`], which turns `tools`,
//! `tool_choice` and `parallel_tool_calls` into one validated description of
//! what the model may call.
//!
//! Custom tools (`{"type":"custom","custom":{...}}`) have no function
//! schema; each reaches the chat template as a function shim with one
//! required string parameter, `input`, so templates that only know function
//! tools render them unchanged.

use std::collections::{BTreeSet, HashMap, HashSet};

use cascadia_metrics::UnenforcedReason;
use cascadia_types::GrammarSpec;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::structured::{
    check_schema, relax_unsatisfiable_bounds, DATA_KEYS, MAX_SCHEMA_DEPTH, SCHEMA_MAPS,
};
use crate::{FunctionCall, Tool, ToolCall};

/// One entry of a request's `tools` list.
///
/// Any entry carrying a `function` key is a function tool whatever its
/// `type` says, as before custom tools existed. Otherwise an entry with a
/// `custom` key or `type: "custom"` is a custom tool, and anything else is
/// read as a function tool, so a malformed entry's error names the field it
/// lacks. The custom form needs `type: "custom"`, which
/// [`ToolPolicy::from_request`] checks.
#[derive(Debug, Clone)]
pub enum RequestTool {
    /// `{"type":"function","function":{"name",...}}`; `function` is opaque
    /// JSON forwarded to the chat template as the client wrote it.
    Function { r#type: String, function: Value },
    /// `{"type":"custom","custom":{"name","description"?,"format"?}}`.
    Custom { r#type: String, custom: Value },
}

#[derive(Deserialize)]
struct FunctionToolForm {
    r#type: String,
    function: Value,
}

#[derive(Deserialize)]
struct CustomToolForm {
    r#type: String,
    custom: Value,
}

/// Whether a `tools` or `tool_calls` entry is in the custom form: no
/// `function` key, and a `custom` key or `type: "custom"`.
fn is_custom_form(v: &Value) -> bool {
    v.get("function").is_none()
        && (v.get("custom").is_some() || v.get("type").and_then(Value::as_str) == Some("custom"))
}

impl<'de> Deserialize<'de> for RequestTool {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        let tool = if is_custom_form(&v) {
            CustomToolForm::deserialize(v).map(|t| RequestTool::Custom {
                r#type: t.r#type,
                custom: t.custom,
            })
        } else {
            FunctionToolForm::deserialize(v).map(|t| RequestTool::Function {
                r#type: t.r#type,
                function: t.function,
            })
        };
        tool.map_err(serde::de::Error::custom)
    }
}

impl RequestTool {
    /// The tool as the chat template and the engine see it: a function tool
    /// unchanged, a custom tool as its function shim (see [`custom_shim`]).
    pub fn to_tool(&self) -> Tool {
        match self {
            RequestTool::Function { r#type, function } => Tool {
                r#type: r#type.clone(),
                function: function.clone(),
            },
            RequestTool::Custom { custom, .. } => custom_shim(custom),
        }
    }

    /// The tool's name, when it has a string one.
    fn name(&self) -> Option<&str> {
        match self {
            RequestTool::Function { function, .. } => function.get("name"),
            RequestTool::Custom { custom, .. } => custom.get("name"),
        }
        .and_then(Value::as_str)
    }
}

/// The function tool a custom tool is rendered as: one required string
/// parameter `input`, described by the tool's description plus, when its
/// `format` is a grammar, that grammar's syntax and definition so the model
/// can follow it. The grammar itself is not enforced inside `input`.
pub fn custom_shim(custom: &Value) -> Tool {
    let name = custom.get("name").and_then(Value::as_str).unwrap_or("");
    let mut description = custom
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if let Some(format) = custom.get("format") {
        if format.get("type").and_then(Value::as_str) == Some("grammar") {
            let grammar = format.get("grammar");
            let field = |k: &str| {
                grammar
                    .and_then(|g| g.get(k))
                    .and_then(Value::as_str)
                    .unwrap_or("")
            };
            description.push_str(&format!(
                "\n\nThe input must follow this {} grammar:\n{}",
                field("syntax"),
                field("definition")
            ));
        }
    }
    Tool {
        r#type: "function".into(),
        function: json!({
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": false
            }
        }),
    }
}

/// What the model may do with the effective tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolMode {
    /// `tool_choice: "none"`: tools may be rendered, never called.
    None,
    /// The model decides (`"auto"`, an absent `tool_choice`, or
    /// `allowed_tools` with mode `auto`).
    Auto,
    /// The model must call at least one tool.
    Required,
    /// The model must call this tool (a function or a custom tool).
    Named(String),
}

/// `tools` + `tool_choice` + `parallel_tool_calls`, validated.
#[derive(Debug, Clone)]
pub struct ToolPolicy {
    /// The effective tools, function-shaped (custom tools as their shims),
    /// narrowed to `allowed_tools` when one was given, in the client's order.
    pub tools: Vec<Tool>,
    pub mode: ToolMode,
    /// Names of the effective tools that are custom tools.
    pub custom_names: BTreeSet<String>,
    /// `parallel_tool_calls: false`: at most one call.
    pub single_call: bool,
    /// `allowed_tools` narrowed the tool list.
    pub narrowed: bool,
    /// The request had a `tool_choice`.
    pub choice_given: bool,
    /// The request had a `tools` field (absent and `[]` render differently).
    pub tools_given: bool,
    /// Indices into `tools` whose `parameters` lost a bound pair no value
    /// can meet (see [`relax_parameters`]).
    pub relaxed: BTreeSet<usize>,
}

/// Which kind of tool a name belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Function,
    Custom,
}

impl Kind {
    /// How an error names a tool of this kind.
    fn what(self) -> &'static str {
        match self {
            Kind::Function => "function",
            Kind::Custom => "custom tool",
        }
    }

    /// "a function tool" / "a custom tool".
    fn a_tool(self) -> &'static str {
        match self {
            Kind::Function => "a function tool",
            Kind::Custom => "a custom tool",
        }
    }
}

/// Why `name`, named as a `wanted` tool, isn't one: `None` when it is.
fn kind_mismatch(by_name: &HashMap<&str, Kind>, name: &str, wanted: Kind) -> Option<String> {
    match by_name.get(name) {
        Some(&kind) if kind == wanted => None,
        Some(&kind) => Some(format!("but '{name}' is {}", kind.a_tool())),
        None => Some("which is not in tools".into()),
    }
}

impl ToolPolicy {
    /// Validates the request's tool fields. An `Err` is the message of a
    /// 400: the caller's mistake, not something an engine should guess at.
    pub fn from_request(
        tools: Option<&[RequestTool]>,
        tool_choice: Option<&Value>,
        parallel_tool_calls: Option<bool>,
    ) -> Result<ToolPolicy, String> {
        let tools_given = tools.is_some();
        let tools = tools.unwrap_or_default();
        // One pass builds the name map every later lookup uses, so a body
        // full of tools validates in linear time.
        let mut by_name: HashMap<&str, Kind> = HashMap::with_capacity(tools.len());
        for (i, tool) in tools.iter().enumerate() {
            let kind = match tool {
                RequestTool::Function { .. } => Kind::Function,
                RequestTool::Custom { r#type, .. } => {
                    if r#type != "custom" {
                        return Err(format!(
                            "tools[{i}] has a custom object but type '{type}'; expected \"custom\"",
                            type = r#type
                        ));
                    }
                    Kind::Custom
                }
            };
            let Some(name) = tool.name().filter(|n| !n.is_empty()) else {
                if kind == Kind::Custom {
                    return Err(format!("tools[{i}] is a custom tool without a name"));
                }
                continue;
            };
            match by_name.get(name) {
                Some(&seen) if seen == Kind::Custom || kind == Kind::Custom => {
                    return Err(format!(
                        "tools[{i}] is named '{name}', which another tool already uses; \
                         a custom tool's name must be unique"
                    ));
                }
                Some(_) => {}
                None => {
                    by_name.insert(name, kind);
                }
            }
        }

        let (mode, allowed) = match tool_choice {
            None => (ToolMode::Auto, None),
            Some(choice) => choice_mode(choice, &by_name)?,
        };

        let mut effective = Vec::with_capacity(tools.len());
        let mut custom_names = BTreeSet::new();
        let mut relaxed = BTreeSet::new();
        for tool in tools {
            if let Some(allowed) = &allowed {
                if !tool.name().is_some_and(|n| allowed.contains(n)) {
                    continue;
                }
            }
            if let (RequestTool::Custom { .. }, Some(name)) = (tool, tool.name()) {
                custom_names.insert(name.to_string());
            }
            let mut tool = tool.to_tool();
            if relax_parameters(&mut tool) {
                relaxed.insert(effective.len());
            }
            effective.push(tool);
        }
        Ok(ToolPolicy {
            tools: effective,
            mode,
            custom_names,
            single_call: parallel_tool_calls == Some(false),
            narrowed: allowed.is_some(),
            choice_given: tool_choice.is_some(),
            tools_given,
            relaxed,
        })
    }

    /// The call grammar of a forced or union turn ([`tool_grammar`]) covers
    /// a tool whose parameters lost a bound pair: part of the client's
    /// schema is not enforced.
    pub fn grammar_relaxed(&self) -> bool {
        let named = match &self.mode {
            ToolMode::Named(name) => Some(name.as_str()),
            _ => None,
        };
        // The tools `tool_schema` puts in the grammar: every one with a
        // name, or the first one a named choice names.
        let mut in_grammar = self.tools.iter().enumerate().filter(|(_, t)| {
            t.function
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|n| named.is_none_or(|want| want == n))
        });
        match named {
            Some(_) => in_grammar
                .next()
                .is_some_and(|(i, _)| self.relaxed.contains(&i)),
            None => in_grammar.any(|(i, _)| self.relaxed.contains(&i)),
        }
    }

    /// The model may call a tool: there is one and `tool_choice` isn't
    /// `"none"`.
    pub fn callable(&self) -> bool {
        !self.tools.is_empty() && self.mode != ToolMode::None
    }

    /// The model must call a tool (`"required"` or a named tool).
    pub fn forced(&self) -> bool {
        self.callable() && matches!(self.mode, ToolMode::Required | ToolMode::Named(_))
    }

    /// `tool_choice` as the engine gets it: `None` when the request had
    /// none, else `"none"`, `"auto"`, `"required"` or a named function (a
    /// custom tool is named as its function shim).
    pub fn engine_tool_choice(&self) -> Option<Value> {
        if !self.choice_given {
            return None;
        }
        Some(match &self.mode {
            ToolMode::None => json!("none"),
            ToolMode::Auto => json!("auto"),
            ToolMode::Required => json!("required"),
            ToolMode::Named(name) => json!({"type": "function", "function": {"name": name}}),
        })
    }

    /// The effective tools as wire JSON for the engine; `None` when the
    /// request had no `tools` field.
    pub fn engine_tools(&self) -> Option<Value> {
        self.prompt_tools()
            .and_then(|ts| serde_json::to_value(ts).ok())
    }

    /// The effective tools as the chat template sees them; `None` when the
    /// request had no `tools` field.
    pub fn prompt_tools(&self) -> Option<&[Tool]> {
        self.tools_given.then_some(self.tools.as_slice())
    }
}

const FUNCTION_CHOICE_SHAPE: &str =
    "tool_choice must be a string or {\"type\":\"function\",\"function\":{\"name\":...}}";
const CUSTOM_CHOICE_SHAPE: &str =
    "tool_choice must be {\"type\":\"custom\",\"custom\":{\"name\":...}}";

/// A `tool_choice` → the mode it sets and, for `allowed_tools`, the names
/// it allows; every name is checked against the request's tools.
fn choice_mode<'a>(
    choice: &'a Value,
    by_name: &HashMap<&str, Kind>,
) -> Result<(ToolMode, Option<HashSet<&'a str>>), String> {
    if let Some(s) = choice.as_str() {
        let mode = match s {
            "auto" => ToolMode::Auto,
            "none" => ToolMode::None,
            "required" => ToolMode::Required,
            other => {
                return Err(format!(
                    "tool_choice '{other}' is not one of \"auto\", \"none\", \"required\""
                ))
            }
        };
        return Ok((mode, None));
    }
    let named_mode = |kind: Kind, key: &str, shape: &str| {
        let name = named(choice, key).ok_or_else(|| shape.to_string())?;
        if let Some(why) = kind_mismatch(by_name, name, kind) {
            return Err(format!("tool_choice names {} '{name}', {why}", kind.what()));
        }
        Ok((ToolMode::Named(name.to_string()), None))
    };
    match choice.get("type").and_then(Value::as_str) {
        Some("function") => named_mode(Kind::Function, "function", FUNCTION_CHOICE_SHAPE),
        Some("custom") => named_mode(Kind::Custom, "custom", CUSTOM_CHOICE_SHAPE),
        Some("allowed_tools") => {
            let (mode, names) = allowed_tools(choice, by_name)?;
            Ok((mode, Some(names)))
        }
        Some(other) => Err(format!(
            "tool_choice type '{other}' is not one of \"function\", \"custom\", \"allowed_tools\""
        )),
        None => Err(FUNCTION_CHOICE_SHAPE.to_string()),
    }
}

/// `{"type":T, T:{"name":N}}` → `N`, when it is a non-empty string.
fn named<'a>(choice: &'a Value, key: &str) -> Option<&'a str> {
    choice
        .get(key)
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
}

/// `{"type":"allowed_tools","allowed_tools":{"mode","tools":[...]}}` → the
/// mode it sets and the names it allows, each checked against `tools`.
fn allowed_tools<'a>(
    choice: &'a Value,
    by_name: &HashMap<&str, Kind>,
) -> Result<(ToolMode, HashSet<&'a str>), String> {
    const FIELD: &str = "tool_choice.allowed_tools";
    let spec = choice
        .get("allowed_tools")
        .filter(|v| v.is_object())
        .ok_or_else(|| format!("{FIELD} must be an object with \"mode\" and \"tools\""))?;
    let mode = match spec.get("mode").and_then(Value::as_str) {
        Some("auto") => ToolMode::Auto,
        Some("required") => ToolMode::Required,
        _ => return Err(format!("{FIELD}.mode must be \"auto\" or \"required\"")),
    };
    let list = spec
        .get("tools")
        .and_then(Value::as_array)
        .filter(|l| !l.is_empty())
        .ok_or_else(|| format!("{FIELD}.tools must be a non-empty list"))?;
    let mut names = HashSet::with_capacity(list.len());
    for (i, entry) in list.iter().enumerate() {
        let (kind, name) = match entry.get("type").and_then(Value::as_str) {
            Some("function") => (Kind::Function, named(entry, "function")),
            Some("custom") => (Kind::Custom, named(entry, "custom")),
            _ => (Kind::Function, None),
        };
        let Some(name) = name else {
            return Err(format!(
                "{FIELD}.tools[{i}] must be {{\"type\":\"function\",\"function\":{{\"name\":...}}}} or {{\"type\":\"custom\",\"custom\":{{\"name\":...}}}}"
            ));
        };
        if let Some(why) = kind_mismatch(by_name, name, kind) {
            return Err(format!("{FIELD} lists {} '{name}', {why}", kind.what()));
        }
        names.insert(name);
    }
    Ok((mode, names))
}

/// A `tool_calls` entry of an assistant message: the function form, or a
/// replayed custom call. The form is chosen as for [`RequestTool`], so a
/// malformed entry's error names the field it lacks.
enum WireToolCall {
    Function(ToolCall),
    Custom {
        id: String,
        r#type: String,
        custom: WireCustomCall,
    },
}

#[derive(Deserialize)]
struct WireCustomForm {
    id: String,
    r#type: String,
    custom: WireCustomCall,
}

impl<'de> Deserialize<'de> for WireToolCall {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        let call = if is_custom_form(&v) {
            WireCustomForm::deserialize(v).map(|c| WireToolCall::Custom {
                id: c.id,
                r#type: c.r#type,
                custom: c.custom,
            })
        } else {
            match ToolCall::deserialize(&v) {
                Ok(call) => Ok(WireToolCall::Function(call)),
                // The untagged parse this replaced fell back to the custom
                // form; keep accepting what it accepted.
                Err(e) if v.get("custom").is_some() => WireCustomForm::deserialize(&v)
                    .map(|c| WireToolCall::Custom {
                        id: c.id,
                        r#type: c.r#type,
                        custom: c.custom,
                    })
                    .map_err(|_| e),
                Err(e) => Err(e),
            }
        };
        call.map_err(serde::de::Error::custom)
    }
}

#[derive(Deserialize)]
struct WireCustomCall {
    name: String,
    input: String,
}

/// `ChatMessage.tool_calls`: function calls as before, plus replayed custom
/// calls (`{"id","type":"custom","custom":{"name","input"}}`), which become
/// the function form the template renders, with arguments `{"input": …}`.
pub(crate) fn de_tool_calls<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<Vec<ToolCall>>, D::Error> {
    let Some(calls) = Option::<Vec<WireToolCall>>::deserialize(d)? else {
        return Ok(None);
    };
    calls
        .into_iter()
        .map(|call| match call {
            WireToolCall::Function(call) => Ok(call),
            WireToolCall::Custom { id, r#type, custom } => {
                if r#type != "custom" {
                    return Err(serde::de::Error::custom(format!(
                        "tool call '{id}' has a custom object but type '{type}'; expected \"custom\"",
                        type = r#type
                    )));
                }
                Ok(ToolCall {
                    id,
                    r#type: "function".into(),
                    function: FunctionCall {
                        name: custom.name,
                        arguments: json!({"input": custom.input}).to_string(),
                    },
                })
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Drop from `tool`'s `parameters` every bound pair no value can meet
/// ([`relax_unsatisfiable_bounds`]), such as `minLength` above `maxLength`,
/// logging each pair with the tool's name; `true` when one was dropped.
///
/// The server must accept such a tool (the client chose it, and a 400
/// would refuse a request the model may never call it in), but no call
/// could match it: a grammar built from it, here or by a downstream server
/// building its own from the forwarded tools (llama-server), decodes until
/// the request is killed. Dropping just that pair keeps the rest of the
/// call shape enforced. A tool with no such pair is left exactly as sent.
fn relax_parameters(tool: &mut Tool) -> bool {
    let Some(parameters) = tool.function.get_mut("parameters") else {
        return false;
    };
    let dropped = relax_unsatisfiable_bounds(parameters);
    let name = tool.function.get("name").and_then(Value::as_str);
    for found in &dropped {
        tracing::warn!(
            tool = name.unwrap_or(""),
            detail = %found.detail,
            "tool parameters hold a bound pair no value can meet; dropping {} and {}",
            found.lower,
            found.upper
        );
    }
    !dropped.is_empty()
}

/// The parameters schema of a tool that declares none: any object.
fn empty_parameters() -> Value {
    json!({"type": "object", "properties": {}})
}

/// The output constraint for a turn with callable tools, or `None` when
/// there is nothing to enforce or it can't be built.
///
/// A forced turn (`"required"` or a named tool) gets the call array: a JSON
/// array of at least one `{"name", "parameters"}` object, `name` one of the
/// tools (only the named one for a named choice), `parameters` that tool's
/// schema; at most one item for a named choice or `parallel_tool_calls:
/// false`. An `auto` turn with a JSON `response_format` (`format_schema`,
/// its schema text) gets `{"anyOf": [<call array>, <format schema>]}`: a
/// call or the asked-for JSON.
///
/// Each nested schema's `$defs` and `definitions` move to the root, renamed
/// with a prefix per source (`t<i>_` for the tool at index `i`, `rf_` for
/// the format schema), and its `$ref`s are rewritten to match; a reference
/// to the schema's own root (`#`) points at a copy stored as
/// `$defs/<prefix>root`. `$schema` and `$id` are dropped from moved schemas.
/// The result passes the same limits and normalisation as a
/// `response_format` schema. A `$ref` of any other form, a `$ref` in a
/// schema that also has a nested `$id` (which would change what the ref
/// resolves against), or a result over those limits, gives `None`, logged and counted: the turn then runs
/// unconstrained. The grammar is best-effort for the same reason. A bound
/// pair no value can meet never gets here: [`ToolPolicy::from_request`]
/// drops it from the tools, and `response_format` refuses it with a 400.
pub fn tool_grammar(policy: &ToolPolicy, format_schema: Option<&str>) -> Option<GrammarSpec> {
    // Under a forced choice the format is moot: the answer is the call.
    let format_schema = match (policy.forced(), format_schema) {
        (true, _) => None,
        (false, Some(format)) if policy.callable() => Some(format),
        (false, _) => return None,
    };
    match tool_schema(policy, format_schema) {
        Ok(body) => body.map(GrammarSpec::best_effort_json_schema),
        Err(reason) => {
            tracing::warn!(
                reason = reason.as_str(),
                "tool-call output constraint not enforced; answering unconstrained"
            );
            cascadia_metrics::count_constraint_unenforced(reason);
            None
        }
    }
}

/// The text of [`tool_grammar`]'s schema; `Ok(None)` when no tool has a
/// name to call.
fn tool_schema(
    policy: &ToolPolicy,
    format_schema: Option<&str>,
) -> Result<Option<String>, UnenforcedReason> {
    let named = match &policy.mode {
        ToolMode::Named(name) => Some(name.as_str()),
        _ => None,
    };
    let mut hoisted = Hoisted::default();
    let mut items = Vec::new();
    for (i, tool) in policy.tools.iter().enumerate() {
        let Some(name) = tool.function.get("name").and_then(Value::as_str) else {
            continue;
        };
        if named.is_some_and(|n| n != name) {
            continue;
        }
        let parameters = match tool.function.get("parameters") {
            None | Some(Value::Null) => empty_parameters(),
            Some(p) => hoisted.take(p.clone(), &format!("t{i}_"))?,
        };
        items.push(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [name]},
                "parameters": parameters
            },
            "required": ["name", "parameters"],
            "additionalProperties": false
        }));
        if named.is_some() {
            break;
        }
    }
    if items.is_empty() {
        return Ok(None);
    }
    let mut calls = Map::new();
    calls.insert("type".into(), json!("array"));
    calls.insert("minItems".into(), json!(1));
    if named.is_some() || policy.single_call {
        calls.insert("maxItems".into(), json!(1));
    }
    calls.insert("items".into(), json!({"anyOf": items}));
    let mut root = match format_schema {
        None => calls,
        Some(text) => {
            // The format schema was parsed once already; it can't fail here.
            let format: Value =
                serde_json::from_str(text).map_err(|_| UnenforcedReason::SchemaRejected)?;
            let format = hoisted.take(format, "rf_")?;
            let mut root = Map::new();
            root.insert("anyOf".into(), json!([Value::Object(calls), format]));
            root
        }
    };
    for (key, defs) in [
        ("$defs", hoisted.defs),
        ("definitions", hoisted.definitions),
    ] {
        if !defs.is_empty() {
            root.insert(key.into(), Value::Object(defs));
        }
    }
    let root = Value::Object(root);
    let text = serde_json::to_string(&root).map_err(|_| UnenforcedReason::TooComplex)?;
    check_schema(&root, &text)
        .map(Some)
        .map_err(|_| UnenforcedReason::TooComplex)
}

/// The definitions moved to the root of a tool grammar, by keyword.
#[derive(Default)]
struct Hoisted {
    defs: Map<String, Value>,
    definitions: Map<String, Value>,
}

impl Hoisted {
    /// `schema`, about to be nested, with its definitions moved here under
    /// `prefix` and its references rewritten to match (see
    /// [`tool_grammar`]).
    fn take(&mut self, schema: Value, prefix: &str) -> Result<Value, UnenforcedReason> {
        let Value::Object(mut root) = schema else {
            return Ok(schema);
        };
        let own = |root: &Map<String, Value>, key: &str| -> HashSet<String> {
            match root.get(key) {
                Some(Value::Object(m)) => m.keys().cloned().collect(),
                _ => HashSet::new(),
            }
        };
        let names = [own(&root, "$defs"), own(&root, "definitions")];
        // The root's own `$id` only names the schema; once nested it means
        // nothing. A nested one starts a new base for the refs below it.
        root.shift_remove("$schema");
        root.shift_remove("$id");
        let (mut self_ref, mut has_ref, mut nested_id) = (false, false, false);
        let mut root_value = Value::Object(std::mem::take(&mut root));
        walk_schemas(&mut root_value, 0, &mut |obj| {
            obj.shift_remove("$schema");
            nested_id |= obj.shift_remove("$id").is_some();
            if let Some(Value::String(r)) = obj.get_mut("$ref") {
                has_ref = true;
                *r = rewrite_ref(r, prefix, &names, &mut self_ref)?;
            }
            Ok(())
        })?;
        if has_ref && nested_id {
            return Err(UnenforcedReason::UnsupportedRef);
        }
        let Value::Object(mut root) = root_value else {
            unreachable!("an object stays an object");
        };
        for (key, into) in [
            ("$defs", &mut self.defs),
            ("definitions", &mut self.definitions),
        ] {
            if !matches!(root.get(key), Some(Value::Object(_))) {
                continue;
            }
            if let Some(Value::Object(defs)) = root.shift_remove(key) {
                for (name, def) in defs {
                    into.insert(format!("{prefix}{name}"), def);
                }
            }
        }
        if !self_ref {
            return Ok(Value::Object(root));
        }
        let name = format!("{prefix}root");
        if self.defs.contains_key(&name) {
            // The schema has its own definition named `root`.
            return Err(UnenforcedReason::UnsupportedRef);
        }
        self.defs.insert(name.clone(), Value::Object(root));
        Ok(json!({"$ref": format!("#/$defs/{name}")}))
    }
}

/// `r` as it reads once its schema is nested and its definitions are moved
/// under `prefix`. `names` are the schema's own `$defs` and `definitions`
/// names; a reference to anything else can't be kept.
fn rewrite_ref(
    r: &str,
    prefix: &str,
    names: &[HashSet<String>; 2],
    self_ref: &mut bool,
) -> Result<String, UnenforcedReason> {
    if r == "#" {
        *self_ref = true;
        return Ok(format!("#/$defs/{prefix}root"));
    }
    for (keyword, names) in ["$defs", "definitions"].iter().zip(names) {
        let Some(rest) = r.strip_prefix(&format!("#/{keyword}/")) else {
            continue;
        };
        let token = rest.split('/').next().unwrap_or_default();
        let name = token.replace("~1", "/").replace("~0", "~");
        if names.contains(&name) {
            return Ok(format!("#/{keyword}/{prefix}{rest}"));
        }
    }
    Err(UnenforcedReason::UnsupportedRef)
}

/// Calls `f` on every schema object in `v`, as the schema checks walk it:
/// values of schema maps (never their names) and every other key's value,
/// except instance data. A schema nested past the depth limit is refused
/// rather than left partly rewritten.
fn walk_schemas(
    v: &mut Value,
    depth: usize,
    f: &mut impl FnMut(&mut Map<String, Value>) -> Result<(), UnenforcedReason>,
) -> Result<(), UnenforcedReason> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(UnenforcedReason::TooComplex);
    }
    match v {
        Value::Array(items) => {
            for item in items {
                walk_schemas(item, depth + 1, f)?;
            }
        }
        Value::Object(obj) => {
            f(obj)?;
            for (key, value) in obj.iter_mut() {
                let key = key.as_str();
                if DATA_KEYS.contains(&key) {
                    continue;
                }
                match value {
                    Value::Object(map) if SCHEMA_MAPS.contains(&key) => {
                        for sub in map.values_mut() {
                            walk_schemas(sub, depth + 1, f)?;
                        }
                    }
                    _ => walk_schemas(value, depth + 1, f)?,
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// The tool calls in a forced turn's answer, when it is exactly the shape
/// the call grammar produces: the whole text (inside at most one ```` ``` ````
/// or ```` ```json ```` fence) a non-empty JSON array whose every item is an
/// object with exactly the keys `name`, one of the effective tools, and
/// `parameters`, an object, which become the call's arguments. Anything else is `None`,
/// left for the model's own tool-call dialect.
pub fn calls_from_forced_output(text: &str, policy: &ToolPolicy) -> Option<Vec<ToolCall>> {
    let text = text.trim();
    let text = text
        .strip_prefix("```")
        .and_then(|t| t.strip_suffix("```"))
        .map(|t| t.strip_prefix("json").unwrap_or(t))
        .unwrap_or(text);
    let Value::Array(items) = serde_json::from_str(text.trim()).ok()? else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    let names: HashSet<&str> = policy
        .tools
        .iter()
        .filter_map(|t| t.function.get("name").and_then(Value::as_str))
        .collect();
    items
        .iter()
        .map(|item| {
            let obj = item.as_object().filter(|o| o.len() == 2)?;
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| names.contains(n))?;
            // Anything but an object (a string of JSON, say) is left to the
            // dialect parser, which reads it as today.
            let parameters = obj.get("parameters").filter(|p| p.is_object())?;
            let arguments = serde_json::to_string(parameters).ok()?;
            Some(ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                r#type: "function".into(),
                function: FunctionCall {
                    name: name.to_owned(),
                    arguments,
                },
            })
        })
        .collect()
}

/// The tool calls in an answer, read the way the turn was constrained.
///
/// When a tool grammar was sent (`grammar_sent`), the call array it
/// produces is read first ([`calls_from_forced_output`]); the engine may
/// have dropped the grammar, so the model's own tool-call dialect is read
/// next. Without a grammar the answer is read by the dialect parser alone,
/// as before tool grammars existed. Under a union grammar
/// (`union_sent`: a call or the client's JSON) only marked calls count
/// ([`crate::parse_tool_calls_marked`]): a bare JSON object with `name` and
/// `parameters` keys is the client's data there, not a call.
///
/// Calls are kept only for the effective tools when `allowed_tools`
/// narrowed them or a grammar was sent (`grammar_sent`); otherwise any name
/// stays, as before. A named `tool_choice` keeps only calls to that tool
/// (another is possible when the engine dropped the grammar).
/// `parallel_tool_calls: false` keeps the first call.
/// `None` when no call is left.
pub fn resolve_calls(
    text: &str,
    policy: &ToolPolicy,
    grammar_sent: bool,
    union_sent: bool,
) -> Option<Vec<ToolCall>> {
    let forced = grammar_sent
        .then(|| calls_from_forced_output(text, policy))
        .flatten();
    let mut calls = forced.or_else(|| {
        if union_sent {
            crate::parse_tool_calls_marked(text)
        } else {
            crate::parse_tool_calls(text)
        }
    })?;
    if policy.narrowed || grammar_sent {
        let names: HashSet<&str> = policy
            .tools
            .iter()
            .filter_map(|t| t.function.get("name").and_then(Value::as_str))
            .collect();
        calls.retain(|c| names.contains(c.function.name.as_str()));
    }
    if let ToolMode::Named(name) = &policy.mode {
        calls.retain(|c| &c.function.name == name);
    }
    if policy.single_call {
        calls.truncate(1);
    }
    (!calls.is_empty()).then_some(calls)
}

/// One tool call in a chat completion's answer.
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum OutputToolCall {
    /// `{"id","type":"function","function":{"name","arguments"}}`.
    Function(ToolCall),
    /// `{"id","type":"custom","custom":{"name","input"}}`.
    Custom {
        id: String,
        r#type: &'static str,
        custom: CustomCall,
    },
}

/// The `custom` object of a custom tool call.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct CustomCall {
    pub name: String,
    pub input: String,
}

impl OutputToolCall {
    /// The called tool's name.
    pub fn name(&self) -> &str {
        match self {
            OutputToolCall::Function(call) => &call.function.name,
            OutputToolCall::Custom { custom, .. } => &custom.name,
        }
    }
}

/// The answer's calls in their output shape: a call to a custom tool
/// becomes `{"id","type":"custom","custom":{"name","input"}}`, every other
/// call stays a function call.
///
/// A custom tool's `input` is the `input` string of the call's arguments;
/// else the arguments object's only string field (a model that renamed the
/// parameter); else the arguments text as written.
pub fn output_calls(calls: Vec<ToolCall>, policy: &ToolPolicy) -> Vec<OutputToolCall> {
    calls
        .into_iter()
        .map(|call| {
            if !policy.custom_names.contains(&call.function.name) {
                return OutputToolCall::Function(call);
            }
            let input = custom_input(&call.function.arguments)
                .unwrap_or_else(|| call.function.arguments.clone());
            OutputToolCall::Custom {
                id: call.id,
                r#type: "custom",
                custom: CustomCall {
                    name: call.function.name,
                    input,
                },
            }
        })
        .collect()
}

/// The `input` a custom call's arguments carry, when one can be found.
fn custom_input(arguments: &str) -> Option<String> {
    let Ok(Value::Object(args)) = serde_json::from_str::<Value>(arguments) else {
        return None;
    };
    if let Some(Value::String(input)) = args.get("input") {
        return Some(input.clone());
    }
    let mut strings = args.values().filter_map(Value::as_str);
    match (strings.next(), strings.next()) {
        (Some(only), None) => Some(only.to_owned()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(v: Value) -> Vec<RequestTool> {
        serde_json::from_value(v).unwrap()
    }

    fn func(name: &str) -> Value {
        json!({"type": "function", "function": {"name": name}})
    }

    fn policy(t: &Value, choice: Option<Value>) -> Result<ToolPolicy, String> {
        let t = tools(t.clone());
        ToolPolicy::from_request(Some(&t), choice.as_ref(), None)
    }

    fn names(p: &ToolPolicy) -> Vec<String> {
        p.tools
            .iter()
            .map(|t| t.function["name"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn absent_or_auto_with_tools_keeps_tools_unchanged() {
        let raw = json!([
            {"type": "function", "function": {"name": "get_weather", "description": "d",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "b", "parameters": {"z": 1.5, "a": [2, 3]}}}
        ]);
        for choice in [None, Some(json!("auto"))] {
            let p = policy(&raw, choice.clone()).unwrap();
            assert_eq!(
                serde_json::to_string(&p.engine_tools().unwrap()).unwrap(),
                serde_json::to_string(&raw).unwrap()
            );
            assert_eq!(
                serde_json::to_string(p.prompt_tools().unwrap()).unwrap(),
                serde_json::to_string(&raw).unwrap()
            );
            assert_eq!(p.mode, ToolMode::Auto);
            assert!(p.callable() && !p.forced() && !p.narrowed && !p.single_call);
            assert_eq!(p.engine_tool_choice(), choice);
        }
        let p = ToolPolicy::from_request(None, None, None).unwrap();
        assert!(p.engine_tools().is_none() && p.prompt_tools().is_none());
        assert!(p.engine_tool_choice().is_none() && !p.callable());
        let p = policy(&json!([]), None).unwrap();
        assert_eq!(p.engine_tools(), Some(json!([])));
        assert!(!p.callable());
    }

    #[test]
    fn none_is_not_callable() {
        let p = policy(&json!([func("a")]), Some(json!("none"))).unwrap();
        assert_eq!(p.mode, ToolMode::None);
        assert!(!p.callable() && !p.forced());
        assert_eq!(names(&p), ["a"]);
    }

    #[test]
    fn required_is_forced() {
        let p = policy(&json!([func("a")]), Some(json!("required"))).unwrap();
        assert_eq!(p.mode, ToolMode::Required);
        assert!(p.callable() && p.forced());
    }

    #[test]
    fn required_without_tools_is_ok_and_not_callable() {
        let p = ToolPolicy::from_request(None, Some(&json!("required")), None).unwrap();
        assert!(!p.callable() && !p.forced());
        let p = policy(&json!([]), Some(json!("required"))).unwrap();
        assert!(!p.callable() && !p.forced());
    }

    #[test]
    fn named_function_is_forced_and_checked() {
        let t = json!([func("a"), func("b")]);
        let p = policy(
            &t,
            Some(json!({"type": "function", "function": {"name": "b"}})),
        )
        .unwrap();
        assert_eq!(p.mode, ToolMode::Named("b".into()));
        assert!(p.forced());
        assert_eq!(names(&p), ["a", "b"], "a named choice doesn't narrow");
        let e = policy(
            &t,
            Some(json!({"type": "function", "function": {"name": "c"}})),
        )
        .unwrap_err();
        assert_eq!(e, "tool_choice names function 'c', which is not in tools");
        let e = policy(&t, Some(json!({"type": "function"}))).unwrap_err();
        assert!(e.starts_with("tool_choice must be"), "{e}");
        let e = policy(&t, Some(json!("sometimes"))).unwrap_err();
        assert_eq!(
            e,
            "tool_choice 'sometimes' is not one of \"auto\", \"none\", \"required\""
        );
        let e = policy(&t, Some(json!({"type": "tool"}))).unwrap_err();
        assert_eq!(
            e,
            "tool_choice type 'tool' is not one of \"function\", \"custom\", \"allowed_tools\""
        );
        // A custom tool's name doesn't satisfy a named function choice.
        let t = json!([{"type": "custom", "custom": {"name": "c"}}]);
        let e = policy(
            &t,
            Some(json!({"type": "function", "function": {"name": "c"}})),
        )
        .unwrap_err();
        assert_eq!(
            e,
            "tool_choice names function 'c', but 'c' is a custom tool"
        );
    }

    fn allowed(mode: &str, list: Value) -> Value {
        json!({"type": "allowed_tools", "allowed_tools": {"mode": mode, "tools": list}})
    }

    #[test]
    fn allowed_tools_narrows_in_client_order_and_sets_mode() {
        let t = json!([func("a"), func("b"), func("c")]);
        let list = json!([func("c"), func("a")]);
        let p = policy(&t, Some(allowed("required", list.clone()))).unwrap();
        assert_eq!(names(&p), ["a", "c"]);
        assert_eq!(p.mode, ToolMode::Required);
        assert!(p.narrowed && p.forced());
        assert_eq!(p.engine_tool_choice(), Some(json!("required")));
        let p = policy(&t, Some(allowed("auto", list))).unwrap();
        assert_eq!(names(&p), ["a", "c"]);
        assert_eq!(p.mode, ToolMode::Auto);
        assert_eq!(p.engine_tool_choice(), Some(json!("auto")));
        assert_eq!(
            p.engine_tools(),
            Some(json!([func("a"), func("c")])),
            "the engine sees the narrowed list"
        );
    }

    #[test]
    fn allowed_tools_lists_a_custom_tool() {
        let t = json!([func("a"), {"type": "custom", "custom": {"name": "patch"}}]);
        let list = json!([{"type": "custom", "custom": {"name": "patch"}}]);
        let p = policy(&t, Some(allowed("required", list))).unwrap();
        assert_eq!(names(&p), ["patch"]);
        assert_eq!(p.custom_names, BTreeSet::from(["patch".to_string()]));
    }

    #[test]
    fn allowed_tools_errors() {
        let t = json!([func("a"), {"type": "custom", "custom": {"name": "patch"}}]);
        for (choice, needle) in [
            (allowed("auto", json!([func("zz")])), "'zz'"),
            (allowed("auto", json!([])), "non-empty"),
            (allowed("sometimes", json!([func("a")])), "mode"),
            (
                json!({"type": "allowed_tools"}),
                "tool_choice.allowed_tools",
            ),
            (
                // a custom tool listed as a function
                allowed("auto", json!([func("patch")])),
                "'patch'",
            ),
            (allowed("auto", json!([{"type": "x"}])), "tools[0]"),
        ] {
            let e = policy(&t, Some(choice.clone())).unwrap_err();
            assert!(
                e.contains("tool_choice.allowed_tools") && e.contains(needle),
                "{choice}: {e}"
            );
        }
        // No tools at all: nothing can be allowed.
        let e = ToolPolicy::from_request(None, Some(&allowed("auto", json!([func("a")]))), None)
            .unwrap_err();
        assert!(e.contains("tool_choice.allowed_tools"), "{e}");
    }

    #[test]
    fn custom_tool_becomes_the_exact_shim() {
        let t = json!([
            {"type": "custom", "custom": {"name": "plain"}},
            {"type": "custom", "custom": {"name": "sql", "description": "Run SQL.",
                "format": {"type": "grammar", "grammar": {"syntax": "lark", "definition": "start: \"SELECT\""}}}},
            {"type": "custom", "custom": {"name": "txt", "description": "Text.", "format": {"type": "text"}}}
        ]);
        let p = policy(&t, None).unwrap();
        let params = json!({"type": "object", "properties": {"input": {"type": "string"}},
            "required": ["input"], "additionalProperties": false});
        let want = json!([
            {"type": "function", "function": {"name": "plain", "description": "", "parameters": params}},
            {"type": "function", "function": {"name": "sql",
                "description": "Run SQL.\n\nThe input must follow this lark grammar:\nstart: \"SELECT\"",
                "parameters": params}},
            {"type": "function", "function": {"name": "txt", "description": "Text.", "parameters": params}}
        ]);
        assert_eq!(
            serde_json::to_string(&p.engine_tools().unwrap()).unwrap(),
            serde_json::to_string(&want).unwrap()
        );
        assert_eq!(p.custom_names.len(), 3);
    }

    #[test]
    fn named_custom_choice_is_forced_and_checked() {
        let t = json!([func("a"), {"type": "custom", "custom": {"name": "patch"}}]);
        let p = policy(
            &t,
            Some(json!({"type": "custom", "custom": {"name": "patch"}})),
        )
        .unwrap();
        assert_eq!(p.mode, ToolMode::Named("patch".into()));
        assert!(p.forced());
        assert_eq!(
            p.engine_tool_choice(),
            Some(json!({"type": "function", "function": {"name": "patch"}}))
        );
        let e = policy(&t, Some(json!({"type": "custom", "custom": {"name": "a"}}))).unwrap_err();
        assert_eq!(
            e,
            "tool_choice names custom tool 'a', but 'a' is a function tool"
        );
        let e = policy(
            &t,
            Some(json!({"type": "custom", "custom": {"name": "zz"}})),
        )
        .unwrap_err();
        assert_eq!(
            e,
            "tool_choice names custom tool 'zz', which is not in tools"
        );
        let e = policy(&t, Some(json!({"type": "custom"}))).unwrap_err();
        assert!(e.starts_with("tool_choice must be"), "{e}");
    }

    #[test]
    fn custom_name_clash_is_rejected() {
        for t in [
            json!([func("x"), {"type": "custom", "custom": {"name": "x"}}]),
            json!([{"type": "custom", "custom": {"name": "x"}}, func("x")]),
            json!([{"type": "custom", "custom": {"name": "x"}}, {"type": "custom", "custom": {"name": "x"}}]),
            json!([{"type": "custom", "custom": {"name": ""}}]),
            json!([{"type": "custom", "custom": {}}]),
            json!([{"type": "function_x", "custom": {"name": "y"}}]),
        ] {
            let e = policy(&t, None).unwrap_err();
            assert!(e.starts_with("tools"), "{t}: {e}");
        }
    }

    #[test]
    fn duplicate_function_names_still_accepted() {
        let t = json!([func("x"), func("x"), {"type": "function", "function": {}}]);
        let p = policy(
            &t,
            Some(json!({"type": "function", "function": {"name": "x"}})),
        )
        .unwrap();
        assert_eq!(p.tools.len(), 3);
        // A tool whose `type` isn't "function" but carries a function object
        // counts as a function tool, as it always has.
        let t = json!([{"type": "other", "function": {"name": "y"}}]);
        let p = policy(
            &t,
            Some(json!({"type": "function", "function": {"name": "y"}})),
        )
        .unwrap();
        assert!(p.forced());
    }

    #[test]
    fn parallel_false_only_when_explicit() {
        let t = tools(json!([func("a")]));
        for (v, want) in [(None, false), (Some(true), false), (Some(false), true)] {
            let p = ToolPolicy::from_request(Some(&t), None, v).unwrap();
            assert_eq!(p.single_call, want, "{v:?}");
        }
    }

    #[test]
    fn engine_tool_choice_maps_each_mode() {
        let t = json!([func("a")]);
        for (choice, want) in [
            (json!("none"), json!("none")),
            (json!("auto"), json!("auto")),
            (json!("required"), json!("required")),
            (
                json!({"type": "function", "function": {"name": "a"}}),
                json!({"type": "function", "function": {"name": "a"}}),
            ),
            (allowed("required", json!([func("a")])), json!("required")),
        ] {
            let p = policy(&t, Some(choice.clone())).unwrap();
            assert_eq!(p.engine_tool_choice(), Some(want), "{choice}");
        }
        assert_eq!(policy(&t, None).unwrap().engine_tool_choice(), None);
    }

    #[test]
    fn allowed_tools_says_when_a_name_is_the_other_kind() {
        let t = json!([func("a"), {"type": "custom", "custom": {"name": "patch"}}]);
        let e = policy(&t, Some(allowed("auto", json!([func("patch")])))).unwrap_err();
        assert_eq!(
            e,
            "tool_choice.allowed_tools lists function 'patch', but 'patch' is a custom tool"
        );
        let e = policy(
            &t,
            Some(allowed(
                "auto",
                json!([{"type": "custom", "custom": {"name": "a"}}]),
            )),
        )
        .unwrap_err();
        assert_eq!(
            e,
            "tool_choice.allowed_tools lists custom tool 'a', but 'a' is a function tool"
        );
    }

    #[test]
    fn malformed_tool_entries_name_the_missing_field() {
        let err = |v: Value| {
            serde_json::from_value::<RequestTool>(v)
                .unwrap_err()
                .to_string()
        };
        assert!(err(json!({"type": "function"})).contains("missing field `function`"));
        assert!(err(json!({"type": "custom"})).contains("missing field `custom`"));
        assert!(err(json!({"custom": {"name": "x"}})).contains("missing field `type`"));
        assert!(err(json!([1])).contains("expected"), "{}", err(json!([1])));
        // What parsed before still parses the same way.
        let tool = |v: Value| serde_json::from_value::<RequestTool>(v).unwrap();
        assert!(matches!(
            tool(json!({"type": "custom", "function": {"name": "f"}})),
            RequestTool::Function { .. }
        ));
        assert!(matches!(
            tool(json!({"type": "other", "function": {"name": "f"}})),
            RequestTool::Function { .. }
        ));
        assert!(matches!(
            tool(json!({"type": "function_x", "custom": {"name": "f"}})),
            RequestTool::Custom { .. }
        ));
    }

    #[test]
    fn malformed_tool_calls_name_the_missing_field() {
        #[derive(Deserialize)]
        struct Calls {
            #[serde(deserialize_with = "de_tool_calls")]
            tool_calls: Option<Vec<ToolCall>>,
        }
        let err = |call: Value| {
            serde_json::from_value::<Calls>(json!({"tool_calls": [call]}))
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default()
        };
        let e = err(json!({"type": "function", "function": {"name": "f", "arguments": "{}"}}));
        assert!(e.contains("missing field `id`"), "{e}");
        let e = err(json!({"type": "custom", "custom": {"name": "f", "input": ""}}));
        assert!(e.contains("missing field `id`"), "{e}");
        let e = err(json!({"id": "c", "type": "custom", "custom": {"name": "f"}}));
        assert!(e.contains("missing field `input`"), "{e}");
        let e = err(json!({"id": "c", "type": "function"}));
        assert!(e.contains("missing field `function`"), "{e}");
        // A function key that doesn't parse, beside a custom object: the
        // custom replay, as the untagged parse read it.
        let replay = serde_json::from_value::<Calls>(json!({"tool_calls": [
            {"id": "c", "type": "custom", "function": "x", "custom": {"name": "f", "input": ""}}]}))
        .unwrap()
        .tool_calls
        .unwrap();
        assert_eq!(replay[0].r#type, "function");
        assert_eq!(replay[0].function.name, "f");
        assert_eq!(replay[0].function.arguments, r#"{"input":""}"#);
        let ok = serde_json::from_value::<Calls>(json!({"tool_calls": [
            {"id": "c", "type": "custom", "function": {"name": "f", "arguments": "{}"}}]}))
        .unwrap();
        assert_eq!(
            ok.tool_calls.unwrap()[0].r#type,
            "custom",
            "a function object wins, as before"
        );
    }

    #[test]
    fn many_tools_validate_in_linear_time() {
        let n = 20_000;
        let mut list: Vec<Value> = (0..n).map(|i| func(&format!("f{i}"))).collect();
        list.push(json!({"type": "custom", "custom": {"name": "c"}}));
        let t = tools(Value::Array(list));
        let allowed_list: Vec<Value> = (0..n).rev().map(|i| func(&format!("f{i}"))).collect();
        let choice = allowed("required", Value::Array(allowed_list));
        let start = std::time::Instant::now();
        let p = ToolPolicy::from_request(Some(&t), Some(&choice), None).unwrap();
        let elapsed = start.elapsed();
        assert_eq!(p.tools.len(), n);
        // Quadratic lookups take tens of seconds here; one name map, well
        // under one even in a debug build on a loaded runner.
        assert!(elapsed < std::time::Duration::from_secs(2), "{elapsed:?}");
    }

    // --- output constraints -------------------------------------------

    use crate::structured::{JSON_OBJECT_SCHEMA, MAX_SCHEMA_BYTES};

    fn unenforced(reason: UnenforcedReason) -> u64 {
        cascadia_metrics::CONSTRAINT_UNENFORCED_TOTAL
            .with_label_values(&[reason.as_str()])
            .get()
    }

    fn weather() -> Value {
        json!({"type": "function", "function": {"name": "get_weather", "description": "Weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                "required": ["city"]}}})
    }

    fn with_params(name: &str, parameters: Value) -> Value {
        json!({"type": "function", "function": {"name": name, "parameters": parameters}})
    }

    fn pol(t: Value, choice: Option<Value>, parallel: Option<bool>) -> ToolPolicy {
        let t = tools(t);
        ToolPolicy::from_request(Some(&t), choice.as_ref(), parallel).unwrap()
    }

    fn item(name: &str, parameters: Value) -> Value {
        json!({"type": "object",
            "properties": {"name": {"type": "string", "enum": [name]}, "parameters": parameters},
            "required": ["name", "parameters"], "additionalProperties": false})
    }

    fn no_params() -> Value {
        json!({"type": "object", "properties": {}})
    }

    fn weather_params() -> Value {
        weather()["function"]["parameters"].clone()
    }

    /// The grammar body, parsed.
    fn body(spec: &GrammarSpec) -> Value {
        serde_json::from_str(&spec.body).unwrap()
    }

    #[test]
    fn required_schema_matches_the_vllm_shape() {
        let p = pol(
            json!([weather(), func("get_time")]),
            Some(json!("required")),
            None,
        );
        let spec = tool_grammar(&p, None).expect("a grammar");
        let want = json!({"type": "array", "minItems": 1, "items": {"anyOf": [
            item("get_weather", weather_params()), item("get_time", no_params())]}});
        assert_eq!(spec.body, serde_json::to_string(&want).unwrap());
        assert!(spec.best_effort);
        assert_eq!(spec.kind, cascadia_types::GrammarKind::JsonSchema);
        // Not forced and no format: nothing to enforce.
        assert!(tool_grammar(&pol(json!([weather()]), None, None), None).is_none());
        assert!(tool_grammar(&pol(json!([weather()]), Some(json!("auto")), None), None).is_none());
        let none = pol(json!([weather()]), Some(json!("none")), None);
        assert!(tool_grammar(&none, None).is_none());
        assert!(tool_grammar(&none, Some(JSON_OBJECT_SCHEMA)).is_none());
        let no_tools = ToolPolicy::from_request(None, Some(&json!("required")), None).unwrap();
        assert!(tool_grammar(&no_tools, Some(JSON_OBJECT_SCHEMA)).is_none());
    }

    #[test]
    fn named_schema_has_one_item_and_max_one() {
        let named = json!({"type": "function", "function": {"name": "get_time"}});
        let p = pol(json!([weather(), func("get_time")]), Some(named), None);
        let spec = tool_grammar(&p, Some(JSON_OBJECT_SCHEMA)).unwrap();
        let want = json!({"type": "array", "minItems": 1, "maxItems": 1,
            "items": {"anyOf": [item("get_time", no_params())]}});
        assert_eq!(
            spec.body,
            serde_json::to_string(&want).unwrap(),
            "the format is ignored"
        );
        // A named custom tool: its shim's parameters.
        let t = json!([weather(), {"type": "custom", "custom": {"name": "patch"}}]);
        let p = pol(
            t,
            Some(json!({"type": "custom", "custom": {"name": "patch"}})),
            None,
        );
        let spec = tool_grammar(&p, None).unwrap();
        assert_eq!(
            body(&spec)["items"]["anyOf"],
            json!([item(
                "patch",
                json!({"type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"], "additionalProperties": false})
            )])
        );
    }

    #[test]
    fn single_call_sets_max_items() {
        let t = json!([weather(), func("get_time")]);
        let p = pol(t.clone(), Some(json!("required")), Some(false));
        let b = body(&tool_grammar(&p, None).unwrap());
        assert_eq!(b["maxItems"], 1);
        assert_eq!(b["items"]["anyOf"].as_array().unwrap().len(), 2);
        let p = pol(t.clone(), Some(json!("required")), Some(true));
        assert!(body(&tool_grammar(&p, None).unwrap())
            .get("maxItems")
            .is_none());
        let p = pol(t, None, Some(false));
        let b = body(&tool_grammar(&p, Some(JSON_OBJECT_SCHEMA)).unwrap());
        assert_eq!(b["anyOf"][0]["maxItems"], 1);
    }

    #[test]
    fn union_wraps_the_format_schema() {
        let format =
            r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#;
        let t = json!([weather(), func("get_time")]);
        let calls = json!({"type": "array", "minItems": 1, "items": {"anyOf": [
            item("get_weather", weather_params()), item("get_time", no_params())]}});
        let want = format!(
            r#"{{"anyOf":[{},{format}]}}"#,
            serde_json::to_string(&calls).unwrap()
        );
        for choice in [None, Some(json!("auto"))] {
            let spec = tool_grammar(&pol(t.clone(), choice, None), Some(format)).unwrap();
            assert_eq!(spec.body, want);
            assert!(spec.best_effort);
        }
        // allowed_tools in auto mode: the union over the allowed tools only.
        let choice = json!({"type": "allowed_tools", "allowed_tools": {"mode": "auto",
            "tools": [func("get_time")]}});
        let spec = tool_grammar(&pol(t, Some(choice), None), Some(format)).unwrap();
        assert_eq!(
            body(&spec)["anyOf"][0]["items"]["anyOf"],
            json!([item("get_time", no_params())])
        );
    }

    #[test]
    fn json_object_format_unions_with_the_object_schema() {
        let spec = tool_grammar(
            &pol(json!([weather()]), None, None),
            Some(JSON_OBJECT_SCHEMA),
        )
        .unwrap();
        let b = body(&spec);
        assert_eq!(
            b["anyOf"][1],
            serde_json::from_str::<Value>(JSON_OBJECT_SCHEMA).unwrap()
        );
        assert_eq!(
            b["anyOf"][0]["items"]["anyOf"],
            json!([item("get_weather", weather_params())])
        );
    }

    #[test]
    fn hoists_defs_with_prefixes_and_rewrites_refs() {
        let addr = json!({"type": "object", "properties": {"street": {"type": "string"}}});
        let other = json!({"type": "integer"});
        let t = json!([
            with_params(
                "a",
                json!({"type": "object",
                "properties": {"home": {"$ref": "#/$defs/Addr"},
                    "street": {"$ref": "#/$defs/Addr/properties/street"}},
                "$defs": {"Addr": addr}})
            ),
            with_params(
                "b",
                json!({"type": "object",
                "properties": {"n": {"$ref": "#/definitions/Addr"}},
                "definitions": {"Addr": other}})
            )
        ]);
        let format = r##"{"type":"object","properties":{"x":{"$ref":"#/$defs/Addr"}},"$defs":{"Addr":{"type":"boolean"}}}"##;
        let b = body(&tool_grammar(&pol(t, None, None), Some(format)).unwrap());
        assert_eq!(
            b["$defs"],
            json!({"t0_Addr": addr, "rf_Addr": {"type": "boolean"}})
        );
        assert_eq!(b["definitions"], json!({"t1_Addr": other}));
        let items = &b["anyOf"][0]["items"]["anyOf"];
        assert_eq!(
            items[0]["properties"]["parameters"],
            json!({"type": "object", "properties": {"home": {"$ref": "#/$defs/t0_Addr"},
                "street": {"$ref": "#/$defs/t0_Addr/properties/street"}}})
        );
        assert_eq!(
            items[1]["properties"]["parameters"],
            json!({"type": "object", "properties": {"n": {"$ref": "#/definitions/t1_Addr"}}})
        );
        assert_eq!(
            b["anyOf"][1],
            json!({"type": "object", "properties": {"x": {"$ref": "#/$defs/rf_Addr"}}})
        );
        // A ref inside a hoisted definition is rewritten too; data is left alone.
        let t = json!([with_params(
            "a",
            json!({"$ref": "#/$defs/List",
            "$defs": {"List": {"type": "array", "items": {"$ref": "#/$defs/Item"},
                "default": [{"$ref": "#/$defs/Item"}]},
                "Item": {"type": "string"}}})
        )]);
        let b = body(&tool_grammar(&pol(t, Some(json!("required")), None), None).unwrap());
        assert_eq!(
            b["$defs"],
            json!({"t0_List": {"type": "array", "items": {"$ref": "#/$defs/t0_Item"},
                "default": [{"$ref": "#/$defs/Item"}]}, "t0_Item": {"type": "string"}})
        );
        assert_eq!(
            b["items"]["anyOf"][0]["properties"]["parameters"],
            json!({"$ref": "#/$defs/t0_List"})
        );
    }

    #[test]
    fn root_self_ref_is_hoisted() {
        let tree = json!({"type": "object", "properties": {"name": {"type": "string"},
            "children": {"type": "array", "items": {"$ref": "#"}}}});
        let t = json!([func("x"), with_params("tree", tree)]);
        let format = r##"{"type":"object","properties":{"next":{"$ref":"#"}}}"##;
        let b = body(&tool_grammar(&pol(t, None, None), Some(format)).unwrap());
        assert_eq!(
            b["$defs"]["t1_root"],
            json!({"type": "object", "properties": {"name": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/t1_root"}}}})
        );
        assert_eq!(
            b["anyOf"][0]["items"]["anyOf"][1]["properties"]["parameters"],
            json!({"$ref": "#/$defs/t1_root"})
        );
        assert_eq!(
            b["$defs"]["rf_root"],
            json!({"type": "object", "properties": {"next": {"$ref": "#/$defs/rf_root"}}})
        );
        assert_eq!(b["anyOf"][1], json!({"$ref": "#/$defs/rf_root"}));
    }

    #[test]
    fn a_bound_pair_no_value_can_meet_is_dropped_from_the_tools() {
        let cases = [
            (
                json!({"type": "object", "properties": {
                    "city": {"type": "string", "minLength": 5, "maxLength": 2, "pattern": "^A"}}}),
                json!({"type": "object", "properties": {
                    "city": {"type": "string", "pattern": "^A"}}}),
            ),
            // An integer range that holds numbers but no integer.
            (
                json!({"type": "object", "properties": {
                    "n": {"type": "integer", "exclusiveMinimum": 2, "exclusiveMaximum": 3}}}),
                json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
            ),
        ];
        for (params, want) in cases {
            let p = pol(
                json!([weather(), with_params("f", params.clone())]),
                Some(json!("required")),
                None,
            );
            // Every path out of the policy carries the relaxed copy.
            assert_eq!(p.tools[0].function, weather()["function"], "{params}");
            assert_eq!(p.tools[1].function["parameters"], want, "{params}");
            assert_eq!(p.engine_tools().unwrap()[1]["function"]["parameters"], want);
            assert_eq!(p.prompt_tools().unwrap()[1].function["parameters"], want);
            // The call stays forced, over the relaxed schema.
            let g = tool_grammar(&p, None).expect("a call grammar");
            let b: Value = serde_json::from_str(&g.body).unwrap();
            assert_eq!(b["items"]["anyOf"][1]["properties"]["parameters"], want);
            assert_eq!(p.relaxed, BTreeSet::from([1]));
            assert!(p.grammar_relaxed());
            // A named choice leaving the relaxed tool out enforces all it sends.
            let named = json!({"type": "function", "function": {"name": "get_weather"}});
            let p = pol(
                json!([weather(), with_params("f", params.clone())]),
                Some(named),
                None,
            );
            assert!(!p.grammar_relaxed());
            let p = pol(
                json!([weather(), with_params("f", params)]),
                Some(json!({"type": "function", "function": {"name": "f"}})),
                None,
            );
            assert!(p.grammar_relaxed());
        }
        // Bounds that some value meets: the tool exactly as sent.
        let fine = json!({"type": "object", "properties": {
            "city": {"type": "string", "minLength": 2, "maxLength": 5},
            "n": {"type": "integer", "exclusiveMinimum": 2, "exclusiveMaximum": 4}}});
        let p = pol(
            json!([with_params("f", fine.clone())]),
            Some(json!("required")),
            None,
        );
        assert_eq!(p.tools[0].function["parameters"], fine);
        assert!(p.relaxed.is_empty() && !p.grammar_relaxed());
    }

    #[test]
    fn unhoistable_ref_gives_no_grammar() {
        for r in [
            "#/properties/a",
            "https://example.com/s.json",
            "other.json#/$defs/A",
            "#/$defs/Missing",
            "#/$defs",
            "#A",
        ] {
            let params = json!({"type": "object",
                "properties": {"a": {"type": "string"}, "b": {"$ref": r}},
                "$defs": {"A": {"type": "string"}}});
            let before = unenforced(UnenforcedReason::UnsupportedRef);
            let p = pol(
                json!([with_params("f", params)]),
                Some(json!("required")),
                None,
            );
            assert!(tool_grammar(&p, None).is_none(), "{r}");
            assert!(unenforced(UnenforcedReason::UnsupportedRef) > before, "{r}");
        }
        // In the format schema too.
        let before = unenforced(UnenforcedReason::UnsupportedRef);
        let format = r##"{"type":"object","properties":{"a":{"type":"string"},"b":{"$ref":"#/properties/a"}}}"##;
        assert!(tool_grammar(&pol(json!([weather()]), None, None), Some(format)).is_none());
        assert!(unenforced(UnenforcedReason::UnsupportedRef) > before);
    }

    #[test]
    fn oversized_or_too_deep_union_gives_no_grammar() {
        // Each tool fits the byte limit alone; together they don't.
        let half = json!({"type": "object", "description": "x".repeat(MAX_SCHEMA_BYTES / 2)});
        assert!(check_schema(&half, &half.to_string()).is_ok());
        let before = unenforced(UnenforcedReason::TooComplex);
        let p = pol(
            json!([
                with_params("a", half.clone()),
                with_params("b", half.clone())
            ]),
            Some(json!("required")),
            None,
        );
        assert!(tool_grammar(&p, None).is_none());
        assert!(unenforced(UnenforcedReason::TooComplex) > before);
        // A tool and a format schema that only exceed it once combined.
        let before = unenforced(UnenforcedReason::TooComplex);
        let p = pol(json!([with_params("a", half.clone())]), None, None);
        assert!(tool_grammar(&p, Some(&half.to_string())).is_none());
        assert!(unenforced(UnenforcedReason::TooComplex) > before);
        // The deepest parameters schema the limits accept alone are too deep
        // once nested in the call array.
        let nest = |k: usize| {
            (0..k).fold(
                json!({"type": "string"}),
                |s, _| json!({"type": "array", "items": s}),
            )
        };
        let deepest = (1..200)
            .take_while(|&k| check_schema(&nest(k), &nest(k).to_string()).is_ok())
            .last()
            .unwrap();
        let before = unenforced(UnenforcedReason::TooComplex);
        let p = pol(
            json!([with_params("a", nest(deepest))]),
            Some(json!("required")),
            None,
        );
        assert!(tool_grammar(&p, None).is_none());
        assert!(unenforced(UnenforcedReason::TooComplex) > before);
    }

    fn validator(spec: &GrammarSpec) -> jsonschema::Validator {
        jsonschema::validator_for(&body(spec)).unwrap()
    }

    #[test]
    fn schemas_validate_their_examples() {
        let t = json!([weather(), func("get_time")]);
        let required =
            validator(&tool_grammar(&pol(t.clone(), Some(json!("required")), None), None).unwrap());
        let call = json!([{"name": "get_weather", "parameters": {"city": "Paris"}}]);
        assert!(required.is_valid(&call));
        assert!(required.is_valid(
            &json!([{"name": "get_weather", "parameters": {"city": "Paris"}},
            {"name": "get_time", "parameters": {}}])
        ));
        assert!(!required.is_valid(&json!([])));
        assert!(!required.is_valid(&json!([{"name": "get_stock", "parameters": {}}])));
        assert!(!required.is_valid(&json!([{"name": "get_weather", "parameters": {}}])));
        assert!(!required.is_valid(&json!([{"name": "get_time", "parameters": {}, "extra": 1}])));
        assert!(!required.is_valid(&json!({"name": "get_time", "parameters": {}})));
        let single = validator(
            &tool_grammar(&pol(t.clone(), Some(json!("required")), Some(false)), None).unwrap(),
        );
        assert!(single.is_valid(&call));
        assert!(
            !single.is_valid(&json!([{"name": "get_time", "parameters": {}},
            {"name": "get_time", "parameters": {}}]))
        );
        let named = json!({"type": "function", "function": {"name": "get_time"}});
        let named = validator(&tool_grammar(&pol(t.clone(), Some(named), None), None).unwrap());
        assert!(named.is_valid(&json!([{"name": "get_time", "parameters": {}}])));
        assert!(!named.is_valid(&call));
        let format =
            r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#;
        let union = validator(&tool_grammar(&pol(t, None, None), Some(format)).unwrap());
        assert!(union.is_valid(&call));
        assert!(union.is_valid(&json!({"answer": "42"})));
        assert!(!union.is_valid(&json!({"answer": 42})));
        assert!(!union.is_valid(&json!([])));
        // Hoisted definitions and a root self-reference resolve.
        let tree = json!({"type": "object", "properties": {"kids": {"type": "array", "items": {"$ref": "#"}},
            "addr": {"$ref": "#/$defs/Addr"}}, "$defs": {"Addr": {"type": "string"}}});
        let hoisted = validator(
            &tool_grammar(
                &pol(
                    json!([with_params("tree", tree)]),
                    Some(json!("required")),
                    None,
                ),
                None,
            )
            .unwrap(),
        );
        assert!(hoisted.is_valid(&json!([{"name": "tree", "parameters":
            {"addr": "a", "kids": [{"addr": "b", "kids": []}]}}])));
        assert!(!hoisted.is_valid(&json!([{"name": "tree", "parameters":
            {"kids": [{"addr": 1}]}}])));
    }

    #[test]
    fn schemas_compile_with_llguidance() {
        let t =
            json!([weather(), func("get_time"), {"type": "custom", "custom": {"name": "patch"}}]);
        let tree = json!({"type": "object", "properties": {"kids": {"type": "array", "items": {"$ref": "#"}},
            "addr": {"$ref": "#/$defs/Addr"}, "n": {"$ref": "#/definitions/N"}},
            "$defs": {"Addr": {"type": "string"}}, "definitions": {"N": {"type": "integer"}}});
        let named = json!({"type": "function", "function": {"name": "get_time"}});
        let format =
            r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#;
        let specs = [
            tool_grammar(&pol(t.clone(), Some(json!("required")), None), None),
            tool_grammar(&pol(t.clone(), Some(json!("required")), Some(false)), None),
            tool_grammar(&pol(t.clone(), Some(named), None), None),
            tool_grammar(&pol(t.clone(), None, None), Some(format)),
            tool_grammar(&pol(t, None, None), Some(JSON_OBJECT_SCHEMA)),
            tool_grammar(
                &pol(
                    json!([with_params("tree", tree)]),
                    Some(json!("required")),
                    None,
                ),
                None,
            ),
        ];
        for spec in specs {
            let spec = spec.expect("a grammar");
            if let Err(e) = cascadia_grammar::GrammarFactory::byte_level().compile(&spec) {
                panic!("{}: {e}", spec.body);
            }
        }
    }

    #[test]
    fn object_allof_in_tool_params_is_merged() {
        // zod's `.and()`: an allOf of plain object schemas.
        let params = json!({"allOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "number"}}, "required": ["b"],
                "additionalProperties": false}
        ]});
        let p = pol(
            json!([with_params("f", params)]),
            Some(json!("required")),
            None,
        );
        let spec = tool_grammar(&p, None).unwrap();
        assert!(!spec.body.contains("allOf"), "{}", spec.body);
        assert_eq!(
            body(&spec)["items"]["anyOf"][0]["properties"]["parameters"],
            json!({"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
                "required": ["a", "b"], "additionalProperties": false})
        );
    }

    #[test]
    fn nested_schema_and_id_keys_are_dropped() {
        let params = json!({"$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "https://example.com/p", "type": "object",
            "properties": {"$id": {"type": "string"}, "a": {"$ref": "#/$defs/A"}},
            "$defs": {"A": {"$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "string"}}});
        // A nested `$id` in a schema without references is dropped too.
        let format = r#"{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"x":{"$id":"x","type":"string"}}}"#;
        let p = pol(json!([with_params("f", params)]), None, None);
        let spec = tool_grammar(&p, Some(format)).unwrap();
        assert!(!spec.body.contains("$schema"), "{}", spec.body);
        let b = body(&spec);
        assert_eq!(
            b["anyOf"][0]["items"]["anyOf"][0]["properties"]["parameters"],
            json!({"type": "object", "properties": {"$id": {"type": "string"},
                "a": {"$ref": "#/$defs/t0_A"}}}),
            "a property named $id stays"
        );
        assert_eq!(b["$defs"]["t0_A"], json!({"type": "string"}));
        assert_eq!(
            b["anyOf"][1],
            json!({"type": "object", "properties": {"x": {"type": "string"}}})
        );
        cascadia_grammar::GrammarFactory::byte_level()
            .compile(&spec)
            .unwrap();
    }

    #[test]
    fn a_nested_id_beside_refs_gives_no_grammar() {
        // Refs under a nested `$id` resolve against it, not the root, so
        // they can't be rewritten safely.
        for params in [
            json!({"type": "object", "properties": {"a": {"$id": "https://example.com/a",
                "type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}}},
                "$defs": {"B": {"type": "string"}}}),
            json!({"type": "object", "properties": {"a": {"$ref": "#/$defs/B"}},
                "$defs": {"B": {"$id": "b", "type": "string"}}}),
        ] {
            let before = unenforced(UnenforcedReason::UnsupportedRef);
            let p = pol(
                json!([with_params("f", params.clone())]),
                Some(json!("required")),
                None,
            );
            assert!(tool_grammar(&p, None).is_none(), "{params}");
            assert!(
                unenforced(UnenforcedReason::UnsupportedRef) > before,
                "{params}"
            );
        }
    }

    #[test]
    fn non_object_parameters_are_left_to_the_dialect_parser() {
        let p = pol(json!([weather()]), Some(json!("required")), None);
        for params in [
            json!("{\"city\":\"Paris\"}"),
            Value::Null,
            json!(3),
            json!([1]),
        ] {
            let text = json!([{"name": "get_weather", "parameters": params}]).to_string();
            assert!(calls_from_forced_output(&text, &p).is_none(), "{text}");
        }
        // Read forced-shape first, then by dialect: a string `parameters`
        // comes out once-encoded.
        let text = r#"[{"name":"get_weather","parameters":"{\"city\":\"Paris\"}"}]"#;
        let calls = calls_from_forced_output(text, &p)
            .or_else(|| crate::parse_tool_calls(text))
            .expect("the dialect parser reads it");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn calls_from_forced_output_reads_only_the_exact_shape() {
        let p = pol(
            json!([weather(), {"type": "custom", "custom": {"name": "patch"}}]),
            Some(json!("required")),
            None,
        );
        let text = r#"[{"name":"get_weather","parameters":{"city":"Paris","unit":"c"}},
            {"name":"patch","parameters":{"input":"x"}}]"#;
        for wrapped in [
            text.to_string(),
            format!("  {text}\n"),
            format!("```json\n{text}\n```"),
            format!("```\n{text}\n```  "),
        ] {
            let calls = calls_from_forced_output(&wrapped, &p).expect(&wrapped);
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].r#type, "function");
            assert!(calls[0].id.starts_with("call_"), "{}", calls[0].id);
            assert_ne!(calls[0].id, calls[1].id);
            assert_eq!(calls[0].function.name, "get_weather");
            assert_eq!(
                calls[0].function.arguments,
                r#"{"city":"Paris","unit":"c"}"#
            );
            assert_eq!(calls[1].function.name, "patch");
            assert_eq!(calls[1].function.arguments, r#"{"input":"x"}"#);
        }
        for bad in [
            r#"{"name":"get_weather","parameters":{"city":"Paris"}}"#,
            r#"[{"name":"get_weather","parameters":{"city":"Paris"},"id":"x"}]"#,
            r#"[{"name":"get_weather"}]"#,
            r#"[{"name":"get_stock","parameters":{}}]"#,
            r#"[{"name":"get_weather","parameters":{"city":"Paris"}}] and more"#,
            r#"Sure: [{"name":"get_weather","parameters":{"city":"Paris"}}]"#,
            r#"[{"name":"get_weather","parameters":{"city":"Paris"}}, 3]"#,
            r#"[{"name":1,"parameters":{}}]"#,
            "```json\n```json\n[{\"name\":\"get_weather\",\"parameters\":{}}]\n```\n```",
            "[]",
            "",
        ] {
            assert!(calls_from_forced_output(bad, &p).is_none(), "{bad}");
        }
    }

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: "call_1".into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    #[test]
    fn resolve_calls_reads_the_forced_shape_first() {
        let p = policy(&json!([func("a"), func("b")]), Some(json!("required"))).unwrap();
        let calls = resolve_calls(r#"[{"name":"b","parameters":{"x":1}}]"#, &p, true, false)
            .expect("a call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "b");
        assert_eq!(calls[0].function.arguments, r#"{"x":1}"#);
        // Fenced, as some servers answer a JSON format.
        let calls = resolve_calls(
            "```json\n[{\"name\":\"a\",\"parameters\":{}}]\n```",
            &p,
            true,
            false,
        )
        .expect("a fenced call");
        assert_eq!(calls[0].function.name, "a");
    }

    #[test]
    fn resolve_calls_under_a_union_ignores_bare_json() {
        let p = policy(&json!([func("a")]), None).unwrap();
        assert_eq!(
            resolve_calls(r#"{"name":"a","parameters":{}}"#, &p, true, true),
            None
        );
        let calls = resolve_calls(
            r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
            &p,
            true,
            true,
        )
        .expect("a marked call");
        assert_eq!(calls[0].function.name, "a");
    }

    #[test]
    fn resolve_calls_filters_to_the_tools_once_a_grammar_applied() {
        let p = policy(&json!([func("a")]), Some(json!("required"))).unwrap();
        let text = r#"<tool_call>{"name":"zzz","arguments":{}}</tool_call>"#;
        assert_eq!(resolve_calls(text, &p, true, false), None);
        // Without a grammar (and without narrowing) the call stays, as today.
        assert_eq!(
            resolve_calls(text, &p, false, false).unwrap()[0]
                .function
                .name,
            "zzz"
        );
    }

    #[test]
    fn resolve_calls_keeps_one_call_when_parallel_is_off() {
        let t = tools(json!([func("a"), func("b")]));
        let p = ToolPolicy::from_request(Some(&t), None, Some(false)).unwrap();
        let text = r#"<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>"#;
        let calls = resolve_calls(text, &p, false, false).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "a");
    }

    #[test]
    fn output_calls_recover_a_custom_input() {
        let p = policy(
            &json!([func("f"), {"type": "custom", "custom": {"name": "c"}}]),
            None,
        )
        .unwrap();
        let input = |arguments: &str| match output_calls(vec![call("c", arguments)], &p).remove(0) {
            OutputToolCall::Custom { id, r#type, custom } => {
                assert_eq!(
                    (id.as_str(), r#type, custom.name.as_str()),
                    ("call_1", "custom", "c")
                );
                custom.input
            }
            other => panic!("not a custom call: {other:?}"),
        };
        assert_eq!(input(r#"{"input":"x"}"#), "x");
        assert_eq!(input(r#"{"patch":"y","n":1}"#), "y");
        assert_eq!(input(r#"{"a":"1","b":"2"}"#), r#"{"a":"1","b":"2"}"#);
        assert_eq!(input("not json"), "not json");
        assert_eq!(input(r#"{"input":5}"#), r#"{"input":5}"#);
        // A function tool's call keeps the function shape.
        assert_eq!(
            output_calls(vec![call("f", "{}")], &p),
            vec![OutputToolCall::Function(call("f", "{}"))]
        );
        let custom =
            serde_json::to_value(output_calls(vec![call("c", r#"{"input":"x"}"#)], &p)).unwrap();
        assert_eq!(
            custom,
            json!([{"id": "call_1", "type": "custom", "custom": {"name": "c", "input": "x"}}])
        );
    }
}
