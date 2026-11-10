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

use serde::Deserialize;
use serde_json::{json, Value};

use crate::{FunctionCall, Tool, ToolCall};

/// One entry of a request's `tools` list.
///
/// The function form is tried first: any entry carrying a `function` object
/// is accepted as a function tool whatever its `type` says, as before custom
/// tools existed. The custom form needs `type: "custom"`, which
/// [`ToolPolicy::from_request`] checks.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum RequestTool {
    /// `{"type":"function","function":{"name",...}}`; `function` is opaque
    /// JSON forwarded to the chat template as the client wrote it.
    Function { r#type: String, function: Value },
    /// `{"type":"custom","custom":{"name","description"?,"format"?}}`.
    Custom { r#type: String, custom: Value },
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
}

/// Which kind of tool a name belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Function,
    Custom,
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
        for tool in tools {
            if let Some(allowed) = &allowed {
                if !tool.name().is_some_and(|n| allowed.contains(n)) {
                    continue;
                }
            }
            if let (RequestTool::Custom { .. }, Some(name)) = (tool, tool.name()) {
                custom_names.insert(name.to_string());
            }
            effective.push(tool.to_tool());
        }
        Ok(ToolPolicy {
            tools: effective,
            mode,
            custom_names,
            single_call: parallel_tool_calls == Some(false),
            narrowed: allowed.is_some(),
            choice_given: tool_choice.is_some(),
            tools_given,
        })
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
    let named_mode = |kind: Kind, key: &str, shape: &str, what: &str| {
        let name = named(choice, key).ok_or_else(|| shape.to_string())?;
        if by_name.get(name) != Some(&kind) {
            return Err(format!(
                "tool_choice names {what} '{name}', which is not in tools"
            ));
        }
        Ok((ToolMode::Named(name.to_string()), None))
    };
    match choice.get("type").and_then(Value::as_str) {
        Some("function") => named_mode(
            Kind::Function,
            "function",
            FUNCTION_CHOICE_SHAPE,
            "function",
        ),
        Some("custom") => named_mode(Kind::Custom, "custom", CUSTOM_CHOICE_SHAPE, "custom tool"),
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
        if by_name.get(name) != Some(&kind) {
            let what = match kind {
                Kind::Function => "function",
                Kind::Custom => "custom tool",
            };
            return Err(format!(
                "{FIELD} lists {what} '{name}', which is not in tools"
            ));
        }
        names.insert(name);
    }
    Ok((mode, names))
}

/// A `tool_calls` entry of an assistant message: the function form, or a
/// replayed custom call.
#[derive(Deserialize)]
#[serde(untagged)]
enum WireToolCall {
    Function(ToolCall),
    Custom {
        id: String,
        r#type: String,
        custom: WireCustomCall,
    },
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
        assert_eq!(e, "tool_choice names function 'c', which is not in tools");
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
            "tool_choice names custom tool 'a', which is not in tools"
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
}
