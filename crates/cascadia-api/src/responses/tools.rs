//! Responses `tools`, `tool_choice` and `include`, mapped onto the chat path.
//!
//! - `function` tools become chat tools as they are.
//! - A `namespace`'s function and custom tools are served flattened as
//!   `<namespace>__<name>`, the shape Codex names its MCP tools in; a call is
//!   mapped back by exact lookup in [`ToolMap::by_flat`], never by splitting
//!   the name.
//! - `custom` (freeform) tools become a function taking one string
//!   parameter `input`; a grammar is put in the description, not
//!   enforced.
//! - Tool types this server cannot serve (OpenAI-hosted ones such as
//!   `web_search`, `file_search`, `mcp`, and the client-side `shell`,
//!   `local_shell`, `apply_patch`, `tool_search` types) are dropped with one
//!   `warn!` per request and left out of the echoed `tools`.

use super::Refusal;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// What a call to a served tool becomes in the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Served {
    /// A `function_call` item.
    Function,
    /// A `custom_tool_call` item, its `input` the call's one string
    /// argument.
    Custom,
}

/// A tool the chat template sees, by its flattened name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServedTool {
    /// What a call to it becomes in the output.
    pub kind: Served,
    /// The name the caller gave it (inside its namespace, if any).
    pub name: String,
    pub namespace: Option<String>,
}

/// A request's `tools`, as served and as echoed.
#[derive(Debug, Default)]
pub struct ToolMap {
    /// Flattened name → what a call to it becomes.
    pub by_flat: HashMap<String, ServedTool>,
    /// The chat tools, in request order.
    pub chat_tools: Vec<crate::Tool>,
    /// The response object's `tools`: Responses-shaped and normalised to the
    /// schema, hosted tools and empty namespaces left out.
    pub echo: Vec<Value>,
    /// The hosted tool types dropped, each once.
    pub dropped: Vec<String>,
}

/// Tool types this server cannot serve: OpenAI-hosted ones and the
/// client-side `shell`, `local_shell`, `apply_patch` and `tool_search`.
fn is_hosted(kind: &str) -> bool {
    kind.starts_with("web_search")
        || matches!(
            kind,
            "file_search"
                | "computer"
                | "computer_use"
                | "computer_use_preview"
                | "code_interpreter"
                | "image_generation"
                | "mcp"
                | "shell"
                | "local_shell"
                | "apply_patch"
                | "tool_search"
                | "programmatic_tool_calling"
        )
}

/// The one parameter a `custom` tool is served with.
fn custom_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    })
}

/// A tool's `name`: a non-empty string, else a refusal naming `<at>.name`.
fn tool_name<'a>(tool: &'a Value, at: &str) -> Result<&'a str, Refusal> {
    match tool.get("name").and_then(Value::as_str) {
        Some(n) if !n.is_empty() => Ok(n),
        _ => Err(Refusal::malformed(
            format!("{at}.name"),
            "a tool needs a non-empty string `name`",
        )),
    }
}

fn tool_type<'a>(tool: &'a Value, at: &str) -> Result<&'a str, Refusal> {
    tool.get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| Refusal::malformed(format!("{at}.type"), "a tool needs a string `type`"))
}

/// A `custom` tool's description for the model: its own, plus the grammar
/// its `input` must follow (shown, not enforced: this path has no
/// constrained decoding).
fn custom_description(tool: &Value) -> String {
    let mut out = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let format = tool.get("format");
    if format.and_then(|f| f.get("type")).and_then(Value::as_str) == Some("grammar") {
        let field = |k: &str| {
            format
                .and_then(|f| f.get(k))
                .and_then(Value::as_str)
                .unwrap_or_default()
        };
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&format!(
            "Put the whole input in the string `input`. It must follow this {} grammar:\n{}",
            field("syntax"),
            field("definition")
        ));
    }
    out
}

/// `tool` with each of `keys` set to `default` where absent or null.
fn with_defaults(tool: &Value, keys: &[&str], default: Value) -> Value {
    let mut out = tool.clone();
    if let Some(obj) = out.as_object_mut() {
        for k in keys {
            let v = obj.entry(*k).or_insert(Value::Null);
            if v.is_null() {
                *v = default.clone();
            }
        }
    }
    out
}

impl ToolMap {
    /// Map a request's `tools`. Absent, `null` or `[]` serves none.
    pub fn from_request(tools: Option<&Value>) -> Result<ToolMap, Refusal> {
        let mut map = ToolMap::default();
        let items = match tools {
            None | Some(Value::Null) => return Ok(map),
            Some(Value::Array(items)) => items,
            Some(_) => return Err(Refusal::malformed("tools", "`tools` must be an array")),
        };
        for (i, tool) in items.iter().enumerate() {
            let at = format!("tools[{i}]");
            match tool_type(tool, &at)? {
                "function" | "custom" => {
                    map.serve(tool, &at, None)?;
                    map.echo.push(map.normalised(tool));
                }
                "namespace" => {
                    let ns = tool_name(tool, &at)?;
                    let nested = match tool.get("tools") {
                        Some(Value::Array(nested)) => nested,
                        None | Some(Value::Null) => continue,
                        Some(_) => {
                            return Err(Refusal::malformed(
                                format!("{at}.tools"),
                                "a namespace's `tools` must be an array",
                            ))
                        }
                    };
                    let mut echoed = Vec::with_capacity(nested.len());
                    for (j, inner) in nested.iter().enumerate() {
                        let inner_at = format!("{at}.tools[{j}]");
                        match tool_type(inner, &inner_at)? {
                            "function" | "custom" => map.serve(inner, &inner_at, Some(ns))?,
                            other => {
                                return Err(Refusal::new(
                                    format!("{inner_at}.type"),
                                    format!(
                                        "a namespace holds only `function` and `custom` tools, \
                                         not `{other}`"
                                    ),
                                ))
                            }
                        }
                        echoed.push(map.normalised(inner));
                    }
                    // An empty namespace offers nothing, and the schema
                    // wants at least one tool in an echoed one.
                    if !echoed.is_empty() {
                        let mut ns_echo = with_defaults(tool, &["description"], json!(""));
                        ns_echo["tools"] = Value::Array(echoed);
                        map.echo.push(ns_echo);
                    }
                }
                hosted if is_hosted(hosted) => {
                    if !map.dropped.iter().any(|d| d == hosted) {
                        map.dropped.push(hosted.to_owned());
                    }
                }
                other => {
                    return Err(Refusal::malformed(
                        format!("{at}.type"),
                        format!(
                            "tool type `{other}` is not supported; `function`, `custom` and \
                             `namespace` tools are served"
                        ),
                    ))
                }
            }
        }
        Ok(map)
    }

    /// Serve one `function` or `custom` tool, flattened into `namespace`.
    fn serve(&mut self, tool: &Value, at: &str, namespace: Option<&str>) -> Result<(), Refusal> {
        let name = tool_name(tool, at)?;
        let flat = Self::flat(namespace, name);
        let custom = tool.get("type").and_then(Value::as_str) == Some("custom");
        let mut function = Map::new();
        function.insert("name".into(), json!(flat));
        if custom {
            let description = custom_description(tool);
            if !description.is_empty() {
                function.insert("description".into(), json!(description));
            }
            function.insert("parameters".into(), custom_parameters());
        } else {
            for key in ["description", "parameters", "strict"] {
                if let Some(v) = tool.get(key).filter(|v| !v.is_null()) {
                    function.insert(key.into(), v.clone());
                }
            }
        }
        let served = ServedTool {
            kind: if custom {
                Served::Custom
            } else {
                Served::Function
            },
            name: name.to_owned(),
            namespace: namespace.map(str::to_owned),
        };
        if self.by_flat.insert(flat.clone(), served).is_some() {
            return Err(Refusal::malformed(
                "tools",
                format!(
                    "more than one tool is served as `{flat}`; tool names must be unique \
                     (a namespace's tools are named `<namespace>__<name>`)"
                ),
            ));
        }
        self.chat_tools.push(crate::Tool {
            r#type: "function".into(),
            function: Value::Object(function),
        });
        Ok(())
    }

    /// A served tool as the response echoes it: a function's `strict` and
    /// `parameters` default to `null`, as the schema requires them.
    fn normalised(&self, tool: &Value) -> Value {
        match tool.get("type").and_then(Value::as_str) {
            Some("function") => with_defaults(tool, &["strict", "parameters"], Value::Null),
            _ => tool.clone(),
        }
    }

    /// The name a tool is served under: `<namespace>__<name>`, or `<name>`.
    pub fn flat(namespace: Option<&str>, name: &str) -> String {
        match namespace {
            Some(ns) => format!("{ns}__{name}"),
            None => name.to_owned(),
        }
    }

    pub fn get(&self, flat: &str) -> Option<&ServedTool> {
        self.by_flat.get(flat)
    }

    /// Whether the model's tool calls are parsed: tools are served and
    /// `tool_choice` is not `"none"`.
    pub fn tools_active(&self, tool_choice: Option<&str>) -> bool {
        !self.chat_tools.is_empty() && tool_choice != Some("none")
    }
}

/// `tool_choice`: `"auto"` and `"none"` are served (`None` when absent).
/// Forcing a call (`"required"`, a named function or custom tool,
/// `allowed_tools`) is refused until it is enforced; a hosted tool
/// can never be called here.
pub fn check_tool_choice(choice: Option<&Value>) -> Result<Option<&'static str>, Refusal> {
    let not_enforced = |what: &str| {
        Err(Refusal::new(
            "tool_choice",
            format!("{what} is not enforced yet on `/v1/responses`; send `\"auto\"` or `\"none\"`"),
        ))
    };
    let kind = match choice {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(s)) => match s.as_str() {
            "auto" => return Ok(Some("auto")),
            "none" => return Ok(Some("none")),
            "required" => return not_enforced("`tool_choice: \"required\"`"),
            other => {
                return Err(Refusal::malformed(
                    "tool_choice",
                    format!("`tool_choice` `{other}` is not one of `auto`, `none`, `required`"),
                ))
            }
        },
        Some(obj) => obj.get("type").and_then(Value::as_str),
    };
    match kind {
        Some(k @ ("function" | "custom")) => {
            not_enforced(&format!("forcing a `{k}` tool with `tool_choice`"))
        }
        Some("allowed_tools") => not_enforced("`tool_choice` of type `allowed_tools`"),
        Some(hosted) if is_hosted(hosted) => Err(Refusal::new(
            "tool_choice",
            format!(
                "`tool_choice` names the hosted tool `{hosted}`, which this server does not run"
            ),
        )),
        _ => Err(Refusal::malformed(
            "tool_choice",
            "`tool_choice` must be `\"auto\"`, `\"none\"`, `\"required\"` or a tool choice object",
        )),
    }
}

/// OpenAI's `IncludeEnum`.
const INCLUDE_VALUES: [&str; 8] = [
    "file_search_call.results",
    "web_search_call.results",
    "web_search_call.action.sources",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
    "code_interpreter_call.outputs",
    "reasoning.encrypted_content",
    "message.output_text.logprobs",
];

/// `include`, checked against [`INCLUDE_VALUES`]. `true` when it asks
/// for `reasoning.encrypted_content`; the other values have nothing to add
/// on this server.
pub(super) fn check_include(include: Option<&Value>) -> Result<bool, Refusal> {
    let items = match include {
        None | Some(Value::Null) => return Ok(false),
        Some(Value::Array(items)) => items,
        Some(_) => {
            return Err(Refusal::malformed(
                "include",
                "`include` must be an array of strings",
            ))
        }
    };
    let mut encrypted = false;
    for item in items {
        match item.as_str() {
            Some(v) if INCLUDE_VALUES.contains(&v) => {
                encrypted |= v == "reasoning.encrypted_content";
            }
            _ => {
                return Err(Refusal::malformed(
                    "include",
                    format!(
                        "`include` value {item} is not supported; expected one of: {}",
                        INCLUDE_VALUES.join(", ")
                    ),
                ))
            }
        }
    }
    Ok(encrypted)
}

/// A `custom` tool call's `input` from the arguments the model wrote for
/// its one-parameter function: `{"input": s}` → `s`; any other single
/// string field → that string; otherwise the arguments text as is.
pub(super) fn custom_input(arguments: &str) -> String {
    if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(arguments) {
        if let Some(Value::String(s)) = obj.get("input") {
            return s.clone();
        }
        if let (1, Some(Value::String(s))) = (obj.len(), obj.values().next()) {
            return s.clone();
        }
    }
    arguments.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(tools: Value) -> ToolMap {
        ToolMap::from_request(Some(&tools)).unwrap()
    }

    fn refused(tools: Value) -> Refusal {
        ToolMap::from_request(Some(&tools)).unwrap_err()
    }

    #[test]
    fn function_becomes_a_chat_tool() {
        let m = map(json!([{"type": "function", "name": "get_weather",
            "description": "the weather", "strict": true,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]));
        assert_eq!(m.chat_tools.len(), 1);
        assert_eq!(m.chat_tools[0].r#type, "function");
        assert_eq!(
            m.chat_tools[0].function,
            json!({"name": "get_weather", "description": "the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                "strict": true})
        );
        assert_eq!(
            m.get("get_weather"),
            Some(&ServedTool {
                kind: Served::Function,
                name: "get_weather".into(),
                namespace: None
            })
        );
        assert_eq!(m.echo.len(), 1);
        assert!(m.dropped.is_empty());
    }

    #[test]
    fn namespace_tools_are_flattened() {
        let m = map(
            json!([{"type": "namespace", "name": "mcp__x", "description": "x server",
            "tools": [
                {"type": "function", "name": "click", "parameters": {"type": "object"}},
                {"type": "function", "name": "get"}]}]),
        );
        let names: Vec<_> = m
            .chat_tools
            .iter()
            .map(|t| t.function["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["mcp__x__click", "mcp__x__get"]);
        assert_eq!(
            m.get("mcp__x__get"),
            Some(&ServedTool {
                kind: Served::Function,
                name: "get".into(),
                namespace: Some("mcp__x".into())
            })
        );
        assert_eq!(ToolMap::flat(Some("mcp__x"), "click"), "mcp__x__click");
        assert_eq!(ToolMap::flat(None, "click"), "click");
        assert_eq!(m.echo.len(), 1);
        assert_eq!(m.echo[0]["type"], "namespace");
        assert_eq!(m.echo[0]["tools"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn custom_tool_takes_one_string_input() {
        let m = map(json!([{"type": "custom", "name": "apply_patch",
            "description": "Apply a patch.",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}}]));
        let f = &m.chat_tools[0].function;
        assert_eq!(m.chat_tools[0].r#type, "function");
        assert_eq!(f["name"], "apply_patch");
        assert_eq!(
            f["parameters"],
            json!({"type": "object", "properties": {"input": {"type": "string"}},
                "required": ["input"]})
        );
        let desc = f["description"].as_str().unwrap();
        assert!(desc.contains("Apply a patch."), "{desc}");
        assert!(desc.contains("start: x"), "{desc}");
        assert_eq!(m.get("apply_patch").unwrap().kind, Served::Custom);
        assert_eq!(m.echo[0]["type"], "custom");
    }

    #[test]
    fn hosted_tools_are_dropped() {
        let m = map(json!([{"type": "web_search", "external_web_access": false},
            {"type": "function", "name": "f"}]));
        assert_eq!(m.chat_tools.len(), 1);
        assert_eq!(m.echo.len(), 1);
        assert_eq!(m.echo[0]["name"], "f");
        assert_eq!(m.dropped, ["web_search"]);
    }

    #[test]
    fn malformed_tools_are_refused_by_path() {
        for (tools, param) in [
            (json!([{"type": "foo"}]), "tools[0].type"),
            (json!([{"name": "f"}]), "tools[0].type"),
            (json!([{"type": "function"}]), "tools[0].name"),
            (json!([{"type": "function", "name": ""}]), "tools[0].name"),
            (
                json!([{"type": "namespace", "name": "n", "tools": [
                    {"type": "function", "name": "a"}, {"type": "function"}]}]),
                "tools[0].tools[1].name",
            ),
            (
                json!([{"type": "namespace", "name": "n", "tools": [{"type": "web_search"}]}]),
                "tools[0].tools[0].type",
            ),
            (
                json!([{"type": "namespace", "tools": [{"type": "function", "name": "a"}]}]),
                "tools[0].name",
            ),
            (
                json!([{"type": "function", "name": "f"}, {"type": "function", "name": "f"}]),
                "tools",
            ),
            (
                json!([{"type": "function", "name": "a__b"},
                    {"type": "namespace", "name": "a", "tools": [{"type": "function", "name": "b"}]}]),
                "tools",
            ),
            (json!({"type": "function"}), "tools"),
        ] {
            assert_eq!(refused(tools.clone()).param, param, "{tools}");
        }
    }

    #[test]
    fn empty_namespace_is_neither_served_nor_echoed() {
        let m = map(json!([{"type": "namespace", "name": "n", "description": "d", "tools": []}]));
        assert!(m.chat_tools.is_empty() && m.echo.is_empty() && m.by_flat.is_empty());
        let m = ToolMap::from_request(None).unwrap();
        assert!(m.chat_tools.is_empty() && m.echo.is_empty());
        let m = map(json!([]));
        assert!(m.chat_tools.is_empty() && m.echo.is_empty());
    }

    #[test]
    fn echo_is_normalised_to_the_schema() {
        let m = map(json!([{"type": "function", "name": "f"},
            {"type": "namespace", "name": "n", "tools": [{"type": "function", "name": "g"}]}]));
        assert_eq!(
            m.echo[0],
            json!({"type": "function", "name": "f", "strict": null, "parameters": null})
        );
        assert_eq!(m.echo[1]["description"], "");
        assert_eq!(m.echo[1]["tools"][0]["strict"], Value::Null);
    }

    #[test]
    fn tools_active_needs_tools_and_a_choice_other_than_none() {
        let m = map(json!([{"type": "function", "name": "f"}]));
        assert!(m.tools_active(None));
        assert!(m.tools_active(Some("auto")));
        assert!(!m.tools_active(Some("none")));
        assert!(!ToolMap::default().tools_active(Some("auto")));
    }

    #[test]
    fn custom_input_recovers_the_string() {
        assert_eq!(custom_input(r#"{"input":"x"}"#), "x");
        assert_eq!(custom_input(r#"{"patch":"y"}"#), "y");
        assert_eq!(custom_input("not json"), "not json");
        assert_eq!(custom_input(r#"{"a":1}"#), r#"{"a":1}"#);
    }

    #[test]
    fn tool_choice_takes_auto_and_none_only() {
        assert_eq!(check_tool_choice(None).unwrap(), None);
        assert_eq!(
            check_tool_choice(Some(&json!("auto"))).unwrap(),
            Some("auto")
        );
        assert_eq!(
            check_tool_choice(Some(&json!("none"))).unwrap(),
            Some("none")
        );
        for choice in [
            json!("required"),
            json!({"type": "function", "name": "f"}),
            json!({"type": "custom", "name": "c"}),
            json!({"type": "allowed_tools", "mode": "auto", "tools": []}),
        ] {
            let r = check_tool_choice(Some(&choice)).unwrap_err();
            assert_eq!(r.param, "tool_choice", "{choice}");
            assert!(
                r.message.contains("not enforced"),
                "{choice}: {}",
                r.message
            );
        }
        for choice in [
            json!({"type": "web_search_preview"}),
            json!(5),
            json!("sometimes"),
        ] {
            let r = check_tool_choice(Some(&choice)).unwrap_err();
            assert_eq!(r.param, "tool_choice", "{choice}");
        }
    }

    #[test]
    fn include_is_checked_against_include_enum() {
        for value in [
            "file_search_call.results",
            "web_search_call.results",
            "web_search_call.action.sources",
            "message.input_image.image_url",
            "computer_call_output.output.image_url",
            "code_interpreter_call.outputs",
            "reasoning.encrypted_content",
            "message.output_text.logprobs",
        ] {
            assert!(check_include(Some(&json!([value]))).is_ok(), "{value}");
        }
        assert!(check_include(Some(&json!(["reasoning.encrypted_content"]))).unwrap());
        assert!(!check_include(Some(&json!(["message.output_text.logprobs"]))).unwrap());
        assert!(!check_include(Some(&json!([]))).unwrap());
        assert!(!check_include(None).unwrap());
        for bad in [json!(["bogus"]), json!("x")] {
            assert_eq!(
                check_include(Some(&bad)).unwrap_err().param,
                "include",
                "{bad}"
            );
        }
    }
}
