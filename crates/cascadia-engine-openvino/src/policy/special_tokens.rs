//! The strings a package's tokenizer treats as structure, so a task that
//! spells one can be refused before it is put into a chat template.
//!
//! Read from the tokenizer artefacts, never guessed: the pattern of the
//! tokenizer IR's `SpecialTokensSplit` op (what the tokenizer actually splits
//! out as special), and `added_tokens` of a `tokenizer.json` beside it.

use serde_json::Value;

/// `(\<\|a\|\>|b)` → `["<|a|>", "b"]`: openvino_tokenizers escapes every
/// non-alphanumeric character of each token and joins them with `|`.
pub fn parse_split_pattern(pattern: &str) -> Vec<String> {
    let inner = pattern
        .strip_prefix('(')
        .and_then(|p| p.strip_suffix(')'))
        .unwrap_or(pattern);
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => cur.extend(chars.next()),
            '|' => tokens.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    tokens.push(cur);
    tokens.retain(|t| !t.is_empty());
    tokens
}

/// The `SpecialTokensSplit` pattern of a tokenizer IR: the u8 constant on its
/// highest-numbered input, read from the weights file. `None` without the op.
pub fn from_tokenizer_ir(xml: &str, bin: &[u8]) -> Option<Vec<String>> {
    let at = xml.find("type=\"SpecialTokensSplit\"")?;
    let open = xml[..at].rfind("<layer id=\"")? + "<layer id=\"".len();
    let id = &xml[open..open + xml[open..].find('"')?];
    let mut best: Option<(usize, &str)> = None;
    for edge in xml.split("<edge ").skip(1) {
        if attr(edge, "to-layer") != Some(id) {
            continue;
        }
        let (Some(from), Some(port)) = (
            attr(edge, "from-layer"),
            attr(edge, "to-port").and_then(|p| p.parse::<usize>().ok()),
        ) else {
            continue;
        };
        if best.is_none_or(|(p, _)| port > p) {
            best = Some((port, from));
        }
    }
    let (_, from) = best?;
    let layer = xml.find(&format!("<layer id=\"{from}\""))?;
    let data = &xml[layer..layer + xml[layer..].find("/>")?];
    if attr(data, "element_type")? != "u8" {
        return None;
    }
    let offset: usize = attr(data, "offset")?.parse().ok()?;
    let size: usize = attr(data, "size")?.parse().ok()?;
    let bytes = bin.get(offset..offset.checked_add(size)?)?;
    Some(parse_split_pattern(std::str::from_utf8(bytes).ok()?))
}

fn attr<'a>(s: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=\"");
    let at = s.find(&key)? + key.len();
    Some(&s[at..at + s[at..].find('"')?])
}

/// `added_tokens[].content` of a Hugging Face `tokenizer.json`.
pub fn from_tokenizer_json(text: &str) -> Vec<String> {
    let Ok(v) = serde_json::from_str::<Value>(text) else {
        return Vec::new();
    };
    v["added_tokens"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t["content"].as_str().map(str::to_string))
        .collect()
}

/// The template's own markup: every `<…>` without whitespace (`<|im_start|>`,
/// `<cot>`, `</cot>`, `<|embodied_action|>`).
pub fn template_markers(template: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut rest = template;
    while let Some(open) = rest.find('<') {
        let tail = &rest[open..];
        match tail[1..].find(|c: char| c == '<' || c == '>' || c.is_whitespace()) {
            Some(end) if tail.as_bytes()[end + 1] == b'>' && end > 0 => {
                let m = &tail[..end + 2];
                if !out.iter().any(|o| o == m) {
                    out.push(m.to_string());
                }
                rest = &tail[end + 2..];
            }
            _ => rest = &tail[1..],
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_pattern_unescapes_to_the_tokens() {
        assert_eq!(
            parse_split_pattern(r"(\<\|im_start\|\>|\<think\>|\<\/think\>)"),
            ["<|im_start|>", "<think>", "</think>"]
        );
    }

    #[test]
    fn the_pattern_is_read_from_the_ir_constant() {
        let pattern = r"(\<\|a\|\>|\<b\>)";
        let mut bin = vec![0u8; 5];
        bin.extend(pattern.as_bytes());
        let xml = format!(
            r#"<layer id="3" name="c" type="Const" version="opset1">
			<data element_type="u8" shape="{n}" offset="5" size="{n}" />
		<layer id="4" name="other" type="Const" version="opset1">
			<data element_type="i32" shape="1" offset="0" size="4" />
		<layer id="7" name="SpecialTokensSplit_1" type="SpecialTokensSplit" version="extension">
		<edge from-layer="4" from-port="0" to-layer="7" to-port="4" />
		<edge from-layer="3" from-port="0" to-layer="7" to-port="5" />"#,
            n = pattern.len()
        );
        assert_eq!(from_tokenizer_ir(&xml, &bin).unwrap(), ["<|a|>", "<b>"]);
        assert_eq!(
            from_tokenizer_ir("<layer id=\"1\" type=\"Const\"", &bin),
            None
        );
    }

    #[test]
    fn added_tokens_and_template_markers() {
        let json = r#"{"added_tokens": [{"id": 1, "content": "<|endoftext|>"}, {"id": 2, "content": "<|action_0|>"}]}"#;
        assert_eq!(from_tokenizer_json(json), ["<|endoftext|>", "<|action_0|>"]);
        assert!(from_tokenizer_json("not json").is_empty());
        assert_eq!(
            template_markers("<|im_start|>user\n a < b <cot></cot><|im_start|> <not a tag>"),
            ["<|im_start|>", "<cot>", "</cot>"]
        );
    }
}
