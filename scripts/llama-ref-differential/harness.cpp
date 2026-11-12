// Runs llama.cpp's own JSON-schema-to-grammar converter (the one llama-server
// uses for `response_format`) on one schema per stdin line and says, per
// line, whether it converted every `$ref` faithfully:
//
//   error <message>     the converter refused the schema (a 4xx at llama-server)
//   degraded <ref>      a `$ref` the converter reached had no recorded target:
//                       it became "any object" (or an `allOf` member was dropped)
//   faithful            every `$ref` it reached resolved to its target
//
// (A ref whose rule name another rule already holds is reused without its
// target being visited; that shows only by comparing grammars, which
// classify.py does with `--grammar` output.)
//
// A `$ref` with no recorded target shows in two ways: `_resolve_ref` and
// `allOf` look it up with `operator[]`, which inserts a null entry while the
// grammar is built; and a ref to an `https://` document is recorded as null
// when refs are resolved (it never fetches). Those null records are swapped
// for a marker schema before the grammar is built, so a ref that reaches
// one (as a rule or as an `allOf` member) leaves the marker in the grammar.
//
// Built against a llama.cpp checkout by run.sh; not part of the Rust build.

#include "json-schema-to-grammar.h"
#include "common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// The converter's class is private to its translation unit, and the refs it
// recorded are private members: read them by compiling it here.
#define private public
#include "json-schema-to-grammar.cpp"
#undef private

// Helpers json-schema-to-grammar.cpp takes from common.cpp, so the harness
// links without the rest of llama.cpp.
std::string string_join(const std::vector<std::string> & v, const std::string & sep) {
    std::string r;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) r += sep;
        r += v[i];
    }
    return r;
}
std::vector<std::string> string_split(const std::string & s, const std::string & d) {
    std::vector<std::string> out;
    size_t a = 0, b;
    while ((b = s.find(d, a)) != std::string::npos) {
        out.push_back(s.substr(a, b - a));
        a = b + d.size();
    }
    out.push_back(s.substr(a));
    return out;
}
std::string string_repeat(const std::string & s, size_t n) {
    std::string r;
    for (size_t i = 0; i < n; i++) r += s;
    return r;
}

// A property name no schema uses: a null-recorded ref's stand-in.
static const char * MARKER = "__llama_ref_unresolved__";

static std::string verdict(const std::string & line, const std::string & root) {
    nlohmann::ordered_json schema = nlohmann::ordered_json::parse(line);
    common_schema_converter converter([](const std::string &) { return nlohmann::ordered_json(); }, false);
    try {
        auto copy = schema;
        converter.resolve_refs(copy, "");
        std::set<std::string> recorded;
        for (auto & kv : converter._refs) {
            recorded.insert(kv.first);
            if (kv.second.is_null()) {
                kv.second = nlohmann::ordered_json{{"properties", {{MARKER, nlohmann::ordered_json::object()}}}};
            }
        }
        converter.visit(copy, root);
        converter.check_errors();
        for (const auto & kv : converter._refs) {
            if (!recorded.count(kv.first)) {
                return "degraded " + kv.first;
            }
        }
        if (converter.format_grammar().find(MARKER) != std::string::npos) {
            return "degraded (a null-recorded ref)";
        }
        return "faithful";
    } catch (const std::exception & e) {
        std::string m = e.what();
        std::replace(m.begin(), m.end(), '\n', ' ');
        return "error " + m.substr(0, 120);
    }
}

// The grammar llama-server would build for `line` with its root schema
// visited under `root`, on one line (each rule `name ::= body` separated
// by a tab), or `error`.
static std::string grammar(const std::string & line, const std::string & root) {
    common_schema_converter converter([](const std::string &) { return nlohmann::ordered_json(); }, false);
    try {
        auto schema = nlohmann::ordered_json::parse(line);
        converter.resolve_refs(schema, "");
        converter.visit(schema, root);
        converter.check_errors();
        std::string g = converter.format_grammar();
        std::replace(g.begin(), g.end(), '\n', '\t');
        return g;
    } catch (const std::exception &) {
        return "error";
    }
}

// The grammar llama-server's chat endpoint builds for a request with a
// `response_format` schema and one tool: one converter, one table of refs
// for both (chat-auto-parser-generator.cpp resolves the tools' refs and
// the response's into the same converter), the response under
// `response-format-schema`, the tool under `tool-f-schema`. `line` is
// `{"response": <schema>, "tool": <schema>}`.
static std::string pair_grammar(const std::string & line) {
    common_schema_converter converter([](const std::string &) { return nlohmann::ordered_json(); }, false);
    try {
        auto both = nlohmann::ordered_json::parse(line);
        auto tool = both["tool"];
        auto response = both["response"];
        converter.resolve_refs(tool, "");
        converter.resolve_refs(response, "");
        converter.visit(response, "response-format-schema");
        converter.visit(tool, "tool-f-schema");
        converter.check_errors();
        std::string g = converter.format_grammar();
        std::replace(g.begin(), g.end(), '\n', '\t');
        return g;
    } catch (const std::exception &) {
        return "error";
    }
}

// `harness [--grammar | --pair] [--root NAME]`: a verdict, the grammar, or
// a pair's grammar, per line. The root name is the one the root schema is
// visited under: llama-server's chat endpoint builds a `response_format`
// grammar through its chat parser, which adds the schema under
// `response-format-schema` (or `response-format`, per template); its plain
// `/completion` endpoint and a `--no-jinja` server under `` (the rule
// `root`).
int main(int argc, char ** argv) {
    bool grammars = false;
    bool pairs = false;
    std::string root;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--grammar") {
            grammars = true;
        } else if (a == "--pair") {
            pairs = true;
        } else if (a == "--root" && i + 1 < argc) {
            root = argv[++i];
        }
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::cout << (pairs ? pair_grammar(line) : grammars ? grammar(line, root) : verdict(line, root)) << "\n";
    }
}
