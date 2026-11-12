// Runs llama.cpp's own JSON-schema-to-grammar converter (the one llama-server
// uses for `response_format`) on one schema per stdin line and says, per
// line, whether it converted every `$ref` faithfully:
//
//   error <message>     the converter refused the schema (a 4xx at llama-server)
//   degraded <ref>      a `$ref` the converter reached had no recorded target:
//                       it became "any object" (or an `allOf` member was dropped)
//   faithful            every `$ref` it reached resolved to its target
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

static std::string verdict(const std::string & line) {
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
        converter.visit(copy, "");
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

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::cout << verdict(line) << "\n";
    }
}
