// Decoding that keeps a model's tool-call marker tokens as text. Plain C++,
// no OpenVINO: shim.cpp includes it, and it compiles on its own.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Token ids written out as literal text when decoding, instead of being
// skipped with the other special tokens: a model's tool-call markers (Gemma 4
// builds its calls out of special tokens, so a plain decode erases them).
using KeepList = std::vector<std::pair<int64_t, std::string>>;

// A keep list is a handful of markers; this only bounds a poisoned count.
constexpr size_t MAX_KEEP_TOKENS = 64;

inline KeepList read_keep_list(const int64_t* ids, const char* const* texts, size_t n) {
    if (n > MAX_KEEP_TOKENS) {
        throw std::invalid_argument("keep token count exceeds MAX_KEEP_TOKENS=64");
    }
    KeepList keep;
    keep.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!texts[i]) throw std::invalid_argument("null keep token text");
        keep.emplace_back(ids[i], std::string(texts[i]));
    }
    return keep;
}

inline const std::string* kept_text(const KeepList& keep, int64_t id) {
    const auto kept =
        std::find_if(keep.begin(), keep.end(), [id](const auto& k) { return k.first == id; });
    return kept == keep.end() ? nullptr : &kept->second;
}

// `decode(ids)`, except kept ids come out as their literal text: the ids are
// split at the kept ids, each run between them is decoded on its own, and the
// kept texts go between the runs. An empty keep list is exactly one
// `decode(ids)`.
template <typename Decode>
std::string decode_keeping(const std::vector<int64_t>& ids, const KeepList& keep,
                           Decode&& decode) {
    if (keep.empty()) return decode(ids);
    std::string text;
    std::vector<int64_t> run;
    for (const int64_t id : ids) {
        const std::string* kept = kept_text(keep, id);
        if (!kept) {
            run.push_back(id);
            continue;
        }
        if (!run.empty()) {
            text += decode(run);
            run.clear();
        }
        text += *kept;
    }
    if (!run.empty()) text += decode(run);
    return text;
}

// The text of a growing id sequence's finished runs: everything up to and
// including its last kept id, and how many ids that covers. A run that a
// kept id has closed decodes the same on every later read.
struct KeepingDecodeCache {
    std::string done_text;
    size_t done_ids = 0;
};

// decode_keeping(ids, keep, decode) for `ids` that only grow between calls
// with the same `cache`: finished runs are decoded once, when their closing
// kept id arrives, and each call decodes only the trailing open run. An
// empty keep list is exactly one `decode(ids)`, the cache untouched.
template <typename Decode>
std::string decode_keeping_cached(const std::vector<int64_t>& ids, const KeepList& keep,
                                  KeepingDecodeCache& cache, Decode&& decode) {
    if (keep.empty()) return decode(ids);
    if (cache.done_ids > ids.size()) cache = KeepingDecodeCache{};
    size_t run_start = cache.done_ids;
    for (size_t i = cache.done_ids; i < ids.size(); ++i) {
        const std::string* kept = kept_text(keep, ids[i]);
        if (!kept) continue;
        if (run_start < i) {
            cache.done_text += decode(std::vector<int64_t>(ids.begin() + run_start, ids.begin() + i));
        }
        cache.done_text += *kept;
        run_start = i + 1;
        cache.done_ids = run_start;
    }
    std::string text = cache.done_text;
    if (cache.done_ids < ids.size()) {
        text += decode(std::vector<int64_t>(ids.begin() + cache.done_ids, ids.end()));
    }
    return text;
}

}  // namespace
