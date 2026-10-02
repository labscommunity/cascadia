//! A small language model as the drafter of the lone-stream speculation path.
//!
//! A pipeline that guesses a lone stream's next tokens produces one token per
//! stage time while its guesses are right and pays a whole trip through every
//! rank when one is wrong: `time per token = a*T + (1-a)*L`. On the 11-box
//! Inkling fleet `L` is ten times `T`, so the share of right guesses `a` is
//! most of the speed. The n-gram drafter ([`crate::ngram_draft`]) is right
//! about three times in ten on text it has not seen; a 0.6B-parameter model
//! that reads the same conversation is right more than five times in ten
//! (measured offline over the fleet's own outputs, autolab experiment 013).
//!
//! The model runs OUTSIDE this process, behind llama.cpp's `/completion`
//! endpoint on localhost (`CASCADIA_STREAMS_SPEC_LM=http://127.0.0.1:PORT`).
//! It need not share the target's vocabulary: it continues the conversation as
//! TEXT, and that text is cut into the target's tokens here. A guess never
//! chooses a token (the pipeline verifies every one), so nothing in this file
//! can change what the model says; it only changes how often work started
//! early is kept.
//!
//! One background thread per [`LmLink`] owns the HTTP connection. The engine
//! thread never blocks on it for longer than it asks to ([`LmLink::next`]'s
//! `wait`): when the model has nothing ready the n-gram drafter answers.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use tokenizers::Tokenizer;

/// How the conversation is laid out for the drafter model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmTemplate {
    /// ChatML as Qwen uses it.
    Qwen,
    /// Llama 3 instruct headers.
    Llama3,
    /// No chat markup at all.
    Plain,
}

impl LmTemplate {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "llama3" | "llama" => Self::Llama3,
            "plain" | "none" => Self::Plain,
            _ => Self::Qwen,
        }
    }

    fn render(self, user: &str) -> String {
        match self {
            Self::Qwen => format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"),
            Self::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Self::Plain => format!("{user}\n\n"),
        }
    }
}

/// Where the drafter lives and how to talk to it.
#[derive(Debug, Clone)]
pub struct LmConfig {
    /// `host:port` of a llama.cpp-compatible server.
    pub addr: String,
    pub template: LmTemplate,
    /// Drafter tokens asked for per generation.
    pub n_predict: usize,
}

impl LmConfig {
    /// `CASCADIA_STREAMS_SPEC_LM=http://host:port` (+ `_TEMPLATE`, `_NPREDICT`).
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("CASCADIA_STREAMS_SPEC_LM").ok()?;
        let addr = url
            .trim()
            .trim_start_matches("http://")
            .trim_end_matches('/')
            .to_string();
        if addr.is_empty() || addr == "0" {
            return None;
        }
        let template = std::env::var("CASCADIA_STREAMS_SPEC_LM_TEMPLATE")
            .map(|s| LmTemplate::parse(&s))
            .unwrap_or(LmTemplate::Qwen);
        let n_predict = std::env::var("CASCADIA_STREAMS_SPEC_LM_NPREDICT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32usize)
            .clamp(4, 256);
        Some(Self {
            addr,
            template,
            n_predict,
        })
    }
}

#[derive(Default)]
struct Shared {
    /// Bumped by every new generation request; a generation whose epoch is
    /// stale is abandoned by the worker.
    epoch: u64,
    /// The drafter's whole prompt for `epoch` (chat markup + the text so far).
    prompt: String,
    /// What the drafter has written for `epoch` so far.
    cont: String,
    /// The generation for `epoch` has ended (length, end of turn, or error).
    done: bool,
    /// A request is waiting for the worker.
    posted: bool,
    shutdown: bool,
    /// Generations started / abandoned before their end / failed.
    started: u64,
    abandoned: u64,
    failed: u64,
    /// The last generation could not reach the server at all.
    unreachable: bool,
    /// The running generation's socket: shut down by a restart, so the worker
    /// (and the server behind it) drop the stale generation at once.
    live: Option<TcpStream>,
}

/// One stream's connection to the drafter model.
pub struct LmLink {
    cfg: LmConfig,
    tok: Arc<Tokenizer>,
    shared: Arc<(Mutex<Shared>, Condvar)>,
    /// Chat markup + user text: fixed for the request.
    head: String,
    /// The assistant tokens the running generation started from.
    base: Vec<i64>,
    /// No generation has been asked for yet (or the last one is unusable).
    idle: bool,
}

impl LmLink {
    /// `prompt_ids` is the request's prompt in the TARGET's tokens; the drafter
    /// sees its text (special tokens dropped) as one user turn.
    pub fn new(cfg: LmConfig, tok: Arc<Tokenizer>, prompt_ids: &[i64]) -> Self {
        let ids: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        let user = last_user_turn(&tok.decode(&ids, false).unwrap_or_default())
            .unwrap_or_else(|| tok.decode(&ids, true).unwrap_or_default());
        let head = cfg.template.render(user.trim());
        let shared = Arc::new((Mutex::new(Shared::default()), Condvar::new()));
        let worker = shared.clone();
        let (addr, n_predict) = (cfg.addr.clone(), cfg.n_predict);
        std::thread::Builder::new()
            .name("spec-lm".into())
            .spawn(move || worker_loop(&worker, &addr, n_predict))
            .ok();
        Self {
            cfg,
            tok,
            shared,
            head,
            base: Vec::new(),
            idle: true,
        }
    }

    /// The drafter's guess for the token after `assistant` (the target's
    /// output so far, including guesses already sent), waiting at most `wait`
    /// for a generation in progress. `None`: nothing usable (yet).
    pub fn next(&mut self, assistant: &[i64], wait: Duration) -> Option<i64> {
        let deadline = Instant::now() + wait;
        loop {
            let (lock, cv) = &*self.shared;
            let mut s = lock.lock().ok()?;
            let follows = !self.idle
                && assistant.len() >= self.base.len()
                && assistant[..self.base.len()] == self.base[..];
            if follows {
                let derived = self.derive(&s.cont, s.done);
                let used = assistant.len() - self.base.len();
                let both = used.min(derived.len());
                if assistant[self.base.len()..self.base.len() + both] == derived[..both] {
                    if used < derived.len() {
                        return Some(derived[used]);
                    }
                    if !s.done {
                        // The drafter agrees with the text as far as it has
                        // written: wait for more, if the caller can. (The text
                        // may be AHEAD of it, after a guess from the tables:
                        // that is no reason to start over.)
                        let left = deadline.saturating_duration_since(Instant::now());
                        if left.is_zero() {
                            return None;
                        }
                        let (g, _) = cv.wait_timeout(s, left).ok()?;
                        drop(g);
                        continue;
                    }
                    if used == 0 {
                        // Ended with nothing for exactly this text (end of
                        // turn, or no server): asking again changes nothing.
                        return None;
                    }
                }
            }
            if Instant::now() > deadline {
                return None;
            }
            // The text moved away from what the drafter was continuing (a
            // wrong guess, an n-gram guess in between) or the generation is
            // used up: start over from the text as it is now.
            let ids: Vec<u32> = assistant.iter().map(|&t| t as u32).collect();
            let text = self.tok.decode(&ids, true).unwrap_or_default();
            if !s.done && !self.idle {
                s.abandoned += 1;
            }
            s.epoch += 1;
            if let Some(live) = s.live.take() {
                let _ = live.shutdown(std::net::Shutdown::Both);
            }
            s.prompt = format!("{}{}", self.head, text);
            s.cont.clear();
            s.done = false;
            s.posted = true;
            s.started += 1;
            self.base = assistant.to_vec();
            self.idle = false;
            cv.notify_all();
            if wait.is_zero() {
                return None;
            }
        }
    }

    /// The drafter could not be reached when last asked: the caller should
    /// use another source until the text changes again.
    pub fn unreachable(&self) -> bool {
        self.shared.0.lock().map(|s| s.unreachable).unwrap_or(true)
    }

    /// `(generations started, abandoned, failed)`.
    pub fn stats(&self) -> (u64, u64, u64) {
        self.shared
            .0
            .lock()
            .map(|s| (s.started, s.abandoned, s.failed))
            .unwrap_or_default()
    }

    /// The drafter's text as target tokens. While the drafter is still
    /// writing, its last word is held back: more letters may yet join it and
    /// change how it is cut.
    fn derive(&self, cont: &str, done: bool) -> Vec<i64> {
        let stable = if done {
            cont
        } else {
            match cont.rfind(|c: char| c.is_whitespace()) {
                // Nothing but a first word yet. Taking it as it stands saves
                // one drafter token of delay after a wrong guess, but a word
                // cut short is a wrong guess AND sends the text away from
                // what the drafter is writing (it has to start over).
                Some(0) | None if gamble_first_word() => cont,
                Some(i) => &cont[..i],
                None => "",
            }
        };
        if stable.is_empty() {
            return Vec::new();
        }
        self.tok
            .encode(stable, false)
            .map(|e| e.get_ids().iter().map(|&t| t as i64).collect())
            .unwrap_or_default()
    }

    pub fn template(&self) -> LmTemplate {
        self.cfg.template
    }
}

impl Drop for LmLink {
    fn drop(&mut self) {
        let (lock, cv) = &*self.shared;
        if let Ok(mut s) = lock.lock() {
            s.shutdown = true;
            s.epoch += 1;
        }
        cv.notify_all();
    }
}

/// `CASCADIA_STREAMS_SPEC_LM_GAMBLE=1`: use the drafter's first word before
/// the next one proves it complete.
fn gamble_first_word() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CASCADIA_STREAMS_SPEC_LM_GAMBLE").is_ok_and(|v| v.trim() == "1")
    })
}

/// The last user message of a prompt rendered with message markers
/// (`<|message_user|>...<|end_message|>`, Inkling's chat template, which also
/// puts a "Thinking effort level" system line first). `None` when the prompt
/// has no such markers.
fn last_user_turn(rendered: &str) -> Option<String> {
    let start = rendered.rfind("<|message_user|>")? + "<|message_user|>".len();
    let rest = &rendered[start..];
    let body = rest.split("<|end_message|>").next().unwrap_or(rest);
    let body = body.strip_prefix("<|content_text|>").unwrap_or(body);
    (!body.trim().is_empty()).then(|| body.to_string())
}

fn worker_loop(shared: &Arc<(Mutex<Shared>, Condvar)>, addr: &str, n_predict: usize) {
    let (lock, cv) = &**shared;
    loop {
        let (epoch, prompt) = {
            let Ok(mut s) = lock.lock() else { return };
            while !s.posted && !s.shutdown {
                s = match cv.wait(s) {
                    Ok(g) => g,
                    Err(_) => return,
                };
            }
            if s.shutdown {
                return;
            }
            s.posted = false;
            (s.epoch, std::mem::take(&mut s.prompt))
        };
        let ok = generate(shared, addr, n_predict, epoch, &prompt);
        if let Ok(mut s) = lock.lock() {
            if s.epoch == epoch {
                s.done = true;
                s.live = None;
                s.unreachable = ok == Some(false);
                if ok != Some(true) {
                    s.failed += 1;
                }
            }
        }
        cv.notify_all();
    }
}

/// One streamed `/completion`; text lands in `shared.cont` while `epoch` is
/// current. `Some(true)`: ran to its end (or was abandoned); `Some(false)`: no
/// server there; `None`: broke off midway.
fn generate(
    shared: &Arc<(Mutex<Shared>, Condvar)>,
    addr: &str,
    n_predict: usize,
    epoch: u64,
    prompt: &str,
) -> Option<bool> {
    let (lock, cv) = &**shared;
    let Some(sock) = addr.to_socket_addrs().ok().and_then(|mut a| a.next()) else {
        return Some(false);
    };
    let Ok(mut conn) = TcpStream::connect_timeout(&sock, Duration::from_millis(300)) else {
        return Some(false);
    };
    let _ = conn.set_nodelay(true);
    let _ = conn.set_read_timeout(Some(Duration::from_millis(25)));
    {
        let mut s = lock.lock().ok()?;
        if s.epoch != epoch {
            return Some(true);
        }
        s.live = conn.try_clone().ok();
    }
    let body = serde_json::json!({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": true,
        "stream": true,
    })
    .to_string();
    let req = format!(
        "POST /completion HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    if conn.write_all(req.as_bytes()).is_err() {
        return None;
    }
    let mut raw: Vec<u8> = Vec::with_capacity(8192);
    let mut buf = [0u8; 4096];
    let mut seen = 0usize; // bytes of `raw` already searched for events
    let started = Instant::now();
    loop {
        match conn.read(&mut buf) {
            Ok(0) => return Some(true),
            Ok(n) => raw.extend_from_slice(&buf[..n]),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => return None,
        }
        // Events are `data: {json}` lines. Chunked transfer framing (hex sizes
        // on lines of their own) never starts with "data: ", so it is skipped.
        let mut stop = false;
        let mut pieces = String::new();
        while let Some(nl) = raw[seen..].iter().position(|&b| b == b'\n') {
            let line = &raw[seen..seen + nl];
            seen += nl + 1;
            let Some(json) = line.strip_prefix(b"data: ") else {
                continue;
            };
            let Ok(v) = serde_json::from_slice::<serde_json::Value>(json) else {
                continue;
            };
            if let Some(c) = v.get("content").and_then(|c| c.as_str()) {
                pieces.push_str(c);
            }
            if v.get("stop").and_then(|s| s.as_bool()).unwrap_or(false) {
                stop = true;
            }
        }
        {
            let mut s = lock.lock().ok()?;
            if s.epoch != epoch {
                return Some(true); // abandoned: closing the socket stops the server's generation
            }
            if !pieces.is_empty() {
                s.cont.push_str(&pieces);
            }
        }
        if !pieces.is_empty() {
            cv.notify_all();
        }
        if stop {
            return Some(true);
        }
        if started.elapsed() > Duration::from_secs(20) {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    /// A stand-in for llama-server: answers every `/completion` with the
    /// scripted continuation of whatever the prompt ends with.
    fn fake_server(script: Vec<(&'static str, &'static str)>) -> String {
        let l = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = l.local_addr().expect("addr").to_string();
        std::thread::spawn(move || {
            for c in l.incoming() {
                let Ok(mut c) = c else { continue };
                let mut r = BufReader::new(c.try_clone().expect("clone"));
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0u8; len];
                if r.read_exact(&mut body).is_err() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                let prompt = v["prompt"].as_str().unwrap_or("").to_string();
                let cont = script
                    .iter()
                    .find(|(tail, _)| prompt.ends_with(tail))
                    .map(|(_, c)| *c)
                    .unwrap_or("");
                let _ = c.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n");
                let words: Vec<&str> = cont.split_inclusive(' ').collect();
                for (i, w) in words.iter().enumerate() {
                    let ev = format!(
                        "data: {}\n\n",
                        serde_json::json!({"content": w, "stop": i + 1 == words.len()})
                    );
                    let _ = c.write_all(format!("{:x}\r\n{ev}\r\n", ev.len()).as_bytes());
                    std::thread::sleep(Duration::from_millis(2));
                }
                let _ = c.write_all(b"0\r\n\r\n");
            }
        });
        addr
    }

    fn word_tokenizer() -> Arc<Tokenizer> {
        // Whitespace-split word-level vocabulary: enough to exercise the bookkeeping.
        let words = [
            "[UNK]", "the", "tide", "rises", "twice", "a", "day", "moon", "pulls", "water", "why",
        ];
        let vocab: serde_json::Map<String, serde_json::Value> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), serde_json::json!(i)))
            .collect();
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": {"type": "Fuse"},
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
        });
        Arc::new(Tokenizer::from_bytes(json.to_string().as_bytes()).expect("tokenizer"))
    }

    fn ids(tok: &Tokenizer, s: &str) -> Vec<i64> {
        tok.encode(s, false)
            .expect("encode")
            .get_ids()
            .iter()
            .map(|&t| t as i64)
            .collect()
    }

    #[test]
    fn follows_the_draft_and_restarts_after_a_wrong_guess() {
        let tok = word_tokenizer();
        let addr = fake_server(vec![
            ("assistant\n", "the tide rises twice a day"),
            ("moon", "pulls the water"),
        ]);
        let cfg = LmConfig {
            addr,
            template: LmTemplate::Qwen,
            n_predict: 16,
        };
        let mut lm = LmLink::new(cfg, tok.clone(), &ids(&tok, "why"));
        let wait = Duration::from_millis(500);
        let mut out: Vec<i64> = Vec::new();
        // The drafter's words come back one by one as target tokens.
        for w in ["the", "tide", "rises"] {
            let g = lm.next(&out, wait).expect("a guess");
            assert_eq!(g, ids(&tok, w)[0]);
            out.push(g);
        }
        // The target says "moon" instead of "twice": the drafter starts over from there.
        out.push(ids(&tok, "moon")[0]);
        let g = lm.next(&out, wait).expect("a guess after the restart");
        assert_eq!(g, ids(&tok, "pulls")[0]);
        let (started, _, failed) = lm.stats();
        assert_eq!((started, failed), (2, 0));
    }

    #[test]
    fn a_missing_server_costs_nothing_but_the_guess() {
        let tok = word_tokenizer();
        let cfg = LmConfig {
            addr: "127.0.0.1:9".into(),
            template: LmTemplate::Plain,
            n_predict: 8,
        };
        let mut lm = LmLink::new(cfg, tok.clone(), &ids(&tok, "why"));
        let t0 = Instant::now();
        assert!(lm.next(&[], Duration::from_millis(30)).is_none());
        assert!(t0.elapsed() < Duration::from_millis(900));
    }
}
