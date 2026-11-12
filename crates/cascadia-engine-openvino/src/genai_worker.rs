//! One stateful ov-genai generation, and the thread that runs them while the engine streams.
//!
//! [`run_job`] is the whole generation of one task: it counts the prompt, runs the polled
//! `generate()`, answers each streamer write (cancel, `stop`, and with streaming the text handed
//! out), then counts the visible answer. The engine runs it inline inside `step()` when it does not
//! stream (one final chunk, as before), or on a [`StatefulWorker`] thread when it does, so `step()`
//! can return the text while the generation runs (one write late, see [`crate::genai_stream`]).

use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cascadia_engine::{InflightCancel, OneShotStop, VisibleArm};
use cascadia_ov_genai_shim::{Error as OvError, GenConfig, PollAction, PollReply, PollWrite};
use cascadia_types::{GenerationTask, SamplingParams};

use crate::genai::{build_gen_config, stateful_max_new, StatefulPipe};
use crate::genai_stream::{visible_text, Finish, StreamText};

/// The stateful pipeline, shared by the engine (warmup, inline generations) and its worker thread.
/// One generation at a time holds it.
pub(crate) type SharedPipe = Arc<Mutex<Box<dyn StatefulPipe>>>;

/// Lock the pipeline. A panic while it was held (a job's, caught by the worker) leaves nothing
/// half-updated on the Rust side, so a poisoned lock is still used.
pub(crate) fn lock_pipe(pipe: &SharedPipe) -> MutexGuard<'_, Box<dyn StatefulPipe>> {
    pipe.lock().unwrap_or_else(|e| e.into_inner())
}

/// The engine's settings a generation needs.
#[derive(Clone, Debug)]
pub(crate) struct GenParams {
    pub speculative_k: u32,
    pub prompt_lookup_ngram: u32,
    pub prompt_pretemplated: bool,
    pub max_tokens_default: u32,
    /// The server's known context window: one `generate()` call makes the
    /// whole answer, so its limit is cut to what the prompt leaves of it.
    pub context_window: Option<u32>,
    /// End a running generation at a `stop` match.
    pub engine_stop_check: bool,
    /// Hand out the answer's text as it is written, one write late ([`Event::Delta`]).
    pub stream_text: bool,
}

impl GenParams {
    /// SEED CAVEAT (validated on-HW, OpenVINO GenAI 2026.1; not re-checked on
    /// the current 2026.2 floor): `seed` is forwarded to
    /// `GenerationConfig.rng_seed`, but it does NOT make ov-genai reproducible
    /// across requests on this reused `LLMPipeline`. OV-GenAI's
    /// `StatefulLLMPipeline` only re-seeds when the seed VALUE CHANGES between
    /// consecutive `generate()` calls (`if (m_sampler.get_seed() !=
    /// config.rng_seed) set_seed(...)`) and never calls `clear_request_info`,
    /// so two identical seeds in a row reuse the advanced RNG and diverge.
    /// Different seeds DO change the output, and the Rust-sampler engines
    /// (ov-runtime/sparse-moe) honor seed fully — they re-seed per request.
    ///
    /// This is a property of `StatefulLLMPipeline`, so it does not describe
    /// [`crate::OvGenaiCbEngine`]: `ContinuousBatchingPipeline` builds a fresh
    /// request context per request. Whether that yields per-request
    /// reproducibility in practice is untested here.
    pub(crate) fn gen_config(
        &self,
        max_tokens: u32,
        temperature: f32,
        enable_thinking: bool,
        sampling: &SamplingParams,
    ) -> GenConfig {
        build_gen_config(
            self.speculative_k,
            self.prompt_lookup_ngram,
            self.prompt_pretemplated,
            max_tokens,
            temperature,
            enable_thinking,
            sampling,
        )
    }
}

/// What a running generation reports before its [`Done`].
#[derive(Debug)]
pub(crate) enum Event {
    /// The prompt is counted and the generation starts.
    Started { prompt_tokens: Option<u32> },
    /// Text handed out, and the tokens it stands for (held writes' included).
    Delta { text: String, n_tokens: usize },
    /// The generation is over.
    Done(Box<Done>),
}

/// How one generation ended.
#[derive(Debug)]
pub(crate) struct Done {
    pub prompt_tokens: Option<u32>,
    /// Tokens the streamer wrote, for the `task cancelled` log line.
    pub polled_tokens: u64,
    pub elapsed_s: f64,
    /// Whether ov-genai ran the streamer on a thread of its own, when this job probed it.
    pub threaded_streamer: Option<bool>,
    pub outcome: Outcome,
}

#[derive(Debug)]
pub(crate) enum Outcome {
    /// The cancel flag was up when `generate()` returned (also after a failure). Streaming: the
    /// tokens of the writes whose text never went out, for the cancelled final.
    Cancelled {
        held_tokens: usize,
    },
    /// `generate()` failed, or the job panicked.
    Failed(OvError),
    Finished(Finished),
}

#[derive(Debug)]
pub(crate) struct Finished {
    /// The answer: the decode with an echoed prompt stripped and trimmed.
    pub visible: String,
    /// The tokenizer's count of `visible`, else ov-genai's generated count.
    pub completion_tokens: u32,
    /// The answer limit the generation ran with.
    pub max_new: u32,
    pub stopped_on_match: bool,
    /// Stopped on a match `visible` does not hold (the stop-text invariant broke).
    pub stop_missing: bool,
    pub ttft_ms: Option<f64>,
    /// With streaming: what is left to send.
    pub stream: Option<Finish>,
}

impl Done {
    fn panicked(message: String) -> Self {
        Self {
            prompt_tokens: None,
            polled_tokens: 0,
            elapsed_s: 0.0,
            threaded_streamer: None,
            outcome: Outcome::Failed(OvError::Native(format!(
                "ov-genai generation panicked: {message}"
            ))),
        }
    }
}

/// Run `task`'s whole generation on `pipe`. `emit` gets [`Event::Started`] and, with
/// [`GenParams::stream_text`], each [`Event::Delta`]; the [`Done`] is returned.
///
/// The `stop` check runs on text the visible answer keeps. Without streaming the shim decodes
/// the newest ids per write (`tail_tokens`) and [`VisibleArm`] arms the check once that decode lies
/// inside the visible text. With streaming no tail is decoded: the check runs on each write's
/// newest visible text ([`crate::genai_stream::Written::fresh`], without the delta's one write of
/// lag), so a stop split across writes is found and a stop inside a stripped echo is not.
pub(crate) fn run_job(
    pipe: &dyn StatefulPipe,
    params: &GenParams,
    inflight: &InflightCancel,
    task: &GenerationTask,
    probe_thread: bool,
    emit: &mut (dyn FnMut(Event) + Send),
) -> Done {
    let max_new = if task.max_tokens > 0 {
        task.max_tokens
    } else {
        params.max_tokens_default
    };
    let prompt_tokens = pipe.count_tokens(&task.prompt);
    emit(Event::Started { prompt_tokens });
    // The pipeline does not stop at the window itself; capped here, the
    // `tokens >= max_new` inference reports `length` there.
    let max_new = stateful_max_new(max_new, prompt_tokens, params.context_window);
    let cfg = params.gen_config(
        max_new,
        task.temperature,
        task.enable_thinking,
        &task.sampling,
    );

    let started = Instant::now();
    // Always the polled generate, so a cancel always works.
    let mut stop = if params.engine_stop_check {
        OneShotStop::new(&task.sampling.stop)
    } else {
        OneShotStop::new(&[])
    };
    // Without streaming, the shim decodes the last `tail_tokens` generated ids
    // per write and the check runs on that suffix decode: it is a substring of
    // the returned text (OneShotStop's input contract), unlike streamer
    // pieces. The +16 covers a run of 0-byte (special) tokens. A draft
    // model or prompt-lookup writes a whole speculative round at once
    // (up to `num_assistant_tokens` + 1 ids), so the window grows by that
    // round too, as sparse-moe's `extra_slack` does. 0 = no decode at all
    // when nothing is checked, and when streaming (the check runs on each
    // write's newest visible text).
    let round_slack = if cfg.num_assistant_tokens > 0 {
        cfg.num_assistant_tokens as usize + 1
    } else {
        0
    };
    let tail_tokens = if stop.is_active() && !params.stream_text {
        stop.max_stop_bytes() + 16 + round_slack
    } else {
        0
    };
    // Arm the tail check only on text the visible answer keeps: an
    // engine stop inside a stripped prompt echo or leading whitespace
    // would give a different answer than a run with the check off.
    let mut arm = VisibleArm::new(&task.prompt, tail_tokens);
    let mut stream = params.stream_text.then(|| StreamText::new(&task.prompt));
    let mut polled_tokens: u64 = 0;
    let mut stopped_on_match = false;
    // Which thread ov-genai calls the streamer on, probed on the first
    // poll of the engine's life only: the thread running this job (the
    // stateful backend) or its own worker thread (the paged-attention / CB
    // adapter, speculative pipelines). The engine logs it.
    let job_thread = std::thread::current().id();
    let mut probe = probe_thread;
    let mut threaded_streamer: Option<bool> = None;
    let generated = {
        // The answer the closure has given once it ended the generation:
        // the shim polls once more at the end (n_tokens 0), and that call
        // must not decide again.
        let mut ended: Option<PollAction> = None;
        let mut poll = |w: PollWrite<'_>| -> PollReply {
            if probe {
                probe = false;
                threaded_streamer = Some(std::thread::current().id() != job_thread);
            }
            polled_tokens += w.n_tokens as u64;
            if let Some(action) = ended {
                return action.into();
            }
            if inflight.cancelled() {
                ended = Some(PollAction::Cancel);
                return PollAction::Cancel.into();
            }
            let hit = match stream.as_mut() {
                Some(s) => {
                    let written = s.on_write(&w);
                    if let Some((text, n_tokens)) = written.delta {
                        emit(Event::Delta { text, n_tokens });
                    }
                    // On the newest text, not one write late: the stop ends
                    // the generation at the write that completes it.
                    stop.is_active() && !written.fresh.is_empty() && stop.push(&written.fresh)
                }
                None => {
                    tail_tokens > 0
                        && w.n_tokens > 0
                        && arm.armed(polled_tokens as usize, w.tail)
                        && stop.matches_tail(w.tail)
                }
            };
            if hit {
                stopped_on_match = true;
                ended = Some(PollAction::Stop);
                return PollAction::Stop.into();
            }
            // Never a `confirm_upto`: a window decoded from a later id can
            // lose a leading space, so every window starts at the first id.
            PollAction::Running.into()
        };
        pipe.generate_polled(
            &task.prompt,
            &cfg,
            tail_tokens,
            params.stream_text,
            &mut poll,
        )
    };
    let elapsed_s = started.elapsed().as_secs_f64();
    let done = |outcome| Done {
        prompt_tokens,
        polled_tokens,
        elapsed_s,
        threaded_streamer,
        outcome,
    };
    if inflight.cancelled() {
        // Also when generate failed after the CANCEL: the task ends as
        // cancelled, never as an error or a partial success.
        let held_tokens = stream.as_ref().map_or(0, StreamText::held_tokens);
        return done(Outcome::Cancelled { held_tokens });
    }
    let result = match generated {
        Ok(r) => r,
        // A panic in the poll closure lands here too (`err.is_poll_panic()`):
        // it cancelled the generation, and its partial text must not read as
        // a success.
        Err(err) => return done(Outcome::Failed(err)),
    };
    let visible = visible_text(&result.text, &task.prompt).to_string();
    // The check ran only on text this visible answer keeps, so it holds the
    // stop. If it doesn't, the stop-text invariant (a window decode is a
    // substring of the final decode) broke: a detokenizer cleanup rule
    // (`clean_up_tokenization_spaces`) can rewrite text based on later tokens.
    let stop_missing = stopped_on_match && !OneShotStop::new(&task.sampling.stop).push(&visible);
    // Prefer the tokenizer's count; fall back to perf_metrics' value.
    let completion_tokens = pipe
        .count_tokens(&visible)
        .unwrap_or(result.generated_tokens);
    let stream = stream.map(|mut s| s.finish(&visible));
    done(Outcome::Finished(Finished {
        visible,
        completion_tokens,
        max_new,
        stopped_on_match,
        stop_missing,
        ttft_ms: result.ttft_ms,
        stream,
    }))
}

/// One generation for the worker.
pub(crate) struct GenJob {
    pub task: GenerationTask,
    pub params: GenParams,
    pub probe_thread: bool,
}

pub(crate) enum Job {
    Generate(Box<GenJob>),
    Shutdown,
}

/// What the engine reads next from its worker.
pub(crate) enum Recv {
    Event(Event),
    /// Nothing yet.
    Idle,
    /// The thread is gone.
    Gone,
}

/// The thread that runs the engine's generations while it streams: one job at a time, each under
/// `catch_unwind`, so a panic costs that task (a [`Done`] with an error), not the engine.
pub(crate) struct StatefulWorker {
    jobs: Sender<Job>,
    events: Receiver<Event>,
    handle: Option<JoinHandle<()>>,
}

impl StatefulWorker {
    pub(crate) fn spawn(pipe: SharedPipe, inflight: Arc<InflightCancel>) -> std::io::Result<Self> {
        let (jobs, job_rx) = mpsc::channel();
        let (event_tx, events) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("ov-genai-stateful".into())
            .spawn(move || worker_loop(&pipe, &inflight, &job_rx, &event_tx))?;
        Ok(Self {
            jobs,
            events,
            handle: Some(handle),
        })
    }

    /// A worker whose thread is already gone (tests of a lost worker).
    #[cfg(test)]
    pub(crate) fn gone() -> Self {
        let (jobs, _) = mpsc::channel();
        let (_, events) = mpsc::channel();
        Self {
            jobs,
            events,
            handle: None,
        }
    }

    /// Hand the thread a job; the job comes back when the thread is gone.
    pub(crate) fn send(&self, job: Job) -> Result<(), Job> {
        self.jobs.send(job).map_err(|e| e.0)
    }

    /// The next event, waiting up to `wait` for it (`None`: not at all).
    pub(crate) fn next(&self, wait: Option<Duration>) -> Recv {
        match wait {
            Some(wait) => match self.events.recv_timeout(wait) {
                Ok(ev) => Recv::Event(ev),
                Err(RecvTimeoutError::Timeout) => Recv::Idle,
                Err(RecvTimeoutError::Disconnected) => Recv::Gone,
            },
            None => match self.events.try_recv() {
                Ok(ev) => Recv::Event(ev),
                Err(TryRecvError::Empty) => Recv::Idle,
                Err(TryRecvError::Disconnected) => Recv::Gone,
            },
        }
    }
}

impl Drop for StatefulWorker {
    /// Ends the thread after its current job (the engine has signalled that job's cancel first)
    /// and waits for it.
    fn drop(&mut self) {
        let _ = self.jobs.send(Job::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(
    pipe: &SharedPipe,
    inflight: &InflightCancel,
    jobs: &Receiver<Job>,
    events: &Sender<Event>,
) {
    while let Ok(job) = jobs.recv() {
        let job = match job {
            Job::Generate(job) => job,
            Job::Shutdown => return,
        };
        let mut emit = |ev: Event| {
            // The engine dropped its end: nothing is waiting for this task.
            let _ = events.send(ev);
        };
        let done = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let pipe = lock_pipe(pipe);
            run_job(
                &**pipe,
                &job.params,
                inflight,
                &job.task,
                job.probe_thread,
                &mut emit,
            )
        }))
        .unwrap_or_else(|payload| Done::panicked(panic_message(payload.as_ref())));
        let _ = events.send(Event::Done(Box::new(done)));
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "(no message)".to_string()
    }
}
