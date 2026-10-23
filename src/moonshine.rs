//! Moonshine Tiny Streaming through the pinned native C library.
//!
//! The library is loaded at runtime with `libloading`; no Moonshine or
//! ONNX Runtime development files are needed to build Dictate. A stream is
//! created once per recording and fed microphone audio while it arrives. The
//! C API returns append-only transcript lines: a line may be revised while it
//! is incomplete, and its text/timing are stable once it is complete.
//!
//! Only complete lines are returned for typing. The last, still-incomplete
//! line is returned separately as [`Update::partial`] so it can be surfaced
//! (currently the daemon logs it) without ever typing provisional text into the
//! focused application. On stop, one forced transcribe call drains the audio;
//! [`LineCursor::finish`] commits anything still open so the final phrase is
//! not lost. Cancel drops the stream without committing anything.

use std::ffi::{CStr, CString, c_char, c_float};
use std::path::Path;
use std::sync::Once;

use anyhow::Context;
use libloading::Library;

use crate::config::InferenceThreads;
use crate::model;

/// Header version this binding was written against. The pinned v0.1.5 runtime
/// reports the same version and emulates older clients if it is ever newer.
const MOONSHINE_HEADER_VERSION: i32 = 30_000;
/// `MOONSHINE_MODEL_ARCH_TINY_STREAMING`.
const MOONSHINE_MODEL_ARCH_TINY_STREAMING: u32 = 2;
/// `MOONSHINE_FLAG_FORCE_UPDATE`.
const MOONSHINE_FLAG_FORCE_UPDATE: u32 = 1;
/// Moonshine expects 16 kHz mono float PCM.
const SAMPLE_RATE: i32 = 16_000;

/// How many inference threads Moonshine will actually use. The pinned v0.1.5
/// C API has no per-session thread argument, so it can only request the
/// runtime's documented single-thread mode or leave ONNX Runtime's own
/// default in place; "exactly two" is not available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Threads {
    /// Set `MOONSHINE_ORT_SINGLE_THREAD` once before the runtime creates its
    /// sessions. Auto picks this for its best measured responsiveness; the
    /// explicit 1 setting does the same.
    Single,
    /// Explicit 2 leaves ONNX Runtime's physical-core default alone. That is
    /// usually more than two threads on a modern CPU, not an exact 2.
    RuntimeDefault,
}

impl Threads {
    pub(crate) fn for_setting(setting: InferenceThreads) -> Self {
        match setting {
            InferenceThreads::Auto | InferenceThreads::One => Self::Single,
            InferenceThreads::Two => Self::RuntimeDefault,
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Single => "single-thread mode",
            Self::RuntimeDefault => {
                "ONNX Runtime default threads (not exactly 2: the pinned API has no per-session thread count)"
            }
        }
    }
}

/// One complete (or provisional) line from the native transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StreamLine {
    pub text: String,
    pub is_complete: bool,
}

/// Text the worker should type now, plus the provisional tail for display.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Update {
    /// Newly completed phrase text. Joined in order; a leading space is added
    /// between typed chunks except before punctuation. Empty when nothing new
    /// completed.
    pub completed: String,
    /// Number of complete lines committed by this update.
    pub completed_lines: usize,
    /// Still-incomplete final line, if any. Never typed by the worker.
    pub partial: Option<String>,
}

/// Remembers which append-only lines were already committed to the focused
/// application, so a revised provisional line is never typed twice.
#[derive(Debug, Default)]
pub(crate) struct LineCursor {
    next: usize,
    typed_anything: bool,
}

impl LineCursor {
    pub(crate) fn reset(&mut self) {
        self.next = 0;
        self.typed_anything = false;
    }

    /// Commit every complete line not committed yet. A complete line's text is
    /// stable, so it is typed once. The last incomplete line is returned as a
    /// provisional tail and left for a later call.
    pub(crate) fn commit(&mut self, lines: &[StreamLine]) -> Update {
        let mut update = Update::default();
        for line in lines.iter().skip(self.next) {
            if !line.is_complete {
                break;
            }
            self.next += 1;
            let text = line.text.trim();
            if text.is_empty() {
                continue;
            }
            append_phrase(&mut update.completed, &mut self.typed_anything, text);
            update.completed_lines += 1;
        }
        update.partial = lines
            .get(self.next)
            .filter(|line| !line.is_complete)
            .map(|line| line.text.trim().to_string())
            .filter(|text| !text.is_empty());
        update
    }

    /// Commit all remaining lines regardless of the incomplete flag. Only for
    /// the final call after `stop` has drained the stream: at that point the
    /// text is final, and this prevents a short last phrase from being lost.
    pub(crate) fn finish(&mut self, lines: &[StreamLine]) -> Update {
        let mut update = Update::default();
        for line in lines.iter().skip(self.next) {
            self.next += 1;
            let text = line.text.trim();
            if text.is_empty() {
                continue;
            }
            append_phrase(&mut update.completed, &mut self.typed_anything, text);
            update.completed_lines += 1;
        }
        update
    }
}

/// Native operations the recording lifecycle drives. [`Moonshine`] is the
/// production implementation; the unit tests use a fake to verify the exact
/// start/feed/stop/cancel sequencing without loading a model.
pub(crate) trait StreamBackend {
    fn start(&mut self) -> anyhow::Result<()>;
    fn feed(&mut self, samples: &[f32]) -> anyhow::Result<Vec<StreamLine>>;
    fn stop(&mut self) -> anyhow::Result<Vec<StreamLine>>;
    fn cancel(&mut self);
    fn close(&mut self);
}

/// One complete recording over a stateful stream backend. This is the single
/// lifecycle used by the daemon worker and the `dictate transcribe
/// --moonshine` test hook: the native stream is opened lazily on the first
/// audio chunk, finalized exactly once on [`finish`](Self::finish), and
/// discarded on [`cancel`](Self::cancel). A session can be reused for the next
/// recording; its line cursor is reset so no text crosses recordings.
pub(crate) struct Session<B: StreamBackend> {
    backend: B,
    cursor: LineCursor,
    open: bool,
}

impl<B: StreamBackend> Session<B> {
    pub(crate) fn new(backend: B) -> Self {
        Self {
            backend,
            cursor: LineCursor::default(),
            open: false,
        }
    }

    /// Feed one chunk and commit newly completed lines. The first chunk opens
    /// a fresh native stream; an error discards the stream and cursor so the
    /// next recording starts clean.
    pub(crate) fn feed(&mut self, samples: &[f32]) -> anyhow::Result<Update> {
        if !self.open {
            if let Err(err) = self.backend.start() {
                self.cursor.reset();
                return Err(err);
            }
            self.open = true;
        }
        match self.backend.feed(samples) {
            Ok(lines) => Ok(self.cursor.commit(&lines)),
            Err(err) => {
                self.backend.cancel();
                self.cursor.reset();
                self.open = false;
                Err(err)
            }
        }
    }

    /// Drain the last audio, commit every remaining line (including the final
    /// incomplete one), and close the stream. A recording with no audio chunks
    /// is a quiet success: no stream is opened and the update is empty.
    pub(crate) fn finish(&mut self) -> anyhow::Result<Update> {
        if !self.open {
            self.cursor.reset();
            return Ok(Update::default());
        }
        match self.backend.stop() {
            Ok(lines) => {
                let update = self.cursor.finish(&lines);
                self.backend.close();
                self.cursor.reset();
                self.open = false;
                Ok(update)
            }
            Err(err) => {
                self.backend.cancel();
                self.cursor.reset();
                self.open = false;
                Err(err)
            }
        }
    }

    /// Discard an in-flight recording. Nothing from it may reach the next
    /// recording, whether it was already returned by `feed` or still open.
    pub(crate) fn cancel(&mut self) {
        if self.open {
            self.backend.cancel();
            self.open = false;
        }
        self.cursor.reset();
    }
}

/// Add one phrase to the output, inserting the single space Dictate's other
/// modes use between phrases. A phrase that starts with punctuation attaches
/// to the previous one.
fn append_phrase(out: &mut String, typed_anything: &mut bool, text: &str) {
    let Some(first) = text.chars().next() else {
        return;
    };
    if *typed_anything && !matches!(first, '.' | ',' | '!' | '?' | ';' | ':' | ')' | ']') {
        out.push(' ');
    }
    out.push_str(text);
    *typed_anything = true;
}

#[repr(C)]
struct MoonshineOption {
    name: *const c_char,
    value: *const c_char,
}

#[repr(C)]
struct SpeakerSpan {
    start_time: c_float,
    duration: c_float,
    speaker_id: u64,
    speaker_index: u32,
    start_char: u64,
    end_char: u64,
}

#[repr(C)]
struct TranscriptWord {
    text: *const c_char,
    start: c_float,
    end: c_float,
    confidence: c_float,
}

/// Full C layout, including fields this binding does not read, because the
/// transcript's `lines` array has this stride.
#[repr(C)]
struct TranscriptLine {
    text: *const c_char,
    audio_data: *const c_float,
    audio_data_count: usize,
    start_time: c_float,
    duration: c_float,
    id: u64,
    is_complete: i8,
    is_updated: i8,
    is_new: i8,
    has_text_changed: i8,
    have_speakers_changed: i8,
    speaker_spans: *const SpeakerSpan,
    speaker_span_count: u64,
    last_transcription_latency_ms: u32,
    words: *const TranscriptWord,
    word_count: u64,
}

#[repr(C)]
struct Transcript {
    lines: *mut TranscriptLine,
    line_count: u64,
}

type GetVersion = unsafe extern "C" fn() -> i32;
type ErrorToString = unsafe extern "C" fn(i32) -> *const c_char;
type LoadTranscriber =
    unsafe extern "C" fn(*const c_char, u32, *const MoonshineOption, u64, i32) -> i32;
type FreeTranscriber = unsafe extern "C" fn(i32);
type CreateStream = unsafe extern "C" fn(i32, u32) -> i32;
type StartStream = unsafe extern "C" fn(i32, i32) -> i32;
type StopStream = unsafe extern "C" fn(i32, i32) -> i32;
type FreeStream = unsafe extern "C" fn(i32, i32) -> i32;
type AddAudio = unsafe extern "C" fn(i32, i32, *const c_float, u64, i32, u32) -> i32;
type TranscribeStream = unsafe extern "C" fn(i32, i32, u32, *mut *mut Transcript) -> i32;

struct Api {
    version: GetVersion,
    error_to_string: ErrorToString,
    load_transcriber: LoadTranscriber,
    free_transcriber: FreeTranscriber,
    create_stream: CreateStream,
    start_stream: StartStream,
    stop_stream: StopStream,
    free_stream: FreeStream,
    add_audio: AddAudio,
    transcribe_stream: TranscribeStream,
}

/// The loaded runtime and live stream. Owned by the daemon worker thread.
pub struct Moonshine {
    _library: Library,
    api: Api,
    transcriber: i32,
    stream: Option<i32>,
    stopped: bool,
}

impl Moonshine {
    /// Load the pinned runtime and model. `threads` describes the policy
    /// selected before process start; the pinned C API has no per-session
    /// thread-count argument. Single-thread mode is selected once at process
    /// start through the library's documented `MOONSHINE_ORT_SINGLE_THREAD`
    /// flag (see [`configure_threads`]).
    pub fn load(model_dir: &Path, threads: Threads) -> anyhow::Result<Self> {
        let library_path = model::moonshine_library_path()?;
        // SAFETY: loading a shared library is inherently unsafe; the pinned
        // archive and the symbols fetched below are checked against header
        // version 30000 and only called through their declared C signatures.
        let library = unsafe { Library::new(&library_path) }.with_context(|| {
            format!(
                "Could not load the Moonshine runtime at {}. Run \"dictate download --moonshine\" first",
                library_path.display()
            )
        })?;
        let api = unsafe { Api::from_library(&library) }?;
        // SAFETY: no preconditions; the function takes no arguments.
        let version = unsafe { (api.version)() };
        if version != MOONSHINE_HEADER_VERSION {
            anyhow::bail!(
                "The Moonshine runtime at {} reports version {version}, but this build expects {MOONSHINE_HEADER_VERSION}. Run \"dictate download --moonshine\" to reinstall the pinned v0.1.5 runtime",
                library_path.display()
            );
        }
        let model_dir = CString::new(model_dir.as_os_str().as_encoded_bytes())
            .context("The Moonshine model path contains an interior NUL byte")?;
        // SAFETY: `model_dir` is a valid NUL-terminated path that outlives the
        // call; there are no options and the header version is pinned.
        let transcriber = unsafe {
            (api.load_transcriber)(
                model_dir.as_ptr(),
                MOONSHINE_MODEL_ARCH_TINY_STREAMING,
                std::ptr::null(),
                0,
                MOONSHINE_HEADER_VERSION,
            )
        };
        if transcriber < 0 {
            anyhow::bail!(
                "Loading the Moonshine model from {} failed: {}",
                model_dir.to_string_lossy(),
                unsafe { api.error_text(transcriber) }
            );
        }
        eprintln!(
            "moonshine: runtime {} loaded ({})",
            version,
            threads.description()
        );
        Ok(Self {
            _library: library,
            api,
            transcriber,
            stream: None,
            stopped: false,
        })
    }

    fn drop_stream(&mut self) {
        if let Some(stream) = self.stream.take() {
            // SAFETY: the stream handle is open and the transcriber is valid.
            unsafe {
                let _ = (self.api.free_stream)(self.transcriber, stream);
            }
        }
    }

    fn open_stream(&self) -> anyhow::Result<i32> {
        self.stream
            .ok_or_else(|| anyhow::anyhow!("Internal error: no Moonshine stream is open"))
    }

    fn transcribe(&self, flags: u32) -> anyhow::Result<Vec<StreamLine>> {
        let stream = self.open_stream()?;
        let mut transcript = std::ptr::null_mut();
        // SAFETY: the handles are valid, `transcript` is a valid out-pointer,
        // and the returned memory is only read before the next API call.
        let status = unsafe {
            (self.api.transcribe_stream)(self.transcriber, stream, flags, &mut transcript)
        };
        self.check(status, "Transcribing the Moonshine stream")?;
        if transcript.is_null() {
            return Ok(Vec::new());
        }
        // SAFETY: on success the library owns a valid transcript for this
        // stream until the next call; line text is copied immediately.
        let lines = unsafe { &*transcript };
        if lines.lines.is_null() || lines.line_count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: the library reports `line_count` contiguous lines and owns
        // their text until the next API call.
        Ok(unsafe { copy_stream_lines(lines.lines, lines.line_count) })
    }

    fn check(&self, status: i32, action: &str) -> anyhow::Result<()> {
        if status == 0 {
            return Ok(());
        }
        anyhow::bail!("{action} failed: {}", unsafe {
            self.api.error_text(status)
        })
    }
}

/// Copy native transcript lines into owned values. A line with a null text
/// pointer keeps its position as an empty placeholder: [`LineCursor`] indexes
/// lines by native position, so dropping the line would shift later lines and
/// break the stable index contract. The placeholder is never typed (empty
/// text), but the cursor still advances past it.
///
/// # Safety
///
/// `lines` must point to `count` contiguous `TranscriptLine` values whose
/// `text` pointers are either null or valid NUL-terminated strings readable
/// for the duration of this call.
unsafe fn copy_stream_lines(lines: *const TranscriptLine, count: u64) -> Vec<StreamLine> {
    let mut out = Vec::with_capacity(count as usize);
    for index in 0..count as usize {
        // SAFETY: the caller guarantees `count` contiguous lines.
        let line = unsafe { &*lines.add(index) };
        let text = if line.text.is_null() {
            String::new()
        } else {
            // SAFETY: non-null text is a valid NUL-terminated string.
            unsafe { CStr::from_ptr(line.text) }
                .to_string_lossy()
                .to_string()
        };
        out.push(StreamLine {
            text,
            is_complete: line.is_complete != 0,
        });
    }
    out
}

impl StreamBackend for Moonshine {
    /// Start a fresh stateful stream. `Session` only calls this when no stream
    /// is open, and a fresh one is always created after finish/cancel.
    fn start(&mut self) -> anyhow::Result<()> {
        if self.stream.is_some() {
            anyhow::bail!("Internal error: a Moonshine stream is already open");
        }
        // SAFETY: the transcriber handle is valid until Drop; flags are 0.
        let stream = unsafe { (self.api.create_stream)(self.transcriber, 0) };
        if stream < 0 {
            anyhow::bail!("Could not create a Moonshine stream: {}", unsafe {
                self.api.error_text(stream)
            });
        }
        // SAFETY: both handles are valid; start returns an error code.
        let status = unsafe { (self.api.start_stream)(self.transcriber, stream) };
        if status < 0 {
            unsafe {
                let _ = (self.api.free_stream)(self.transcriber, stream);
            }
            anyhow::bail!("Could not start a Moonshine stream: {}", unsafe {
                self.api.error_text(status)
            });
        }
        self.stream = Some(stream);
        self.stopped = false;
        Ok(())
    }

    /// Feed one chunk of 16 kHz mono audio and return any newly completed
    /// text. The C API caches analyses internally, so calling this for every
    /// 100 ms chunk does not repeatedly transcribe a growing buffer.
    fn feed(&mut self, samples: &[f32]) -> anyhow::Result<Vec<StreamLine>> {
        let stream = self.open_stream()?;
        if !samples.is_empty() {
            // SAFETY: `samples` is a valid slice and `stream` is open.
            let status = unsafe {
                (self.api.add_audio)(
                    self.transcriber,
                    stream,
                    samples.as_ptr(),
                    samples.len() as u64,
                    SAMPLE_RATE,
                    0,
                )
            };
            self.check(status, "Adding audio to the Moonshine stream")?;
        }
        self.transcribe(0)
    }

    /// Stop the stream and run one forced analysis, which drains all leftover
    /// audio, including a final chunk shorter than the normal 200 ms cache.
    fn stop(&mut self) -> anyhow::Result<Vec<StreamLine>> {
        let Some(stream) = self.stream else {
            return Ok(Vec::new());
        };
        if !self.stopped {
            // SAFETY: `stream` is open and the transcriber is valid.
            let status = unsafe { (self.api.stop_stream)(self.transcriber, stream) };
            self.check(status, "Stopping the Moonshine stream")?;
            self.stopped = true;
        }
        self.transcribe(MOONSHINE_FLAG_FORCE_UPDATE)
    }

    /// Discard an open stream without finalizing it. Used by cancel and by a
    /// mode switch that drops the model; no provisional text is committed.
    fn cancel(&mut self) {
        self.drop_stream();
        self.stopped = false;
    }

    /// Close a stopped stream after its final transcript was consumed.
    fn close(&mut self) {
        self.drop_stream();
    }
}

impl Drop for Moonshine {
    fn drop(&mut self) {
        self.drop_stream();
        // SAFETY: the transcriber handle was returned by load and is freed
        // exactly once, before the library is unloaded.
        unsafe {
            (self.api.free_transcriber)(self.transcriber);
        }
    }
}

impl Api {
    unsafe fn from_library(library: &Library) -> anyhow::Result<Self> {
        macro_rules! symbol {
            ($name:literal, $type:ty) => {{
                // SAFETY: the caller guarantees the library is loaded and the
                // pinned runtime exports these C symbols.
                let symbol: libloading::Symbol<'_, $type> =
                    unsafe { library.get(concat!($name, "\0").as_bytes()) }.with_context(|| {
                        format!(
                            "The Moonshine runtime is missing the {} symbol. Run \"dictate download --moonshine\" to reinstall it",
                            $name
                        )
                    })?;
                *symbol
            }};
        }
        Ok(Self {
            version: symbol!("moonshine_get_version", GetVersion),
            error_to_string: symbol!("moonshine_error_to_string", ErrorToString),
            load_transcriber: symbol!("moonshine_load_transcriber_from_files", LoadTranscriber),
            free_transcriber: symbol!("moonshine_free_transcriber", FreeTranscriber),
            create_stream: symbol!("moonshine_create_stream", CreateStream),
            start_stream: symbol!("moonshine_start_stream", StartStream),
            stop_stream: symbol!("moonshine_stop_stream", StopStream),
            free_stream: symbol!("moonshine_free_stream", FreeStream),
            add_audio: symbol!("moonshine_transcribe_add_audio_to_stream", AddAudio),
            transcribe_stream: symbol!("moonshine_transcribe_stream", TranscribeStream),
        })
    }

    unsafe fn error_text(&self, code: i32) -> String {
        // SAFETY: the runtime returns a static NUL-terminated message.
        let pointer = unsafe { (self.error_to_string)(code) };
        if pointer.is_null() {
            return format!("error {code}");
        }
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .to_string()
    }
}

/// Select Moonshine's execution policy once, before any worker thread exists:
/// the pinned C API has no per-session thread argument, and the native runtime
/// reads `MOONSHINE_ORT_SINGLE_THREAD` when it creates its ONNX Runtime
/// sessions. Auto and 1 request true single-thread mode (Auto is the better
/// measured default); explicit 2 leaves ONNX Runtime's conservative
/// physical-core default in place, which is not an exact two. Whisper uses its
/// own exact 1/2 API setting.
pub(crate) fn configure_threads(threads: Threads) {
    static CONFIGURED: Once = Once::new();
    CONFIGURED.call_once(|| {
        // SAFETY: called on the main thread before model loading or other
        // threads start, so no concurrent environment access can race. The
        // app setting is authoritative over a stale flag inherited from the
        // parent environment.
        unsafe {
            match threads {
                Threads::Single => std::env::set_var("MOONSHINE_ORT_SINGLE_THREAD", "1"),
                Threads::RuntimeDefault => {
                    std::env::remove_var("MOONSHINE_ORT_SINGLE_THREAD");
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, complete: bool) -> StreamLine {
        StreamLine {
            text: text.to_string(),
            is_complete: complete,
        }
    }

    #[test]
    fn provisional_text_is_not_committed_until_complete() {
        let mut cursor = LineCursor::default();
        let first = cursor.commit(&[line(" ask not", false)]);
        assert_eq!(first.completed, "");
        assert_eq!(first.partial.as_deref(), Some("ask not"));

        let second = cursor.commit(&[line(" ask not what", true)]);
        assert_eq!(second.completed, "ask not what");
        assert_eq!(second.completed_lines, 1);
        assert_eq!(second.partial, None);

        // The same complete line is never typed twice.
        let third = cursor.commit(&[line(" ask not what", true)]);
        assert_eq!(third.completed, "");
    }

    #[test]
    fn revised_complete_line_is_not_duplicated_and_new_line_is_spaced() {
        let mut cursor = LineCursor::default();
        cursor.commit(&[line(" your country", true)]);
        let update = cursor.commit(&[
            line(" your country", true),
            line(" can do", true),
            line(" for you", false),
        ]);
        assert_eq!(update.completed, " can do");
        assert_eq!(update.completed_lines, 1);
        assert_eq!(update.partial.as_deref(), Some("for you"));
    }

    #[test]
    fn finish_commits_open_last_line_so_stop_does_not_lose_a_phrase() {
        let mut cursor = LineCursor::default();
        cursor.commit(&[line(" one", true), line(" two", false)]);
        let update = cursor.finish(&[line(" one", true), line(" two", false)]);
        assert_eq!(update.completed, " two");
        assert_eq!(update.completed_lines, 1);
        assert_eq!(
            cursor
                .finish(&[line(" one", true), line(" two", false)])
                .completed,
            ""
        );
        assert_eq!(
            cursor
                .finish(&[line(" one", true), line(" two", false)])
                .completed_lines,
            0
        );
    }

    #[test]
    fn punctuation_attaches_without_a_space() {
        let mut cursor = LineCursor::default();
        cursor.commit(&[line(" hello", true), line(" , world", true)]);
        // The first chunk has no leading space because nothing was typed yet.
        let update = cursor.commit(&[
            line(" hello", true),
            line(" , world", true),
            line(" !", true),
        ]);
        assert_eq!(update.completed, "!");
    }

    #[test]
    fn cancel_reset_forgets_typed_lines() {
        let mut cursor = LineCursor::default();
        cursor.commit(&[line(" hello", true)]);
        cursor.reset();
        let update = cursor.commit(&[line(" hello", true)]);
        assert_eq!(update.completed, "hello");
    }

    #[derive(Default)]
    struct FakeBackend {
        events: Vec<&'static str>,
        start_count: usize,
        feed_lines: Vec<StreamLine>,
        stop_lines: Vec<StreamLine>,
        fail_next_feed: bool,
    }

    impl StreamBackend for FakeBackend {
        fn start(&mut self) -> anyhow::Result<()> {
            self.events.push("start");
            self.start_count += 1;
            Ok(())
        }

        fn feed(&mut self, _samples: &[f32]) -> anyhow::Result<Vec<StreamLine>> {
            self.events.push("feed");
            if std::mem::take(&mut self.fail_next_feed) {
                anyhow::bail!("fake feed failed");
            }
            Ok(self.feed_lines.clone())
        }

        fn stop(&mut self) -> anyhow::Result<Vec<StreamLine>> {
            self.events.push("stop");
            Ok(self.stop_lines.clone())
        }

        fn cancel(&mut self) {
            self.events.push("cancel");
        }

        fn close(&mut self) {
            self.events.push("close");
        }
    }

    #[test]
    fn session_opens_lazily_and_finalizes_once() {
        let backend = FakeBackend {
            feed_lines: vec![line(" hello", true)],
            stop_lines: vec![line(" hello", true), line(" world", false)],
            ..FakeBackend::default()
        };
        let mut session = Session::new(backend);
        assert_eq!(session.feed(&[0.0]).unwrap().completed, "hello");
        // Stop commits the still-open last line exactly once.
        assert_eq!(session.finish().unwrap().completed, " world");
        assert_eq!(session.finish().unwrap(), Update::default());
        assert_eq!(
            session.backend.events,
            ["start", "feed", "stop", "close"],
            "stream must be opened by the session and closed after finishing"
        );
    }

    #[test]
    fn second_recording_gets_a_fresh_stream_and_cursor() {
        let backend = FakeBackend {
            feed_lines: vec![line(" hello", true)],
            ..FakeBackend::default()
        };
        let mut session = Session::new(backend);
        assert_eq!(session.feed(&[0.0]).unwrap().completed, "hello");
        assert_eq!(session.finish().unwrap(), Update::default());

        session.backend.feed_lines = vec![line(" hello", true)];
        // The same phrase must not be suppressed by the previous recording's
        // line cursor, and the native stream must be created again.
        assert_eq!(session.feed(&[0.0]).unwrap().completed, "hello");
        session.cancel();
        assert_eq!(session.backend.start_count, 2);
        assert_eq!(
            session.backend.events,
            ["start", "feed", "stop", "close", "start", "feed", "cancel"]
        );
    }

    #[test]
    fn silent_recording_without_chunks_is_a_quiet_finish() {
        let mut session = Session::new(FakeBackend::default());
        assert_eq!(session.finish().unwrap(), Update::default());
        assert!(session.backend.events.is_empty(), "no stream is opened");
    }

    #[test]
    fn cancel_discards_open_stream_and_never_leaks_into_the_next_recording() {
        let backend = FakeBackend::default();
        let mut session = Session::new(backend);
        session.backend.feed_lines = vec![line(" stale", false)];
        let update = session.feed(&[0.0]).unwrap();
        assert_eq!(update.completed, "");
        assert_eq!(update.partial.as_deref(), Some("stale"));
        session.cancel();
        assert_eq!(session.backend.events, ["start", "feed", "cancel"]);

        session.backend.events.clear();
        session.backend.feed_lines = vec![line(" fresh", true)];
        assert_eq!(session.feed(&[0.0]).unwrap().completed, "fresh");
        assert_eq!(session.backend.events, ["start", "feed"]);
    }

    #[test]
    fn feed_error_discards_the_stream_and_the_next_recording_starts_clean() {
        let backend = FakeBackend {
            fail_next_feed: true,
            ..FakeBackend::default()
        };
        let mut session = Session::new(backend);
        let err = session.feed(&[0.0]).unwrap_err();
        assert!(format!("{err:#}").contains("fake feed failed"));
        assert_eq!(session.backend.events, ["start", "feed", "cancel"]);

        session.backend.events.clear();
        session.backend.feed_lines = vec![line(" recovered", true)];
        assert_eq!(session.feed(&[0.0]).unwrap().completed, "recovered");
        assert_eq!(session.backend.start_count, 2);
    }

    #[test]
    fn moonshine_thread_policy_is_single_for_auto_and_explicit_two_stays_default() {
        assert_eq!(
            Threads::for_setting(InferenceThreads::Auto),
            Threads::Single
        );
        assert_eq!(Threads::for_setting(InferenceThreads::One), Threads::Single);
        assert_eq!(
            Threads::for_setting(InferenceThreads::Two),
            Threads::RuntimeDefault
        );
        assert!(
            Threads::RuntimeDefault
                .description()
                .contains("not exactly 2")
        );
    }

    fn raw_line(text: *const c_char, complete: bool) -> TranscriptLine {
        TranscriptLine {
            text,
            audio_data: std::ptr::null(),
            audio_data_count: 0,
            start_time: 0.0,
            duration: 0.0,
            id: 0,
            is_complete: i8::from(complete),
            is_updated: 0,
            is_new: 0,
            has_text_changed: 0,
            have_speakers_changed: 0,
            speaker_spans: std::ptr::null(),
            speaker_span_count: 0,
            last_transcription_latency_ms: 0,
            words: std::ptr::null(),
            word_count: 0,
        }
    }

    #[test]
    fn null_text_line_keeps_its_native_index_as_an_empty_placeholder() {
        let one = CString::new(" one").unwrap();
        let three = CString::new(" three").unwrap();
        let native = [
            raw_line(one.as_ptr(), true),
            // The runtime can expose a line before assigning text to it.
            raw_line(std::ptr::null(), true),
            raw_line(three.as_ptr(), true),
        ];
        let copied = unsafe { copy_stream_lines(native.as_ptr(), native.len() as u64) };
        assert_eq!(copied.len(), native.len(), "native indices must not shift");
        assert_eq!(copied[0].text, " one");
        assert_eq!(copied[1].text, "");
        assert_eq!(copied[2].text, " three");

        let mut cursor = LineCursor::default();
        assert_eq!(cursor.commit(&copied).completed, "one three");
    }

    /// Opt-in check of the actual native runtime and the same Session the
    /// daemon owns. Set DICTATE_MODEL_DIR to the downloaded model tree and
    /// MOONSHINE_ORT_SINGLE_THREAD=1 before launching the test process.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires downloaded Moonshine runtime/model and 11 seconds of paced audio"]
    fn native_session_reuses_model_after_finish_and_cancel() -> anyhow::Result<()> {
        use std::time::{Duration, Instant};

        crate::model::check_moonshine_ready()?;
        let model = Moonshine::load(
            &crate::model::moonshine_model_dir_checked()?,
            Threads::Single,
        )?;
        let mut session = Session::new(model);
        let samples = crate::audio::load_wav_16k_mono(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/jfk.wav"
        )))?;
        let assert_sentence = |text: &str| {
            let normalized: String = text
                .chars()
                .map(|ch| if ch.is_alphanumeric() { ch } else { ' ' })
                .collect::<String>()
                .to_lowercase();
            let words = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
            assert_eq!(words.matches("what your country can do for you").count(), 1);
        };

        // Warm the resident model, then verify that a finished stream cannot
        // emit its final phrase again.
        let mut first = String::new();
        for chunk in samples.chunks(1_600) {
            first.push_str(&session.feed(chunk)?.completed);
        }
        first.push_str(&session.finish()?.completed);
        assert_sentence(&first);
        assert_eq!(session.finish()?, Update::default());

        // Cancel an unfinished second recording. Its provisional words must
        // not survive into finish or the next stream over the same model.
        session.feed(&samples[..samples.len().min(11_200)])?;
        session.cancel();
        assert_eq!(session.finish()?, Update::default());

        let started = Instant::now();
        let mut first_partial = None;
        let mut first_complete = None;
        let mut final_text = String::new();
        for (index, chunk) in samples.chunks(1_600).enumerate() {
            let deadline = started + Duration::from_millis((index as u64 + 1) * 100);
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
            let update = session.feed(chunk)?;
            if update.partial.is_some() && first_partial.is_none() {
                first_partial = Some(started.elapsed());
            }
            if !update.completed.is_empty() && first_complete.is_none() {
                first_complete = Some(started.elapsed());
            }
            final_text.push_str(&update.completed);
        }
        let stopped = Instant::now();
        final_text.push_str(&session.finish()?.completed);
        let stop_compute = stopped.elapsed();
        let after_audio = started
            .elapsed()
            .saturating_sub(Duration::from_secs_f64(samples.len() as f64 / 16_000.0));
        assert_sentence(&final_text);
        assert_eq!(session.finish()?, Update::default());
        eprintln!(
            "native Moonshine paced: first partial {first_partial:?}, first complete {first_complete:?}, stop flush {stop_compute:?}, final result {after_audio:?} after audio end"
        );
        Ok(())
    }
}
