//! Headless contract 1 transcription service.
use crate::config::Settings;
use crate::http_transport::{Request, Server};
use serde_json::{Value, json};
use std::collections::{BTreeSet, VecDeque};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tiny_http::{Header, Response};
thread_local! { static RETRY_AFTER: std::cell::Cell<u64> = const { std::cell::Cell::new(1) }; }
const VERSION: &str = env!("CARGO_PKG_VERSION");
pub(crate) const USAGE: &str = "Usage: dictate serve [--listen ADDR:PORT] [--token-file PATH] [--max-concurrency N] [--max-queue N] [--max-handlers N] [--linger-drain-bytes N] [--linger-secs N] [--max-lingering-rejections N] [--read-timeout-secs N] [--max-header-bytes N] [--max-upload-bytes N] [--max-body-bytes N] [--max-duration-secs N] [--idle-unload-secs N] [--accept-poll-ms N] [--retry-after-secs N]\nFlags override persisted server.* settings for one run. No models are downloaded. Without a token, anything that reaches the port can use it.";
#[derive(Debug)]
pub struct ApiError {
    status: u16,
    code: Option<u8>,
    kind: &'static str,
    message: String,
}
impl ApiError {
    fn new(status: u16, code: Option<u8>, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            kind,
            message: message.into(),
        }
    }
    fn bad(message: &str) -> Self {
        Self::new(400, Some(2), "invalid_request", message)
    }
    fn internal(stage: &str, detail: &str) -> Self {
        // Engine errors can include input text. Log stage only, never raw engine output.
        if stage == "encoding" {
            eprintln!("error: HTTP encoding failed: {detail}");
        } else {
            eprintln!(
                "error: HTTP {stage} failed (input-bearing detail redacted, detail_bytes={})",
                detail.len()
            );
        }
        Self::new(
            500,
            Some(6),
            "transcription_failed",
            "Transcription failed; check the service logs",
        )
    }
}

#[derive(Debug)]
struct Transcribe {
    file: Vec<u8>,
    model: &'static Model,
    max_duration: f64,
    segments: bool,
}
#[derive(Debug)]
struct Model {
    id: &'static str,
    engine: &'static str,
    label: &'static str,
    flag: &'static str,
}
const MODELS: &[Model] = &[
    Model {
        id: "parakeet-tdt-0-6b-v2",
        engine: "standard",
        label: "Parakeet TDT 0.6B v2",
        flag: "",
    },
    Model {
        id: "whisper-tiny-en",
        engine: "whisper",
        label: "Whisper Tiny English",
        flag: " --whisper",
    },
    Model {
        id: "fluid-speech-mini",
        engine: "fluid",
        label: "Fluid Speech Mini",
        flag: " --fluid",
    },
    Model {
        id: "nemotron-streaming-en",
        engine: "nemotron",
        label: "Nemotron Streaming English",
        flag: " --live",
    },
    Model {
        id: "moonshine-tiny-streaming-en",
        engine: "moonshine",
        label: "Moonshine Tiny Streaming English",
        flag: " --moonshine",
    },
];
fn installed(m: &Model) -> bool {
    use crate::model::*;
    match m.engine {
        "standard" => model_dir_checked(),
        "whisper" => whisper_model_dir_checked(),
        "fluid" => fluid_model_dir_checked(),
        "nemotron" => live_model_dir_checked(),
        _ => moonshine_model_dir_checked().and_then(|dir| {
            check_moonshine_runtime()?;
            Ok(dir)
        }),
    }
    .is_ok()
}
fn model_dir(m: &Model) -> anyhow::Result<std::path::PathBuf> {
    use crate::model::*;
    match m.engine {
        "standard" => model_dir(),
        "whisper" => whisper_model_dir(),
        "fluid" => fluid_model_dir(),
        "nemotron" => live_model_dir(),
        _ => moonshine_model_dir(),
    }
}
fn size(m: &Model) -> u64 {
    if !installed(m) {
        return crate::model::download_size(m.engine);
    }
    model_dir(m)
        .ok()
        .and_then(|p| std::fs::read_dir(p).ok())
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or_else(|| crate::model::download_size(m.engine))
}
fn default_model(settings: &Settings) -> &'static Model {
    let choice = crate::config::effective_selection(settings).model;
    MODELS
        .iter()
        .find(|m| m.engine == choice.as_str())
        .expect("model catalog")
}
fn device(m: &Model) -> &'static str {
    if m.engine == "whisper"
        && matches!(
            crate::whisper_device::last_for_thread().or_else(crate::whisper_device::last),
            Some(crate::whisper_device::WhisperDevice::Gpu { .. })
        )
    {
        "gpu"
    } else {
        "cpu"
    }
}
fn missing(m: &Model) -> ApiError {
    ApiError::new(
        503,
        Some(3),
        "model_missing",
        format!(
            "{} is not installed; run `dictate download{}`",
            m.label, m.flag
        ),
    )
}
fn empty() -> ApiError {
    eprintln!("error: HTTP audio validation failed (input details redacted)");
    ApiError::new(
        422,
        Some(4),
        "invalid_audio",
        "Audio is empty, silent or unreadable; provide a supported recording",
    )
}
fn too_long() -> ApiError {
    ApiError::new(
        413,
        Some(5),
        "duration_limit",
        "Audio exceeds the duration limit",
    )
}
fn decode_audio(bytes: &[u8], max_duration: f64) -> Result<(Vec<f32>, f64), ApiError> {
    let mut file =
        tempfile::NamedTempFile::new().map_err(|e| ApiError::internal("spool", &e.to_string()))?;
    file.write_all(bytes)
        .map_err(|e| ApiError::internal("spool", &e.to_string()))?;
    let mut source = crate::media::Audio::open(file.path().to_str().expect("temporary path"))
        .map_err(|_| empty())?;
    if source.duration.is_some_and(|s| s > max_duration) {
        return Err(too_long());
    }
    let mut resampler = if source.rate == crate::audio::TARGET_SAMPLE_RATE {
        None
    } else {
        Some(crate::audio::StreamResampler::new(source.rate).map_err(|_| empty())?)
    };
    let mut samples = Vec::new();
    loop {
        let part = source.next().map_err(|_| empty())?;
        let done = part.is_none();
        let part =
            crate::media::Audio::resample_packet(&mut resampler, part).map_err(|_| empty())?;
        samples.extend(part);
        if samples.len() as f64 / crate::audio::TARGET_SAMPLE_RATE as f64 > max_duration {
            return Err(too_long());
        }
        if done {
            break;
        }
    }
    if samples.is_empty()
        || samples.iter().any(|s| !s.is_finite())
        || !samples.iter().any(|s| *s != 0.0)
    {
        return Err(empty());
    }
    let duration = samples.len() as f64 / crate::audio::TARGET_SAMPLE_RATE as f64;
    Ok((samples, duration))
}
trait Transcriber {
    fn transcribe(&mut self, request: &Transcribe) -> Result<Value, ApiError>;
    fn unload(&mut self);
}
struct Native {
    engine: Option<crate::transcription::Engine>,
    id: Option<&'static str>,
    loaded: Arc<Mutex<BTreeSet<String>>>,
    worker: usize,
}
impl Transcriber for Native {
    fn unload(&mut self) {
        self.engine = None;
        if let Some(id) = self.id.take() {
            self.loaded
                .lock()
                .expect("loaded lock")
                .remove(&format!("{}:{id}", self.worker));
        }
    }
    fn transcribe(&mut self, r: &Transcribe) -> Result<Value, ApiError> {
        if !installed(r.model) {
            return Err(missing(r.model));
        }
        let (samples, duration) = decode_audio(&r.file, r.max_duration)?;
        let load = Instant::now();
        if self.id != Some(r.model.id) {
            self.unload();
            self.engine = Some(
                crate::transcription::Engine::load(r.model.engine)
                    .map_err(|e| ApiError::internal("model load", &e.to_string()))?,
            );
            self.id = Some(r.model.id);
            self.loaded
                .lock()
                .expect("loaded lock")
                .insert(format!("{}:{}", self.worker, r.model.id));
        }
        let load_ms = load.elapsed().as_millis();
        let start = Instant::now();
        // Share the CLI's bounded pause feeder and engine dispatch. VAD is required only for long audio.
        let pieces = if samples.len() > crate::segment::MAX_SEGMENT_SAMPLES {
            if crate::model::vad_model_path_checked().is_err() {
                return Err(ApiError::new(
                    503,
                    Some(3),
                    "model_missing",
                    "Speech segmentation model is missing; run `dictate download`",
                ));
            }
            let mut feeder = crate::segment::Feeder::new(
                crate::audio::TARGET_SAMPLE_RATE,
                crate::vad::Vad::load_default()
                    .map_err(|e| ApiError::internal("VAD load", &e.to_string()))?,
            )
            .map_err(|e| ApiError::internal("segmentation", &e.to_string()))?;
            let mut pieces = feeder
                .push(&samples)
                .map_err(|e| ApiError::internal("segmentation", &e.to_string()))?;
            pieces.extend(
                feeder
                    .finish()
                    .map_err(|e| ApiError::internal("segmentation", &e.to_string()))?,
            );
            pieces
        } else {
            vec![crate::segment::Segment {
                audio: samples,
                context: Vec::new(),
                offset: 0,
                start: crate::segment::SegmentStart::Recording,
            }]
        };
        let mut segments = Vec::new();
        let mut text = String::new();
        for piece in pieces {
            let decoded = self
                .engine
                .as_mut()
                .expect("loaded engine")
                .decode(&piece, false);
            let (part, _) = match decoded {
                Ok(part) => part,
                Err(error) => {
                    // A streaming engine may retain partial state after an error.
                    self.unload();
                    return Err(ApiError::internal("decode", &error.to_string()));
                }
            };
            if crate::asr::is_empty_transcript(&part) {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&part);
            segments.push(json!({"start":(piece.offset as f64 / crate::audio::TARGET_SAMPLE_RATE as f64).min(duration),"end":((piece.offset as usize+piece.audio.len()) as f64/crate::audio::TARGET_SAMPLE_RATE as f64).min(duration),"text":part}));
        }
        if text.trim().is_empty() {
            return Err(empty());
        }
        let mut value = json!({"contract":1,"version":VERSION,"text":text,"language":"en","languageProbability":null,"durationSec":duration,"model":r.model.id,"device":device(r.model),"timings":{"loadMs":load_ms,"transcribeMs":start.elapsed().as_millis()}});
        if r.segments {
            value["segments"] = json!(segments);
        }
        Ok(value)
    }
}
/// Hash both values to equal-sized inputs before ring's constant-time comparison.
fn authorized(expected: Option<&[u8; 32]>, header: Option<&str>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    let candidate = header.and_then(|s| s.strip_prefix("Bearer ")).unwrap_or("");
    let digest = ring::digest::digest(&ring::digest::SHA256, candidate.as_bytes());
    // ring 0.17 exposes this deprecated entry point; keep the comparison constant-time.
    #[allow(deprecated)]
    ring::constant_time::verify_slices_are_equal(expected, digest.as_ref()).is_ok()
}
fn token(path: &str) -> Result<Option<[u8; 32]>, String> {
    if path.is_empty() {
        return Ok(None);
    }
    let value = std::fs::read_to_string(path).map_err(|_| "couldn't read token file")?;
    let value = value.trim();
    if value.is_empty() {
        return Err("token file is empty".into());
    }
    Ok(Some(
        ring::digest::digest(&ring::digest::SHA256, value.as_bytes())
            .as_ref()
            .try_into()
            .expect("SHA256"),
    ))
}
struct Reply {
    status: u16,
    bytes: Vec<u8>,
    headers: Vec<(&'static str, String)>,
}
impl Reply {
    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            bytes: value.to_string().into_bytes(),
            headers: vec![("Content-Type", "application/json".into())],
        }
    }
    fn error(error: ApiError) -> Self {
        let mut reply = Self::json(
            error.status,
            json!({"contract":1,"error":{"code":error.code,"kind":error.kind,"message":error.message}}),
        );
        if reply.status == 401 {
            reply.headers.push(("WWW-Authenticate", "Bearer".into()));
        }
        if reply.status == 429 {
            reply
                .headers
                .push(("Retry-After", RETRY_AFTER.with(|n| n.get()).to_string()));
        }
        reply
    }
}

/// Split header parameters without treating quoted filename semicolons as separators.
fn multipart_parameters(value: &str) -> Result<Vec<&str>, ApiError> {
    let mut quoted = false;
    let mut escaped = false;
    let mut start = 0;
    let mut parts = Vec::new();
    for (index, byte) in value.bytes().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if quoted && byte == b'\\' {
            escaped = true;
            continue;
        }
        if byte == b'"' {
            quoted = !quoted;
        }
        if byte == b';' && !quoted {
            parts.push(value[start..index].trim());
            start = index + 1;
        }
    }
    if quoted || escaped {
        return Err(ApiError::bad("Unclosed multipart parameter quote"));
    }
    parts.push(value[start..].trim());
    Ok(parts)
}
fn validate(body: &[u8], content_type: &str, settings: &Settings) -> Result<Transcribe, ApiError> {
    let bad = || ApiError::bad("Invalid multipart body, duplicate or unknown part");
    let parameters = multipart_parameters(content_type)?;
    let mut params = parameters.into_iter();
    if !params
        .next()
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("multipart/form-data"))
    {
        return Err(ApiError::new(
            415,
            Some(4),
            "content_type",
            "Use multipart/form-data",
        ));
    }
    let mut boundary = None;
    for param in params {
        let (k, v) = param.trim().split_once('=').ok_or_else(bad)?;
        if !k.eq_ignore_ascii_case("boundary") || boundary.is_some() {
            return Err(bad());
        }
        boundary = Some(v.trim().trim_matches('"'));
    }
    let boundary = boundary.ok_or_else(bad)?;
    if boundary.is_empty()
        || boundary.len() > 70
        || boundary.ends_with(' ')
        || !boundary
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"'()+_,-./:=? ".contains(&b))
    {
        return Err(bad());
    }
    let marker = format!("--{boundary}");
    let delimiter = format!("\r\n--{boundary}");
    let mut rest = body.strip_prefix(marker.as_bytes()).ok_or_else(bad)?;
    let mut fields = std::collections::BTreeMap::new();
    loop {
        if let Some(tail) = rest.strip_prefix(b"--") {
            if tail != b"\r\n" && !tail.is_empty() {
                return Err(bad());
            }
            break;
        }
        rest = rest.strip_prefix(b"\r\n").ok_or_else(bad)?;
        let pos = rest
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(bad)?;
        if pos > settings.server.max_header_bytes {
            return Err(bad());
        }
        let headers = std::str::from_utf8(&rest[..pos]).map_err(|_| bad())?;
        let mut name = None;
        for line in headers.split("\r\n") {
            let (k, v) = line.split_once(':').ok_or_else(bad)?;
            if k.eq_ignore_ascii_case("Content-Disposition") {
                if name.is_some() {
                    return Err(bad());
                }
                let parameters = multipart_parameters(v.trim())?;
                let mut params = parameters.into_iter();
                if params.next() != Some("form-data") {
                    return Err(bad());
                }
                for param in params {
                    let (k, v) = param.trim().split_once('=').ok_or_else(bad)?;
                    if k.eq_ignore_ascii_case("name") {
                        if name.replace(v.trim_matches('"')).is_some() {
                            return Err(bad());
                        }
                    } else if !k.eq_ignore_ascii_case("filename") {
                        return Err(bad());
                    }
                }
            } else if !k.eq_ignore_ascii_case("Content-Type") {
                return Err(bad());
            }
        }
        let name = name.ok_or_else(bad)?;
        if !["file", "language", "model", "maxDurationSec", "segments"].contains(&name) {
            return Err(bad());
        }
        rest = &rest[pos + 4..];
        let end = rest
            .windows(delimiter.len())
            .enumerate()
            .find(|(i, w)| {
                *w == delimiter.as_bytes()
                    && (rest[*i + delimiter.len()..].starts_with(b"--")
                        || rest[*i + delimiter.len()..].starts_with(b"\r\n"))
            })
            .map(|(i, _)| i)
            .ok_or_else(bad)?;
        if fields.insert(name, rest[..end].to_vec()).is_some() {
            return Err(bad());
        }
        rest = &rest[end + delimiter.len()..];
    }
    let file = fields.remove("file").ok_or_else(bad)?;
    if file.len() > settings.server.max_upload_bytes {
        return Err(ApiError::new(
            413,
            Some(5),
            "upload_limit",
            "Audio exceeds the upload limit",
        ));
    }
    if file.is_empty() {
        return Err(empty());
    }
    if file.windows(8).any(|w| w == b"OpusHead") || file.windows(6).any(|w| w == b"A_OPUS") {
        return Err(ApiError::new(
            415,
            Some(4),
            "unsupported_codec",
            "Opus is not supported; convert the recording to WAV",
        ));
    }
    let field = |name| -> Result<Option<&str>, ApiError> {
        fields
            .get(name)
            .map(|b| std::str::from_utf8(b).map_err(|_| bad()))
            .transpose()
    };
    let model = match field("model")? {
        None | Some("") => default_model(settings),
        Some(id) => MODELS
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| ApiError::bad("Unknown model; see /v1/capabilities"))?,
    };
    match field("language")?.unwrap_or("auto") {
        "en" | "auto" => {}
        "fa" => {
            return Err(ApiError::new(
                422,
                Some(7),
                "language_unsupported",
                "This model supports English only",
            ));
        }
        _ => return Err(ApiError::bad("language must be fa, en or auto")),
    }
    let max_duration = field("maxDurationSec")?
        .map(|s| s.parse::<f64>().map_err(|_| bad()))
        .transpose()?
        .unwrap_or(settings.server.max_duration_secs as f64);
    if !max_duration.is_finite()
        || max_duration <= 0.
        || max_duration > settings.server.max_duration_secs as f64
    {
        return Err(ApiError::bad(
            "maxDurationSec must be positive and at most the service limit",
        ));
    }
    let segments = match field("segments")?.unwrap_or("false") {
        "true" => true,
        "false" => false,
        _ => return Err(bad()),
    };
    Ok(Transcribe {
        file,
        model,
        max_duration,
        segments,
    })
}
fn capabilities(settings: &Settings) -> Value {
    json!({"contract":1,"version":VERSION,"models":MODELS.iter().map(|m|json!({"id":m.id,"label":m.label,"languages":["en"],"installed":installed(m),"sizeBytes":size(m)})).collect::<Vec<_>>(),"inputFormats":["wav","mp3","flac","ogg/vorbis","aac","m4a","webm/vorbis","mkv","mp4"],"languages":["en","auto"],"device":device(default_model(settings)),"limits":{"maxDurationSec":settings.server.max_duration_secs,"maxUploadBytes":settings.server.max_upload_bytes,"maxConcurrency":settings.server.max_concurrency,"maxQueue":settings.server.max_queue,"maxHandlers":settings.server.max_handlers}})
}
fn health_json(missing: Value, auth: bool, models: BTreeSet<String>) -> Value {
    json!({"contract":1,"version":VERSION,"ready":missing.as_array().is_some_and(|m| m.is_empty()),"missing":missing,"auth":if auth {"token"} else {"none"},"modelsLoaded":models})
}
fn route(method: &str, path: &str, auth: bool) -> Result<bool, ApiError> {
    if method == "GET" && path == "/v1/health" {
        return Ok(false);
    }
    if path != "/v1/health" && !auth {
        return Err(ApiError::new(
            401,
            None,
            "unauthorized",
            "A valid bearer token is required",
        ));
    }
    let expected = match path {
        "/v1/health" | "/v1/capabilities" => "GET",
        "/v1/transcribe" => "POST",
        _ => return Err(ApiError::new(404, None, "not_found", "Route not found")),
    };
    if method != expected {
        return Err(ApiError::new(
            405,
            None,
            "method_not_allowed",
            "Method not allowed",
        ));
    }
    Ok(path == "/v1/transcribe")
}
struct Job {
    request: Request,
    started: Instant,
    transcription: Transcribe,
}
fn respond(request: Request, reply: Reply, started: Instant, chars: usize) {
    let method = request.method().to_string();
    // Never log query strings (they may contain caller secrets).
    let path = request.url().split('?').next().unwrap_or("/");
    let path = match path {
        "/v1/health" | "/v1/capabilities" | "/v1/transcribe" => path,
        _ => "<unknown>",
    }
    .to_string();
    let status = reply.status;
    let mut response = Response::from_data(reply.bytes).with_status_code(status);
    for (key, value) in reply.headers {
        response.add_header(Header::from_bytes(key, value).expect("safe response header"));
    }
    if let Err(e) = request.respond(response) {
        eprintln!("error: HTTP response write failed: {e}");
    }
    eprintln!(
        "{method} {path} {status} {}ms bytes={chars}",
        started.elapsed().as_millis()
    );
}
#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    closed: bool,
    active: usize,
}
type SharedQueue = Arc<(Mutex<Queue>, Condvar)>;
fn worker(mut synth: Box<dyn Transcriber>, queue: SharedQueue, idle: u64) {
    let mut last_used = Instant::now();
    loop {
        let (lock, wake) = &*queue;
        let mut state = lock.lock().expect("queue lock");
        while state.jobs.is_empty() && !state.closed {
            if idle == 0 {
                state = wake.wait(state).expect("queue wait");
            } else {
                let remaining = Duration::from_secs(idle).saturating_sub(last_used.elapsed());
                let (next, timeout) = wake.wait_timeout(state, remaining).expect("queue wait");
                state = next;
                if timeout.timed_out() {
                    synth.unload();
                    last_used = Instant::now();
                }
            }
        }
        let Some(job) = state.jobs.pop_front() else {
            break;
        };
        state.active += 1;
        drop(state);
        let chars = job.transcription.file.len();
        let reply = match synth.transcribe(&job.transcription) {
            Ok(value) => Reply::json(200, value),
            Err(error) => Reply::error(error),
        };
        respond(job.request, reply, job.started, chars);
        queue.0.lock().expect("queue lock").active -= 1;
        last_used = Instant::now();
    }
}

pub(crate) fn parse_flags(
    settings: &mut crate::server_settings::ServerSettings,
    args: &[String],
) -> Result<(), String> {
    if !args.len().is_multiple_of(2) {
        return Err("serve flags require values".into());
    }
    for pair in args.as_chunks::<2>().0 {
        let key = pair[0]
            .strip_prefix("--")
            .ok_or("Expected serve flag")?
            .replace('-', "_");
        settings.set(&format!("server.{key}"), &pair[1])?;
    }
    Ok(())
}
pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let (mut settings, problem) = crate::config::load();
    if let Some(problem) = problem {
        return Err(problem);
    }
    parse_flags(&mut settings.server, args)?;
    crate::config::init_effective(&settings);
    // The pinned Moonshine API uses an environment switch. Set it before
    // ctrlc or HTTP workers create threads; Engine::load's Once then is a no-op.
    crate::moonshine::configure_threads(crate::moonshine::Threads::for_setting(
        crate::config::inference_threads(),
    ));
    let token = token(&settings.server.token_file)?;
    let server =
        Server::http(&settings.server.listen).map_err(|e| format!("HTTP listen failed: {e}"))?;
    let stop = Arc::new(AtomicBool::new(false));
    let signal = stop.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .map_err(|e| format!("Signal handler failed: {e}"))?;
    eprintln!(
        "HTTP listening={} auth={} concurrency={} queue={} handlers={} upload_bytes={} duration_secs={} read_timeout_secs={} body_bytes={} header_bytes={} idle_unload_secs={} accept_poll_ms={} retry_after_secs={}",
        server.server_addr(),
        if token.is_some() { "token" } else { "none" },
        settings.server.max_concurrency,
        settings.server.max_queue,
        settings.server.max_handlers,
        settings.server.max_upload_bytes,
        settings.server.max_duration_secs,
        settings.server.read_timeout_secs,
        settings.server.max_body_bytes,
        settings.server.max_header_bytes,
        settings.server.idle_unload_secs,
        settings.server.accept_poll_ms,
        settings.server.retry_after_secs
    );
    let loaded = Arc::new(Mutex::new(BTreeSet::new()));
    let catalog = settings.clone();
    let worker_loaded = loaded.clone();
    serve_http(
        server,
        settings,
        token,
        stop,
        loaded,
        move |health| {
            if health {
                let m = default_model(&catalog);
                json!(if installed(m) {
                    Vec::<&str>::new()
                } else {
                    vec![m.label]
                })
            } else {
                capabilities(&catalog)
            }
        },
        move |worker| {
            Box::new(Native {
                engine: None,
                id: None,
                loaded: worker_loaded.clone(),
                worker,
            })
        },
    )
}
/// Shared transport loop; tests inject a device-free fake without encoders or models.
fn serve_http(
    server: Server,
    settings: Settings,
    token: Option<[u8; 32]>,
    stop: Arc<AtomicBool>,
    loaded: Arc<Mutex<BTreeSet<String>>>,
    caps: impl Fn(bool) -> Value + Send + Sync + 'static,
    factory: impl FnMut(usize) -> Box<dyn Transcriber> + Send + 'static,
) -> Result<(), String> {
    RETRY_AFTER.with(|n| n.set(settings.server.retry_after_secs));
    let queue: SharedQueue = Arc::default();
    let mut workers = Vec::new();
    let factory = Arc::new(Mutex::new(factory));
    for worker_id in 0..settings.server.max_concurrency {
        let factory = factory.clone();
        let queue = queue.clone();
        let idle = settings.server.idle_unload_secs;
        workers.push(std::thread::spawn(move || {
            let native = factory.lock().expect("factory lock")(worker_id);
            worker(native, queue, idle)
        }));
    }
    let caps = Arc::new(caps);
    let settings = Arc::new(settings);
    let handlers = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut intake = Vec::new();
    let mut receive_error = None;
    let mut rejections = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        rejections.retain(|thread: &std::thread::JoinHandle<()>| !thread.is_finished());
        intake.retain(|thread: &std::thread::JoinHandle<()>| !thread.is_finished());
        let request =
            match server.recv_timeout(Duration::from_millis(settings.server.accept_poll_ms)) {
                Ok(Some(request)) => request,
                Ok(None) => continue,
                Err(error) => {
                    receive_error = Some(format!("HTTP receive failed: {error}"));
                    break;
                }
            };
        let mut request = request;
        request.configure_close(
            settings.server.linger_drain_bytes,
            Duration::from_secs(settings.server.linger_secs),
        );
        if handlers.load(Ordering::Acquire) >= settings.server.max_handlers {
            if rejections.len() >= settings.server.max_lingering_rejections {
                eprintln!(
                    "warn: fallback=true HTTP rejection capacity exhausted; closing connection without response"
                );
                drop(request);
                continue;
            }
            let retry_after = settings.server.retry_after_secs;
            rejections.push(std::thread::spawn(move || {
                RETRY_AFTER.with(|n| n.set(retry_after));
                respond(
                    request,
                    Reply::error(ApiError::new(
                        429,
                        None,
                        "queue_full",
                        "Service is busy; retry later",
                    )),
                    Instant::now(),
                    0,
                );
            }));
            continue;
        }
        handlers.fetch_add(1, Ordering::AcqRel);
        let handlers = handlers.clone();
        let settings = settings.clone();
        let queue = queue.clone();
        let loaded = loaded.clone();
        let caps = caps.clone();
        intake.push(std::thread::spawn(move || {
            struct Handler(Arc<std::sync::atomic::AtomicUsize>);
            impl Drop for Handler {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::AcqRel);
                }
            }
            let _handler = Handler(handlers);
            RETRY_AFTER.with(|n| n.set(settings.server.retry_after_secs));
            let mut request = request;
            let started = Instant::now();
            if let Err(error) = request.read_headers(
                Duration::from_secs(settings.server.read_timeout_secs),
                settings.server.max_header_bytes,
            ) {
                eprintln!("error: HTTP request headers failed: {error}");
                respond(
                    request,
                    Reply::error(ApiError::bad(
                        "Could not read request headers; use Content-Length and check service limits",
                    )),
                    started,
                    0,
                );
                return;
            }
            let path = request.url().split('?').next().unwrap_or("/");
            let auth_header = request
                .headers()
                .iter()
                .find(|h| h.field.equiv("Authorization"))
                .map(|h| h.value.as_str());
            match route(
                request.method().as_str(),
                path,
                authorized(token.as_ref(), auth_header),
            ) {
                Err(error) => respond(request, Reply::error(error), started, 0),
                Ok(false) => {
                    let reply = if path == "/v1/health" {
                        let ids: BTreeSet<String> = loaded
                            .lock()
                            .expect("loaded lock")
                            .iter()
                            .filter_map(|s| s.split_once(':').map(|(_, id)| id.to_owned()))
                            .collect();
                        let missing = caps(true);
                        Reply::json(
                            200,
                            health_json(missing, token.is_some(), ids),
                        )
                    } else {
                        Reply::json(200, caps(false))
                    };
                    respond(request, reply, started, 0);
                }
                Ok(true) => {
                    let content_type = request
                        .headers()
                        .iter()
                        .find(|h| h.field.equiv("Content-Type"))
                        .map(|h| h.value.as_str().to_owned());
                    if !content_type
                        .as_ref()
                        .is_some_and(|s| s.split(';').next().is_some_and(|mime| mime.trim().eq_ignore_ascii_case("multipart/form-data")))
                    {
                        respond(
                            request,
                            Reply::error(ApiError::new(
                                415,
                                Some(4),
                                "content_type",
                                "Use Content-Type: multipart/form-data",
                            )),
                            started,
                            0,
                        );
                        return;
                    }
                    if request
                        .body_length()
                        .is_some_and(|n| n > settings.server.max_body_bytes)
                    {
                        respond(
                            request,
                            Reply::error(ApiError::new(
                                413,
                                Some(5),
                                "body_limit",
                                "Request body exceeds the service limit",
                            )),
                            started,
                            0,
                        );
                        return;
                    }
                    let mut body = Vec::new();
                    if let Err(e) = request
                        .as_reader()
                        .take(settings.server.max_body_bytes as u64 + 1)
                        .read_to_end(&mut body)
                    {
                        eprintln!("error: HTTP body read failed: {e}");
                        respond(
                            request,
                            Reply::error(ApiError::bad("Could not read request body")),
                            started,
                            0,
                        );
                        return;
                    }
                    match validate(&body, content_type.as_deref().unwrap_or(""), &settings) {
                        Err(error) => respond(request, Reply::error(error), started, 0),
                        Ok(speak) => {
                            let job = Job {
                                request,
                                started,
                                transcription: speak,
                            };
                            let mut state = queue.0.lock().expect("queue lock");
                            if state.closed {
                                drop(state);
                                respond(
                                    job.request,
                                    Reply::error(ApiError::new(
                                        503,
                                        None,
                                        "shutting_down",
                                        "Service is shutting down; retry later",
                                    )),
                                    started,
                                    0,
                                );
                            } else if state.jobs.len() + state.active
                                >= settings.server.max_queue + settings.server.max_concurrency
                            {
                                drop(state);
                                respond(
                                    job.request,
                                    Reply::error(ApiError::new(
                                        429,
                                        None,
                                        "queue_full",
                                        "Service is busy; retry later",
                                    )),
                                    started,
                                    job.transcription.file.len(),
                                );
                            } else {
                                state.jobs.push_back(job);
                                queue.1.notify_one();
                            }
                        }
                    }
                }
            }
        }));
    }
    for rejection in rejections {
        rejection
            .join()
            .map_err(|_| "HTTP rejection handler panicked")?;
    }
    queue.0.lock().expect("queue lock").closed = true;
    queue.1.notify_all();
    for worker in workers {
        worker.join().map_err(|_| "HTTP worker panicked")?;
    }
    for handler in intake {
        handler.join().map_err(|_| "HTTP handler panicked")?;
    }
    if let Some(error) = receive_error {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpStream;
    use std::sync::mpsc;
    struct Fake;
    impl Transcriber for Fake {
        fn transcribe(&mut self, r: &Transcribe) -> Result<Value, ApiError> {
            if r.file == b"missing" {
                return Err(missing(r.model));
            }
            let mut v = json!({"contract":1,"version":VERSION,"text":"hello","language":"en","languageProbability":null,"model":r.model.id,"durationSec":1.,"device":"cpu","timings":{"loadMs":0,"transcribeMs":0}});
            if r.segments {
                v["segments"] = json!([{"start":0.,"end":1.,"text":"hello"}]);
            }
            Ok(v)
        }
        fn unload(&mut self) {}
    }
    struct TestServer {
        address: String,
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<Result<(), String>>>,
    }
    impl TestServer {
        fn start(
            settings: Settings,
            token: Option<[u8; 32]>,
            factory: impl FnMut(usize) -> Box<dyn Transcriber> + Send + 'static,
        ) -> Self {
            let server = Server::http("127.0.0.1:0").unwrap();
            let address = server.server_addr().to_string();
            let stop = Arc::new(AtomicBool::new(false));
            let signal = stop.clone();
            let thread = std::thread::spawn(move || {
                serve_http(
                    server,
                    settings,
                    token,
                    signal,
                    Arc::default(),
                    |health| {
                        if health {
                            json!([])
                        } else {
                            capabilities(&Settings::default())
                        }
                    },
                    factory,
                )
            });
            Self {
                address,
                stop,
                thread: Some(thread),
            }
        }
        fn connect(&self, method: &str, path: &str, body: &[u8], extra: &str) -> TcpStream {
            let mut s = TcpStream::connect(&self.address).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            write!(s,"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=x\r\nContent-Length: {}\r\n{extra}\r\n",body.len()).unwrap();
            s.write_all(body).unwrap();
            s
        }
        fn request(
            &self,
            method: &str,
            path: &str,
            body: &[u8],
            extra: &str,
        ) -> (u16, Value, String) {
            read_reply(&mut self.connect(method, path, body, extra))
        }
    }
    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            self.thread.take().unwrap().join().unwrap().unwrap();
        }
    }
    fn read_reply(s: &mut TcpStream) -> (u16, Value, String) {
        let mut h = Vec::new();
        while !h.ends_with(b"\r\n\r\n") {
            let mut b = [0];
            s.read_exact(&mut b).unwrap();
            h.push(b[0]);
        }
        let h = String::from_utf8(h).unwrap();
        let len = h
            .lines()
            .find_map(|l| {
                l.split_once(':')
                    .filter(|(k, _)| k.eq_ignore_ascii_case("Content-Length"))
                    .map(|(_, v)| v.trim().parse::<usize>().unwrap())
            })
            .unwrap();
        let mut b = vec![0; len];
        s.read_exact(&mut b).unwrap();
        (
            h.split_whitespace().nth(1).unwrap().parse().unwrap(),
            serde_json::from_slice(&b).unwrap(),
            h,
        )
    }
    fn body(parts: &[(&str, &[u8])]) -> Vec<u8> {
        let mut b = Vec::new();
        for (name, data) in parts {
            write!(
                b,
                "--x\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            )
            .unwrap();
            b.extend_from_slice(data);
            b.extend_from_slice(b"\r\n");
        }
        b.extend_from_slice(b"--x--\r\n");
        b
    }
    fn digest(s: &str) -> [u8; 32] {
        ring::digest::digest(&ring::digest::SHA256, s.as_bytes())
            .as_ref()
            .try_into()
            .unwrap()
    }
    #[test]
    fn rejected_upload_does_not_hold_health_while_lingering() {
        let mut settings = Settings::default();
        settings.server.max_body_bytes = 16;
        settings.server.linger_secs = 1;
        settings.server.linger_drain_bytes = usize::MAX;
        settings.server.max_handlers = 2;
        let server = TestServer::start(settings, None, |_| Box::new(Fake));
        let mut upload = TcpStream::connect(&server.address).unwrap();
        upload
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        write!(upload, "POST /v1/transcribe HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=x\r\nContent-Length: 100000\r\n\r\npartial").unwrap();
        let mut response = String::new();
        upload.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 413"), "{response}");
        let (_, body) = response.split_once("\r\n\r\n").unwrap();
        let json: Value = serde_json::from_str(body).unwrap();
        assert!(json.get("error").is_some(), "{json}");
        assert_eq!(server.request("GET", "/v1/health", b"", "").0, 200);
        let mut sender = upload.try_clone().unwrap();
        sender
            .set_write_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let (done, completion) = std::sync::mpsc::channel();
        let writer = std::thread::spawn(move || {
            while sender.write_all(&[0; 4096]).is_ok() {}
            done.send(()).unwrap();
        });
        assert_eq!(server.request("GET", "/v1/health", b"", "").0, 200);
        completion
            .recv_timeout(Duration::from_secs(3))
            .expect("continuous upload outlived linger deadline");
        writer.join().unwrap();
        assert_eq!(server.request("GET", "/v1/health", b"", "").0, 200);
    }
    #[test]
    fn auth_and_routes() {
        let server = TestServer::start(Settings::default(), Some(digest("secret")), |_| {
            Box::new(Fake)
        });
        let (s, v, _) = server.request("GET", "/v1/health", b"", "");
        assert_eq!(s, 200);
        assert_eq!(v["auth"], "token");
        assert_eq!(v["ready"], true);
        let (s, _, h) = server.request("GET", "/v1/capabilities", b"", "");
        assert_eq!(s, 401);
        assert!(h.contains("WWW-Authenticate: Bearer"));
        for (method, path, status) in [
            ("GET", "/v1/capabilities", 200),
            ("GET", "/unknown", 404),
            ("POST", "/v1/health", 405),
            ("GET", "/v1/transcribe", 405),
        ] {
            assert_eq!(
                server
                    .request(method, path, b"", "Authorization: Bearer secret\r\n")
                    .0,
                status
            );
        }
        assert!(authorized(Some(&digest("secret")), Some("Bearer secret")));
        for candidate in [
            None,
            Some("Bearer secreu"),
            Some("Bearer secret extra"),
            Some("Basic secret"),
        ] {
            assert!(!authorized(Some(&digest("secret")), candidate));
        }
    }
    #[test]
    fn multipart_validation_and_shape() {
        let server = TestServer::start(Settings::default(), None, |_| Box::new(Fake));
        for (parts, status, code) in [
            (vec![("file", b"audio".as_slice())], 200, None),
            (
                vec![("file", b"audio".as_slice()), ("unknown", b"x")],
                400,
                Some(2),
            ),
            (
                vec![("file", b"a".as_slice()), ("file", b"b")],
                400,
                Some(2),
            ),
            (
                vec![("file", b"a".as_slice()), ("model", b"unknown")],
                400,
                Some(2),
            ),
            (vec![("file", b"missing".as_slice())], 503, Some(3)),
            (
                vec![("file", b"a".as_slice()), ("language", b"fa")],
                422,
                Some(7),
            ),
            (vec![("file", b"OpusHead".as_slice())], 415, Some(4)),
            (vec![("file", b"".as_slice())], 422, Some(4)),
            (
                vec![("file", b"a".as_slice()), ("maxDurationSec", b"601")],
                400,
                Some(2),
            ),
        ] {
            let (s, v, _) = server.request("POST", "/v1/transcribe", &body(&parts), "");
            assert_eq!(s, status, "{v}");
            if let Some(code) = code {
                assert_eq!(v["error"]["code"], code);
            } else {
                assert!(v.get("segments").is_none());
                assert!(v["languageProbability"].is_null());
                assert_eq!(v["contract"], 1);
            }
        }
        let (_, v, _) = server.request(
            "POST",
            "/v1/transcribe",
            &body(&[("file", b"a"), ("segments", b"true")]),
            "",
        );
        assert_eq!(v["segments"][0]["end"], 1.);
        let r = validate(
            &body(&[("file", b"a\r\n--xNOTBOUNDARY")]),
            "multipart/form-data; boundary=\"x\"",
            &Settings::default(),
        )
        .unwrap();
        assert_eq!(r.file, b"a\r\n--xNOTBOUNDARY");
        for data in [b"bad".as_slice(), b"--x\r\n\r\n--x--"] {
            assert!(
                validate(
                    data,
                    "multipart/form-data; boundary=x",
                    &Settings::default()
                )
                .is_err()
            );
        }
    }
    #[test]
    fn stalled_client_does_not_block_health() {
        let server = TestServer::start(Settings::default(), None, |_| Box::new(Fake));
        let mut stalled = TcpStream::connect(&server.address).unwrap();
        stalled
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        write!(stalled,"POST /v1/transcribe HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=x\r\nContent-Length: 4096\r\nExpect: 100-continue\r\n\r\n--x").unwrap();
        let mut interim = [0; 25];
        stalled.read_exact(&mut interim).unwrap();
        let mut health = server.connect("GET", "/v1/health", b"", "");
        health
            .set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read_reply(&mut health)));
        stalled.shutdown(std::net::Shutdown::Both).unwrap();
        assert_eq!(result.unwrap().0, 200);
    }
    struct Blocking {
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }
    impl Transcriber for Blocking {
        fn transcribe(&mut self, r: &Transcribe) -> Result<Value, ApiError> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            Fake.transcribe(r)
        }
        fn unload(&mut self) {}
    }
    #[test]
    fn queue_full_is_immediate() {
        let (entered, wait) = mpsc::channel();
        let (release, receive) = mpsc::channel();
        let mut receive = Some(receive);
        let mut settings = Settings::default();
        settings.server.max_queue = 0;
        settings.server.retry_after_secs = 2;
        let server = TestServer::start(settings, None, move |_| {
            Box::new(Blocking {
                entered: entered.clone(),
                release: receive.take().unwrap(),
            })
        });
        let first = server.connect("POST", "/v1/transcribe", &body(&[("file", b"a")]), "");
        wait.recv_timeout(Duration::from_secs(2)).unwrap();
        let (status, _, headers) =
            server.request("POST", "/v1/transcribe", &body(&[("file", b"b")]), "");
        release.send(()).unwrap();
        assert_eq!(status, 429);
        assert!(headers.contains("Retry-After: 2"));
        assert_eq!(read_reply(&mut { first }).0, 200);
    }
    #[test]
    fn limits_and_framing() {
        let mut settings = Settings::default();
        settings.server.max_upload_bytes = 1;
        settings.server.max_body_bytes = 250;
        let server = TestServer::start(settings, None, |_| Box::new(Fake));
        assert_eq!(
            server
                .request("POST", "/v1/transcribe", &body(&[("file", b"aa")]), "")
                .0,
            413
        );
        assert_eq!(
            server.request("POST", "/v1/transcribe", &[b'a'; 251], "").0,
            413
        );
        let mut s = TcpStream::connect(&server.address).unwrap();
        s.write_all(b"POST /v1/transcribe HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        assert_eq!(read_reply(&mut s).0, 400);
        let mut s = TcpStream::connect(&server.address).unwrap();
        s.write_all(b"POST /v1/transcribe HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n")
            .unwrap();
        assert_eq!(read_reply(&mut s).0, 400);
    }
    #[test]
    fn headers_and_handlers_are_bounded() {
        let mut settings = Settings::default();
        settings.server.max_header_bytes = 256;
        settings.server.max_handlers = 1;
        settings.server.read_timeout_secs = 1;
        let server = TestServer::start(settings.clone(), None, |_| Box::new(Fake));
        let mut oversized = TcpStream::connect(&server.address).unwrap();
        write!(
            oversized,
            "GET /v1/health HTTP/1.1\r\nX-Long: {}\r\n\r\n",
            "a".repeat(256)
        )
        .unwrap();
        oversized.shutdown(std::net::Shutdown::Write).unwrap();
        assert_eq!(read_reply(&mut oversized).0, 400);
        drop(oversized);
        drop(server);
        // A separate instance isolates handler capacity from the header
        // rejection's lingering cleanup, which runs after sending FIN.
        let server = TestServer::start(settings, None, |_| Box::new(Fake));
        let mut stalled = TcpStream::connect(&server.address).unwrap();
        stalled
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        write!(stalled,"POST /v1/transcribe HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=x\r\nContent-Length: 100\r\nExpect: 100-continue\r\n\r\n").unwrap();
        let mut interim = [0; 25];
        stalled.read_exact(&mut interim).unwrap();
        assert_eq!(server.request("GET", "/v1/health", b"", "").0, 429);
        stalled.shutdown(std::net::Shutdown::Both).unwrap();
    }
    #[test]
    fn health_readiness_is_honest() {
        let value = health_json(json!(["Whisper Tiny English"]), false, BTreeSet::new());
        assert_eq!(value["ready"], false);
        assert_eq!(value["missing"][0], "Whisper Tiny English");
        assert_eq!(value["auth"], "none");
        let value = health_json(json!([]), true, BTreeSet::from(["whisper-tiny-en".into()]));
        assert_eq!(value["ready"], true);
        assert_eq!(value["modelsLoaded"][0], "whisper-tiny-en");
    }
    #[test]
    fn audio_validation_without_models_or_devices() {
        assert_eq!(decode_audio(b"not audio", 1.).unwrap_err().code, Some(4));
        let make_wav = |samples: &[f32]| {
            let mut cursor = std::io::Cursor::new(Vec::new());
            let mut wav = hound::WavWriter::new(
                &mut cursor,
                hound::WavSpec {
                    channels: 2,
                    sample_rate: 48000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for sample in samples {
                wav.write_sample(*sample).unwrap();
            }
            wav.finalize().unwrap();
            cursor.into_inner()
        };
        let silent = make_wav(&vec![0.; 9600]);
        assert_eq!(decode_audio(&silent, 1.).unwrap_err().code, Some(4));
        let signal = make_wav(
            &(0..9600)
                .map(|i| (i as f32 * 0.1).sin() * 0.2)
                .collect::<Vec<_>>(),
        );
        let (samples, duration) = decode_audio(&signal, 1.).unwrap();
        assert!(!samples.is_empty());
        assert!((duration - 0.1).abs() < 0.01);
        let error = decode_audio(&signal, 0.01).unwrap_err();
        assert_eq!((error.status, error.code), (413, Some(5)));
    }
    #[test]
    fn token_file_is_trimmed_and_empty_is_rejected() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"  \n").unwrap();
        assert!(token(file.path().to_str().unwrap()).is_err());
        file.as_file_mut().set_len(0).unwrap();
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(b"  secret\n").unwrap();
        assert_eq!(
            token(file.path().to_str().unwrap()).unwrap(),
            Some(digest("secret"))
        );
        assert!(token("").unwrap().is_none());
    }
    #[test]
    fn quoted_multipart_filename_parameters() {
        assert_eq!(
            multipart_parameters("form-data; name=\"file\"; filename=\"a;b.wav\"").unwrap(),
            vec!["form-data", "name=\"file\"", "filename=\"a;b.wav\""]
        );
        assert!(multipart_parameters("form-data; name=\"file").is_err());
        let spaced = String::from_utf8(body(&[("file", b"audio")]))
            .unwrap()
            .replace("--x", "--a b");
        assert!(
            validate(
                spaced.as_bytes(),
                "Multipart/Form-Data; Boundary=\"a b\"",
                &Settings::default()
            )
            .is_ok()
        );
        let bytes=b"--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a;b.wav\"\r\n\r\naudio\r\n--x--\r\n";
        assert_eq!(
            validate(
                bytes,
                "multipart/form-data; boundary=x",
                &Settings::default()
            )
            .unwrap()
            .file,
            b"audio"
        );
    }
}
