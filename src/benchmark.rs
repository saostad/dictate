//! Offline, single-engine benchmark harness.
//!
//! This module deliberately loads one engine per process, keeps it resident,
//! and drives the same decoders, stream lifecycles, VAD feeder and text join
//! the daemon uses. It never types into a focused application and it never
//! downloads a model. `scripts/benchmark-models.py` starts one process per
//! engine/thread configuration and turns this JSON into tables and charts.
//!
//! The `benchmark` command is wired from `src/main.rs`:
//!
//! ```text
//! mod benchmark;
//! ...
//! "benchmark" => benchmark::run(args),
//! ```
//!
//! Usage:
//!
//! ```text
//! dictate benchmark --engine standard --threads 1 \
//!   --corpus tests/fixtures/benchmark-corpus.json --runs 5 --paced-runs 3
//! ```

use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::asr::{self, Asr, JoinMode};
use crate::audio;
use crate::config;
use crate::live::{self, Live};
use crate::model;
use crate::moonshine::{self, Moonshine, Session, Threads as MoonshineThreads};
use crate::segment::{Feeder, Segment};
use crate::vad::Vad;

const SAMPLE_RATE: u32 = 16_000;
const DEFAULT_CORPUS: &str = "tests/fixtures/benchmark-corpus.json";
const DEFAULT_JFK: &str = "tests/fixtures/jfk.wav";
const MOONSHINE_CHUNK_SAMPLES: usize = 1_600; // 100 ms, the daemon feeder's nominal chunk
const SYNTHETIC_SECONDS: usize = 5;
const SHORT_JFK_SECONDS: usize = 3;
const LONG_JFK_REPEATS: usize = 4;
const LONG_JFK_GAP_SAMPLES: usize = SAMPLE_RATE as usize / 4; // 0.25 s

const USAGE: &str = "\
Usage: dictate benchmark --engine <standard|fluid|whisper|live|moonshine> [options]

One process measures one engine and one thread setting, loading the model once.
JSON goes to stdout; progress and diagnostics go to stderr.

Options:
  --engine <name>       standard, fluid, whisper, live, or moonshine (required)
  --threads <setting>   default, 1, 2; moonshine also accepts runtime-default.
                        Moonshine has no exact 2-thread API.
  --corpus <manifest>   portable corpus manifest (default tests/fixtures/benchmark-corpus.json)
  --jfk <file.wav>      public-domain JFK reference (default tests/fixtures/jfk.wav)
  --runs <n>            warmed JFK repetitions (default 5)
  --paced-runs <n>      real-time JFK runs for live/moonshine (default 3)
  --model-dir <dir>     model tree; defaults to DICTATE_MODEL_DIR or the app data dir
  --quick               small smoke run: 1 JFK run, 1 paced run, first 2 corpus clips
  --help                show this help

The corpus manifest's clip paths are relative to the manifest file. A prepared
corpus can be built once with scripts/prepare-benchmark-corpus.py; neither this
command nor the orchestration script downloads models or corpus audio.";

/// One engine per process. The same enum is used for flags and JSON labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Engine {
    Standard,
    Fluid,
    Whisper,
    Live,
    Moonshine,
}

impl Engine {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "standard" => Some(Self::Standard),
            "fluid" => Some(Self::Fluid),
            "whisper" => Some(Self::Whisper),
            "live" => Some(Self::Live),
            "moonshine" => Some(Self::Moonshine),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Fluid => "fluid",
            Self::Whisper => "whisper",
            Self::Live => "live",
            Self::Moonshine => "moonshine",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Standard => "Parakeet TDT 0.6B v2 int8",
            Self::Fluid => "Fluid Speech Mini int8",
            Self::Whisper => "Whisper Tiny English Q8_0 (whisper.cpp)",
            Self::Live => "Nemotron streaming en 0.6B int8",
            Self::Moonshine => "Moonshine Tiny Streaming (native C)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThreadChoice {
    Default,
    One,
    Two,
    RuntimeDefault,
}

impl ThreadChoice {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "default" | "auto" => Some(Self::Default),
            "1" => Some(Self::One),
            "2" => Some(Self::Two),
            "runtime-default" => Some(Self::RuntimeDefault),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::One => "1",
            Self::Two => "2",
            Self::RuntimeDefault => "runtime-default",
        }
    }
}

#[derive(Debug, Clone)]
struct Options {
    engine: Engine,
    threads: ThreadChoice,
    corpus: PathBuf,
    jfk: PathBuf,
    runs: usize,
    paced_runs: usize,
    quick: bool,
    model_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct ThreadInfo {
    requested: &'static str,
    /// Exact native thread count, when the API has one. Moonshine's runtime
    /// default is deliberately `None`: it is not exactly two threads.
    effective: Option<usize>,
    policy: String,
    moonshine: Option<MoonshineThreads>,
}

impl ThreadInfo {
    fn json(&self) -> Value {
        let mut value = json!({
            "requested": self.requested,
            "effective": self.effective,
            "policy": self.policy,
        });
        if let Some(policy) = self.moonshine {
            value["moonshine_policy"] = json!(match policy {
                MoonshineThreads::Single => {
                    "single-thread mode (MOONSHINE_ORT_SINGLE_THREAD=1)"
                }
                MoonshineThreads::RuntimeDefault => {
                    "ONNX Runtime default threads (not exactly 2; the pinned C API has no count)"
                }
            });
        }
        value
    }
}

/// Entry point. `args` are the arguments after the `benchmark` command.
pub(crate) fn run(args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let args: Vec<String> = args.collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let options = parse_args(&args)?;
    if let Some(root) = &options.model_dir {
        apply_model_root(root)?;
    }

    match execute(&options) {
        Ok(document) => {
            emit(&document)?;
            Ok(())
        }
        Err(err) => {
            // A missing model or a bad corpus should still produce a machine-
            // readable error row before the process exits nonzero, so the
            // orchestration script can keep the failure instead of dropping it.
            let document = error_document(&options, "benchmark", &err);
            let _ = emit(&document);
            Err(err).context("benchmark failed")
        }
    }
}

fn emit(document: &Value) -> anyhow::Result<()> {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, document).context("Writing benchmark JSON")?;
    use std::io::Write;
    writeln!(lock).context("Writing benchmark JSON")?;
    Ok(())
}

fn parse_args(args: &[String]) -> anyhow::Result<Options> {
    let mut engine = None;
    let mut threads = ThreadChoice::Default;
    let mut corpus = PathBuf::from(DEFAULT_CORPUS);
    let mut jfk = PathBuf::from(DEFAULT_JFK);
    let mut runs = 5usize;
    let mut paced_runs = 3usize;
    let mut quick = false;
    let mut model_dir = None;
    let mut index = 0usize;

    while let Some(argument) = args.get(index) {
        let (flag, inline) = match argument.split_once('=') {
            Some((flag, value)) => (flag, Some(value)),
            None => (argument.as_str(), None),
        };
        match flag {
            "--engine" => {
                let value = flag_value(args, &mut index, inline, flag)?;
                engine =
                    Some(Engine::parse(&value).ok_or_else(|| {
                        anyhow::anyhow!("unknown --engine \"{value}\"\n\n{USAGE}")
                    })?);
            }
            "--threads" => {
                let value = flag_value(args, &mut index, inline, flag)?;
                threads = ThreadChoice::parse(&value)
                    .ok_or_else(|| anyhow::anyhow!("unknown --threads \"{value}\"\n\n{USAGE}"))?;
            }
            "--corpus" => {
                corpus = PathBuf::from(flag_value(args, &mut index, inline, flag)?);
            }
            "--jfk" => {
                jfk = PathBuf::from(flag_value(args, &mut index, inline, flag)?);
            }
            "--runs" => {
                runs = parse_count(&flag_value(args, &mut index, inline, flag)?, flag)?;
            }
            "--paced-runs" => {
                paced_runs = parse_count(&flag_value(args, &mut index, inline, flag)?, flag)?;
            }
            "--model-dir" => {
                model_dir = Some(PathBuf::from(flag_value(args, &mut index, inline, flag)?));
            }
            "--quick" => {
                if inline.is_some() {
                    bail!("--quick takes no value\n\n{USAGE}");
                }
                quick = true;
            }
            unknown => bail!("unknown benchmark argument \"{unknown}\"\n\n{USAGE}"),
        }
        index += 1;
    }

    let engine =
        engine.context("missing --engine; choose standard, fluid, whisper, live or moonshine")?;
    if quick {
        runs = 1;
        paced_runs = 1;
    }
    Ok(Options {
        engine,
        threads,
        corpus,
        jfk,
        runs,
        paced_runs,
        quick,
        model_dir,
    })
}

fn flag_value(
    args: &[String],
    index: &mut usize,
    inline: Option<&str>,
    flag: &str,
) -> anyhow::Result<String> {
    if let Some(value) = inline {
        if value.is_empty() {
            bail!("{flag} needs a value\n\n{USAGE}");
        }
        return Ok(value.to_string());
    }
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{flag} needs a value\n\n{USAGE}"))
}

fn parse_count(value: &str, flag: &str) -> anyhow::Result<usize> {
    let count: usize = value
        .parse()
        .with_context(|| format!("{flag} expects a positive integer, got \"{value}\""))?;
    if count == 0 {
        bail!("{flag} must be at least 1");
    }
    Ok(count)
}

/// `--model-dir` is the models root (the directory containing the Parakeet
/// child and its siblings), but the parakeet child itself is accepted too.
/// Setting this once, before any model or worker thread exists, keeps the
/// benchmark out of the user's normal model tree without touching config files.
fn apply_model_root(root: &Path) -> anyhow::Result<()> {
    if !root.is_dir() {
        bail!(
            "--model-dir {} is not a directory. Pass the models folder (the parent of parakeet-tdt-0.6b-v2-int8 and its siblings)",
            root.display()
        );
    }
    let parakeet = root.join("parakeet-tdt-0.6b-v2-int8");
    let selected = if parakeet.is_dir() {
        parakeet
    } else {
        root.to_path_buf()
    };
    // SAFETY: this runs before any engine/model worker thread exists, and the
    // process is dedicated to the benchmark. No user file is changed.
    unsafe {
        std::env::set_var("DICTATE_MODEL_DIR", &selected);
    }
    Ok(())
}

fn resolve_threads(engine: Engine, choice: ThreadChoice) -> anyhow::Result<ThreadInfo> {
    match engine {
        Engine::Standard | Engine::Fluid | Engine::Live => {
            let effective = match choice {
                ThreadChoice::Default => asr::intra_threads(),
                ThreadChoice::One => 1,
                ThreadChoice::Two => 2,
                ThreadChoice::RuntimeDefault => bail!(
                    "--threads runtime-default applies only to Moonshine; use default, 1 or 2 for {}",
                    engine.as_str()
                ),
            };
            Ok(ThreadInfo {
                requested: choice.as_str(),
                effective: Some(effective),
                policy: if choice == ThreadChoice::Default {
                    "app default: ONNX Runtime intra-op threads = available parallelism clamped to 1..=16".to_string()
                } else {
                    "explicit native intra-op thread count".to_string()
                },
                moonshine: None,
            })
        }
        Engine::Whisper => {
            let effective = match choice {
                ThreadChoice::Default => config::small_model_threads(),
                ThreadChoice::One => 1,
                ThreadChoice::Two => 2,
                ThreadChoice::RuntimeDefault => bail!(
                    "--threads runtime-default applies only to Moonshine; use default, 1 or 2 for whisper"
                ),
            };
            Ok(ThreadInfo {
                requested: choice.as_str(),
                effective: Some(effective),
                policy: if choice == ThreadChoice::Default {
                    "app default: half the logical CPUs, capped at 2 (isolated from user config)"
                        .to_string()
                } else {
                    "explicit whisper.cpp n_threads".to_string()
                },
                moonshine: None,
            })
        }
        Engine::Moonshine => {
            let selected = match choice {
                ThreadChoice::Default | ThreadChoice::One => MoonshineThreads::Single,
                ThreadChoice::RuntimeDefault => MoonshineThreads::RuntimeDefault,
                ThreadChoice::Two => bail!(
                    "Moonshine's pinned native API has no exact two-thread setting. Use --threads 1 or --threads runtime-default"
                ),
            };
            Ok(ThreadInfo {
                requested: choice.as_str(),
                effective: (selected == MoonshineThreads::Single).then_some(1),
                policy: match selected {
                    MoonshineThreads::Single => {
                        "app default / explicit 1: MOONSHINE_ORT_SINGLE_THREAD=1".to_string()
                    }
                    MoonshineThreads::RuntimeDefault => {
                        "runtime default: leave ONNX Runtime's own thread count in place (not exactly 2)"
                            .to_string()
                    }
                },
                moonshine: Some(selected),
            })
        }
    }
}

fn execute(options: &Options) -> anyhow::Result<Value> {
    let preloaded = preload_audio(options)?;
    let threads = resolve_threads(options.engine, options.threads)?;
    match options.engine {
        Engine::Standard | Engine::Fluid | Engine::Whisper => {
            benchmark_asr(options, &threads, &preloaded)
        }
        Engine::Live => benchmark_live(options, &threads, &preloaded),
        Engine::Moonshine => benchmark_moonshine(options, &threads, &preloaded),
    }
}

// ---------------------------------------------------------------------------
// Audio and corpus preloading
// ---------------------------------------------------------------------------

struct CorpusManifest {
    path: PathBuf,
    name: String,
    source_url: Option<String>,
    dataset: Option<String>,
    revision: Option<String>,
    license: Option<String>,
    selection: Option<String>,
    clips: Vec<CorpusClip>,
}

struct CorpusClip {
    id: String,
    path: PathBuf,
    reference: String,
    expected_duration: Option<f64>,
    expected_wav_sha256: Option<String>,
    speaker_id: Option<i64>,
    speaker_gender: Option<String>,
    dataset_row: Option<i64>,
}

struct PreloadedClip {
    id: String,
    reference: String,
    samples: Vec<f32>,
    duration_seconds: f64,
    speaker_id: Option<i64>,
    speaker_gender: Option<String>,
    dataset_row: Option<i64>,
}

struct PreloadedCorpus {
    manifest: CorpusManifest,
    clips: Vec<PreloadedClip>,
    audio_seconds: f64,
}

struct PreloadedAudio {
    corpus: PreloadedCorpus,
    jfk: Vec<f32>,
    jfk_seconds: f64,
    preload_ms: u128,
}

fn preload_audio(options: &Options) -> anyhow::Result<PreloadedAudio> {
    let started = Instant::now();
    let manifest = load_manifest(&options.corpus)?;
    let mut clips = Vec::with_capacity(manifest.clips.len());
    let mut audio_seconds = 0.0f64;
    for clip in &manifest.clips {
        validate_pcm16_wav(&clip.path)?;
        if let Some(expected) = &clip.expected_wav_sha256 {
            let actual = sha256_file(&clip.path)?;
            if !actual.eq_ignore_ascii_case(expected) {
                bail!(
                    "Corpus WAV SHA-256 mismatch for {}: got {actual}, expected {expected}. Re-run scripts/prepare-benchmark-corpus.py to rebuild the corpus",
                    clip.path.display()
                );
            }
        }
        let samples = audio::load_wav_16k_mono(&clip.path)
            .with_context(|| format!("Could not preload {}", clip.path.display()))?;
        if samples.is_empty() {
            bail!("Corpus clip {} is empty", clip.path.display());
        }
        let duration = samples.len() as f64 / SAMPLE_RATE as f64;
        if let Some(expected) = clip.expected_duration
            && (duration - expected).abs() > 0.02
        {
            bail!(
                "Corpus clip {} is {duration:.3} s but the manifest says {expected:.3} s; the fixture and audio do not match",
                clip.path.display()
            );
        }
        audio_seconds += duration;
        clips.push(PreloadedClip {
            id: clip.id.clone(),
            reference: clip.reference.clone(),
            samples,
            duration_seconds: duration,
            speaker_id: clip.speaker_id,
            speaker_gender: clip.speaker_gender.clone(),
            dataset_row: clip.dataset_row,
        });
    }

    let jfk = audio::load_wav_16k_mono(&options.jfk)
        .with_context(|| format!("Could not preload JFK reference {}", options.jfk.display()))?;
    if jfk.is_empty() {
        bail!("JFK reference {} is empty", options.jfk.display());
    }
    let jfk_seconds = jfk.len() as f64 / SAMPLE_RATE as f64;

    Ok(PreloadedAudio {
        corpus: PreloadedCorpus {
            manifest,
            clips,
            audio_seconds,
        },
        jfk,
        jfk_seconds,
        preload_ms: started.elapsed().as_millis(),
    })
}

fn load_manifest(path: &Path) -> anyhow::Result<CorpusManifest> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("Could not read corpus manifest {}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .with_context(|| format!("Could not parse corpus manifest {}", path.display()))?;
    if value.get("schema_version").and_then(Value::as_u64) != Some(1) {
        bail!(
            "Corpus manifest {} must have schema_version 1 (see docs/model-benchmarks.md)",
            path.display()
        );
    }
    let clips_value = value
        .get("clips")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Corpus manifest {} has no clips list", path.display()))?;
    if clips_value.is_empty() {
        bail!("Corpus manifest {} has no clips", path.display());
    }
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    let mut clips = Vec::with_capacity(clips_value.len());
    let mut ids = HashSet::new();
    for (index, clip) in clips_value.iter().enumerate() {
        let id = clip
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("corpus clip {index} has no string id"))?
            .to_string();
        if !ids.insert(id.clone()) {
            bail!(
                "Corpus manifest {} has duplicate clip id {id}",
                path.display()
            );
        }
        let file = clip
            .get("file")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("corpus clip {id} has no file"))?;
        let file_path = PathBuf::from(file);
        let file_path = if file_path.is_absolute() {
            file_path
        } else {
            root.join(file_path)
        };
        let reference = clip
            .get("reference")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let expected_duration = clip.get("duration_seconds").and_then(Value::as_f64);
        let expected_wav_sha256 = clip
            .get("wav_sha256")
            .and_then(Value::as_str)
            .map(str::to_string);
        let speaker_id = clip.get("speaker_id").and_then(Value::as_i64);
        let speaker_gender = clip
            .get("speaker_gender")
            .and_then(Value::as_str)
            .map(str::to_string);
        let dataset_row = clip.get("dataset_row").and_then(Value::as_i64);
        clips.push(CorpusClip {
            id,
            path: file_path,
            reference,
            expected_duration,
            expected_wav_sha256,
            speaker_id,
            speaker_gender,
            dataset_row,
        });
    }
    let source = value.get("source").cloned().unwrap_or(Value::Null);
    Ok(CorpusManifest {
        path: path.to_path_buf(),
        name: value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("unnamed corpus")
            .to_string(),
        source_url: source
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_string),
        dataset: source
            .get("dataset")
            .and_then(Value::as_str)
            .map(str::to_string),
        revision: source
            .get("revision")
            .and_then(Value::as_str)
            .map(str::to_string),
        license: source
            .get("license")
            .and_then(Value::as_str)
            .map(str::to_string),
        selection: source
            .get("selection")
            .and_then(Value::as_str)
            .map(str::to_string),
        clips,
    })
}

fn validate_pcm16_wav(path: &Path) -> anyhow::Result<()> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Could not open {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_rate != SAMPLE_RATE
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        bail!(
            "Corpus audio {} must be 16 kHz mono PCM16, found {} Hz, {} channel(s), {} bit {:?}. Run scripts/prepare-benchmark-corpus.py",
            path.display(),
            spec.sample_rate,
            spec.channels,
            spec.bits_per_sample,
            spec.sample_format
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("Could not open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("Could not read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

// ---------------------------------------------------------------------------
// Common report shape
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct MemorySnapshot {
    rss_bytes: Option<u64>,
    peak_rss_bytes: Option<u64>,
}

impl MemorySnapshot {
    fn json(self) -> Value {
        json!({
            "rss_bytes": self.rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
        })
    }
}

#[cfg(target_os = "linux")]
fn memory_snapshot() -> MemorySnapshot {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return MemorySnapshot::default();
    };
    let mut snapshot = MemorySnapshot::default();
    for line in status.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let kib = value
            .split_whitespace()
            .next()
            .and_then(|number| number.parse::<u64>().ok());
        let Some(kib) = kib else { continue };
        match name {
            "VmRSS" => snapshot.rss_bytes = Some(kib * 1024),
            "VmHWM" => snapshot.peak_rss_bytes = Some(kib * 1024),
            _ => {}
        }
    }
    snapshot
}

#[cfg(not(target_os = "linux"))]
fn memory_snapshot() -> MemorySnapshot {
    MemorySnapshot::default()
}

fn model_files(dirs: &[PathBuf]) -> anyhow::Result<(Vec<Value>, u64)> {
    let mut files = Vec::new();
    for dir in dirs {
        collect_dir_files(dir, 0, &mut files)?;
    }
    files.sort_by(|left, right| {
        left.get("path")
            .and_then(Value::as_str)
            .cmp(&right.get("path").and_then(Value::as_str))
    });
    let total = files
        .iter()
        .filter_map(|file| file.get("bytes").and_then(Value::as_u64))
        .sum();
    Ok((files, total))
}

fn collect_dir_files(dir: &Path, depth: usize, out: &mut Vec<Value>) -> anyhow::Result<()> {
    if depth > 6 {
        return Ok(());
    }
    let mut entries = fs::read_dir(dir)
        .with_context(|| format!("Could not list model directory {}", dir.display()))?
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("Could not list model directory {}", dir.display()))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        let metadata = entry
            .metadata()
            .with_context(|| format!("Could not stat {}", path.display()))?;
        if metadata.is_dir() {
            collect_dir_files(&path, depth + 1, out)?;
        } else if metadata.is_file() {
            out.push(json!({
                "path": path.display().to_string(),
                "bytes": metadata.len(),
            }));
        }
    }
    Ok(())
}

fn engine_dirs(engine: Engine) -> anyhow::Result<(PathBuf, Vec<PathBuf>)> {
    let dir = match engine {
        Engine::Standard => model::model_dir_checked()?,
        Engine::Fluid => model::fluid_model_dir_checked()?,
        Engine::Whisper => model::whisper_model_dir_checked()?,
        Engine::Live => model::live_model_dir_checked()?,
        Engine::Moonshine => {
            model::check_moonshine_runtime()?;
            model::moonshine_model_dir_checked()?
        }
    };
    let mut dirs = vec![dir.clone()];
    if engine == Engine::Moonshine {
        dirs.push(model::moonshine_runtime_dir()?);
    }
    Ok((dir, dirs))
}

fn runtime_json(dirs: &[PathBuf], model_dir: &Path) -> anyhow::Result<Value> {
    let (files, bytes) = model_files(dirs)?;
    Ok(json!({
        "package_version": env!("CARGO_PKG_VERSION"),
        "executable": std::env::current_exe()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "unknown".to_string()),
        "model_dir": model_dir.display().to_string(),
        "extra_model_dirs": dirs.iter().skip(1).map(|dir| dir.display().to_string()).collect::<Vec<_>>(),
        "model_files": files,
        "model_bytes": bytes,
    }))
}

fn options_json(options: &Options) -> Value {
    json!({
        "engine": options.engine.as_str(),
        "threads": options.threads.as_str(),
        "corpus": options.corpus.display().to_string(),
        "jfk": options.jfk.display().to_string(),
        "runs": options.runs,
        "paced_runs": options.paced_runs,
        "quick": options.quick,
        "model_dir": options.model_dir.as_ref().map(|path| path.display().to_string()),
    })
}

fn preload_json(options: &Options, audio: &PreloadedAudio) -> Value {
    json!({
        "ms": audio.preload_ms,
        "corpus_manifest": options.corpus.display().to_string(),
        "corpus_clips": audio.corpus.clips.len(),
        "corpus_audio_seconds": audio.corpus.audio_seconds,
        "jfk_path": options.jfk.display().to_string(),
        "jfk_seconds": audio.jfk_seconds,
    })
}

fn corpus_json(corpus: &PreloadedCorpus) -> Value {
    json!({
        "path": corpus.manifest.path.display().to_string(),
        "name": corpus.manifest.name,
        "source_url": corpus.manifest.source_url,
        "dataset": corpus.manifest.dataset,
        "revision": corpus.manifest.revision,
        "license": corpus.manifest.license,
        "selection": corpus.manifest.selection,
        "clips": corpus.clips.len(),
        "audio_seconds": corpus.audio_seconds,
    })
}

fn base_document(
    options: &Options,
    threads: &ThreadInfo,
    runtime: Value,
    audio: &PreloadedAudio,
    memory_after_load: MemorySnapshot,
    model_load_ms: u128,
) -> Value {
    json!({
        "schema_version": 1,
        "kind": "dictate-engine-benchmark",
        "status": "ok",
        "generated_unix_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0),
        "engine": options.engine.as_str(),
        "engine_label": options.engine.label(),
        "threads": threads.json(),
        "options": options_json(options),
        "runtime": runtime,
        "preload": preload_json(options, audio),
        "memory": {
            "units": "bytes; Linux /proc/self/status VmRSS and VmHWM for the whole process, including runtime and scratch, never weights-only RAM",
            "after_load": memory_after_load.json(),
            "at_end": Value::Null,
        },
        "load": {
            "model_load_ms": model_load_ms,
            "first_decode_ms": Value::Null,
            "first_chunk_ms": Value::Null,
            "warmup_transcript": Value::Null,
            "warmup_total_ms": Value::Null,
        },
        "jfk": {
            "path": options.jfk.display().to_string(),
            "audio_seconds": audio.jfk_seconds,
            "runs": [],
        },
        "whole_vs_paused_jfk": Value::Null,
        "quality": {
            "corpus": corpus_json(&audio.corpus),
            "runs": [],
        },
        "stress": [],
        "paced": Value::Null,
        "engine_errors": [],
    })
}

fn error_document(options: &Options, stage: &str, err: &anyhow::Error) -> Value {
    json!({
        "schema_version": 1,
        "kind": "dictate-engine-benchmark",
        "status": "error",
        "stage": stage,
        "engine": options.engine.as_str(),
        "engine_label": options.engine.label(),
        "threads": {
            "requested": options.threads.as_str(),
            "effective": Value::Null,
        },
        "options": options_json(options),
        "error": format!("{err:#}"),
    })
}

fn finish_document(document: &mut Value, memory_end: MemorySnapshot, errors: Vec<Value>) {
    document["memory"]["at_end"] = memory_end.json();
    if errors.is_empty() {
        document["status"] = json!("ok");
    } else {
        document["status"] = json!("partial");
        document["engine_errors"] = Value::Array(errors);
    }
}

fn push_engine_error(errors: &mut Vec<Value>, stage: &str, id: &str, error: impl Into<String>) {
    errors.push(json!({
        "stage": stage,
        "id": id,
        "error": error.into(),
    }));
}

// ---------------------------------------------------------------------------
// Whole/Pauses engines
// ---------------------------------------------------------------------------

fn benchmark_asr(
    options: &Options,
    threads: &ThreadInfo,
    audio: &PreloadedAudio,
) -> anyhow::Result<Value> {
    let (dir, dirs) = engine_dirs(options.engine)?;
    let runtime = runtime_json(&dirs, &dir)?;
    let load_started = Instant::now();
    let mut asr = match options.engine {
        Engine::Standard => Asr::load_with_threads(&dir, threads.effective.unwrap_or(1))?,
        Engine::Fluid => Asr::load_fluid_with_threads(&dir, threads.effective.unwrap_or(1))?,
        Engine::Whisper => Asr::load_whisper_with_threads(&dir, threads.effective.unwrap_or(1))?,
        _ => unreachable!("benchmark_asr only handles Whole/Pauses engines"),
    };
    let model_load_ms = load_started.elapsed().as_millis();
    let memory_after_load = memory_snapshot();
    let mut document = base_document(
        options,
        threads,
        runtime,
        audio,
        memory_after_load,
        model_load_ms,
    );
    let mut errors = Vec::new();

    // One warmed decode, with the first decode timed separately from load.
    let warmup_started = Instant::now();
    let warmup = asr.transcribe(&audio.jfk);
    let warmup_total_ms = warmup_started.elapsed().as_millis();
    match warmup {
        Ok(text) => {
            document["load"]["first_decode_ms"] = json!(warmup_total_ms);
            document["load"]["warmup_transcript"] = json!(text);
        }
        Err(err) => {
            document["load"]["first_decode_error"] = json!(format!("{err:#}"));
            push_engine_error(&mut errors, "warmup", "jfk", format!("{err:#}"));
        }
    }
    document["load"]["warmup_total_ms"] = json!(warmup_total_ms);

    let mut jfk_runs = Vec::with_capacity(options.runs);
    for run in 1..=options.runs {
        let started = Instant::now();
        let result = asr.transcribe(&audio.jfk);
        let decode_ms = started.elapsed().as_millis();
        match result {
            Ok(transcript) => jfk_runs.push(json!({
                "run": run,
                "status": "ok",
                "decode_ms": decode_ms,
                "rtf": rtf(decode_ms as f64, audio.jfk_seconds),
                "transcript": transcript,
            })),
            Err(err) => {
                let error = format!("{err:#}");
                push_engine_error(&mut errors, "jfk_run", &run.to_string(), &error);
                jfk_runs.push(json!({
                    "run": run,
                    "status": "error",
                    "decode_ms": decode_ms,
                    "rtf": rtf(decode_ms as f64, audio.jfk_seconds),
                    "transcript": Value::Null,
                    "error": error,
                }));
            }
        }
    }
    document["jfk"]["runs"] = Value::Array(jfk_runs);

    // A single paired warm-model Whole vs the production VAD feeder Pauses
    // path. This is a reference comparison, not a third quality corpus and not
    // a statistical distribution.
    let whole_started = Instant::now();
    let whole_result = asr.transcribe(&audio.jfk);
    let whole_ms = whole_started.elapsed().as_millis();
    let paused = paused_jfk(&mut asr, &audio.jfk);
    let paused_json = match paused {
        Ok(mut paused) => {
            paused["whole_ms"] = json!(whole_ms);
            match &whole_result {
                Ok(text) => {
                    paused["whole_status"] = json!("ok");
                    paused["whole_transcript"] = json!(text);
                }
                Err(err) => {
                    let error = format!("{err:#}");
                    paused["whole_status"] = json!("error");
                    paused["whole_error"] = json!(error);
                    push_engine_error(&mut errors, "paused_whole", "jfk", error);
                }
            }
            paused
        }
        Err(err) => {
            let error = format!("{err:#}");
            push_engine_error(&mut errors, "paused_pipeline", "jfk", &error);
            json!({
                "status": "error",
                "whole_ms": whole_ms,
                "whole_status": if whole_result.is_ok() { "ok" } else { "error" },
                "whole_transcript": whole_result.ok(),
                "error": error,
            })
        }
    };
    if paused_json.get("status").and_then(Value::as_str) == Some("partial") {
        let failures = paused_json
            .get("paused_failures")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        push_engine_error(
            &mut errors,
            "paused_pipeline",
            "jfk",
            format!("{failures} paused segment decode(s) failed; see whole_vs_paused_jfk"),
        );
    }
    document["whole_vs_paused_jfk"] = paused_json;

    let quality_limit = if options.quick { 2 } else { usize::MAX };
    document["quality"] = quality_asr(&mut asr, &audio.corpus, quality_limit, &mut errors);
    document["stress"] = Value::Array(stress_asr(&mut asr, &audio.jfk, &mut errors));
    finish_document(&mut document, memory_snapshot(), errors);
    Ok(document)
}

/// Run the real `segment::Feeder` (VAD included) and the real segment decode +
/// join path over JFK, returning both the VAD/scoring cost and decode cost.
fn paused_jfk(asr: &mut Asr, samples: &[f32]) -> anyhow::Result<Value> {
    let vad_started = Instant::now();
    let vad = Vad::load_default().context("Loading the Silero VAD for the paused pipeline")?;
    let vad_load_ms = vad_started.elapsed().as_millis();
    let mut feeder = Feeder::new(SAMPLE_RATE, vad)?;
    let mut state = PauseJoin::default();
    let mut stats = PauseStats::default();
    let started = Instant::now();
    // VAD pushes are usually below one millisecond. Preserve their fractional
    // time before summing so a recording cannot incorrectly report zero cost.
    let mut vad_ms = 0.0f64;
    for piece in samples.chunks((SAMPLE_RATE / 10) as usize) {
        let push_started = Instant::now();
        let segments = feeder.push(piece)?;
        vad_ms += push_started.elapsed().as_secs_f64() * 1000.0;
        process_segments(asr, &segments, &mut state, &mut stats);
    }
    let push_started = Instant::now();
    let segments = feeder.finish()?;
    vad_ms += push_started.elapsed().as_secs_f64() * 1000.0;
    process_segments(asr, &segments, &mut state, &mut stats);
    let total_ms = started.elapsed().as_millis();

    if !state.typed_words.is_empty()
        && let Some(punct) = state.pending
    {
        state.joined.push(punct);
    }
    Ok(json!({
        "status": if stats.failures == 0 { "ok" } else { "partial" },
        "trials": 1,
        "note": "single paired warm JFK trial; these are raw values, not a statistical distribution",
        "vad_fresh_load": true,
        "vad_load_in_paused_total": false,
        "vad_load_ms": vad_load_ms,
        "paused_total_ms": total_ms,
        "paused_vad_ms": vad_ms,
        "paused_decode_ms": stats.decode_ms,
        "paused_segments": stats.segments,
        "paused_failures": stats.failures,
        "paused_transcript": state.joined,
        "errors": stats.errors,
    }))
}

#[derive(Default)]
struct PauseJoin {
    joined: String,
    typed_words: Vec<String>,
    pending: Option<char>,
}

#[derive(Default)]
struct PauseStats {
    segments: usize,
    decode_ms: u128,
    failures: usize,
    errors: Vec<String>,
}

fn process_segments(
    asr: &mut Asr,
    segments: &[Segment],
    state: &mut PauseJoin,
    stats: &mut PauseStats,
) {
    for segment in segments {
        stats.segments += 1;
        let started = Instant::now();
        let piece = asr.transcribe_segment_pieces(&segment.context, &segment.audio, segment.start);
        stats.decode_ms += started.elapsed().as_millis();
        let piece = match piece {
            Ok(piece) => piece,
            Err(err) => {
                stats.failures += 1;
                stats.errors.push(format!("{err:#}"));
                continue;
            }
        };
        if piece.words.is_empty() {
            continue;
        }
        let tail_start = state.typed_words.len().saturating_sub(8);
        let (to_type, new_pending) = asr::join_piece(
            &state.typed_words[tail_start..],
            state.pending,
            piece.boundary,
            &piece.words,
            JoinMode::from_disjoint(piece.disjoint),
        );
        state.pending = new_pending;
        if to_type.is_empty() {
            continue;
        }
        state.joined.push_str(&to_type);
        asr::append_typed_words(&mut state.typed_words, &to_type);
    }
}

fn quality_asr(
    asr: &mut Asr,
    corpus: &PreloadedCorpus,
    limit: usize,
    errors: &mut Vec<Value>,
) -> Value {
    let mut runs = Vec::new();
    for clip in corpus.clips.iter().take(limit) {
        let started = Instant::now();
        let result = asr.transcribe(&clip.samples);
        let decode_ms = started.elapsed().as_millis();
        let mut run = json!({
            "clip_id": clip.id,
            "reference": clip.reference,
            "duration_seconds": clip.duration_seconds,
            "speaker_id": clip.speaker_id,
            "speaker_gender": clip.speaker_gender,
            "dataset_row": clip.dataset_row,
            "decode_ms": decode_ms,
            "rtf": rtf(decode_ms as f64, clip.duration_seconds),
        });
        match result {
            Ok(transcript) => {
                run["status"] = json!("ok");
                run["transcript"] = json!(transcript);
            }
            Err(err) => {
                let error = format!("{err:#}");
                push_engine_error(errors, "quality", &clip.id, &error);
                run["status"] = json!("error");
                run["transcript"] = Value::Null;
                run["error"] = json!(error);
            }
        }
        runs.push(run);
    }
    json!({
        "corpus": corpus_json(corpus),
        "runs": runs,
    })
}

struct SyntheticCase {
    id: &'static str,
    kind: &'static str,
    label: &'static str,
    samples: Vec<f32>,
}

fn synthetic_cases(jfk: &[f32]) -> Vec<SyntheticCase> {
    let silence = vec![0.0f32; SYNTHETIC_SECONDS * SAMPLE_RATE as usize];
    let mut state = 0x9E37_79B9u32;
    let noise = (0..SYNTHETIC_SECONDS * SAMPLE_RATE as usize)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 0.02 - 0.01
        })
        .collect::<Vec<f32>>();

    let short_len = (SHORT_JFK_SECONDS * SAMPLE_RATE as usize).min(jfk.len());
    let short = jfk[..short_len].to_vec();

    let mut repeated =
        Vec::with_capacity(jfk.len() * LONG_JFK_REPEATS + LONG_JFK_GAP_SAMPLES * LONG_JFK_REPEATS);
    for repeat in 0..LONG_JFK_REPEATS {
        if repeat > 0 {
            repeated.extend(std::iter::repeat_n(0.0, LONG_JFK_GAP_SAMPLES));
        }
        repeated.extend_from_slice(jfk);
    }

    vec![
        SyntheticCase {
            id: "silence-5s",
            kind: "silence",
            label: "five seconds of digital silence; any words are false words",
            samples: silence,
        },
        SyntheticCase {
            id: "noise-5s",
            kind: "noise",
            label: "five seconds of deterministic low-level noise (uniform +/-0.01); any words are false words",
            samples: noise,
        },
        SyntheticCase {
            id: "jfk-short-3s",
            kind: "short_speech",
            label: "first three seconds of the public-domain JFK reference (synthetic truncation)",
            samples: short,
        },
        SyntheticCase {
            id: "jfk-repeated-4x",
            kind: "long_repeated_speech",
            label: "the JFK reference repeated four times with 0.25 s gaps (synthetic; not included in corpus WER)",
            samples: repeated,
        },
    ]
}

fn stress_asr(asr: &mut Asr, jfk: &[f32], errors: &mut Vec<Value>) -> Vec<Value> {
    let cases = synthetic_cases(jfk);
    let mut out = Vec::with_capacity(cases.len());
    for case in cases {
        let duration = case.samples.len() as f64 / SAMPLE_RATE as f64;
        let started = Instant::now();
        let result = asr.transcribe(&case.samples);
        let decode_ms = started.elapsed().as_millis();
        let mut row = json!({
            "case_id": case.id,
            "kind": case.kind,
            "label": case.label,
            "synthetic": true,
            "duration_seconds": duration,
            "decode_ms": decode_ms,
            "rtf": rtf(decode_ms as f64, duration),
        });
        match result {
            Ok(transcript) => {
                row["status"] = json!("ok");
                row["transcript"] = json!(transcript);
            }
            Err(err) => {
                let error = format!("{err:#}");
                push_engine_error(errors, "stress", case.id, &error);
                row["status"] = json!("error");
                row["transcript"] = Value::Null;
                row["error"] = json!(error);
            }
        }
        out.push(row);
    }
    out
}

// ---------------------------------------------------------------------------
// Streaming engines
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct StreamMetrics {
    status: &'static str,
    error: Option<String>,
    transcript: String,
    decode_ms: f64,
    chunks: usize,
    first_chunk_ms: Option<f64>,
    first_committed_ms: Option<f64>,
    first_committed_audio_ms: Option<f64>,
    first_committed_text: Option<String>,
    first_provisional_ms: Option<f64>,
    first_provisional_text: Option<String>,
    audio_arrival_end_ms: f64,
    stop_to_final_ms: Option<f64>,
    finalization_compute_ms: f64,
    max_backlog_ms: f64,
    max_lag_ms: f64,
    final_partial: Option<String>,
}

impl StreamMetrics {
    fn new(audio_seconds: f64) -> Self {
        Self {
            status: "ok",
            error: None,
            transcript: String::new(),
            decode_ms: 0.0,
            chunks: 0,
            first_chunk_ms: None,
            first_committed_ms: None,
            first_committed_audio_ms: None,
            first_committed_text: None,
            first_provisional_ms: None,
            first_provisional_text: None,
            audio_arrival_end_ms: audio_seconds * 1000.0,
            stop_to_final_ms: None,
            finalization_compute_ms: 0.0,
            max_backlog_ms: 0.0,
            max_lag_ms: 0.0,
            final_partial: None,
        }
    }

    fn mark_error(&mut self, error: impl Into<String>) {
        self.status = "error";
        self.error = Some(error.into());
    }
}

fn rtf(process_ms: f64, audio_seconds: f64) -> f64 {
    if audio_seconds <= 0.0 {
        0.0
    } else {
        process_ms / (audio_seconds * 1000.0)
    }
}

fn stream_run_json(metrics: &StreamMetrics, run: usize, audio_seconds: f64, paced: bool) -> Value {
    let mut value = json!({
        "run": run,
        "status": metrics.status,
        "decode_ms": metrics.decode_ms,
        "rtf": rtf(metrics.decode_ms, audio_seconds),
        "chunks": metrics.chunks,
        "transcript": metrics.transcript,
        "error": metrics.error,
    });
    if paced {
        value["first_chunk_ms"] = json!(metrics.first_chunk_ms);
        value["first_provisional_ms"] = json!(metrics.first_provisional_ms);
        value["first_provisional_text"] = json!(metrics.first_provisional_text);
        value["first_committed_ms"] = json!(metrics.first_committed_ms);
        value["first_committed_audio_ms"] = json!(metrics.first_committed_audio_ms);
        value["first_committed_text"] = json!(metrics.first_committed_text);
        value["audio_arrival_end_ms"] = json!(metrics.audio_arrival_end_ms);
        value["stop_to_final_ms"] = json!(metrics.stop_to_final_ms);
        value["finalization_compute_ms"] = json!(metrics.finalization_compute_ms);
        value["max_backlog_ms"] = json!(metrics.max_backlog_ms);
        value["max_lag_ms"] = json!(metrics.max_lag_ms);
        value["final_partial"] = json!(metrics.final_partial);
    }
    value
}

fn benchmark_live(
    options: &Options,
    threads: &ThreadInfo,
    audio: &PreloadedAudio,
) -> anyhow::Result<Value> {
    let (dir, dirs) = engine_dirs(options.engine)?;
    let runtime = runtime_json(&dirs, &dir)?;
    let load_started = Instant::now();
    let environment = asr::shared_environment()?;
    let mut live = Live::load(&dir, &environment, threads.effective.unwrap_or(1))?;
    let model_load_ms = load_started.elapsed().as_millis();
    let memory_after_load = memory_snapshot();
    let mut document = base_document(
        options,
        threads,
        runtime,
        audio,
        memory_after_load,
        model_load_ms,
    );
    let mut errors = Vec::new();

    let warmup = run_live_stream(&mut live, &audio.jfk, false);
    // `first_decode_ms` is the compute for the whole first JFK recording, so
    // it is comparable with the Whole/Pauses engines. The first streaming
    // chunk is reported separately as `first_chunk_ms`.
    document["load"]["first_decode_ms"] = json!(warmup.decode_ms);
    document["load"]["first_chunk_ms"] = json!(warmup.first_chunk_ms);
    document["load"]["warmup_transcript"] = json!(warmup.transcript);
    document["load"]["warmup_total_ms"] = json!(warmup.decode_ms);
    if warmup.status != "ok" {
        push_engine_error(
            &mut errors,
            "warmup",
            "jfk",
            warmup
                .error
                .unwrap_or_else(|| "unknown streaming error".to_string()),
        );
    }

    let mut jfk_runs = Vec::with_capacity(options.runs);
    for run in 1..=options.runs {
        let metrics = run_live_stream(&mut live, &audio.jfk, false);
        if metrics.status != "ok" {
            push_engine_error(
                &mut errors,
                "jfk_run",
                &run.to_string(),
                metrics
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown streaming error".to_string()),
            );
        }
        jfk_runs.push(stream_run_json(&metrics, run, audio.jfk_seconds, false));
    }
    document["jfk"]["runs"] = Value::Array(jfk_runs);

    let quality_limit = if options.quick { 2 } else { usize::MAX };
    document["quality"] = quality_live(&mut live, &audio.corpus, quality_limit, &mut errors);
    document["stress"] = Value::Array(stress_live(&mut live, &audio.jfk, &mut errors));

    // Paced after warmup, at n=3 by default. Slower-than-real-time is recorded,
    // never gated: a large stop-to-final time is a result.
    let mut paced = Vec::with_capacity(options.paced_runs);
    for run in 1..=options.paced_runs {
        let metrics = run_live_stream(&mut live, &audio.jfk, true);
        if metrics.status != "ok" {
            push_engine_error(
                &mut errors,
                "paced_run",
                &run.to_string(),
                metrics
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown streaming error".to_string()),
            );
        }
        paced.push(stream_run_json(&metrics, run, audio.jfk_seconds, true));
    }
    document["paced"] = json!({
        "applies": true,
        "method": "audio fed at real time; the remainder is padded and followed by 3 zero chunks, then the model is reset",
        "runs": paced,
    });

    finish_document(&mut document, memory_snapshot(), errors);
    Ok(document)
}

fn append_live_output(text: &mut String, first: &mut bool, out: &str) -> Option<String> {
    if out.is_empty() {
        return None;
    }
    if *first {
        *first = false;
        let shown = out.trim_start();
        text.push_str(shown);
        Some(shown.to_string())
    } else {
        text.push_str(out);
        Some(out.to_string())
    }
}

fn decode_live_chunk(live: &mut Live, piece: &[f32]) -> (Result<String, String>, f64) {
    let started = Instant::now();
    let result = live
        .transcribe_chunk(piece)
        .map_err(|err| format!("{err:#}"));
    (result, started.elapsed().as_secs_f64() * 1000.0)
}

fn run_live_stream(live: &mut Live, samples: &[f32], paced: bool) -> StreamMetrics {
    let audio_seconds = samples.len() as f64 / SAMPLE_RATE as f64;
    let mut metrics = StreamMetrics::new(audio_seconds);
    live.reset();
    let chunk = live.chunk_samples().max(1);
    let full_chunks = samples.len() / chunk;
    let run_started = Instant::now();
    let mut first_output = true;
    let mut failed = false;

    for index in 0..full_chunks {
        let piece = &samples[index * chunk..(index + 1) * chunk];
        let arrival_ms = ((index + 1) * chunk) as f64 * 1000.0 / SAMPLE_RATE as f64;
        let deadline = run_started + Duration::from_secs_f64(arrival_ms / 1000.0);
        if paced {
            sleep_until(deadline);
            metrics.max_backlog_ms = metrics.max_backlog_ms.max(
                Instant::now()
                    .saturating_duration_since(deadline)
                    .as_secs_f64()
                    * 1000.0,
            );
        }
        let (result, decode_ms) = decode_live_chunk(live, piece);
        metrics.decode_ms += decode_ms;
        metrics.chunks += 1;
        if metrics.first_chunk_ms.is_none() {
            metrics.first_chunk_ms = Some(decode_ms);
        }
        if paced {
            metrics.max_lag_ms = metrics.max_lag_ms.max(
                Instant::now()
                    .saturating_duration_since(deadline)
                    .as_secs_f64()
                    * 1000.0,
            );
        }
        match result {
            Ok(out) => {
                if let Some(shown) =
                    append_live_output(&mut metrics.transcript, &mut first_output, &out)
                    && metrics.first_committed_ms.is_none()
                {
                    metrics.first_committed_ms = Some(run_started.elapsed().as_secs_f64() * 1000.0);
                    metrics.first_committed_audio_ms = Some(arrival_ms);
                    metrics.first_committed_text = Some(shown);
                }
            }
            Err(error) => {
                metrics.mark_error(error);
                failed = true;
                break;
            }
        }
    }

    if !failed {
        if paced {
            let deadline = run_started + Duration::from_secs_f64(audio_seconds);
            sleep_until(deadline);
            metrics.max_backlog_ms = metrics.max_backlog_ms.max(
                Instant::now()
                    .saturating_duration_since(deadline)
                    .as_secs_f64()
                    * 1000.0,
            );
        }
        // The final partial chunk is padded exactly as LiveEnd does. Three zero
        // chunks flush the model; all of that is finalization compute.
        let mut stop_pieces = Vec::new();
        if !samples.len().is_multiple_of(chunk) {
            let mut remainder = samples[full_chunks * chunk..].to_vec();
            remainder.resize(chunk, 0.0);
            stop_pieces.push(remainder);
        }
        stop_pieces.extend((0..live::FLUSH_CHUNKS).map(|_| vec![0.0; chunk]));
        for piece in &stop_pieces {
            let (result, decode_ms) = decode_live_chunk(live, piece);
            metrics.decode_ms += decode_ms;
            metrics.finalization_compute_ms += decode_ms;
            metrics.chunks += 1;
            if metrics.first_chunk_ms.is_none() {
                metrics.first_chunk_ms = Some(decode_ms);
            }
            match result {
                Ok(out) => {
                    if let Some(shown) =
                        append_live_output(&mut metrics.transcript, &mut first_output, &out)
                        && metrics.first_committed_ms.is_none()
                    {
                        metrics.first_committed_ms =
                            Some(run_started.elapsed().as_secs_f64() * 1000.0);
                        metrics.first_committed_audio_ms = Some(audio_seconds * 1000.0);
                        metrics.first_committed_text = Some(shown);
                    }
                }
                Err(error) => {
                    metrics.mark_error(error);
                    break;
                }
            }
        }
        if paced {
            metrics.stop_to_final_ms = Some(
                (run_started.elapsed().as_secs_f64() * 1000.0 - audio_seconds * 1000.0).max(0.0),
            );
            metrics.max_lag_ms = metrics.max_lag_ms.max(
                (run_started.elapsed().as_secs_f64() * 1000.0 - audio_seconds * 1000.0).max(0.0),
            );
        }
    }
    live.reset();
    metrics
}

fn quality_live(
    live: &mut Live,
    corpus: &PreloadedCorpus,
    limit: usize,
    errors: &mut Vec<Value>,
) -> Value {
    let mut runs = Vec::new();
    for clip in corpus.clips.iter().take(limit) {
        let metrics = run_live_stream(live, &clip.samples, false);
        if metrics.status != "ok" {
            push_engine_error(
                errors,
                "quality",
                &clip.id,
                metrics
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown streaming error".to_string()),
            );
        }
        let mut run = stream_run_json(&metrics, 0, clip.duration_seconds, false);
        run["clip_id"] = json!(clip.id);
        run["reference"] = json!(clip.reference);
        run["duration_seconds"] = json!(clip.duration_seconds);
        run["speaker_id"] = json!(clip.speaker_id);
        run["speaker_gender"] = json!(clip.speaker_gender);
        run["dataset_row"] = json!(clip.dataset_row);
        run["status"] = json!(metrics.status);
        run["transcript"] = json!(metrics.transcript);
        run["error"] = json!(metrics.error);
        run.as_object_mut().unwrap().remove("run");
        runs.push(run);
    }
    json!({
        "corpus": corpus_json(corpus),
        "runs": runs,
    })
}

fn stress_live(live: &mut Live, jfk: &[f32], errors: &mut Vec<Value>) -> Vec<Value> {
    synthetic_cases(jfk)
        .into_iter()
        .map(|case| {
            let duration = case.samples.len() as f64 / SAMPLE_RATE as f64;
            let metrics = run_live_stream(live, &case.samples, false);
            if metrics.status != "ok" {
                push_engine_error(
                    errors,
                    "stress",
                    case.id,
                    metrics
                        .error
                        .clone()
                        .unwrap_or_else(|| "unknown streaming error".to_string()),
                );
            }
            stream_stress_json(&metrics, case.id, case.kind, case.label, duration)
        })
        .collect()
}

fn stream_stress_json(
    metrics: &StreamMetrics,
    id: &str,
    kind: &str,
    label: &str,
    duration: f64,
) -> Value {
    json!({
        "case_id": id,
        "kind": kind,
        "label": label,
        "synthetic": true,
        "duration_seconds": duration,
        "status": metrics.status,
        "decode_ms": metrics.decode_ms,
        "rtf": rtf(metrics.decode_ms, duration),
        "chunks": metrics.chunks,
        "transcript": metrics.transcript,
        "error": metrics.error,
    })
}

fn benchmark_moonshine(
    options: &Options,
    threads: &ThreadInfo,
    audio: &PreloadedAudio,
) -> anyhow::Result<Value> {
    let (dir, dirs) = engine_dirs(options.engine)?;
    let runtime = runtime_json(&dirs, &dir)?;
    // The pinned C API reads this once, before its ONNX Runtime sessions
    // exist. Resolve it here and never mutate the environment afterwards.
    let policy = threads
        .moonshine
        .context("internal error: missing Moonshine thread policy")?;
    moonshine::configure_threads(policy);
    let load_started = Instant::now();
    let model = Moonshine::load(&dir, policy)?;
    let model_load_ms = load_started.elapsed().as_millis();
    let memory_after_load = memory_snapshot();
    let mut document = base_document(
        options,
        threads,
        runtime,
        audio,
        memory_after_load,
        model_load_ms,
    );
    let mut session = Session::new(model);
    let mut errors = Vec::new();

    let warmup = run_moonshine_stream(&mut session, &audio.jfk, false);
    // Match the Whole/Pauses engines: the warmup decode covers the entire
    // first JFK recording. The first 100 ms chunk is a separate diagnostic.
    document["load"]["first_decode_ms"] = json!(warmup.decode_ms);
    document["load"]["first_chunk_ms"] = json!(warmup.first_chunk_ms);
    document["load"]["warmup_transcript"] = json!(warmup.transcript);
    document["load"]["warmup_total_ms"] = json!(warmup.decode_ms);
    if warmup.status != "ok" {
        push_engine_error(
            &mut errors,
            "warmup",
            "jfk",
            warmup
                .error
                .unwrap_or_else(|| "unknown streaming error".to_string()),
        );
    }

    let mut jfk_runs = Vec::with_capacity(options.runs);
    for run in 1..=options.runs {
        let metrics = run_moonshine_stream(&mut session, &audio.jfk, false);
        if metrics.status != "ok" {
            push_engine_error(
                &mut errors,
                "jfk_run",
                &run.to_string(),
                metrics
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown streaming error".to_string()),
            );
        }
        jfk_runs.push(stream_run_json(&metrics, run, audio.jfk_seconds, false));
    }
    document["jfk"]["runs"] = Value::Array(jfk_runs);

    let quality_limit = if options.quick { 2 } else { usize::MAX };
    document["quality"] =
        quality_moonshine(&mut session, &audio.corpus, quality_limit, &mut errors);
    document["stress"] = Value::Array(stress_moonshine(&mut session, &audio.jfk, &mut errors));

    let mut paced = Vec::with_capacity(options.paced_runs);
    for run in 1..=options.paced_runs {
        let metrics = run_moonshine_stream(&mut session, &audio.jfk, true);
        if metrics.status != "ok" {
            push_engine_error(
                &mut errors,
                "paced_run",
                &run.to_string(),
                metrics
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown streaming error".to_string()),
            );
        }
        paced.push(stream_run_json(&metrics, run, audio.jfk_seconds, true));
    }
    document["paced"] = json!({
        "applies": true,
        "method": "audio fed at real time in 100 ms chunks; Session::finish() drains the stream and only completed lines are joined",
        "runs": paced,
    });

    finish_document(&mut document, memory_snapshot(), errors);
    Ok(document)
}

fn run_moonshine_stream(
    session: &mut Session<Moonshine>,
    samples: &[f32],
    paced: bool,
) -> StreamMetrics {
    let audio_seconds = samples.len() as f64 / SAMPLE_RATE as f64;
    let mut metrics = StreamMetrics::new(audio_seconds);
    // Discard anything left by a previous failed run; after a successful
    // finish this is a no-op. Provisional text is never part of transcript.
    session.cancel();
    let run_started = Instant::now();
    let mut failed = false;

    for (index, piece) in samples.chunks(MOONSHINE_CHUNK_SAMPLES).enumerate() {
        let end_sample = ((index + 1) * MOONSHINE_CHUNK_SAMPLES).min(samples.len());
        let arrival_ms = end_sample as f64 * 1000.0 / SAMPLE_RATE as f64;
        let deadline = run_started + Duration::from_secs_f64(arrival_ms / 1000.0);
        if paced {
            sleep_until(deadline);
            metrics.max_backlog_ms = metrics.max_backlog_ms.max(
                Instant::now()
                    .saturating_duration_since(deadline)
                    .as_secs_f64()
                    * 1000.0,
            );
        }
        let started = Instant::now();
        let result = session.feed(piece).map_err(|err| format!("{err:#}"));
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
        metrics.decode_ms += decode_ms;
        metrics.chunks += 1;
        if metrics.first_chunk_ms.is_none() {
            metrics.first_chunk_ms = Some(decode_ms);
        }
        if paced {
            metrics.max_lag_ms = metrics.max_lag_ms.max(
                Instant::now()
                    .saturating_duration_since(deadline)
                    .as_secs_f64()
                    * 1000.0,
            );
        }
        match result {
            Ok(update) => {
                if metrics.first_provisional_ms.is_none()
                    && let Some(partial) = update.partial.as_deref()
                {
                    metrics.first_provisional_ms =
                        Some(run_started.elapsed().as_secs_f64() * 1000.0);
                    metrics.first_provisional_text = Some(partial.to_string());
                }
                if !update.completed.is_empty() {
                    if metrics.first_committed_ms.is_none() {
                        metrics.first_committed_ms =
                            Some(run_started.elapsed().as_secs_f64() * 1000.0);
                        metrics.first_committed_audio_ms = Some(arrival_ms);
                        metrics.first_committed_text = Some(update.completed.clone());
                    }
                    metrics.transcript.push_str(&update.completed);
                }
            }
            Err(error) => {
                metrics.mark_error(error);
                session.cancel();
                failed = true;
                break;
            }
        }
    }

    if !failed {
        if paced {
            // The last audio chunk (including the short tail) has arrived.
            let deadline = run_started + Duration::from_secs_f64(audio_seconds);
            sleep_until(deadline);
        }
        let started = Instant::now();
        let result = session.finish().map_err(|err| format!("{err:#}"));
        let finish_ms = started.elapsed().as_secs_f64() * 1000.0;
        metrics.decode_ms += finish_ms;
        metrics.finalization_compute_ms = finish_ms;
        match result {
            Ok(update) => {
                if metrics.first_provisional_ms.is_none()
                    && let Some(partial) = update.partial.as_deref()
                {
                    metrics.first_provisional_ms =
                        Some(run_started.elapsed().as_secs_f64() * 1000.0);
                    metrics.first_provisional_text = Some(partial.to_string());
                }
                if !update.completed.is_empty() {
                    if metrics.first_committed_ms.is_none() {
                        metrics.first_committed_ms =
                            Some(run_started.elapsed().as_secs_f64() * 1000.0);
                        metrics.first_committed_audio_ms = Some(audio_seconds * 1000.0);
                        metrics.first_committed_text = Some(update.completed.clone());
                    }
                    metrics.transcript.push_str(&update.completed);
                }
                metrics.final_partial = update.partial;
            }
            Err(error) => metrics.mark_error(error),
        }
        if paced {
            let finished_at = run_started.elapsed().as_secs_f64() * 1000.0;
            metrics.stop_to_final_ms = Some((finished_at - audio_seconds * 1000.0).max(0.0));
            metrics.max_lag_ms = metrics
                .max_lag_ms
                .max((finished_at - audio_seconds * 1000.0).max(0.0));
        }
    }
    session.cancel();
    metrics
}

fn quality_moonshine(
    session: &mut Session<Moonshine>,
    corpus: &PreloadedCorpus,
    limit: usize,
    errors: &mut Vec<Value>,
) -> Value {
    let mut runs = Vec::new();
    for clip in corpus.clips.iter().take(limit) {
        let metrics = run_moonshine_stream(session, &clip.samples, false);
        if metrics.status != "ok" {
            push_engine_error(
                errors,
                "quality",
                &clip.id,
                metrics
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown streaming error".to_string()),
            );
        }
        let mut run = stream_run_json(&metrics, 0, clip.duration_seconds, false);
        run["clip_id"] = json!(clip.id);
        run["reference"] = json!(clip.reference);
        run["duration_seconds"] = json!(clip.duration_seconds);
        run["speaker_id"] = json!(clip.speaker_id);
        run["speaker_gender"] = json!(clip.speaker_gender);
        run["dataset_row"] = json!(clip.dataset_row);
        run["status"] = json!(metrics.status);
        run["transcript"] = json!(metrics.transcript);
        run["error"] = json!(metrics.error);
        run.as_object_mut().unwrap().remove("run");
        runs.push(run);
    }
    json!({
        "corpus": corpus_json(corpus),
        "runs": runs,
    })
}

fn stress_moonshine(
    session: &mut Session<Moonshine>,
    jfk: &[f32],
    errors: &mut Vec<Value>,
) -> Vec<Value> {
    synthetic_cases(jfk)
        .into_iter()
        .map(|case| {
            let duration = case.samples.len() as f64 / SAMPLE_RATE as f64;
            let metrics = run_moonshine_stream(session, &case.samples, false);
            if metrics.status != "ok" {
                push_engine_error(
                    errors,
                    "stress",
                    case.id,
                    metrics
                        .error
                        .clone()
                        .unwrap_or_else(|| "unknown streaming error".to_string()),
                );
            }
            stream_stress_json(&metrics, case.id, case.kind, case.label, duration)
        })
        .collect()
}

fn sleep_until(deadline: Instant) {
    let now = Instant::now();
    if deadline > now {
        std::thread::sleep(deadline - now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_default_runs_and_quick_caps() {
        let args = vec![
            "--engine".to_string(),
            "standard".to_string(),
            "--threads".to_string(),
            "1".to_string(),
        ];
        let options = parse_args(&args).unwrap();
        assert_eq!(options.engine, Engine::Standard);
        assert_eq!(options.threads, ThreadChoice::One);
        assert_eq!(options.runs, 5);
        assert_eq!(options.paced_runs, 3);
        assert!(!options.quick);

        let args = vec![
            "--engine=moonshine".to_string(),
            "--threads".to_string(),
            "1".to_string(),
            "--quick".to_string(),
        ];
        let options = parse_args(&args).unwrap();
        assert_eq!(options.runs, 1);
        assert_eq!(options.paced_runs, 1);
        assert!(options.quick);
    }

    #[test]
    fn parse_rejects_missing_engine_and_zero_runs() {
        assert!(parse_args(&[]).is_err());
        let args = vec![
            "--engine".to_string(),
            "whisper".to_string(),
            "--runs".to_string(),
            "0".to_string(),
        ];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn moonshine_rejects_exact_two_threads() {
        let error = resolve_threads(Engine::Moonshine, ThreadChoice::Two).unwrap_err();
        assert!(format!("{error:#}").contains("no exact two-thread"));
        let one = resolve_threads(Engine::Moonshine, ThreadChoice::One).unwrap();
        assert_eq!(one.effective, Some(1));
        let runtime = resolve_threads(Engine::Moonshine, ThreadChoice::RuntimeDefault).unwrap();
        assert_eq!(runtime.effective, None);
        assert!(
            runtime.json()["moonshine_policy"]
                .as_str()
                .unwrap()
                .contains("not exactly 2")
        );
    }

    #[test]
    fn non_moonshine_rejects_runtime_default() {
        for engine in [
            Engine::Standard,
            Engine::Fluid,
            Engine::Whisper,
            Engine::Live,
        ] {
            assert!(resolve_threads(engine, ThreadChoice::RuntimeDefault).is_err());
        }
    }

    #[test]
    fn synthetic_noise_is_deterministic_and_bounded() {
        let jfk = vec![0.1f32; SAMPLE_RATE as usize];
        let first = synthetic_cases(&jfk);
        let second = synthetic_cases(&jfk);
        let noise = first.iter().find(|case| case.kind == "noise").unwrap();
        assert_eq!(
            noise.samples,
            second
                .iter()
                .find(|case| case.kind == "noise")
                .unwrap()
                .samples
        );
        assert!(noise.samples.iter().all(|sample| sample.abs() <= 0.01));
        assert_eq!(
            noise.samples.len(),
            SYNTHETIC_SECONDS * SAMPLE_RATE as usize
        );
        let repeated = first
            .iter()
            .find(|case| case.kind == "long_repeated_speech")
            .unwrap();
        assert_eq!(
            repeated.samples.len(),
            jfk.len() * LONG_JFK_REPEATS + LONG_JFK_GAP_SAMPLES * (LONG_JFK_REPEATS - 1)
        );
    }

    #[test]
    fn live_output_spacing_keeps_first_space_rule() {
        let mut text = String::new();
        let mut first = true;
        assert_eq!(
            append_live_output(&mut text, &mut first, " hello").as_deref(),
            Some("hello")
        );
        assert_eq!(
            append_live_output(&mut text, &mut first, " world").as_deref(),
            Some(" world")
        );
        assert_eq!(text, "hello world");
    }

    #[test]
    fn rtf_is_process_time_over_audio_time() {
        assert!((rtf(1000.0, 10.0) - 0.1).abs() < 1e-9);
        assert_eq!(rtf(10.0, 0.0), 0.0);
    }

    #[test]
    fn stream_json_keeps_first_chunk_separate_from_warm_decode() {
        let mut metrics = StreamMetrics::new(11.0);
        metrics.first_chunk_ms = Some(7.5);
        metrics.decode_ms = 1500.0;
        let value = stream_run_json(&metrics, 1, 11.0, true);
        assert_eq!(value["first_chunk_ms"], 7.5);
        assert_eq!(value["decode_ms"], 1500.0);
        assert!(value.get("first_decode_ms").is_none());
    }
}
