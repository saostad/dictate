#[cfg(all(feature = "bundled-ort", feature = "system-ort"))]
compile_error!(
    "features \"bundled-ort\" and \"system-ort\" cannot be enabled together; build with default features for the bundled ONNX Runtime, or with --no-default-features --features system-ort for the system ONNX Runtime"
);

#[cfg(not(any(feature = "bundled-ort", feature = "system-ort")))]
compile_error!(
    "one of the features \"bundled-ort\" or \"system-ort\" must be enabled; build with default features for the bundled ONNX Runtime, or with --no-default-features --features system-ort for the system ONNX Runtime"
);

mod asr;
mod audio;
mod autostart;
mod benchmark;
mod config;
mod cue;
mod daemon;
mod fluid;
mod hotkey;
mod live;
mod mic;
mod model;
#[cfg(target_os = "macos")]
mod modtap;
mod moonshine;
mod notify;
mod segment;
mod selection;
mod settings_ui;
mod tray;
mod typer;
mod vad;
#[cfg(target_os = "linux")]
mod wayland_shortcut;
mod whisper;

use std::path::Path;

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
    Ok(())
}

fn run() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        // Opened as Dictate.app (Finder, Spotlight, login item): run the daemon.
        if launched_as_macos_app() {
            return run_daemon(false);
        }
        eprintln!("{}", usage());
        std::process::exit(2);
    };
    match command.as_str() {
        "download" => {
            let (mut live, mut fluid, mut whisper, mut moonshine) = (false, false, false, false);
            loop {
                match args.next().as_deref() {
                    None => break,
                    Some("--live") => live = true,
                    Some("--fluid") => fluid = true,
                    Some("--whisper") => whisper = true,
                    Some("--moonshine") => moonshine = true,
                    Some(_) => {
                        eprintln!(
                            "Usage: dictate download [--live | --fluid | --whisper | --moonshine]"
                        );
                        std::process::exit(2);
                    }
                }
            }
            let selected = [live, fluid, whisper, moonshine]
                .into_iter()
                .filter(|selected| *selected)
                .count();
            if selected > 1 {
                eprintln!("Usage: dictate download [--live | --fluid | --whisper | --moonshine]");
                std::process::exit(2);
            }
            if live {
                model::download_live()
            } else if fluid {
                model::download_fluid()
            } else if whisper {
                model::download_whisper()
            } else if moonshine {
                model::download_moonshine()
            } else {
                model::download()
            }
        }
        "listen" => listen(),
        "mics" => mics(),
        "autostart" => {
            let Some(action) = args.next() else {
                eprintln!("Usage: dictate autostart on|off|status");
                std::process::exit(2);
            };
            if args.next().is_some() {
                eprintln!("Usage: dictate autostart on|off|status");
                std::process::exit(2);
            }
            autostart_cmd(&action)
        }
        "daemon" => {
            let background = match args.next().as_deref() {
                None => false,
                Some("--background") => true,
                Some(_) => {
                    eprintln!("Usage: dictate daemon [--background]");
                    std::process::exit(2);
                }
            };
            if args.next().is_some() {
                eprintln!("Usage: dictate daemon [--background]");
                std::process::exit(2);
            }
            run_daemon(background)
        }
        "toggle" | "start" | "stop" | "cancel" | "status" | "settings" => run_client(&command),
        #[cfg(target_os = "linux")]
        "shortcut" => {
            let display = args
                .next()
                .context("Usage: dictate shortcut <Ctrl+Shift+Space|other shortcut>")?;
            anyhow::ensure!(args.next().is_none(), "Usage: dictate shortcut <shortcut>");
            hotkey::resolve_file_hotkey(&display, false).map_err(anyhow::Error::msg)?;
            run_client(&format!("shortcut {display}"))
        }
        "benchmark" => benchmark::run(args),
        "transcribe" => {
            let Some(first) = args.next() else {
                eprintln!(
                    "Usage: dictate transcribe [--live | --fluid | --whisper | --moonshine] <file.wav>"
                );
                std::process::exit(2);
            };
            match first.as_str() {
                "--segmented" => {
                    let Some(file) = args.next() else {
                        eprintln!("Usage: dictate transcribe [--segmented] <file.wav>");
                        std::process::exit(2);
                    };
                    transcribe_segmented(Path::new(&file))
                }
                "--live" => {
                    let Some(file) = args.next() else {
                        eprintln!("Usage: dictate transcribe [--live] <file.wav>");
                        std::process::exit(2);
                    };
                    transcribe_live(Path::new(&file))
                }
                "--fluid" => {
                    let Some(file) = args.next() else {
                        eprintln!("Usage: dictate transcribe [--fluid] <file.wav>");
                        std::process::exit(2);
                    };
                    transcribe(Path::new(&file), Some(config::SpeechModel::Fluid))
                }
                "--whisper" => {
                    let Some(file) = args.next() else {
                        eprintln!("Usage: dictate transcribe [--whisper] <file.wav>");
                        std::process::exit(2);
                    };
                    transcribe(Path::new(&file), Some(config::SpeechModel::Whisper))
                }
                "--moonshine" => {
                    let Some(file) = args.next() else {
                        eprintln!("Usage: dictate transcribe [--moonshine] <file.wav>");
                        std::process::exit(2);
                    };
                    transcribe_moonshine(Path::new(&file))
                }
                _ => transcribe(Path::new(&first), None),
            }
        }
        "help" | "--help" => {
            println!("{}", usage());
            Ok(())
        }
        unknown => {
            eprintln!("Unknown command \"{unknown}\". Run \"dictate help\" for usage.");
            std::process::exit(2);
        }
    }
}

fn init_config() {
    let (settings, problem) = config::load();
    config::init_effective(&settings);
    if let Some(problem) = problem {
        eprintln!("error: {problem} Using defaults.");
    }
}

fn transcribe(path: &Path, model: Option<config::SpeechModel>) -> anyhow::Result<()> {
    init_config();
    let samples = audio::load_wav_16k_mono(path)?;
    let start = std::time::Instant::now();
    let mut asr = match model {
        Some(config::SpeechModel::Fluid) => {
            asr::Asr::load_fluid(&model::fluid_model_dir_checked()?)?
        }
        Some(config::SpeechModel::Standard) => asr::Asr::load(&model::model_dir_checked()?)?,
        Some(config::SpeechModel::Whisper) => {
            asr::Asr::load_whisper(&model::whisper_model_dir_checked()?)?
        }
        None => asr::Asr::load_configured()?,
    };
    eprintln!("model loaded in {} ms", start.elapsed().as_millis());
    let seconds = samples.len() as f32 / 16_000.0;
    let start = std::time::Instant::now();
    let text = asr.transcribe(&samples)?;
    eprintln!(
        "transcribed {seconds:.1} s of audio in {} ms",
        start.elapsed().as_millis()
    );
    if asr::is_empty_transcript(&text) {
        eprintln!("{}", asr::no_speech_message(seconds));
        return Ok(());
    }
    println!("{text}");
    Ok(())
}

/// Test hook: transcribe a file through the same feeder path the streaming
/// daemon uses (100 ms pushes at the file's rate, per-segment transcribe,
/// same join rule). Prints the joined text to stdout and per-segment timings
/// to stderr.
fn transcribe_segmented(path: &Path) -> anyhow::Result<()> {
    init_config();
    let (mono, rate) = audio::load_wav_mono(path)?;
    let vad = vad::Vad::load_default()?;
    let start = std::time::Instant::now();
    let mut asr = asr::Asr::load_configured()?;
    eprintln!("model loaded in {} ms", start.elapsed().as_millis());
    let mut feeder = segment::Feeder::new(rate, vad)?;
    let mut run = SegmentRun::default();
    let chunk = (rate as usize / 10).max(1);
    for piece in mono.chunks(chunk) {
        let segments = feeder.push(piece)?;
        transcribe_segments(&mut asr, &segments, &mut run)?;
    }
    let segments = feeder.finish()?;
    transcribe_segments(&mut asr, &segments, &mut run)?;
    if run.joined.is_empty() {
        let seconds = mono.len() as f32 / rate.max(1) as f32;
        eprintln!("{}", asr::no_speech_message(seconds));
        return Ok(());
    }
    if let Some(punct) = run.pending {
        run.joined.push(punct);
    }
    println!("{}", run.joined);
    Ok(())
}

/// Joined-text state for one `--segmented` run.
#[derive(Default)]
struct SegmentRun {
    seq: usize,
    joined: String,
    typed_words: Vec<String>,
    pending: Option<char>,
}

/// Transcribe feeder segments like the daemon worker: decode with left
/// context, skip empties silently, join with the same rule, log per-segment
/// timings to stderr.
fn transcribe_segments(
    asr: &mut asr::Asr,
    segments: &[segment::Segment],
    run: &mut SegmentRun,
) -> anyhow::Result<()> {
    for segment in segments {
        run.seq += 1;
        let seconds = segment.audio.len() as f32 / 16_000.0;
        let context_s = segment.context.len() as f32 / 16_000.0;
        let start = std::time::Instant::now();
        let piece =
            asr.transcribe_segment_pieces(&segment.context, &segment.audio, segment.start)?;
        let ms = start.elapsed().as_millis();
        if piece.words.is_empty() {
            continue;
        }
        let decode = if piece.disjoint {
            "disjoint decode"
        } else {
            "context decode"
        };
        eprintln!(
            "segment {}: {seconds:.1} s audio (context {context_s:.1} s, {}, {decode}), transcribed in {ms} ms",
            run.seq,
            segment.start.label()
        );
        let tail_start = run.typed_words.len().saturating_sub(8);
        let (to_type, new_pending) = asr::join_piece(
            &run.typed_words[tail_start..],
            run.pending,
            piece.boundary,
            &piece.words,
            asr::JoinMode::from_disjoint(piece.disjoint),
        );
        run.pending = new_pending;
        if to_type.is_empty() {
            continue;
        }
        run.joined.push_str(&to_type);
        asr::append_typed_words(&mut run.typed_words, &to_type);
    }
    Ok(())
}

/// Transcribe a file with the Live streaming model: the same chunk loop the
/// daemon runs (8960-sample chunks, padded remainder, 3 zero chunks to flush).
/// Prints the text to stdout and the average chunk time to stderr.
fn transcribe_live(path: &Path) -> anyhow::Result<()> {
    init_config();
    let dir = model::live_model_dir_checked()?;
    let samples = audio::load_wav_16k_mono(path)?;
    let env = asr::shared_environment()?;
    let threads = asr::intra_threads();
    let start = std::time::Instant::now();
    let mut live = live::Live::load(&dir, &env, threads)?;
    eprintln!("model loaded in {} ms", start.elapsed().as_millis());
    let chunk = live.chunk_samples().max(1);
    let mut text = String::new();
    let mut first = true;
    let mut chunks: usize = 0;
    let mut total_ms: u128 = 0;
    for piece in samples.chunks(chunk) {
        let mut full = piece.to_vec();
        full.resize(chunk, 0.0);
        let start = std::time::Instant::now();
        let out = live.transcribe_chunk(&full)?;
        total_ms += start.elapsed().as_millis();
        chunks += 1;
        append_live_output(&mut text, &mut first, &out);
    }
    let zeros = vec![0.0; chunk];
    for _ in 0..live::FLUSH_CHUNKS {
        let start = std::time::Instant::now();
        let out = live.transcribe_chunk(&zeros)?;
        total_ms += start.elapsed().as_millis();
        chunks += 1;
        append_live_output(&mut text, &mut first, &out);
    }
    live.reset();
    let avg_ms = total_ms as f64 / chunks.max(1) as f64;
    eprintln!(
        "live: avg {} ms per 560 ms chunk (RTF {:.2})",
        total_ms / chunks.max(1) as u128,
        live::rtf(avg_ms)
    );
    if asr::is_empty_transcript(&text) {
        let seconds = samples.len() as f32 / 16_000.0;
        eprintln!("{}", asr::no_speech_message(seconds));
        return Ok(());
    }
    println!("{text}");
    Ok(())
}

/// Test hook: stream a WAV through Moonshine in 100 ms chunks, exactly as the
/// daemon feeder does, and finish the stream the way stop does. Prints only
/// committed final text; provisional tails go to stderr for inspection.
fn transcribe_moonshine(path: &Path) -> anyhow::Result<()> {
    init_config();
    let threads = moonshine::Threads::for_setting(config::inference_threads());
    moonshine::configure_threads(threads);
    let dir = model::moonshine_model_dir_checked()?;
    model::check_moonshine_runtime()?;
    let samples = audio::load_wav_16k_mono(path)?;
    let load_start = std::time::Instant::now();
    let model = moonshine::Moonshine::load(&dir, threads)?;
    eprintln!("model loaded in {} ms", load_start.elapsed().as_millis());
    // Same session lifecycle as the daemon worker: the first chunk opens the
    // native stream, finish drains the last phrase, cancel discards it.
    let mut session = moonshine::Session::new(model);
    let mut text = String::new();
    let mut total_ms: u128 = 0;
    let mut calls: usize = 0;
    for piece in samples.chunks(1_600) {
        let start = std::time::Instant::now();
        let update = session.feed(piece)?;
        total_ms += start.elapsed().as_millis();
        calls += 1;
        text.push_str(&update.completed);
        if let Some(partial) = update.partial {
            eprintln!("partial: {partial}");
        }
    }
    let start = std::time::Instant::now();
    let update = session.finish()?;
    total_ms += start.elapsed().as_millis();
    text.push_str(&update.completed);
    let audio_ms = samples.len() as f64 / 16.0;
    let avg_ms = total_ms as f64 / calls.max(1) as f64;
    eprintln!(
        "moonshine: {calls} chunks, avg {avg_ms:.0} ms per 100 ms chunk (RTF {:.2} inference, excluding load)",
        total_ms as f64 / audio_ms.max(1.0)
    );
    if asr::is_empty_transcript(&text) {
        let seconds = samples.len() as f32 / 16_000.0;
        eprintln!("{}", asr::no_speech_message(seconds));
        return Ok(());
    }
    println!("{text}");
    Ok(())
}

/// Append one Live chunk output: empty outputs are skipped, the first
/// non-empty output of a recording loses its leading space, later outputs are
/// kept as-is (the model's text carries its own spaces).
fn append_live_output(text: &mut String, first: &mut bool, out: &str) {
    if out.is_empty() {
        return;
    }
    if *first {
        *first = false;
        text.push_str(out.trim_start());
    } else {
        text.push_str(out);
    }
}

fn listen() -> anyhow::Result<()> {
    init_config();
    let settings = config::load().0;
    let mic = config::resolve_microphone(
        &settings.microphone,
        std::env::var("DICTATE_MICROPHONE").ok().as_deref(),
    );
    let recorder = mic::Recorder::start(mic.as_deref())?;
    let loader = std::thread::spawn(asr::Asr::load_configured);
    eprintln!("Recording... press Enter to stop.");
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .context("Could not read from stdin")?;
    let samples = recorder.stop()?;
    if mic::recording_too_short(&samples) {
        let ms = samples.len() as u64 * 1000 / 16_000;
        anyhow::bail!("Recording too short ({ms} ms); nothing to transcribe.");
    }
    let mut asr = match loader.join() {
        Ok(Ok(asr)) => asr,
        Ok(Err(err)) => return Err(err),
        Err(_) => anyhow::bail!("Model loading thread panicked"),
    };
    let seconds = samples.len() as f32 / 16_000.0;
    let start = std::time::Instant::now();
    let text = asr.transcribe(&samples)?;
    eprintln!(
        "recorded {seconds:.1} s, transcribed in {} ms",
        start.elapsed().as_millis()
    );
    if asr::is_empty_transcript(&text) {
        eprintln!("{}", asr::no_speech_message(seconds));
        return Ok(());
    }
    println!("{text}");
    Ok(())
}

/// List the microphones: each with its default/selected marks and its id
/// on the next line, plus a hint for choosing one.
fn mics() -> anyhow::Result<()> {
    init_config();
    let settings = config::load().0;
    let mics = mic::list_microphones()?;
    if mics.is_empty() {
        println!("No microphones found.");
    }
    for mic in &mics {
        let mut line = format!("* {}", mic.name);
        if mic.is_default {
            line.push_str(" [default]");
        }
        if settings.microphone == mic.id {
            line.push_str(" [selected]");
        }
        println!("{line}");
        println!("    {}", mic.id);
    }
    println!("Choose one in the Settings window.");
    Ok(())
}

/// Turn start-at-login on or off, or show whether it is on.
fn autostart_cmd(action: &str) -> anyhow::Result<()> {
    match action {
        "on" => autostart::enable(),
        "off" => autostart::disable(),
        "status" => autostart_status(),
        _ => {
            eprintln!("Usage: dictate autostart on|off|status");
            std::process::exit(2);
        }
    }
}

/// Print `Start at login: on (<target>)` (plus a note when the entry points
/// elsewhere and will be updated) or `Start at login: off`.
fn autostart_status() -> anyhow::Result<()> {
    let state = autostart::status()?;
    if !state.enabled {
        println!("Start at login: off");
        return Ok(());
    }
    if state.stale {
        let current = std::env::current_exe()
            .map(|exe| exe.display().to_string())
            .unwrap_or_else(|_| state.target.clone().unwrap_or_default());
        let old = state.target.as_deref().unwrap_or("?");
        println!("Start at login: on ({current}) (entry points to {old}; will be updated)");
    } else if let Some(target) = &state.target {
        println!("Start at login: on ({target})");
    } else {
        println!("Start at login: on");
    }
    Ok(())
}

fn run_daemon(background: bool) -> anyhow::Result<()> {
    daemon::run_daemon(background)
}

#[cfg(unix)]
fn run_client(cmd: &str) -> anyhow::Result<()> {
    daemon::socket::client_command(cmd)
}

#[cfg(not(unix))]
fn run_client(cmd: &str) -> anyhow::Result<()> {
    anyhow::bail!("\"dictate {cmd}\" is not supported on this OS yet.")
}

/// Whether this binary is the main executable of a macOS app bundle
/// (`Dictate.app/Contents/MacOS/dictate`).
fn launched_as_macos_app() -> bool {
    cfg!(target_os = "macos")
        && std::env::current_exe().is_ok_and(|exe| is_app_bundle_executable(&exe))
}

pub(crate) fn is_app_bundle_executable(exe: &Path) -> bool {
    let Some(macos_dir) = exe.parent() else {
        return false;
    };
    let Some(contents) = macos_dir.parent() else {
        return false;
    };
    macos_dir.file_name().is_some_and(|name| name == "MacOS")
        && contents.file_name().is_some_and(|name| name == "Contents")
        && contents
            .parent()
            .and_then(Path::extension)
            .is_some_and(|ext| ext == "app")
}

fn usage() -> &'static str {
    "Usage: dictate <command> [args]\n\nCommands:\n  download [--live | --fluid | --whisper | --moonshine]  Download the default speech model (about 660 MB), or only the model named by the flag\n  transcribe [--live | --fluid | --whisper | --moonshine] <file.wav>  Transcribe a WAV file with the configured model or the named one\n  transcribe --segmented <file.wav>  Transcribe a WAV through the Pauses streaming path\n  benchmark --engine <name> ...  Measure one local ASR engine and print raw JSON (see docs/model-benchmarks.md)\n  listen                 Record from the microphone and print the text\n  mics                   List the microphones and the selected one\n  daemon [--background]  Run the dictation daemon (keeps the model loaded)
  autostart on|off|status  Turn start-at-login on or off, or show the setting\n  toggle                 Start/stop recording via the daemon\n  start                  Start recording via the daemon\n  stop                   Stop recording and transcribe via the daemon\n  cancel                 Discard the recording via the daemon\n  status                 Show whether the daemon is recording\n  settings               Open the Settings window via the daemon\n  help                   Show this help"
}

#[cfg(test)]
mod tests {
    use super::is_app_bundle_executable;
    use std::path::Path;

    #[test]
    fn detects_app_bundle_executable() {
        assert!(is_app_bundle_executable(Path::new(
            "/Applications/Dictate.app/Contents/MacOS/dictate"
        )));
        assert!(!is_app_bundle_executable(Path::new(
            "/Users/me/.local/bin/dictate"
        )));
        assert!(!is_app_bundle_executable(Path::new(
            "/opt/Dictate/Contents/MacOS/dictate"
        )));
    }
}
