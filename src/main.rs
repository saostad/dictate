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
mod config;
mod cue;
mod daemon;
mod hotkey;
mod live;
mod mic;
mod model;
mod notify;
mod segment;
mod tray;
mod typer;
mod vad;

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
        eprintln!("{}", usage());
        std::process::exit(2);
    };
    match command.as_str() {
        "download" => {
            let live = match args.next().as_deref() {
                None => false,
                Some("--live") => true,
                Some(_) => {
                    eprintln!("Usage: dictate download [--live]");
                    std::process::exit(2);
                }
            };
            if args.next().is_some() {
                eprintln!("Usage: dictate download [--live]");
                std::process::exit(2);
            }
            if live {
                model::download_live()
            } else {
                model::download()
            }
        }
        "listen" => listen(),
        "daemon" => run_daemon(),
        "toggle" | "start" | "stop" | "cancel" | "status" => run_client(&command),
        "transcribe" => {
            let Some(first) = args.next() else {
                eprintln!("Usage: dictate transcribe [--live] <file.wav>");
                std::process::exit(2);
            };
            if first == "--segmented" {
                let Some(file) = args.next() else {
                    eprintln!("Usage: dictate transcribe [--segmented] <file.wav>");
                    std::process::exit(2);
                };
                transcribe_segmented(Path::new(&file))
            } else if first == "--live" {
                let Some(file) = args.next() else {
                    eprintln!("Usage: dictate transcribe [--live] <file.wav>");
                    std::process::exit(2);
                };
                transcribe_live(Path::new(&file))
            } else {
                transcribe(Path::new(&first))
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

fn transcribe(path: &Path) -> anyhow::Result<()> {
    let dir = model::model_dir_checked()?;
    let samples = audio::load_wav_16k_mono(path)?;
    let start = std::time::Instant::now();
    let mut asr = asr::Asr::load(&dir)?;
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
    let dir = model::model_dir_checked()?;
    let (mono, rate) = audio::load_wav_mono(path)?;
    let vad = vad::Vad::load_default()?;
    let start = std::time::Instant::now();
    let mut asr = asr::Asr::load(&dir)?;
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
        let (kept_texts, boundary) =
            asr.transcribe_segment_pieces(&segment.context, &segment.audio)?;
        let ms = start.elapsed().as_millis();
        if kept_texts.is_empty() {
            continue;
        }
        eprintln!(
            "segment {}: {seconds:.1} s audio (context {context_s:.1} s), transcribed in {ms} ms",
            run.seq
        );
        let tail_start = run.typed_words.len().saturating_sub(2);
        let (to_type, new_pending) = asr::join_piece(
            &run.typed_words[tail_start..],
            run.pending,
            boundary,
            &kept_texts,
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
    let dir = model::model_dir_checked()?;
    let recorder = mic::Recorder::start()?;
    let loader = std::thread::spawn(move || asr::Asr::load(&dir));
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

fn run_daemon() -> anyhow::Result<()> {
    daemon::run_daemon()
}

#[cfg(unix)]
fn run_client(cmd: &str) -> anyhow::Result<()> {
    daemon::socket::client_command(cmd)
}

#[cfg(not(unix))]
fn run_client(cmd: &str) -> anyhow::Result<()> {
    anyhow::bail!("\"dictate {cmd}\" is not supported on this OS yet.")
}

fn usage() -> &'static str {
    "Usage: dictate <command> [args]\n\nCommands:\n  download [--live]      Download the speech model (about 660 MB; --live also fetches the Live model, about 900 MB)\n  transcribe [--live] <file.wav>  Transcribe a WAV file and print the text\n  listen                 Record from the microphone and print the text\n  daemon                 Run the dictation daemon (keeps the model loaded)\n  toggle                 Start/stop recording via the daemon\n  start                  Start recording via the daemon\n  stop                   Stop recording and transcribe via the daemon\n  cancel                 Discard the recording via the daemon\n  status                 Show whether the daemon is recording\n  help                   Show this help"
}
