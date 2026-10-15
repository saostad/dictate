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
mod mic;
mod model;
mod notify;
mod stream;
mod tray;
mod typer;

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
        "download" => model::download(),
        "listen" => listen(),
        "daemon" => run_daemon(),
        "toggle" | "start" | "stop" | "cancel" | "status" => run_client(&command),
        "transcribe" => {
            let Some(first) = args.next() else {
                eprintln!("Usage: dictate transcribe <file.wav>");
                std::process::exit(2);
            };
            if first == "--segmented" {
                let Some(file) = args.next() else {
                    eprintln!("Usage: dictate transcribe [--segmented] <file.wav>");
                    std::process::exit(2);
                };
                transcribe_segmented(Path::new(&file))
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

/// Test hook: transcribe a file through the same buffered-streaming engine
/// the daemon uses (simulated 100 ms chunks at the file's rate, resampled
/// once, decoded synchronously when triggered, same commit and join rules).
/// Prints the joined text to stdout and per-decode timings to stderr.
fn transcribe_segmented(path: &Path) -> anyhow::Result<()> {
    let dir = model::model_dir_checked()?;
    let (mono, rate) = audio::load_wav_mono(path)?;
    if rate == 0 {
        anyhow::bail!("Invalid sample rate 0 Hz");
    }
    let start = std::time::Instant::now();
    let mut asr = asr::Asr::load(&dir)?;
    eprintln!("model loaded in {} ms", start.elapsed().as_millis());
    let mut resampler = if rate == audio::TARGET_SAMPLE_RATE {
        None
    } else {
        Some(audio::StreamResampler::new(rate)?)
    };
    let mut run = StreamRun::default();
    let chunk = (rate as usize / 10).max(1);
    for piece in mono.chunks(chunk) {
        let out16 = match &mut resampler {
            Some(resampler) => resampler.push(piece)?,
            None => piece.to_vec(),
        };
        if out16.is_empty() {
            continue;
        }
        run.stream.push(&out16);
        if run.stream.should_decode() {
            decode_mid(&mut asr, &mut run)?;
        }
    }
    let tail16 = match &mut resampler {
        Some(resampler) => resampler.finish()?,
        None => Vec::new(),
    };
    if !tail16.is_empty() {
        run.stream.push(&tail16);
    }
    if !run.stream.is_empty() {
        decode_final(&mut asr, &mut run)?;
    }
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

/// Joined-text state for one `--segmented` run: the same engine state the
/// daemon worker holds, appending to a string instead of typing.
#[derive(Default)]
struct StreamRun {
    stream: stream::Stream,
    joined: String,
    typed_words: Vec<String>,
    pending: Option<char>,
}

/// Decode the current window like the daemon worker: commit with the guard,
/// join with the same rule, append, log one `decode N` line to stderr.
fn decode_mid(asr: &mut asr::Asr, run: &mut StreamRun) -> anyhow::Result<()> {
    let window = run.stream.window();
    let words = transcribe_window(asr, &run.stream)?;
    let took_ms = words.1;
    let words = words.0;
    let boundary = asr::boundary_punct_before(&words, window.ctx_s);
    let (committed, new_end) =
        stream::commit_words(&words, window.ctx_s, window.window_s, window.guard_s);
    run.stream.mark_decoded();
    if let Some(end_s) = new_end {
        run.stream.advance(window.start, end_s);
    }
    let decode_n = run.stream.decodes();
    if committed.is_empty() {
        eprintln!(
            "decode {decode_n}: window {:.1} s (ctx {:.1} s), took {took_ms} ms, committed 0 words, typed 0 chars",
            window.window_s, window.ctx_s
        );
        return Ok(());
    }
    let kept_texts: Vec<String> = committed.iter().map(|word| word.text.clone()).collect();
    append_committed(
        run,
        &kept_texts,
        boundary,
        decode_n,
        &window,
        took_ms,
        committed.len(),
    );
    Ok(())
}

/// Final decode like the daemon worker: commit everything remaining.
fn decode_final(asr: &mut asr::Asr, run: &mut StreamRun) -> anyhow::Result<()> {
    let window = run.stream.window();
    let words = transcribe_window(asr, &run.stream)?;
    let took_ms = words.1;
    let words = words.0;
    let boundary = asr::boundary_punct_before(&words, window.ctx_s);
    let kept = asr::keep_words_after(&words, window.ctx_s);
    let new_end = kept.last().map(|word| word.end);
    run.stream.mark_decoded();
    if let Some(end_s) = new_end {
        run.stream.advance(window.start, end_s);
    }
    let decode_n = run.stream.decodes();
    if kept.is_empty() {
        eprintln!(
            "decode {decode_n}: window {:.1} s (ctx {:.1} s), took {took_ms} ms, committed 0 words, typed 0 chars",
            window.window_s, window.ctx_s
        );
        return Ok(());
    }
    let kept_texts: Vec<String> = kept.iter().map(|word| word.text.clone()).collect();
    append_committed(
        run,
        &kept_texts,
        boundary,
        decode_n,
        &window,
        took_ms,
        kept.len(),
    );
    Ok(())
}

/// Transcribe the stream's current window, returning the words plus the
/// decode time in milliseconds.
fn transcribe_window(
    asr: &mut asr::Asr,
    stream: &stream::Stream,
) -> anyhow::Result<(Vec<asr::Word>, u128)> {
    let window_audio = stream.window_audio().to_vec();
    let start = std::time::Instant::now();
    let words = asr.transcribe_words(&window_audio)?;
    Ok((words, start.elapsed().as_millis()))
}

/// Join committed words onto the run text with the same rule the daemon
/// worker types with, and log one `decode N` line (`typed` counts appended
/// chars here).
fn append_committed(
    run: &mut StreamRun,
    kept_texts: &[String],
    boundary: Option<char>,
    decode_n: usize,
    window: &stream::Window,
    took_ms: u128,
    committed_words: usize,
) {
    let tail_start = run.typed_words.len().saturating_sub(2);
    let (to_type, new_pending) = asr::join_piece(
        &run.typed_words[tail_start..],
        run.pending,
        boundary,
        kept_texts,
    );
    run.pending = new_pending;
    if to_type.is_empty() {
        eprintln!(
            "decode {decode_n}: window {:.1} s (ctx {:.1} s), took {took_ms} ms, committed {committed_words} words, typed 0 chars",
            window.window_s, window.ctx_s
        );
        return;
    }
    let chars = to_type.chars().count();
    run.joined.push_str(&to_type);
    asr::append_typed_words(&mut run.typed_words, &to_type);
    eprintln!(
        "decode {decode_n}: window {:.1} s (ctx {:.1} s), took {took_ms} ms, committed {committed_words} words, typed {chars} chars",
        window.window_s, window.ctx_s
    );
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
    "Usage: dictate <command> [args]\n\nCommands:\n  download               Download the speech model (about 660 MB)\n  transcribe <file.wav>  Transcribe a WAV file and print the text\n  listen                 Record from the microphone and print the text\n  daemon                 Run the dictation daemon (keeps the model loaded)\n  toggle                 Start/stop recording via the daemon\n  start                  Start recording via the daemon\n  stop                   Stop recording and transcribe via the daemon\n  cancel                 Discard the recording via the daemon\n  status                 Show whether the daemon is recording\n  help                   Show this help"
}
