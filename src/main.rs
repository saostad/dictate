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
#[cfg(unix)]
mod daemon;
mod mic;
mod model;
#[cfg(unix)]
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
            let Some(file) = args.next() else {
                eprintln!("Usage: dictate transcribe <file.wav>");
                std::process::exit(2);
            };
            transcribe(Path::new(&file))
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

#[cfg(unix)]
fn run_daemon() -> anyhow::Result<()> {
    daemon::run_daemon()
}

#[cfg(not(unix))]
fn run_daemon() -> anyhow::Result<()> {
    anyhow::bail!("\"dictate daemon\" is not supported on this OS yet.")
}

#[cfg(unix)]
fn run_client(cmd: &str) -> anyhow::Result<()> {
    daemon::client_command(cmd)
}

#[cfg(not(unix))]
fn run_client(cmd: &str) -> anyhow::Result<()> {
    anyhow::bail!("\"dictate {cmd}\" is not supported on this OS yet.")
}

fn usage() -> &'static str {
    "Usage: dictate <command> [args]\n\nCommands:\n  download               Download the speech model (about 660 MB)\n  transcribe <file.wav>  Transcribe a WAV file and print the text\n  listen                 Record from the microphone and print the text\n  daemon                 Run the dictation daemon (keeps the model loaded)\n  toggle                 Start/stop recording via the daemon\n  start                  Start recording via the daemon\n  stop                   Stop recording and transcribe via the daemon\n  cancel                 Discard the recording via the daemon\n  status                 Show whether the daemon is recording\n  help                   Show this help"
}
