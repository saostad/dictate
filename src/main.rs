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
mod cli;
mod config;
mod config_cli;
mod cue;
mod daemon;
mod fluid;
mod hotkey;
mod http_transport;
mod live;
mod media;
mod mic;
mod model;
#[cfg(target_os = "macos")]
mod modtap;
mod moonshine;
mod notify;
mod segment;
mod selection;
mod server;
mod server_settings;
mod settings_ui;
mod transcription;
mod tray;
mod typer;
mod vad;
#[cfg(target_os = "linux")]
mod wayland_shortcut;
mod whisper;
mod whisper_device;

use std::path::Path;

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    if let Err(err) = run() {
        if err.is::<cli::UsageError>() {
            eprintln!("{err}");
            std::process::exit(2);
        }
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
    Ok(())
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = match cli::parse(&args)? {
        cli::Action::NoArgs => {
            // Opened as Dictate.app (Finder, Spotlight, login item): run the daemon.
            if launched_as_macos_app() {
                return run_daemon(false);
            }
            return Err(cli::UsageError(usage().to_owned()).into());
        }
        cli::Action::Help(help) => {
            println!("{help}");
            if args.first().is_some_and(|arg| arg == "config") {
                config_cli::print_values();
            }
            return Ok(());
        }
        cli::Action::Version => {
            println!("dictate {}", env!("CARGO_PKG_VERSION"));
            if args.first().is_some_and(|arg| arg == "version") {
                println!("{}", settings_ui::page::RELEASES_URL);
            }
            return Ok(());
        }
        cli::Action::Run(command, rest) => (command, rest),
    };
    // The pure CLI parser has validated command names, flags and arity.
    match command {
        "download" => match rest.first().map(String::as_str) {
            Some("--live") => model::download_live(),
            Some("--fluid") => model::download_fluid(),
            Some("--whisper") => model::download_whisper(),
            Some("--moonshine") => model::download_moonshine(),
            _ => model::download(),
        },
        "serve" => server::run(rest).map_err(anyhow::Error::msg),
        "listen" => listen(),
        "mics" => mics(),
        "config" => config_cli::run(rest),
        "log" => {
            let path = daemon::daemon_log_path()?;
            println!("{}", path.display());
            if !rest.is_empty() {
                tray::open_target(&path.to_string_lossy())?;
            }
            Ok(())
        }
        "autostart" => autostart_cmd(&rest[0]),
        "daemon" => run_daemon(!rest.is_empty()),
        "toggle" | "start" | "stop" | "cancel" | "status" | "settings" => run_client(command),
        #[cfg(target_os = "linux")]
        "shortcut" => {
            if rest.is_empty() {
                config_cli::run(&["get".into(), "shortcut".into()])?;
                if hotkey::hyprland_session() {
                    for bind in daemon::query_hyprland_binds()? {
                        println!("{bind}");
                    }
                }
                return Ok(());
            }
            if rest[0] == "--configure" {
                let reply =
                    daemon::socket::set_if_running("configure-shortcut")?.ok_or_else(|| {
                        anyhow::anyhow!("Start the daemon first to configure desktop shortcuts")
                    })?;
                println!("{reply}");
                return Ok(());
            }
            let display = &rest[0];
            hotkey::resolve_file_hotkey(display, false).map_err(|err| {
                cli::UsageError(format!("{err}\n{}", cli::command_usage(command).unwrap()))
            })?;
            run_client(&format!("shortcut {display}"))
        }
        "benchmark" => benchmark::run(rest.iter().cloned()),
        "transcribe" => transcription::run(rest),
        _ => unreachable!("validated command"),
    }
}

fn init_config() {
    let (settings, problem) = config::load();
    config::init_effective(&settings);
    if let Some(problem) = problem {
        eprintln!("error: {problem} Using defaults.");
    }
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
    eprintln!("Choose one in the Settings window.");
    Ok(())
}

/// Turn start-at-login on or off, or show whether it is on.
fn autostart_cmd(action: &str) -> anyhow::Result<()> {
    match action {
        "on" => autostart::enable(),
        "off" => autostart::disable(),
        "status" => autostart_status(),
        _ => Err(cli::UsageError(cli::command_usage("autostart").unwrap().to_owned()).into()),
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
    "Usage: dictate <command> [args]\n\nCommands:\n  download [--live | --fluid | --whisper | --moonshine]  Download the default speech model (about 660 MB), or only the model named by the flag\n  transcribe [ENGINE] [--format txt|json|srt|vtt] [-o FILE] <path | ->  Transcribe audio (see transcribe --help)\n  benchmark --engine <name> ...  Measure one local ASR engine and print raw JSON (see docs/model-benchmarks.md)\n  serve [FLAGS]          Headless HTTP transcription API (see serve --help)\n  listen                 Record from the microphone and print the text\n  mics                   List the microphones and the selected one\n  daemon [--background]  Run the dictation daemon (keeps the model loaded)
  autostart on|off|status  Turn start-at-login on or off, or show the setting\n  toggle                 Start/stop recording via the daemon\n  start                  Start recording via the daemon\n  stop                   Stop recording and transcribe via the daemon\n  cancel                 Discard the recording via the daemon\n  status                 Show whether the daemon is recording\n  config [list|get|set|path|values]  Read or change settings (see config --help)\n  log [--open]           Print or open the daemon log\n  settings               Open the Settings window via the daemon\n  shortcut <shortcut>    Set the daemon hotkey (Linux only)\n  help, --help, -h       Show this help\n  version, --version, -V  Show the version\n\nRun dictate <command> --help (or -h) for command usage.\n\nExit codes: 0 success, 2 usage error, 1 runtime failure."
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
