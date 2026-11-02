//! Side-effect-free CLI validation, shared help and usage errors.
use std::fmt;

#[derive(Debug)]
pub(crate) struct UsageError(pub String);

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UsageError {}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Action<'a> {
    NoArgs,
    Help(&'static str),
    Version,
    Run(&'a str, &'a [String]),
}

pub(crate) fn command_usage(command: &str) -> Option<&'static str> {
    Some(match command {
        "download" => "Usage: dictate download [--live | --fluid | --whisper | --moonshine]",
        "transcribe" => {
            "Usage: dictate transcribe [--live | --fluid | --whisper | --moonshine] [--segmented] [--format txt|json|srt|vtt] [-o FILE | --output FILE] <path | ->\nFormats: WAV, MP3, AAC-LC/M4A, FLAC, OGG/Vorbis, audio in MP4/MOV/MKV/WebM. Opus is not supported. Stdin (-) is spooled to disk.\nLong files use pause segmentation; --segmented also segments short clips. Whisper/Live/Moonshine use segment timestamps. Output files are saved atomically; input cannot be overwritten."
        }
        "listen" => "Usage: dictate listen",
        "mics" => "Usage: dictate mics",
        "config" => {
            "Usage: dictate config [list] [--json] | get <key> | set <key> <value> | path | values\nKeys: microphone, mode, model, inference-threads, unload-after, shortcut, sounds, notifications, gpu, autostart\nValues: see dictate config values; microphone: id or default; booleans: on/off/true/false; autostart: on/off"
        }
        "log" => "Usage: dictate log [--open]",
        "daemon" => "Usage: dictate daemon [--background]",
        "autostart" => "Usage: dictate autostart on|off|status",
        "toggle" => "Usage: dictate toggle",
        "start" => "Usage: dictate start",
        "stop" => "Usage: dictate stop",
        "cancel" => "Usage: dictate cancel",
        "status" => "Usage: dictate status",
        "settings" => "Usage: dictate settings",
        #[cfg(target_os = "linux")]
        "shortcut" => "Usage: dictate shortcut [<Ctrl+Shift+Space|other shortcut> | --configure]",
        "benchmark" => crate::benchmark::USAGE,
        "help" => crate::usage(),
        "version" => "Usage: dictate version",
        _ => return None,
    })
}

fn model_flag(flag: &str) -> bool {
    matches!(flag, "--live" | "--fluid" | "--whisper" | "--moonshine")
}

pub(crate) fn parse(args: &[String]) -> Result<Action<'_>, UsageError> {
    let Some((command, rest)) = args.split_first() else {
        return Ok(Action::NoArgs);
    };
    if matches!(command.as_str(), "help" | "--help" | "-h") && rest.is_empty() {
        return Ok(Action::Help(crate::usage()));
    }
    if matches!(command.as_str(), "version" | "--version" | "-V") && rest.is_empty() {
        return Ok(Action::Version);
    }
    let usage = command_usage(command).ok_or_else(|| {
        UsageError(format!(
            "Unknown command \"{command}\". Run \"dictate help\" for usage."
        ))
    })?;
    if rest
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        return Ok(Action::Help(usage));
    }
    let valid = match command.as_str() {
        "config" => {
            match rest
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .as_slice()
            {
                [] | ["list"] | ["--json"] | ["list", "--json"] | ["path"] | ["values"] => true,
                ["get", key] => crate::config_cli::KEYS.contains(key),
                ["set", key, value] => {
                    crate::config_cli::parse(key, value)?;
                    true
                }
                _ => false,
            }
        }
        "log" => rest.is_empty() || rest == ["--open"],
        "download" => rest.is_empty() || (rest.len() == 1 && model_flag(&rest[0])),
        "transcribe" => {
            crate::transcription::Options::parse(rest)?;
            true
        }
        "daemon" => rest.is_empty() || (rest.len() == 1 && rest[0] == "--background"),
        "autostart" => rest.len() == 1 && matches!(rest[0].as_str(), "on" | "off" | "status"),
        #[cfg(target_os = "linux")]
        "shortcut" => {
            rest.is_empty()
                || rest.len() == 1 && (rest[0] == "--configure" || !rest[0].starts_with('-'))
        }
        // Benchmark owns its detailed option parser and reports UsageError itself.
        "benchmark" => true,
        _ => rest.is_empty(),
    };
    if !valid {
        return Err(UsageError(usage.to_owned()));
    }
    Ok(Action::Run(command, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn help_and_version_do_not_execute_commands() {
        for flag in ["--version", "-V", "version"] {
            assert_eq!(parse(&args(&[flag])).unwrap(), Action::Version);
        }
        assert_eq!(parse(&args(&["-h"])).unwrap(), Action::Help(crate::usage()));
        for command in [
            "download",
            "transcribe",
            "listen",
            "mics",
            "daemon",
            "autostart",
            "toggle",
            "start",
            "stop",
            "cancel",
            "status",
            "settings",
            "benchmark",
            "help",
            "version",
            #[cfg(target_os = "linux")]
            "shortcut",
        ] {
            for flag in ["--help", "-h"] {
                assert_eq!(
                    parse(&args(&[command, flag])).unwrap(),
                    Action::Help(command_usage(command).unwrap())
                );
            }
        }
        assert_eq!(parse(&[]).unwrap(), Action::NoArgs);
    }

    #[test]
    fn transcribe_rejects_unknown_flags_and_extra_arguments() {
        for arguments in [
            vec!["transcribe"],
            vec!["transcribe", "--bogus", "x.wav"],
            vec!["transcribe", "x.wav", "extra.wav"],
            vec!["transcribe", "--fluid", "--bogus"],
            vec!["transcribe", "--live", "x.wav", "extra.wav"],
        ] {
            assert!(parse(&args(&arguments)).is_err(), "{arguments:?}");
        }
        for arguments in [
            vec!["transcribe", "x.wav"],
            vec!["transcribe", "--segmented", "x.wav"],
            vec!["transcribe", "--fluid", "x.wav"],
        ] {
            assert!(matches!(
                parse(&args(&arguments)),
                Ok(Action::Run("transcribe", _))
            ));
        }
    }
}
