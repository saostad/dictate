//! Shared validation and file edits for Settings clients.
use crate::config::{self, Selection, Settings};
use crate::settings_ui::page::PageAction;

pub(crate) const KEYS: &[&str] = &[
    "microphone",
    "mode",
    "model",
    "inference-threads",
    "unload-after",
    "shortcut",
    "sounds",
    "notifications",
    "gpu",
    "autostart",
];

pub(crate) fn parse(key: &str, value: &str) -> Result<PageAction, crate::cli::UsageError> {
    use PageAction::*;
    let bad = || {
        crate::cli::UsageError(format!(
            "Invalid value for {key}: {value:?}. Run dictate config --help."
        ))
    };
    if value.contains(['\n', '\r', '\0']) {
        return Err(bad());
    }
    let boolean = || match value {
        "on" | "true" => Ok(true),
        "off" | "false" => Ok(false),
        _ => Err(bad()),
    };
    Ok(match key {
        "microphone" => SelectMic(if value == "default" {
            String::new()
        } else {
            value.to_owned()
        }),
        "mode" => SelectMode(match config::Mode::parse(value).ok_or_else(bad)? {
            config::Mode::Whole => config::DictationMode::Whole,
            config::Mode::Pauses => config::DictationMode::Pauses,
            config::Mode::Live => config::DictationMode::Live,
            config::Mode::Moonshine => return Err(bad()),
        }),
        "model" => SelectSpeechModel(config::ModelChoice::parse(value).ok_or_else(bad)?),
        "inference-threads" => {
            SelectInferenceThreads(config::InferenceThreads::parse(value).ok_or_else(bad)?)
        }
        "unload-after" => SelectUnloadAfter(config::UnloadAfter::parse(value).ok_or_else(bad)?),
        "shortcut" => {
            crate::hotkey::resolve_file_hotkey(value, cfg!(target_os = "macos"))
                .map_err(crate::cli::UsageError)?;
            SelectShortcut(value.to_owned())
        }
        "sounds" => SetSounds(boolean()?),
        "notifications" => SetNotifications(boolean()?),
        "gpu" => SetGpu(boolean()?),
        "autostart" => SetAutostart(match value {
            "on" => true,
            "off" => false,
            _ => return Err(bad()),
        }),
        _ => {
            return Err(crate::cli::UsageError(format!(
                "Unknown setting {key:?}; expected {}",
                KEYS.join(", ")
            )));
        }
    })
}

pub(crate) fn validate_mic(
    id: &str,
    mics: &[crate::mic::Mic],
) -> Result<(), crate::cli::UsageError> {
    if id.is_empty() || mics.iter().any(|mic| mic.id == id) {
        return Ok(());
    }
    Err(crate::cli::UsageError(format!(
        "Unknown microphone {id:?}; available ids: default, {}",
        mics.iter()
            .map(|mic| mic.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

/// Apply the same compatible selection rules as the page. No I/O.
pub(crate) fn edit(settings: &mut Settings, action: &PageAction) -> anyhow::Result<()> {
    use PageAction::*;
    match action {
        SelectMic(id) => settings.microphone.clone_from(id),
        SelectMode(mode) => Selection::new(*mode, Selection::from_settings(settings).model)
            .map_err(anyhow::Error::msg)?
            .apply(settings),
        SelectSpeechModel(model) => {
            anyhow::ensure!(
                model.available(),
                "This model is unavailable on this platform"
            );
            Selection::from_settings(settings)
                .with_model(*model)
                .apply(settings);
        }
        SelectInferenceThreads(value) => settings.inference_threads = *value,
        SelectUnloadAfter(value) => settings.unload_after = *value,
        SelectShortcut(value) => settings.hotkey.clone_from(value),
        SetSounds(value) => settings.sounds = *value,
        SetNotifications(value) => settings.notifications = *value,
        SetGpu(value) => settings.gpu = *value,
        SetAutostart(_) => {}
        _ => anyhow::bail!("Not a setting change"),
    }
    Ok(())
}

pub(crate) fn file_value(settings: &Settings, key: &str) -> anyhow::Result<String> {
    if key.starts_with("server.") {
        return settings.server.value(key).map_err(anyhow::Error::msg);
    }
    Ok(match key {
        "microphone" => {
            if settings.microphone.is_empty() {
                "default".to_owned()
            } else {
                settings.microphone.clone()
            }
        }
        "mode" => Selection::from_settings(settings).mode.as_str().to_owned(),
        "model" => Selection::from_settings(settings).model.as_str().to_owned(),
        "inference-threads" => settings.inference_threads.as_str().to_owned(),
        "unload-after" => settings.unload_after.as_str().to_owned(),
        "shortcut" => settings.hotkey.clone(),
        "sounds" => settings.sounds.to_string(),
        "notifications" => settings.notifications.to_string(),
        "gpu" => settings.gpu.to_string(),
        "autostart" => if crate::autostart::status()?.enabled {
            "on"
        } else {
            "off"
        }
        .to_owned(),
        _ => anyhow::bail!("Unknown setting {key:?}"),
    })
}

fn effective(settings: &Settings, key: &str) -> anyhow::Result<(String, Vec<String>)> {
    let vars: &[&str] = match key {
        "mode" | "model" => &["DICTATE_MODE", "DICTATE_STREAMING", "DICTATE_SPEECH_MODEL"],
        "microphone" => &["DICTATE_MICROPHONE"],
        "shortcut" => &["DICTATE_HOTKEY"],
        "sounds" => &["DICTATE_SOUNDS"],
        "notifications" => &["DICTATE_NOTIFY"],
        "gpu" => &["DICTATE_GPU"],
        "inference-threads" => &["DICTATE_INFERENCE_THREADS"],
        _ => &[],
    };
    let overrides = vars
        .iter()
        .filter_map(|var| {
            std::env::var(var)
                .ok()
                .map(|value| format!("{var}={value}"))
        })
        .collect();
    let value = match key {
        "mode" => config::effective_selection(settings)
            .mode
            .as_str()
            .to_owned(),
        "model" => config::effective_selection(settings)
            .model
            .as_str()
            .to_owned(),
        "microphone" => config::resolve_microphone(
            &settings.microphone,
            std::env::var("DICTATE_MICROPHONE").ok().as_deref(),
        )
        .unwrap_or_else(|| "default".to_owned()),
        "sounds" => config::apply_env(
            settings.sounds,
            std::env::var("DICTATE_SOUNDS").ok().as_deref(),
        )
        .to_string(),
        "notifications" => config::apply_env(
            settings.notifications,
            std::env::var("DICTATE_NOTIFY").ok().as_deref(),
        )
        .to_string(),
        "gpu" => config::apply_env(settings.gpu, std::env::var("DICTATE_GPU").ok().as_deref())
            .to_string(),
        "shortcut" => std::env::var("DICTATE_HOTKEY")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| settings.hotkey.clone()),
        "inference-threads" => {
            let value = std::env::var("DICTATE_INFERENCE_THREADS")
                .ok()
                .and_then(|v| config::InferenceThreads::parse(&v))
                .unwrap_or(settings.inference_threads);
            value.as_str().to_owned()
        }
        _ => file_value(settings, key)?,
    };
    Ok((value, overrides))
}

pub(crate) fn print_values() {
    let server = crate::server_settings::ServerSettings::default();
    for key in crate::server_settings::ServerSettings::KEYS {
        println!(
            "{key}: default {} (see dictate serve --help)",
            server.value(key).expect("server key")
        );
    }
    use crate::settings_ui::page::{INFERENCE_THREAD_OPTIONS, MODE_OPTIONS, SPEECH_MODEL_OPTIONS};
    println!(
        "mode: {}",
        MODE_OPTIONS
            .iter()
            .map(|(v, _, _)| v.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "model: {} (parakeet is an alias for standard)",
        SPEECH_MODEL_OPTIONS
            .iter()
            .map(|(v, _, _)| v.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "inference-threads: {}",
        INFERENCE_THREAD_OPTIONS
            .iter()
            .map(|(v, _)| v.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "unload-after: {}",
        config::UnloadAfter::ALL
            .iter()
            .map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "shortcut: a platform shortcut such as Ctrl+Shift+Space; microphone: id from dictate mics, default or empty; sounds/notifications/gpu: on/off/true/false; autostart: on/off"
    );
}

pub(crate) fn run(args: &[String]) -> anyhow::Result<()> {
    if args.first().is_some_and(|arg| arg == "values") {
        print_values();
        return Ok(());
    }
    if args.first().is_some_and(|arg| arg == "path") {
        println!(
            "{}",
            config::config_path()
                .ok_or_else(|| anyhow::anyhow!("No config directory"))?
                .display()
        );
        return Ok(());
    }
    if args.get(1).is_some_and(|key| key.starts_with("server.")) {
        let (mut settings, problem) = config::load();
        if let Some(problem) = problem {
            anyhow::bail!("{problem}");
        }
        if args[0] == "set" {
            settings
                .server
                .set(&args[1], &args[2])
                .map_err(crate::cli::UsageError)?;
            config::save(&settings)?;
        } else {
            println!(
                "{}",
                settings
                    .server
                    .value(&args[1])
                    .map_err(anyhow::Error::msg)?
            );
        }
        return Ok(());
    }
    if args.first().is_some_and(|arg| arg == "set") {
        let action = parse(&args[1], &args[2])?;
        #[cfg(unix)]
        if let Some(reply) =
            crate::daemon::socket::set_if_running(&format!("set {} {}", args[1], args[2]))?
        {
            println!("{reply}");
            return Ok(());
        }
        let (mut settings, problem) = config::load();
        if let Some(problem) = problem {
            anyhow::bail!("{problem}");
        }
        if let PageAction::SelectMic(id) = &action
            && !id.is_empty()
        {
            validate_mic(id, &crate::mic::list_microphones()?)?;
        }
        edit(&mut settings, &action)?;
        if let PageAction::SetAutostart(on) = action {
            if on {
                crate::autostart::enable()?;
            } else {
                crate::autostart::disable()?;
            }
            eprintln!("Autostart updated");
            return Ok(());
        } else {
            config::save(&settings)?;
        }
        #[cfg(unix)]
        eprintln!("Saved; takes effect when the daemon starts");
        #[cfg(not(unix))]
        eprintln!(
            "Saved; a running daemon must be restarted to pick it up (live control unavailable on this platform)"
        );
        return Ok(());
    }
    let (settings, problem) = config::load();
    if let Some(problem) = problem {
        anyhow::bail!("{problem}");
    }
    if args.first().is_some_and(|arg| arg == "get") {
        println!("{}", effective(&settings, &args[1])?.0);
        return Ok(());
    }
    let mut json = serde_json::Map::new();
    for key in KEYS
        .iter()
        .chain(crate::server_settings::ServerSettings::KEYS.iter())
    {
        let file = file_value(&settings, key)?;
        let (value, overrides) = effective(&settings, key)?;
        if args.iter().any(|arg| arg == "--json") {
            json.insert(
                (*key).to_owned(),
                serde_json::json!({"value": value, "file": file, "environment": overrides}),
            );
        } else if overrides.is_empty() {
            println!("{key} = {value}");
        } else {
            println!(
                "{key} = {value} (file: {file}; environment: {})",
                overrides.join(", ")
            );
        }
    }
    if !json.is_empty() {
        println!("{}", serde_json::Value::Object(json));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_key_parses_and_rejects_invalid_values() {
        for (key, value) in [
            ("microphone", "default"),
            ("mode", "whole"),
            ("model", "nemotron"),
            ("inference-threads", "auto"),
            ("unload-after", "never"),
            ("shortcut", "Ctrl+Shift+Space"),
            ("sounds", "off"),
            ("notifications", "true"),
            ("gpu", "off"),
            ("autostart", "on"),
        ] {
            assert!(parse(key, value).is_ok(), "{key}");
            let args = ["config", "set", key, value].map(str::to_owned);
            assert!(crate::cli::parse(&args).is_ok(), "{key}");
            if key != "microphone" {
                assert!(parse(key, "bogus").is_err(), "{key}");
                assert!(
                    crate::cli::parse(&["config", "set", key, "bogus"].map(str::to_owned)).is_err(),
                    "{key}"
                );
            }
        }
        assert!(parse("unknown", "off").is_err());
        assert!(parse("sounds", "off\nset mode live").is_err());
        assert!(validate_mic("unknown", &[]).is_err());
        assert!(validate_mic("", &[]).is_ok());
        for args in [
            vec!["config", "get", "bad"],
            vec!["config", "set", "sounds"],
            vec!["config", "set", "mode", "bogus"],
        ] {
            assert!(
                crate::cli::parse(&args.into_iter().map(str::to_owned).collect::<Vec<_>>())
                    .is_err()
            );
        }
    }

    #[test]
    fn explicit_ui_actions_and_file_round_trip_are_idempotent() {
        let dir = std::env::temp_dir().join(format!("dictate-config-cli-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                std::fs::remove_dir_all(&self.0).expect("remove isolated config fixture");
            }
        }
        let _cleanup = Cleanup(dir.clone());
        let path = dir.join("config.toml");
        let mut settings = Settings::default();
        let action = parse("sounds", "off").unwrap();
        assert_eq!(action, PageAction::SetSounds(false));
        assert_eq!(
            parse("notifications", "off").unwrap(),
            PageAction::SetNotifications(false)
        );
        for _ in 0..2 {
            edit(&mut settings, &action).unwrap();
            config::save_to(&path, &settings).unwrap();
            settings = config::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
            assert!(!settings.sounds);
        }
        edit(&mut settings, &PageAction::SetNotifications(false)).unwrap();
        assert!(!settings.notifications);
        assert!(edit(&mut settings, &parse("mode", "live").unwrap()).is_err());
        edit(&mut settings, &parse("model", "nemotron").unwrap()).unwrap();
        assert_eq!(
            Selection::from_settings(&settings).mode,
            config::DictationMode::Live
        );
    }
}
