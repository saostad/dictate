//! User choices are a compatible mode/model pair. The older `Mode` codes
//! remain an internal backend dispatch detail and are accepted for migration.

use crate::config::{Mode, Settings, SpeechModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DictationMode {
    Whole,
    Pauses,
    Live,
}
impl DictationMode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Whole => "whole",
            Self::Pauses => "pauses",
            Self::Live => "live",
        }
    }
    pub(crate) const fn display_name(self) -> &'static str {
        match self {
            Self::Whole => "Whole recording",
            Self::Pauses => "Type at pauses",
            Self::Live => "Live",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelChoice {
    Parakeet,
    Fluid,
    Whisper,
    Nemotron,
    Moonshine,
}
impl ModelChoice {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Parakeet => "standard",
            Self::Fluid => "fluid",
            Self::Whisper => "whisper",
            Self::Nemotron => "nemotron",
            Self::Moonshine => "moonshine",
        }
    }
    pub(crate) const fn display_name(self) -> &'static str {
        match self {
            Self::Parakeet => SpeechModel::Standard.display_name(),
            Self::Fluid => SpeechModel::Fluid.display_name(),
            Self::Whisper => SpeechModel::Whisper.display_name(),
            Self::Nemotron => "Nemotron Streaming",
            Self::Moonshine => "Moonshine Tiny Streaming",
        }
    }
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "standard" | "parakeet" => Some(Self::Parakeet),
            "fluid" => Some(Self::Fluid),
            "whisper" => Some(Self::Whisper),
            "nemotron" => Some(Self::Nemotron),
            "moonshine" => Some(Self::Moonshine),
            _ => None,
        }
    }
    pub(crate) const fn batch_model(self) -> Option<SpeechModel> {
        match self {
            Self::Parakeet => Some(SpeechModel::Standard),
            Self::Fluid => Some(SpeechModel::Fluid),
            Self::Whisper => Some(SpeechModel::Whisper),
            Self::Nemotron | Self::Moonshine => None,
        }
    }
    pub(crate) const fn supports(self, mode: DictationMode) -> bool {
        matches!(
            (self.batch_model().is_some(), mode),
            (true, DictationMode::Whole | DictationMode::Pauses) | (false, DictationMode::Live)
        )
    }
    pub(crate) const fn available(self) -> bool {
        !matches!(self, Self::Moonshine)
            || cfg!(all(
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Selection {
    pub mode: DictationMode,
    pub model: ModelChoice,
}
impl Selection {
    pub(crate) fn new(mode: DictationMode, model: ModelChoice) -> Result<Self, String> {
        if !model.supports(mode) {
            return Err(format!(
                "{} does not support {}",
                model.display_name(),
                mode.display_name()
            ));
        }
        Ok(Self { mode, model })
    }
    pub(crate) fn with_model(self, model: ModelChoice) -> Self {
        let mode = if model.supports(self.mode) {
            self.mode
        } else if model.batch_model().is_some() {
            DictationMode::Whole
        } else {
            DictationMode::Live
        };
        Self { mode, model }
    }
    pub(crate) fn from_settings(settings: &Settings) -> Self {
        Self::from_runtime(settings.mode, settings.model)
    }
    pub(crate) fn from_runtime(runtime: Mode, batch: SpeechModel) -> Self {
        let mode = match runtime {
            Mode::Whole => DictationMode::Whole,
            Mode::Pauses => DictationMode::Pauses,
            Mode::Live | Mode::Moonshine => DictationMode::Live,
        };
        let model = match runtime {
            Mode::Live => ModelChoice::Nemotron,
            Mode::Moonshine => ModelChoice::Moonshine,
            _ => match batch {
                SpeechModel::Standard => ModelChoice::Parakeet,
                SpeechModel::Fluid => ModelChoice::Fluid,
                SpeechModel::Whisper => ModelChoice::Whisper,
            },
        };
        Self { mode, model }
    }
    pub(crate) fn runtime_mode(self) -> Mode {
        match self.mode {
            DictationMode::Whole => Mode::Whole,
            DictationMode::Pauses => Mode::Pauses,
            DictationMode::Live => {
                if self.model == ModelChoice::Moonshine {
                    Mode::Moonshine
                } else {
                    Mode::Live
                }
            }
        }
    }
    /// Apply only after the worker acknowledges its successfully loaded model.
    /// Keep the last batch model for returning from Live and startup fallback.
    pub(crate) fn apply(self, settings: &mut Settings) {
        settings.mode = self.runtime_mode();
        if let Some(model) = self.model.batch_model() {
            settings.model = model;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compatibility_matrix_and_model_changes_never_create_invalid_pairs() {
        for model in [
            ModelChoice::Parakeet,
            ModelChoice::Fluid,
            ModelChoice::Whisper,
            ModelChoice::Nemotron,
            ModelChoice::Moonshine,
        ] {
            for mode in [
                DictationMode::Whole,
                DictationMode::Pauses,
                DictationMode::Live,
            ] {
                assert_eq!(Selection::new(mode, model).is_ok(), model.supports(mode));
                let current = Selection::from_settings(&Settings::default());
                let next = current.with_model(model);
                assert!(next.model.supports(next.mode));
                assert_eq!(next.model, model);
            }
        }
    }
    #[test]
    fn streaming_models_are_both_live_and_preserve_the_batch_preference() {
        let mut settings = Settings {
            mode: Mode::Pauses,
            model: SpeechModel::Whisper,
            ..Settings::default()
        };
        let original = Selection::from_settings(&settings);
        for model in [ModelChoice::Nemotron, ModelChoice::Moonshine] {
            let next = original.with_model(model);
            assert_eq!(next.mode, DictationMode::Live);
            next.apply(&mut settings);
            assert_eq!(settings.model, SpeechModel::Whisper);
            assert_eq!(Selection::from_settings(&settings), next);
        }
        let returned = Selection::from_settings(&settings).with_model(ModelChoice::Fluid);
        assert_eq!(returned.mode, DictationMode::Whole);
        returned.apply(&mut settings);
        assert_eq!(settings.model, SpeechModel::Fluid);
        assert_eq!(settings.mode, Mode::Whole);
    }
    #[test]
    fn batch_model_changes_preserve_pauses() {
        let current = Selection::new(DictationMode::Pauses, ModelChoice::Whisper).unwrap();
        assert_eq!(
            current.with_model(ModelChoice::Parakeet).mode,
            DictationMode::Pauses
        );
    }
}
