//! Whisper Tiny English through whisper.cpp (`whisper-rs`).
//!
//! Whole mode decodes the finished recording; whisper.cpp divides long audio
//! into its own 30-second decoder windows. Pauses mode decodes each bounded
//! VAD segment with up to 1.5 seconds of the audio immediately before it.
//! Only the first segment has no context and therefore decodes alone. Later
//! segments decode context + segment, enable whisper.cpp's token timestamps,
//! and trim the words before the context boundary so the repeated context is
//! never typed. This is deliberately not skipped after a natural pause: a
//! segment-only decode of the reference clip changed real words even with the
//! full encoder window, so the context decode and its cost are retained.
//!
//! It does not invent word timestamps: the existing word-timestamp path is
//! not used for Whisper.

use std::ffi::{CStr, c_char, c_void};
use std::io::Write;
use std::path::Path;
use std::sync::Once;

use anyhow::Context;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperSegment,
};

/// One token with whisper.cpp's token-level timing (seconds). Only used
/// internally to discard repeated context, never exposed as word timestamps.
/// The text is kept as raw UTF-8 bytes: whisper.cpp BPE can split a multi-byte
/// character across tokens, and decoding each token separately would corrupt
/// accented names before the fragments are joined.
#[derive(Debug, Clone)]
pub(crate) struct TimedToken {
    pub bytes: Vec<u8>,
    pub start: f32,
    pub end: f32,
}

/// Text that whisper.cpp emits for silence, music or noise instead of leaving
/// the segment empty. Exact bracketed markers only, so ordinary words in
/// brackets are not dropped.
pub(crate) fn is_noise_text(text: &str) -> bool {
    let trimmed = text.trim();
    // whisper.cpp marks non-speech in brackets or parentheses. A bare word
    // such as "music" is ordinary dictation, not a marker.
    let bracketed = (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')'));
    if !bracketed {
        return false;
    }
    let inner = trimmed.trim_matches(['[', ']', '(', ')']).trim();
    matches!(
        inner.to_ascii_lowercase().as_str(),
        "blank_audio"
            | "silence"
            | "music"
            | "applause"
            | "laughs"
            | "inaudible"
            | "no speech"
            | "background noise"
    )
}

/// Remove whisper.cpp's special tokens that can appear when token-level
/// timestamps or debug output are enabled (`[_TT_123]`, `[_BEG_]`, ...). They
/// are metadata, never spoken text, and must not reach the focused app.
fn strip_special_tokens(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '[' && chars.peek() == Some(&'_') {
            for inner in chars.by_ref() {
                if inner == ']' {
                    break;
                }
            }
            continue;
        }
        if ch == '<' && chars.peek() == Some(&'|') {
            let mut previous = '\0';
            for inner in chars.by_ref() {
                if previous == '|' && inner == '>' {
                    break;
                }
                previous = inner;
            }
            continue;
        }
        out.push(ch);
    }
    out
}

/// Clean one decoder segment: remove special tokens, trim, drop known
/// non-speech markers, and collapse internal whitespace to the single spaces
/// Dictate types.
pub(crate) fn clean_segment_text(text: &str) -> Option<String> {
    let text = strip_special_tokens(text);
    let text = text.trim();
    if text.is_empty() || is_noise_text(text) {
        return None;
    }
    Some(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Whether one whisper.cpp token is metadata rather than spoken text. Special
/// tokens are ASCII; bytes that are not valid UTF-8 are spoken fragments (for
/// example a BPE token holding half of a multi-byte character), not metadata.
fn is_special_token(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    let text = text.trim();
    (text.starts_with("[_") && text.ends_with(']'))
        || (text.starts_with("<|") && text.ends_with("|>"))
}

/// One whole word assembled from one or more BPE tokens, with the token
/// range's real timing. whisper.cpp token timestamps are token-level; trimming
/// per token can split a word (for example " Amer" + "icans").
#[derive(Debug, Clone, PartialEq)]
struct TimedWord {
    text: String,
    start: f32,
    end: f32,
}

/// Whether every spoken token has a usable whisper.cpp timestamp. Special
/// tokens are ignored: some carry -1 and must not force the untimed fallback.
/// Timing must be finite, nonnegative and ordered (`end >= start`); anything
/// else is treated as no timing rather than trusted or fabricated.
fn has_token_timings(tokens: &[TimedToken]) -> bool {
    let mut any = false;
    let mut previous_start = f32::NEG_INFINITY;
    for token in tokens
        .iter()
        .filter(|token| !is_special_token(&token.bytes))
    {
        any = true;
        if !token.start.is_finite()
            || !token.end.is_finite()
            || token.start < 0.0
            || token.end < token.start
            || token.start < previous_start
        {
            return false;
        }
        previous_start = token.start;
    }
    any
}

/// Group whisper.cpp BPE tokens into whole words. A token that starts with
/// whitespace begins a new word; a token without leading whitespace continues
/// the current word. Special tokens are dropped. Fragments are concatenated as
/// raw bytes and decoded once per whole word, so a multi-byte character split
/// across BPE tokens (a real case for accented names) survives intact.
fn group_timed_words(tokens: &[TimedToken]) -> Vec<TimedWord> {
    let mut words = Vec::new();
    let mut current: Option<(Vec<u8>, f32, f32)> = None;
    for token in tokens {
        if is_special_token(&token.bytes) {
            continue;
        }
        let bytes = token.bytes.as_slice();
        if bytes.is_empty() {
            continue;
        }
        if bytes.iter().all(u8::is_ascii_whitespace) {
            // A standalone whitespace token ends the current word.
            if let Some((text, start, end)) = current.take() {
                words.push(decode_timed_word(text, start, end));
            }
            continue;
        }
        let starts_word = bytes.first().is_some_and(u8::is_ascii_whitespace);
        if starts_word || current.is_none() {
            if let Some((text, start, end)) = current.take() {
                words.push(decode_timed_word(text, start, end));
            }
            let first = bytes
                .iter()
                .position(|byte| !byte.is_ascii_whitespace())
                .unwrap_or(bytes.len());
            current = Some((bytes[first..].to_vec(), token.start, token.end));
        } else if let Some((text, _, end)) = current.as_mut() {
            text.extend_from_slice(bytes);
            // The word ends when its last fragment ends.
            *end = token.end;
        }
    }
    if let Some((text, start, end)) = current {
        words.push(decode_timed_word(text, start, end));
    }
    words
        .into_iter()
        .filter(|word| !word.text.is_empty())
        .collect()
}

/// Decode one assembled word's raw bytes. This is the only UTF-8 decode in the
/// grouping path, so split characters are recombined before conversion.
fn decode_timed_word(bytes: Vec<u8>, start: f32, end: f32) -> TimedWord {
    TimedWord {
        text: String::from_utf8_lossy(&bytes).into_owned(),
        start,
        end,
    }
}

/// Keep whole words whose midpoint is at/after `boundary_s`, the way the
/// existing word path trims repeated context. Word boundaries come from
/// grouping BPE fragments first, so a cut can never emit a word fragment such
/// as "icans". Returns the kept word texts plus the trailing punctuation of
/// the last context word, if any. This uses whisper.cpp's token timestamps, not
/// word timestamps.
pub(crate) fn trim_context_tokens(
    tokens: &[TimedToken],
    boundary_s: f32,
) -> (Vec<String>, Option<char>) {
    let mut kept = Vec::new();
    let mut context_last: Option<TimedWord> = None;
    for word in group_timed_words(tokens) {
        if (word.start + word.end) / 2.0 >= boundary_s {
            kept.push(word.text);
        } else {
            context_last = Some(word);
        }
    }
    let boundary = context_last
        .as_ref()
        .and_then(|word| word.text.trim_end().chars().last())
        .filter(|last| matches!(last, '.' | ',' | '!' | '?' | ';' | ':'));
    (kept, boundary)
}

/// Keep whisper.cpp's model/init chatter out of the daemon log, while still
/// passing warnings and errors through. The callback is process-global, so it
/// is installed once, before the context is created.
/// ggml log levels: NONE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CONT=5.
/// Pass warnings and errors (and their continuation lines) through, but keep
/// whisper.cpp's INFO chatter out of the daemon log.
const GGML_LOG_WARNING: u32 = 3;

unsafe extern "C" fn log_warnings_and_errors(
    level: u32,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    if should_log_ggml(level) && !text.is_null() {
        let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
        let _ = std::io::stderr().write_all(bytes);
    }
}

fn should_log_ggml(level: u32) -> bool {
    level >= GGML_LOG_WARNING
}

fn quiet_whisper_logs() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        whisper_rs::set_log_callback(Some(log_warnings_and_errors), std::ptr::null_mut());
    });
}

/// The loaded Whisper Tiny English context. The model stays resident in the
/// daemon worker; each decode gets a fresh state.
pub struct Whisper {
    context: WhisperContext,
    threads: usize,
}

/// Encoder positions needed for a standalone clip, with one second of margin.
/// Each position covers 20 ms. whisper.cpp otherwise always encodes a padded
/// 30 s window; on a short Whole recording or a segment-only fallback that is
/// most of the cost, especially on an old CPU. The cap is the model's 1500
/// positions, so long Whole recordings still cover the full 30 s window.
pub(crate) fn audio_ctx_for(samples: usize) -> i32 {
    let seconds = samples as f32 / 16_000.0;
    ((seconds * 50.0).ceil() as i32 + 64).clamp(512, 1500)
}

/// Encoder positions for a Pauses piece that includes left context. Context
/// audio is decoded with the segment and then removed using whisper.cpp's
/// token timestamps. `0` selects the model's own full 1500-position window.
///
/// Round 3 tried to reduce `Whisper Pauses` cost by decoding natural-pause
/// pieces alone (including shorter encoder windows). That was rejected rather
/// than tuned to the fixture: 512 positions changed "Ask not" to "S not", and
/// 768, 1024 and the full 1500 positions all changed "my" to "am I" when the
/// piece was decoded without its left context. Keep context decodes at the
/// full window.
const CONTEXT_AUDIO_CTX: i32 = 0;

/// Segment-level no-speech probability above which the decode is discarded.
/// whisper.cpp's own skip needs both this probability and a low average
/// log-probability; a silent clip can still produce a confident " you" with
/// no_speech=0.988. Dictate therefore drops any segment above the same
/// conservative 0.6 threshold on its own.
const NO_SPEECH_PROBABILITY: f32 = 0.6;

/// Whether a whisper.cpp segment's no-speech probability means silence.
fn is_no_speech(probability: f32) -> bool {
    probability > NO_SPEECH_PROBABILITY
}

/// One decoded whisper.cpp segment with its real timestamps.
struct DecodedSegment {
    /// Segment end in seconds, or `None` when whisper.cpp reported an
    /// unusable value. Never invent one: an absent end forces the safe
    /// segment-only fallback instead of guessing whether text is context.
    end_s: Option<f32>,
    text: String,
    tokens: Vec<TimedToken>,
}

impl Whisper {
    pub fn load(path: &Path, threads: usize) -> anyhow::Result<Self> {
        quiet_whisper_logs();
        let mut params = WhisperContextParameters::default();
        params.use_gpu(false);
        params.flash_attn(false);
        let context = WhisperContext::new_with_params(path, params).with_context(|| {
            format!(
                "Loading Whisper Tiny English from {} failed. Delete that file and run \"dictate download --whisper\" again",
                path.display()
            )
        })?;
        eprintln!("whisper: using {threads} threads");
        Ok(Self { context, threads })
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> anyhow::Result<String> {
        let segments = self.decode(samples, false, audio_ctx_for(samples.len()))?;
        let mut parts = Vec::new();
        for segment in segments {
            if let Some(text) = clean_segment_text(&segment.text) {
                parts.push(text);
            }
        }
        Ok(parts.join(" "))
    }

    /// Decode one bounded streaming piece. The first piece has no `context`
    /// and decodes alone. Every later piece decodes context + segment with
    /// token timestamps and keeps only text at/after the context boundary.
    ///
    /// `_starts_in_known_silence` is the segment's start provenance, retained
    /// for boundary logging and benchmarking. It deliberately does not skip
    /// the context decode: decoding a natural-pause piece alone changed words
    /// on the reference clip (see [`CONTEXT_AUDIO_CTX`]).
    ///
    /// Returns word texts (for the existing punctuation/overlap join), the
    /// boundary punctuation from the context, and whether the returned text
    /// came from a segment-only decode with no overlapping audio.
    pub(crate) fn transcribe_pieces(
        &mut self,
        context: &[f32],
        segment: &[f32],
        _starts_in_known_silence: bool,
    ) -> anyhow::Result<crate::asr::Piece> {
        if context.is_empty() {
            let text = self.transcribe(segment)?;
            let words = if text.is_empty() {
                Vec::new()
            } else {
                text.split_whitespace().map(str::to_string).collect()
            };
            return Ok(crate::asr::Piece {
                words,
                boundary: None,
                disjoint: true,
            });
        }
        let mut audio = Vec::with_capacity(context.len() + segment.len());
        audio.extend_from_slice(context);
        audio.extend_from_slice(segment);
        let context_s = context.len() as f32 / 16_000.0;
        let decoded = self.decode(&audio, true, CONTEXT_AUDIO_CTX)?;
        let mut words: Vec<String> = Vec::new();
        let mut boundary: Option<char> = None;
        let mut needs_segment_only = false;
        for decoded_segment in decoded {
            let Some(text) = clean_segment_text(&decoded_segment.text) else {
                continue;
            };
            if has_token_timings(&decoded_segment.tokens) {
                let (kept, punct) = trim_context_tokens(&decoded_segment.tokens, context_s);
                if let Some(punct) = punct {
                    boundary = Some(punct);
                }
                words.extend(kept);
            } else if decoded_segment.end_s.is_some_and(|end| end <= context_s) {
                // Fully in the context: contributes punctuation only.
                if let Some(last) = text.split_whitespace().last()
                    && let Some(punct) = last
                        .chars()
                        .last()
                        .filter(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':'))
                {
                    boundary = Some(punct);
                }
            } else {
                // The segment crosses the boundary but has no usable timing.
                // Never return its text as-is: that text includes repeated
                // context. Re-decode the segment alone below, which is safe
                // and measured; fabricating a boundary would be worse.
                needs_segment_only = true;
            }
        }
        if needs_segment_only {
            // Discard anything the context decode contributed, so the fresh
            // segment-only text cannot duplicate it.
            words.clear();
            let alone = self.decode(segment, false, audio_ctx_for(segment.len()))?;
            for decoded_segment in alone {
                if let Some(text) = clean_segment_text(&decoded_segment.text) {
                    words.extend(text.split_whitespace().map(str::to_string));
                }
            }
        }
        // The timing fallback replaced the context decode with a fresh
        // segment-only decode. Report it as disjoint so the join does not
        // drop an intentional repeated word at the boundary; the context
        // decode's punctuation remains a fallback boundary.
        Ok(crate::asr::Piece {
            words,
            boundary,
            disjoint: needs_segment_only,
        })
    }

    fn decode(
        &self,
        samples: &[f32],
        token_timestamps: bool,
        audio_ctx: i32,
    ) -> anyhow::Result<Vec<DecodedSegment>> {
        let mut state = self
            .context
            .create_state()
            .context("Could not create a Whisper decoder state")?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads as i32);
        params.set_audio_ctx(audio_ctx);
        params.set_translate(false);
        params.set_language(Some("en"));
        params.set_no_context(true);
        params.set_single_segment(false);
        params.set_n_max_text_ctx(0);
        params.set_token_timestamps(token_timestamps);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.0);
        params.set_no_speech_thold(0.6);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        state
            .full(params, samples)
            .context("Whisper transcription failed")?;
        let mut out = Vec::new();
        for segment in state.as_iter() {
            if is_no_speech(segment.no_speech_probability()) {
                continue;
            }
            out.push(decoded_segment(&segment)?);
        }
        Ok(out)
    }
}

/// Convert whisper.cpp's centisecond integer to seconds. Negative and
/// non-finite values are not timestamps Dictate can use.
fn timestamp_seconds(raw: i64) -> Option<f32> {
    let seconds = raw as f32 / 100.0;
    (seconds.is_finite() && seconds >= 0.0).then_some(seconds)
}

fn decoded_segment(segment: &WhisperSegment<'_>) -> anyhow::Result<DecodedSegment> {
    let text = segment.to_str_lossy()?.to_string();
    let mut tokens = Vec::new();
    for index in 0..segment.n_tokens() {
        let Some(token) = segment.get_token(index) else {
            anyhow::bail!("Whisper returned no token at index {index}");
        };
        let data = token.token_data();
        tokens.push(TimedToken {
            bytes: token.to_bytes()?.to_vec(),
            // Keep invalid raw values intact; `has_token_timings` rejects them
            // and the caller falls back to a segment-only decode.
            start: data.t0 as f32 / 100.0,
            end: data.t1 as f32 / 100.0,
        });
    }
    Ok(DecodedSegment {
        end_s: timestamp_seconds(segment.end_timestamp()),
        text,
        tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(text: &str, start: f32, end: f32) -> TimedToken {
        TimedToken {
            bytes: text.as_bytes().to_vec(),
            start,
            end,
        }
    }

    fn token_bytes(bytes: &[u8], start: f32, end: f32) -> TimedToken {
        TimedToken {
            bytes: bytes.to_vec(),
            start,
            end,
        }
    }

    #[test]
    fn audio_ctx_covers_the_audio_without_padding_a_full_window() {
        assert_eq!(audio_ctx_for(0), 512);
        assert_eq!(audio_ctx_for(16_000), 512);
        assert_eq!(audio_ctx_for(6 * 16_000), 512);
        assert_eq!(audio_ctx_for(9 * 16_000), 514);
        assert_eq!(audio_ctx_for(11 * 16_000), 614);
        assert_eq!(audio_ctx_for(30 * 16_000), 1500);
        assert_eq!(audio_ctx_for(10 * 60 * 16_000), 1500);
    }

    #[test]
    fn token_trim_drops_context_and_keeps_boundary_punctuation() {
        let tokens = vec![
            token(" for", 0.2, 0.4),
            token(" you,", 0.4, 0.8),
            token(" ask", 1.6, 1.8),
            token(" what", 1.8, 2.0),
        ];
        let (words, punct) = trim_context_tokens(&tokens, 1.5);
        assert_eq!(words, vec!["ask", "what"]);
        assert_eq!(punct, Some(','));
    }

    #[test]
    fn token_trim_keeps_token_at_boundary() {
        let tokens = vec![token(" ask", 1.4, 1.6)];
        let (words, punct) = trim_context_tokens(&tokens, 1.5);
        assert_eq!(words, vec!["ask"]);
        assert_eq!(punct, None);
    }

    #[test]
    fn token_trim_never_emits_a_fragment_of_a_straddling_word() {
        // BPE tokens can split a word; their concatenated text is the word.
        // The boundary falls after the whole word's midpoint, so the word is
        // context and none of its fragments ("icans") may leak into the text.
        let tokens = vec![
            token(" Amer", 0.8, 1.0),
            token("icans", 1.0, 1.2),
            token(" ask", 1.6, 1.8),
        ];
        let (words, _) = trim_context_tokens(&tokens, 1.1);
        assert_eq!(words, vec!["ask"]);

        // A whole word straddling the cut is kept as a whole when its
        // midpoint is at/after the boundary.
        let tokens = vec![
            token(" Amer", 0.8, 1.0),
            token("icans", 1.0, 1.2),
            token(" ask", 1.6, 1.8),
        ];
        let (words, _) = trim_context_tokens(&tokens, 0.95);
        assert_eq!(words, vec!["Americans", "ask"]);
    }

    #[test]
    fn token_trim_preserves_deliberate_repeated_phrases() {
        // Deliberate repetition is speech, not a hallucination. The context
        // decode repeats "I agree with you" because the new audio says it
        // again; both occurrences after the boundary stay in the output.
        let tokens = vec![
            token(" I", 0.2, 0.4),
            token(" agree", 0.4, 0.6),
            token(" with", 0.6, 0.8),
            token(" you", 0.8, 1.0),
            token(" I", 1.2, 1.4),
            token(" agree", 1.4, 1.6),
            token(" with", 1.6, 1.8),
            token(" you", 1.8, 2.0),
        ];
        let (words, _) = trim_context_tokens(&tokens, 1.1);
        assert_eq!(words, vec!["I", "agree", "with", "you"]);
        let (words, _) = trim_context_tokens(&tokens, 0.9);
        assert_eq!(
            words,
            vec!["you", "I", "agree", "with", "you"],
            "overlap dedupe belongs to the joiner, not context trimming"
        );
    }

    #[test]
    fn token_trim_preserves_utf8_and_attached_punctuation() {
        let tokens = vec![
            token(" déjà", 0.2, 0.4),
            token(" vu,", 0.4, 0.6),
            token(" café", 1.6, 1.8),
        ];
        let (words, punct) = trim_context_tokens(&tokens, 1.5);
        assert_eq!(words, vec!["café"]);
        assert_eq!(punct, Some(','));
    }

    #[test]
    fn token_trim_reassembles_a_utf8_character_split_across_bpe_tokens() {
        // whisper.cpp BPE can split a multi-byte character between tokens.
        // "café" is 63 61 66 c3 a9; a token boundary after c3 must not be
        // decoded separately or the name becomes "caf�".
        let tokens = vec![
            token_bytes(b" caf\xc3", 0.2, 0.4),
            token_bytes(b"\xa9", 0.4, 0.6),
            token(" please", 1.6, 1.8),
        ];
        let (words, _) = trim_context_tokens(&tokens, 0.1);
        assert_eq!(words, vec!["café", "please"]);
        let (words, _) = trim_context_tokens(&tokens, 1.5);
        assert_eq!(words, vec!["please"]);
    }

    #[test]
    fn a_special_token_without_timing_does_not_disable_trimming() {
        let tokens = vec![
            token("[_BEG_]", -1.0, -1.0),
            token(" for", 0.2, 0.4),
            token(" ask", 1.6, 1.8),
        ];
        assert!(has_token_timings(&tokens));
        let (words, _) = trim_context_tokens(&tokens, 1.5);
        assert_eq!(words, vec!["ask"]);
    }

    #[test]
    fn token_timings_must_be_finite_nonnegative_and_ordered() {
        assert!(!has_token_timings(&[]));
        assert!(!has_token_timings(&[token(" ask", -1.0, 1.0)]));
        assert!(!has_token_timings(&[token(" ask", 1.0, -1.0)]));
        assert!(!has_token_timings(&[token(" ask", 2.0, 1.0)]));
        assert!(!has_token_timings(&[token(" ask", f32::NAN, 1.0)]));
        assert!(!has_token_timings(&[token(" ask", 0.0, f32::INFINITY)]));
        assert!(has_token_timings(&[token(" ask", 1.0, 1.0)]));
        assert!(!has_token_timings(&[
            token(" ask", 1.0, 1.2),
            token(" what", 0.8, 1.0),
        ]));
    }

    #[test]
    fn unusable_segment_end_is_not_fabricated() {
        assert_eq!(timestamp_seconds(0), Some(0.0));
        assert_eq!(timestamp_seconds(123), Some(1.23));
        assert_eq!(timestamp_seconds(-1), None);
    }

    #[test]
    fn info_logs_are_filtered_but_warnings_and_errors_pass() {
        assert!(!should_log_ggml(0));
        assert!(!should_log_ggml(1));
        assert!(!should_log_ggml(2));
        assert!(should_log_ggml(3));
        assert!(should_log_ggml(4));
    }

    #[test]
    fn high_no_speech_probability_is_silence() {
        assert!(!is_no_speech(0.0));
        assert!(!is_no_speech(0.6));
        assert!(is_no_speech(0.61));
        assert!(is_no_speech(0.988));
    }

    #[test]
    fn special_tokens_are_removed_from_clean_text() {
        assert_eq!(
            clean_segment_text(" Ask not.[_TT_169] ").as_deref(),
            Some("Ask not.")
        );
        assert_eq!(
            clean_segment_text("[_BEG_] And so my fellow Americans.<|endoftext|>").as_deref(),
            Some("And so my fellow Americans.")
        );
    }

    #[test]
    fn timing_tokens_do_not_leak_into_context_trimming() {
        let tokens = vec![
            token(" you", 1.0, 1.4),
            token("[_TT_140]", 1.4, 1.4),
            token(" ask", 1.6, 1.8),
        ];
        let (words, _) = trim_context_tokens(&tokens, 1.5);
        assert_eq!(words, vec!["ask"]);
    }

    #[test]
    fn noise_markers_are_dropped_but_normal_brackets_are_kept() {
        assert!(is_noise_text("[BLANK_AUDIO]"));
        assert!(is_noise_text(" (silence) "));
        assert!(is_noise_text("[Music]"));
        assert!(!is_noise_text("[inaudible word]"));
        assert!(!is_noise_text("music"));
        assert!(!is_noise_text("silence"));
        assert_eq!(
            clean_segment_text("  hello   world  ").as_deref(),
            Some("hello world")
        );
        assert_eq!(clean_segment_text("[BLANK_AUDIO]"), None);
    }
}
