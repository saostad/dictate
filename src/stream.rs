//! Buffered streaming engine for type-while-speaking.
//!
//! Instead of cutting the recording at VAD pauses (a cut lands inside a word
//! whenever speech and noise have similar energy), the worker accumulates the
//! whole 16 kHz recording and re-decodes a growing window every [`STEP_S`] of
//! new audio. Words clear of the right guard are committed (typed) and never
//! re-decoded; the guard region is re-decoded next time with more right
//! context. At stop, the final decode commits everything remaining.

use crate::asr::Word;
use crate::audio::TARGET_SAMPLE_RATE;

/// New audio that triggers another decode.
pub(crate) const STEP_S: f32 = 3.0;
/// Left context decoded ahead of the committed position, for accuracy.
pub(crate) const CONTEXT_S: f32 = 1.5;
/// Right guard: words ending inside it are re-decoded next time.
pub(crate) const RIGHT_GUARD_S: f32 = 1.0;
/// Guard when the window exceeds [`MAX_WINDOW_S`], so decoding always
/// makes progress.
pub(crate) const BACKLOG_GUARD_S: f32 = 0.3;
/// Window length that switches to the backlog guard.
pub(crate) const MAX_WINDOW_S: f32 = 12.0;

const STEP_SAMPLES: usize = (STEP_S * TARGET_SAMPLE_RATE as f32) as usize;
const CONTEXT_SAMPLES: usize = (CONTEXT_S * TARGET_SAMPLE_RATE as f32) as usize;

/// One decode window: its start in the recording plus its lengths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Window {
    pub(crate) start: usize,
    pub(crate) ctx_s: f32,
    pub(crate) window_s: f32,
    pub(crate) guard_s: f32,
}

/// Buffered 16 kHz audio of one recording plus the commit positions.
#[derive(Debug)]
pub(crate) struct Stream {
    audio: Vec<f32>,
    committed_until: usize,
    last_decode_len: usize,
    decodes: usize,
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}

impl Stream {
    /// Empty stream for one recording.
    pub(crate) fn new() -> Self {
        Self {
            audio: Vec::new(),
            committed_until: 0,
            last_decode_len: 0,
            decodes: 0,
        }
    }

    /// Append 16 kHz samples.
    pub(crate) fn push(&mut self, chunk: &[f32]) {
        self.audio.extend_from_slice(chunk);
    }

    /// Whole buffered audio (for the final dump).
    pub(crate) fn audio(&self) -> &[f32] {
        &self.audio
    }

    /// True when no audio arrived yet.
    pub(crate) fn is_empty(&self) -> bool {
        self.audio.is_empty()
    }

    /// Successful decodes so far (the `N` in `decode N`).
    pub(crate) fn decodes(&self) -> usize {
        self.decodes
    }

    /// True when at least [`STEP_S`] of new audio arrived since the last
    /// decode (successful or not).
    pub(crate) fn should_decode(&self) -> bool {
        self.audio.len().saturating_sub(self.last_decode_len) >= STEP_SAMPLES
    }

    /// Current decode window: from [`CONTEXT_S`] before the committed
    /// position (or sample 0) to the buffered end, with the backlog guard
    /// when it exceeds [`MAX_WINDOW_S`].
    pub(crate) fn window(&self) -> Window {
        let start = self.committed_until.saturating_sub(CONTEXT_SAMPLES);
        let window_s = (self.audio.len().saturating_sub(start)) as f32 / TARGET_SAMPLE_RATE as f32;
        let ctx_s = (self.committed_until.saturating_sub(start)) as f32 / TARGET_SAMPLE_RATE as f32;
        let guard_s = if window_s > MAX_WINDOW_S {
            BACKLOG_GUARD_S
        } else {
            RIGHT_GUARD_S
        };
        Window {
            start,
            ctx_s,
            window_s,
            guard_s,
        }
    }

    /// Audio of [`Stream::window`], the exact samples sent to the model.
    pub(crate) fn window_audio(&self) -> &[f32] {
        let start = self.window().start.min(self.audio.len());
        &self.audio[start..]
    }

    /// Record a successful decode: the next trigger needs [`STEP_S`] more.
    pub(crate) fn mark_decoded(&mut self) {
        self.last_decode_len = self.audio.len();
        self.decodes += 1;
    }

    /// Record a failed decode: the next trigger needs [`STEP_S`] more, but
    /// the decode count (and dump numbering) only covers successes.
    pub(crate) fn mark_failed(&mut self) {
        self.last_decode_len = self.audio.len();
    }

    /// Advance the committed position to `new_end_s` (window-relative end of
    /// the last committed word), clamped to the buffered audio.
    pub(crate) fn advance(&mut self, window_start: usize, new_end_s: f32) {
        let end = window_start + (new_end_s * TARGET_SAMPLE_RATE as f32).round() as usize;
        self.committed_until = end.min(self.audio.len());
    }
}

/// Commit words from one window decode, in order.
///
/// Words whose midpoint is before `ctx_s` are already committed and skipped.
/// The rest commit while their `end` stays at or before
/// `window_s - guard_s`, stopping at the first that isn't (uncommitted words
/// are re-decoded next time with more right context).
///
/// Returns the committed words plus the window-relative end of the last one
/// (`None` when nothing committed).
pub(crate) fn commit_words(
    words: &[Word],
    ctx_s: f32,
    window_s: f32,
    guard_s: f32,
) -> (Vec<Word>, Option<f32>) {
    let mut committed = Vec::new();
    let limit = window_s - guard_s;
    for word in words {
        if (word.start + word.end) / 2.0 < ctx_s {
            continue;
        }
        if word.end > limit {
            break;
        }
        committed.push(word.clone());
    }
    let new_end = committed.last().map(|word| word.end);
    (committed, new_end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f32, end: f32) -> Word {
        Word {
            text: text.to_string(),
            start,
            end,
        }
    }

    fn texts(words: &[Word]) -> Vec<String> {
        words.iter().map(|word| word.text.clone()).collect()
    }

    #[test]
    fn commit_skips_context_words_but_keeps_the_boundary_word() {
        let words = vec![
            word("country", 0.5, 0.9),
            word("you", 1.2, 1.6),
            word("ask", 1.4, 1.6),
            word("what", 1.7, 1.9),
        ];
        // "you" straddles the 1.5 s boundary with its midpoint (1.4) before
        // it, so it stays context; "ask" sits exactly on it and commits.
        let (committed, new_end) = commit_words(&words, 1.5, 4.0, 1.0);
        assert_eq!(texts(&committed), vec!["ask", "what"]);
        assert_eq!(new_end, Some(1.9));
    }

    #[test]
    fn commit_stops_at_the_guard_and_keeps_order() {
        let words = vec![
            word("and", 0.0, 1.0),
            word("so", 1.0, 2.0),
            word("my", 2.0, 3.0),
            word("fellow", 3.0, 3.5),
        ];
        // Window 4 s, guard 1 s: words ending at or before 3.0 commit, in
        // order, stopping at the first that doesn't.
        let (committed, new_end) = commit_words(&words, 0.0, 4.0, 1.0);
        assert_eq!(texts(&committed), vec!["and", "so", "my"]);
        assert_eq!(new_end, Some(3.0));
    }

    #[test]
    fn commit_empty_is_empty() {
        assert_eq!(commit_words(&[], 1.5, 4.0, 1.0), (Vec::new(), None));
    }

    #[test]
    fn commit_with_backlog_guard_makes_progress() {
        let words = vec![word("country", 11.5, 12.5)];
        // A 13 s window with the default 1 s guard commits nothing (12.5 >
        // 12.0); the 0.3 s backlog guard commits it (12.5 <= 12.7).
        assert_eq!(
            commit_words(&words, 0.0, 13.0, RIGHT_GUARD_S),
            (Vec::new(), None)
        );
        let (committed, new_end) = commit_words(&words, 0.0, 13.0, BACKLOG_GUARD_S);
        assert_eq!(texts(&committed), vec!["country"]);
        assert_eq!(new_end, Some(12.5));
    }

    #[test]
    fn decode_trigger_needs_three_seconds_of_new_audio() {
        let mut stream = Stream::new();
        assert!(!stream.should_decode());
        stream.push(&vec![0.0; STEP_SAMPLES - 1]);
        assert!(!stream.should_decode());
        stream.push(&[0.0]);
        assert!(stream.should_decode());
        stream.mark_decoded();
        assert!(!stream.should_decode());
        stream.mark_failed();
        assert!(!stream.should_decode());
    }

    #[test]
    fn window_starts_1_5s_before_the_committed_position() {
        let mut stream = Stream::new();
        stream.push(&vec![0.0; 8 * TARGET_SAMPLE_RATE as usize]);
        stream.advance(0, 5.0);
        let window = stream.window();
        assert_eq!(
            window.start,
            ((5.0 - CONTEXT_S) * TARGET_SAMPLE_RATE as f32) as usize
        );
        assert!((window.ctx_s - CONTEXT_S).abs() < 1e-6);
        assert!((window.window_s - 4.5).abs() < 1e-6);
        assert_eq!(window.guard_s, RIGHT_GUARD_S);
        assert_eq!(
            stream.window_audio().len(),
            (window.window_s * TARGET_SAMPLE_RATE as f32) as usize
        );
    }

    #[test]
    fn window_near_the_start_has_short_context() {
        let mut stream = Stream::new();
        stream.push(&vec![0.0; 3 * TARGET_SAMPLE_RATE as usize]);
        let window = stream.window();
        assert_eq!(window.start, 0);
        assert_eq!(window.ctx_s, 0.0);
        assert!((window.window_s - 3.0).abs() < 1e-6);
    }

    #[test]
    fn window_over_12s_uses_the_backlog_guard() {
        let mut stream = Stream::new();
        stream.push(&vec![0.0; 12 * TARGET_SAMPLE_RATE as usize]);
        assert_eq!(stream.window().guard_s, RIGHT_GUARD_S);
        stream.push(&[0.0]);
        assert_eq!(stream.window().guard_s, BACKLOG_GUARD_S);
    }

    #[test]
    fn advance_clamps_to_the_buffered_audio() {
        let mut stream = Stream::new();
        stream.push(&vec![0.0; TARGET_SAMPLE_RATE as usize]);
        stream.advance(0, 99.0);
        assert_eq!(stream.window().ctx_s, 1.0);
    }
}
