//! Pause-based audio segmenter for type-while-speaking.
//!
//! Splits mono 16 kHz audio at natural pauses so the daemon can transcribe
//! and type each piece while the user keeps talking. Silero VAD scores
//! 512-sample frames (32 ms each); speech starts at p >= 0.5 and ends at
//! p < 0.35 (hysteresis: uncertain frames keep the current state). Cuts land
//! at the middle of pauses of >= 400 ms, force-cuts at 15 s after the
//! lowest-prob frame in the last 3 s. Segments are a contiguous, sample-exact
//! partition of the whole recording: the first starts at sample 0, and only
//! segments with no speech frames are skipped.
//!
//! `Feeder` is the shared streaming path (daemon + `transcribe --segmented`):
//! it resamples the continuous stream to 16 kHz once (stateful, so there are
//! no per-segment resample edges) and drives a `Segmenter` at 16 kHz. Each
//! emitted segment also carries up to `CONTEXT_S` of trailing audio before
//! it (empty for the first), decoded as left context.
//!
//! Every emitted segment also records where it *starts*: at the recording
//! start, after a confirmed natural pause, or after a forced 15 s cut. This
//! is the provenance of the previous boundary, not of the segment's own
//! ending cut. The feeder attaches left context for every engine and every
//! segment after the first. Whisper tried skipping that context decode when
//! the start was known silence, but decoding the piece alone changed real
//! words on the reference clip; the provenance is retained for boundary logs
//! and benchmarks, not as a decoder shortcut.
//!
//! Cut decisions read only the prob sequence (via [`Scorer`]), so the tests
//! drive the segmenter with synthetic probs and no model.

/// Non-speech frames that trigger a pause cut: 13 x 32 ms = 416 ms >= 400 ms.
const SILENCE_CUT_FRAMES: usize = 13;
/// Segment length that triggers a force-cut: 15 s at 16 kHz.
const MAX_SEGMENT_SAMPLES: usize = 240_000;
/// Force-cut search window: the lowest-prob frame in the last ~3 s
/// (93 x 32 ms = 2976 ms).
const FORCE_SEARCH_FRAMES: usize = 93;
/// Speech starts at p >= 0.5 ...
const SPEECH_START_PROB: f32 = 0.5;
/// ... and ends at p < 0.35 (hysteresis).
const SPEECH_END_PROB: f32 = 0.35;

/// Left context decoded ahead of every segment after the first: seconds of
/// trailing 16 kHz audio before the segment, for accuracy at the cut.
pub(crate) const CONTEXT_S: f32 = 1.5;

/// Samples of [`CONTEXT_S`] at the feeder's 16 kHz output rate.
const CONTEXT_SAMPLES: usize = (CONTEXT_S * crate::audio::TARGET_SAMPLE_RATE as f32) as usize;

/// Where a segment starts, relative to the previous cut (the segment's START
/// provenance, never the provenance of its own ending cut).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmentStart {
    /// The recording began here (possibly with leading silence). No previous
    /// cut exists, so the segment never overlaps already-typed speech.
    Recording,
    /// The previous cut was in a confirmed natural pause (>= 416 ms of
    /// silence, cut at its middle). This segment starts with the pause
    /// remainder, so its beginning is known silence and it does not overlap
    /// already-typed speech. Later audio is still decoded with left context
    /// for accuracy; this provenance is used for logs and benchmarks.
    AfterPause,
    /// The previous cut was forced at the 15 s limit. This segment may start
    /// in the middle of speech, so a decoder needs left context.
    AfterForce,
}

impl SegmentStart {
    /// True when the segment begins in confirmed silence: the recording
    /// start or the second half of a natural pause. Kept for boundary
    /// diagnostics; decoders still use left context when it is available.
    pub(crate) fn in_known_silence(self) -> bool {
        matches!(self, Self::Recording | Self::AfterPause)
    }

    /// Short label for the per-segment timing log.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Recording => "recording start",
            Self::AfterPause => "after pause",
            Self::AfterForce => "after forced cut",
        }
    }
}

/// One segment emitted by [`Segmenter`], before the feeder attaches context.
#[derive(Debug)]
pub(crate) struct Cut {
    pub(crate) audio: Vec<f32>,
    pub(crate) start: SegmentStart,
}

/// One finished streaming segment at 16 kHz, plus the trailing audio
/// immediately before it (empty for the first segment), decoded as left
/// context.
#[derive(Debug)]
pub(crate) struct Segment {
    pub(crate) audio: Vec<f32>,
    pub(crate) context: Vec<f32>,
    pub(crate) start: SegmentStart,
}

/// Per-frame speech probability at 16 kHz. The segmenter reads only this
/// prob sequence, so tests script it without the model.
pub(crate) trait Scorer {
    fn prob(&mut self, frame: &[f32]) -> anyhow::Result<f32>;
}

impl Scorer for crate::vad::Vad {
    fn prob(&mut self, frame: &[f32]) -> anyhow::Result<f32> {
        crate::vad::Vad::prob(self, frame)
    }
}

/// Streaming feeder path shared by the daemon and `transcribe --segmented`:
/// resamples the continuous device-rate stream to 16 kHz once (stateful) and
/// segments at 16 kHz, so live segmentation equals file segmentation.
pub(crate) struct Feeder<S> {
    resampler: Option<crate::audio::StreamResampler>,
    segmenter: Segmenter<S>,
    /// Trailing audio of previously emitted segments (at most
    /// `CONTEXT_SAMPLES`), attached as left context to the next one.
    history: Vec<f32>,
}

impl<S: Scorer> Feeder<S> {
    /// New feeder for `device_rate` Hz mono input scored by `scorer`;
    /// segments come out at 16 kHz.
    pub(crate) fn new(device_rate: u32, scorer: S) -> anyhow::Result<Self> {
        if device_rate == 0 {
            anyhow::bail!("Invalid sample rate 0 Hz");
        }
        let resampler = if device_rate == crate::audio::TARGET_SAMPLE_RATE {
            None
        } else {
            Some(crate::audio::StreamResampler::new(device_rate)?)
        };
        Ok(Self {
            resampler,
            segmenter: Segmenter::new(scorer),
            history: Vec::new(),
        })
    }

    /// Push device-rate samples; returns finished 16 kHz segments.
    pub(crate) fn push(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Segment>> {
        match &mut self.resampler {
            Some(resampler) => {
                let mono16 = resampler.push(samples)?;
                let audios = self.segmenter.push(&mono16)?;
                Ok(self.wrap(audios))
            }
            None => {
                let audios = self.segmenter.push(samples)?;
                Ok(self.wrap(audios))
            }
        }
    }

    /// Finish the recording: returns finished segments, including the tail
    /// when it holds speech. (The flushed tail audio can still complete a
    /// segment, which is why this returns a list and not just the tail.)
    pub(crate) fn finish(&mut self) -> anyhow::Result<Vec<Segment>> {
        let tail16 = match &mut self.resampler {
            Some(resampler) => resampler.finish()?,
            None => Vec::new(),
        };
        let mut audios = self.segmenter.push(&tail16)?;
        if let Some(tail) = self.segmenter.finish() {
            audios.push(tail);
        }
        Ok(self.wrap(audios))
    }

    /// The scorer back (the daemon reuses one Vad across recordings).
    pub(crate) fn into_scorer(self) -> S {
        self.segmenter.into_scorer()
    }

    /// Attach left context to freshly emitted segments, in order: the first
    /// ever emitted gets none, each later one gets the trailing audio of all
    /// previously emitted segments (segments partition the stream, so that is
    /// exactly the audio before it).
    fn wrap(&mut self, cuts: Vec<Cut>) -> Vec<Segment> {
        let mut out = Vec::with_capacity(cuts.len());
        for cut in cuts {
            let Cut { audio, start } = cut;
            let context = self.history.clone();
            self.history.extend_from_slice(&audio);
            let drop = self.history.len().saturating_sub(CONTEXT_SAMPLES);
            self.history.drain(..drop);
            out.push(Segment {
                audio,
                context,
                start,
            });
        }
        out
    }
}

/// Splits mono 16 kHz audio at pauses, scored by a [`Scorer`]. Usually driven
/// through [`Feeder`].
pub(crate) struct Segmenter<S> {
    scorer: S,
    pending: Vec<f32>,
    samples: Vec<f32>,
    probs: Vec<f32>,
    speech: Vec<bool>,
    has_speech: bool,
    trailing_silence: usize,
    /// Start provenance of the currently buffered (not yet emitted) audio.
    start: SegmentStart,
}

impl<S: Scorer> Segmenter<S> {
    /// New segmenter for 16 kHz mono audio scored by `scorer`.
    pub(crate) fn new(scorer: S) -> Self {
        Self {
            scorer,
            pending: Vec::new(),
            samples: Vec::new(),
            probs: Vec::new(),
            speech: Vec::new(),
            has_speech: false,
            trailing_silence: 0,
            start: SegmentStart::Recording,
        }
    }

    /// Push 16 kHz samples; returns finished segments.
    pub(crate) fn push(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Cut>> {
        const FRAME: usize = crate::vad::FRAME_LEN;
        let mut out = Vec::new();
        let mut offset = 0;
        // Finish the pending partial frame first.
        if !self.pending.is_empty() {
            let need = FRAME - self.pending.len();
            let take = need.min(samples.len() - offset);
            self.pending
                .extend_from_slice(&samples[offset..offset + take]);
            offset += take;
            if self.pending.len() == FRAME {
                let frame = std::mem::take(&mut self.pending);
                if let Some(segment) = self.push_frame(&frame)? {
                    out.push(segment);
                }
            }
        }
        while offset + FRAME <= samples.len() {
            let frame = &samples[offset..offset + FRAME];
            offset += FRAME;
            if let Some(segment) = self.push_frame(frame)? {
                out.push(segment);
            }
        }
        if offset < samples.len() {
            self.pending.extend_from_slice(&samples[offset..]);
        }
        Ok(out)
    }

    /// Finish the recording; returns the tail when it holds speech.
    /// Tails with no speech frames are skipped (the only audio ever dropped).
    pub(crate) fn finish(&mut self) -> Option<Cut> {
        if !self.has_speech {
            return None;
        }
        let mut tail = std::mem::take(&mut self.samples);
        tail.extend_from_slice(&self.pending);
        self.pending.clear();
        self.probs.clear();
        self.speech.clear();
        self.has_speech = false;
        self.trailing_silence = 0;
        let start = self.start;
        self.start = SegmentStart::Recording;
        Some(Cut { audio: tail, start })
    }

    /// The scorer back.
    pub(crate) fn into_scorer(self) -> S {
        self.scorer
    }

    /// Score one complete frame and maybe emit a segment.
    fn push_frame(&mut self, frame: &[f32]) -> anyhow::Result<Option<Cut>> {
        let prob = self.scorer.prob(frame)?;
        // Hysteresis: speech starts at p >= 0.5 and ends at p < 0.35.
        let was_speech = self.speech.last().copied().unwrap_or(false);
        let is_speech = if was_speech {
            prob >= SPEECH_END_PROB
        } else {
            prob >= SPEECH_START_PROB
        };
        self.samples.extend_from_slice(frame);
        self.probs.push(prob);
        self.speech.push(is_speech);
        if is_speech {
            self.has_speech = true;
            self.trailing_silence = 0;
        } else {
            self.trailing_silence += 1;
        }
        if !self.has_speech {
            // No speech yet: keep everything so the first segment starts at
            // sample 0, and never cut silence alone.
            return Ok(None);
        }
        if self.trailing_silence >= SILENCE_CUT_FRAMES {
            return Ok(Some(self.cut_pause()));
        }
        if self.samples.len() >= MAX_SEGMENT_SAMPLES {
            return Ok(self.cut_force());
        }
        Ok(None)
    }

    /// Cut at the middle of the trailing pause; the pause remainder carries
    /// over. The head always holds the speech (the remainder is pause only).
    /// Nothing is trimmed: the head plus the remainder exactly partition the
    /// buffered audio. The next segment starts in the confirmed pause.
    fn cut_pause(&mut self) -> Cut {
        let silence = self.trailing_silence;
        let keep_in_first = silence / 2;
        let remainder_frames = silence - keep_in_first;
        let cut = self.samples.len() - remainder_frames * crate::vad::FRAME_LEN;
        let first_frames = self.speech.len() - remainder_frames;
        let head = self.samples[..cut].to_vec();
        let remainder = self.samples[cut..].to_vec();
        let remainder_probs = self.probs[first_frames..].to_vec();
        let remainder_speech = self.speech[first_frames..].to_vec();
        self.samples = remainder;
        self.probs = remainder_probs;
        self.speech = remainder_speech;
        self.has_speech = self.speech.iter().any(|&speech| speech);
        self.trailing_silence = trailing_count(&self.speech);
        let start = self.start;
        self.start = SegmentStart::AfterPause;
        Cut { audio: head, start }
    }

    /// Force-cut a long segment after the lowest-prob frame in the last 3 s.
    /// A first half with no speech frames is skipped (`None`); otherwise
    /// nothing is trimmed. The next segment may start mid-speech, so it keeps
    /// the forced-cut provenance even when the skipped head was silence.
    fn cut_force(&mut self) -> Option<Cut> {
        let frames = self.speech.len();
        let window = FORCE_SEARCH_FRAMES.min(frames);
        let window_start = frames - window;
        let mut best = window_start;
        for index in window_start + 1..frames {
            if self.probs[index] < self.probs[best] {
                best = index;
            }
        }
        let cut_frames = (best + 1).min(frames);
        let cut = cut_frames * crate::vad::FRAME_LEN;
        let cut = cut.min(self.samples.len());
        let first_has_speech = self.speech[..cut_frames].iter().any(|&speech| speech);
        let head = self.samples[..cut].to_vec();
        let remainder = self.samples[cut..].to_vec();
        let remainder_probs = self.probs[cut_frames..].to_vec();
        let remainder_speech = self.speech[cut_frames..].to_vec();
        self.samples = remainder;
        self.probs = remainder_probs;
        self.speech = remainder_speech;
        self.has_speech = self.speech.iter().any(|&speech| speech);
        self.trailing_silence = trailing_count(&self.speech);
        let start = self.start;
        self.start = SegmentStart::AfterForce;
        if !first_has_speech {
            return None;
        }
        Some(Cut { audio: head, start })
    }
}

/// Count trailing non-speech frames.
fn trailing_count(speech: &[bool]) -> usize {
    speech.iter().rev().take_while(|&&speech| !speech).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;
    const FRAME: usize = crate::vad::FRAME_LEN;

    /// Scorer scripted with a prob sequence: every cut decision in these
    /// tests comes from synthetic probs, no model needed.
    struct Scripted {
        probs: Vec<f32>,
        index: usize,
    }

    impl Scripted {
        fn new(probs: Vec<f32>) -> Self {
            Self { probs, index: 0 }
        }
    }

    impl Scorer for Scripted {
        fn prob(&mut self, _frame: &[f32]) -> anyhow::Result<f32> {
            let prob = self.probs.get(self.index).copied().unwrap_or(0.0);
            self.index += 1;
            Ok(prob)
        }
    }

    /// Prob sequence from (prob, frames) runs.
    fn seq(runs: &[(f32, usize)]) -> Vec<f32> {
        let mut probs = Vec::new();
        for &(prob, frames) in runs {
            probs.extend(std::iter::repeat_n(prob, frames));
        }
        probs
    }

    /// `frames` frames of distinct samples, so partition tests catch wrong
    /// offsets (uniform audio would hide them).
    fn ramp(frames: usize) -> Vec<f32> {
        (0..frames * FRAME).map(|i| i as f32).collect()
    }

    /// A segmenter scripted with `runs`, plus exactly the audio for it.
    fn scripted(runs: &[(f32, usize)]) -> (Segmenter<Scripted>, Vec<f32>) {
        let probs = seq(runs);
        let frames = probs.len();
        (Segmenter::new(Scripted::new(probs)), ramp(frames))
    }

    /// Sine burst (speech-like) at 440 Hz, amplitude 0.2.
    fn tone(secs: f32) -> Vec<f32> {
        tone_at(RATE, 0.2, secs)
    }

    fn tone_at(rate: u32, amplitude: f32, secs: f32) -> Vec<f32> {
        let len = (secs * rate as f32) as usize;
        (0..len)
            .map(|i| {
                amplitude * (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / rate as f32).sin()
            })
            .collect()
    }

    /// Near-silence: faint 440 Hz at amplitude 0.001 (RMS ~0.0007).
    fn silence(secs: f32) -> Vec<f32> {
        silence_at(RATE, secs)
    }

    fn silence_at(rate: u32, secs: f32) -> Vec<f32> {
        let len = (secs * rate as f32) as usize;
        (0..len)
            .map(|i| 0.001 * (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / rate as f32).sin())
            .collect()
    }

    /// Push `audio` in 100 ms chunks, like the feeder thread.
    fn push_chunked<S: Scorer>(segmenter: &mut Segmenter<S>, audio: &[f32]) -> Vec<Cut> {
        let chunk = RATE as usize / 10;
        let mut out = Vec::new();
        for piece in audio.chunks(chunk) {
            out.extend(segmenter.push(piece).unwrap());
        }
        out
    }

    fn concat(mut segments: Vec<Cut>, tail: Option<Cut>) -> Vec<f32> {
        if let Some(tail) = tail {
            segments.push(tail);
        }
        segments
            .into_iter()
            .flat_map(|segment| segment.audio)
            .collect()
    }

    #[test]
    fn pause_splits_into_two_segments() {
        let (mut segmenter, audio) = scripted(&[(0.9, 100), (0.1, 15), (0.9, 100)]);
        let mut segments = push_chunked(&mut segmenter, &audio);
        if let Some(tail) = segmenter.finish() {
            segments.push(tail);
        }
        assert_eq!(segments.len(), 2, "a 480 ms pause must split");
        for (index, segment) in segments.iter().enumerate() {
            assert!(!segment.audio.is_empty(), "segment {index} must hold audio");
        }
    }

    #[test]
    fn short_pause_does_not_split() {
        // 9 frames = 288 ms (~300 ms), 12 frames = 384 ms (just under 400).
        for pause_frames in [9, 12] {
            let (mut segmenter, audio) = scripted(&[(0.9, 100), (0.1, pause_frames), (0.9, 100)]);
            let segments = push_chunked(&mut segmenter, &audio);
            assert!(
                segments.is_empty(),
                "{} silence frames ({} ms) must not split ({} segments)",
                pause_frames,
                pause_frames * 32,
                segments.len()
            );
            let tail = segmenter.finish().expect("the whole recording is one tail");
            assert_eq!(tail.audio, audio, "nothing may be dropped");
        }
    }

    #[test]
    fn pause_cut_lands_at_pause_middle() {
        // Exactly 13 silence frames (416 ms): the cut triggers at the 13th
        // and keeps 6 in the first segment.
        let (mut segmenter, audio) = scripted(&[(0.9, 100), (0.1, 13), (0.9, 100)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert_eq!(segments.len(), 1, "the pause must cut once");
        assert_eq!(
            segments[0].audio.len(),
            (100 + 6) * FRAME,
            "the cut must land at the pause middle"
        );
        let tail = segmenter.finish().expect("second segment");
        assert_eq!(
            segments[0].audio.len() + tail.audio.len(),
            audio.len(),
            "the cut must not drop samples"
        );
    }

    #[test]
    fn long_speech_force_cuts_every_fifteen_seconds() {
        // 20 s of speech: 1250 frames. Each force-cut triggers at 240128
        // samples (frame 469 of the buffer); all probs equal, so it lands
        // after the first frame of the last-3-s window (frame 377).
        let (mut segmenter, audio) = scripted(&[(0.9, 1250)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert_eq!(segments.len(), 3, "20 s of speech must force-cut 3 times");
        for (index, segment) in segments.iter().enumerate() {
            assert_eq!(
                segment.audio.len(),
                377 * FRAME,
                "force-cut {} must land at 15 s minus the 3 s window",
                index + 1
            );
        }
        let tail = segmenter.finish().expect("the remainder is the tail");
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.audio.len())
                .sum::<usize>()
                + tail.audio.len(),
            audio.len(),
            "force-cuts must not drop samples"
        );
    }

    #[test]
    fn force_cut_lands_after_lowest_prob_frame() {
        // 16 s of speech with one uncertain frame (0.05) at index 400, inside
        // the last-3-s window (indices 376..469) when the force-cut triggers.
        let mut probs = vec![0.9; 512];
        probs[400] = 0.05;
        let mut segmenter = Segmenter::new(Scripted::new(probs));
        let audio = ramp(512);
        let segments = push_chunked(&mut segmenter, &audio);
        assert_eq!(segments.len(), 1, "16 s of speech must force-cut once");
        assert_eq!(
            segments[0].audio.len(),
            401 * FRAME,
            "the force-cut must land after the lowest-prob frame"
        );
    }

    #[test]
    fn force_cut_skips_head_without_speech() {
        // 400 silence frames then 100 speech frames: the force-cut triggers
        // at frame 469 with the lowest-prob frame at the window start (frame
        // 376), so the head is pure silence and is skipped, not emitted.
        let (mut segmenter, audio) = scripted(&[(0.1, 400), (0.9, 100)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(segments.is_empty(), "a speechless head must be skipped");
        let tail = segmenter.finish().expect("the remainder holds the speech");
        assert_eq!(
            tail.audio.len(),
            123 * FRAME,
            "only the skipped head may be gone"
        );
        assert_eq!(
            tail.audio,
            audio[audio.len() - tail.audio.len()..],
            "the tail must be the recording end"
        );
    }

    #[test]
    fn uncertain_frames_after_speech_keep_speech() {
        // 0.4 sits between the 0.5 start and 0.35 end thresholds: after
        // speech it reads as speech, so 15 uncertain frames never cut.
        let (mut segmenter, audio) = scripted(&[(0.9, 100), (0.4, 15)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(segments.is_empty(), "uncertain frames must not cut speech");
        let tail = segmenter.finish().expect("one uncut tail");
        assert_eq!(tail.audio, audio, "nothing may be dropped");
    }

    #[test]
    fn uncertain_frames_after_silence_keep_silence() {
        // 0.4 after silence reads as silence: 6 silence frames plus 10
        // uncertain ones reach the 13-frame pause cut.
        let (mut segmenter, audio) = scripted(&[(0.9, 100), (0.1, 6), (0.4, 10), (0.9, 50)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert_eq!(segments.len(), 1, "the pause must cut once");
    }

    #[test]
    fn pure_silence_yields_nothing() {
        let (mut segmenter, audio) = scripted(&[(0.1, 100)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(segments.is_empty(), "silence must emit no segments");
        assert!(segmenter.finish().is_none(), "a silent tail is dropped");
    }

    #[test]
    fn leading_silence_starts_at_sample_zero() {
        let (mut segmenter, audio) = scripted(&[(0.1, 30), (0.9, 100)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(segments.is_empty(), "no pause, so no mid-stream cut");
        let tail = segmenter.finish().expect("speech tail");
        assert_eq!(tail.audio, audio, "segment 1 starts at sample 0");
    }

    /// Segments are a contiguous partition of the whole recording: emitted
    /// segments plus the tail equal the input sample-exact (only all-silence
    /// segments are ever skipped, and none of these fixtures has one). The
    /// ramp audio pins every offset: uniform samples would hide a wrong cut.
    #[test]
    fn segments_partition_the_whole_recording() {
        let fixtures = [
            ("pause split", vec![(0.9, 100), (0.1, 15), (0.9, 100)]),
            ("short pause", vec![(0.9, 100), (0.1, 9), (0.9, 100)]),
            ("leading silence", vec![(0.1, 30), (0.9, 100)]),
            ("long speech", vec![(0.9, 1250)]),
        ];
        for (name, runs) in &fixtures {
            let (mut segmenter, audio) = scripted(runs);
            let segments = push_chunked(&mut segmenter, &audio);
            let tail = segmenter.finish();
            let out = concat(segments, tail);
            assert_eq!(out, audio, "{name}: concatenation must equal the input");
        }
        // Pure silence is the skipped case: nothing emitted, nothing lost.
        let (mut segmenter, audio) = scripted(&[(0.1, 100)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(segments.is_empty(), "silence must emit no segments");
        assert!(segmenter.finish().is_none(), "a silent tail is skipped");
    }

    #[test]
    fn feeder_segments_device_rate_audio_at_16k() {
        let mut audio = tone_at(48_000, 0.2, 2.0);
        audio.extend(silence_at(48_000, 0.6));
        audio.extend(tone_at(48_000, 0.2, 2.0));
        // ~4.6 s at 16 kHz; the trailing run is generous so the script covers
        // the resampler output whatever its exact length.
        let scorer = Scripted::new(seq(&[(0.9, 62), (0.1, 20), (0.9, 80)]));
        let mut feeder = Feeder::new(48_000, scorer).unwrap();
        let mut segments = Vec::new();
        for piece in audio.chunks(4800) {
            segments.extend(feeder.push(piece).unwrap());
        }
        segments.extend(feeder.finish().unwrap());
        assert_eq!(segments.len(), 2, "the pause must cut once");
        let pieces = segments;
        // 16 kHz output: ~4.6 s total (nothing is trimmed).
        let total: usize = pieces.iter().map(|segment| segment.audio.len()).sum();
        let expected = 16_000.0 * 4.6;
        assert!(
            (total as f32 - expected).abs() / expected < 0.02,
            "unexpected total length: {total}"
        );
        // Speech survives the resample + split (a third of the samples now).
        let input_speech = audio.iter().filter(|sample| sample.abs() > 0.05).count();
        let output_speech: usize = pieces
            .iter()
            .map(|segment| {
                segment
                    .audio
                    .iter()
                    .filter(|sample| sample.abs() > 0.05)
                    .count()
            })
            .sum();
        let expected_speech = input_speech as f32 / 3.0;
        assert!(
            (output_speech as f32 - expected_speech).abs() / expected_speech < 0.05,
            "speech lost: {output_speech} vs {expected_speech:.0} expected"
        );
    }

    #[test]
    fn feeder_attaches_left_context_after_the_first_segment() {
        let mut audio = tone(2.0);
        audio.extend(silence(0.6));
        audio.extend(tone(2.0));
        let scorer = Scripted::new(seq(&[(0.9, 62), (0.1, 20), (0.9, 62)]));
        let mut feeder = Feeder::new(RATE, scorer).unwrap();
        let mut segments = Vec::new();
        for piece in audio.chunks(RATE as usize / 10) {
            segments.extend(feeder.push(piece).unwrap());
        }
        segments.extend(feeder.finish().unwrap());
        assert_eq!(segments.len(), 2, "the pause must cut once");
        assert_eq!(segments[0].start, SegmentStart::Recording);
        assert_eq!(segments[1].start, SegmentStart::AfterPause);
        assert!(segments[0].context.is_empty(), "segment 1 has no context");
        // Segment 1 is ~2.2 s, so the context is capped at 1.5 s of its tail.
        assert_eq!(segments[1].context.len(), CONTEXT_SAMPLES);
        let tail = &segments[0].audio[segments[0].audio.len() - CONTEXT_SAMPLES..];
        assert_eq!(
            segments[1].context, tail,
            "segment 2 context must be the trailing 1.5 s before it"
        );
    }

    #[test]
    fn feeder_keeps_forced_cut_context_and_marks_the_start() {
        // 20 s without a natural pause: one forced cut. Standard/Fluid must
        // still get their left context; only the start provenance changes.
        let audio = tone(20.0);
        let mut feeder = Feeder::new(RATE, Scripted::new(seq(&[(0.9, 700)]))).unwrap();
        let mut segments = Vec::new();
        for piece in audio.chunks(RATE as usize / 10) {
            segments.extend(feeder.push(piece).unwrap());
        }
        segments.extend(feeder.finish().unwrap());
        assert_eq!(segments.len(), 2, "20 s of speech must force-cut once");
        assert_eq!(segments[0].start, SegmentStart::Recording);
        assert_eq!(segments[1].start, SegmentStart::AfterForce);
        assert!(!segments[1].start.in_known_silence());
        assert_eq!(
            segments[1].context.len(),
            CONTEXT_SAMPLES,
            "forced-cut context must not be dropped"
        );
    }

    #[test]
    fn first_segment_start_provenance_is_the_recording() {
        let (mut segmenter, audio) = scripted(&[(0.1, 30), (0.9, 100)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(segments.is_empty(), "no pause, so no mid-stream cut");
        let tail = segmenter.finish().expect("speech tail");
        assert_eq!(tail.start, SegmentStart::Recording);
    }

    #[test]
    fn segment_after_a_natural_pause_starts_in_known_silence() {
        let (mut segmenter, audio) = scripted(&[(0.9, 100), (0.1, 15), (0.9, 100)]);
        let mut segments = push_chunked(&mut segmenter, &audio);
        if let Some(tail) = segmenter.finish() {
            segments.push(tail);
        }
        assert_eq!(segments.len(), 2, "a 480 ms pause must split");
        assert_eq!(segments[0].start, SegmentStart::Recording);
        assert_eq!(segments[1].start, SegmentStart::AfterPause);
        assert!(segments[1].start.in_known_silence());
    }

    #[test]
    fn segment_after_a_forced_cut_is_not_known_silence() {
        let (mut segmenter, audio) = scripted(&[(0.9, 500)]);
        let segments = push_chunked(&mut segmenter, &audio);
        assert_eq!(segments.len(), 1, "the first forced cut emits the head");
        assert_eq!(segments[0].start, SegmentStart::Recording);
        let tail = segmenter.finish().expect("the remainder is the tail");
        assert_eq!(tail.start, SegmentStart::AfterForce);
        assert!(!tail.start.in_known_silence());
    }

    #[test]
    fn tail_after_each_cut_inherits_that_cuts_start_provenance() {
        // Tail after a natural pause.
        let (mut segmenter, audio) = scripted(&[(0.9, 100), (0.1, 15), (0.9, 30)]);
        assert_eq!(push_chunked(&mut segmenter, &audio).len(), 1);
        let tail = segmenter.finish().expect("tail after the pause");
        assert_eq!(tail.start, SegmentStart::AfterPause);

        // Tail after a forced cut.
        let (mut segmenter, audio) = scripted(&[(0.9, 500)]);
        assert_eq!(push_chunked(&mut segmenter, &audio).len(), 1);
        let tail = segmenter.finish().expect("tail after the forced cut");
        assert_eq!(tail.start, SegmentStart::AfterForce);
    }

    #[test]
    fn one_push_can_emit_multiple_cuts_with_their_own_start_provenance() {
        // One push spans a natural pause, a forced cut, then another natural
        // pause. Each emitted segment's start is the *previous* boundary.
        let (mut segmenter, audio) =
            scripted(&[(0.9, 100), (0.1, 13), (0.9, 500), (0.1, 13), (0.9, 50)]);
        let segments = segmenter.push(&audio).unwrap();
        let starts: Vec<SegmentStart> = segments.iter().map(|cut| cut.start).collect();
        assert_eq!(
            starts,
            vec![
                SegmentStart::Recording,
                SegmentStart::AfterPause,
                SegmentStart::AfterForce,
            ]
        );
        let tail = segmenter.finish().expect("the final speech is the tail");
        assert_eq!(tail.start, SegmentStart::AfterPause);
        assert_eq!(
            concat(segments, Some(tail)),
            audio,
            "cuts must still partition the recording sample-exact"
        );
    }

    #[test]
    fn feeder_rejects_rate_zero() {
        let Err(err) = Feeder::new(0, Scripted::new(Vec::new())) else {
            panic!("sample rate 0 must fail");
        };
        assert!(format!("{err:#}").contains('0'), "unexpected: {err:#}");
    }
}
