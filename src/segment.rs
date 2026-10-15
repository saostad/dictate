//! Pause-based audio segmenter for type-while-speaking.
//!
//! Splits device-rate mono audio at natural pauses so the daemon can
//! transcribe and type each piece while the user keeps talking. Energy VAD on
//! 30 ms frames (RMS) with an adaptive threshold; cuts at ≥ 450 ms of silence
//! after speech (segment ≥ 1.2 s) in the middle of the silence, force-cuts at
//! 8 s at the quietest frame in the last 2 s, drops pure-silence segments, and
//! keeps 150 ms of pre-roll before the first speech frame.

/// Length of one VAD frame.
const FRAME_MS: u32 = 30;
/// Consecutive silence that triggers a pause cut.
const SILENCE_CUT_MS: u32 = 450;
/// Minimum segment length for a pause cut.
const MIN_SEGMENT_MS: u32 = 1200;
/// Segment length that triggers a force-cut.
const MAX_SEGMENT_MS: u32 = 8000;
/// Pre-roll kept before the first speech frame.
const PREROLL_MS: u32 = 150;
/// Search window for the force-cut point.
const FORCE_SEARCH_MS: u32 = 2000;
/// Floor of the adaptive speech threshold.
const MIN_THRESHOLD: f32 = 0.008;
/// Speech threshold is this multiple of the noise floor.
const THRESHOLD_RATIO: f32 = 3.0;
/// Starting noise floor (frame RMS).
const INITIAL_NOISE_FLOOR: f32 = 0.01;

/// Splits mono audio at pauses. All audio is at the device rate; the caller
/// resamples finished segments to 16 kHz.
pub(crate) struct Segmenter {
    frame_len: usize,
    noise_floor: f32,
    pending: Vec<f32>,
    samples: Vec<f32>,
    rms: Vec<f32>,
    speech: Vec<bool>,
    has_speech: bool,
    trailing_silence: usize,
    preroll_frames: usize,
    silence_cut_frames: usize,
    min_segment_samples: usize,
    max_segment_samples: usize,
    force_search_frames: usize,
}

impl Segmenter {
    /// New segmenter for `sample_rate` Hz mono audio.
    pub(crate) fn new(sample_rate: u32) -> Self {
        let frame_len = ((sample_rate as usize * FRAME_MS as usize) / 1000).max(1);
        let preroll_frames =
            ((sample_rate as usize * PREROLL_MS as usize) / 1000 / frame_len).max(1);
        let silence_cut_frames =
            ((sample_rate as usize * SILENCE_CUT_MS as usize) / 1000 / frame_len).max(1);
        let min_segment_samples = sample_rate as usize * MIN_SEGMENT_MS as usize / 1000;
        let max_segment_samples = sample_rate as usize * MAX_SEGMENT_MS as usize / 1000;
        let force_search_frames =
            ((sample_rate as usize * FORCE_SEARCH_MS as usize) / 1000 / frame_len).max(1);
        Self {
            frame_len,
            noise_floor: INITIAL_NOISE_FLOOR,
            pending: Vec::new(),
            samples: Vec::new(),
            rms: Vec::new(),
            speech: Vec::new(),
            has_speech: false,
            trailing_silence: 0,
            preroll_frames,
            silence_cut_frames,
            min_segment_samples,
            max_segment_samples,
            force_search_frames,
        }
    }

    /// Push device-rate samples; returns finished segments (device rate).
    pub(crate) fn push(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        let mut offset = 0;
        // Finish the pending partial frame first.
        if !self.pending.is_empty() {
            let need = self.frame_len - self.pending.len();
            let take = need.min(samples.len() - offset);
            self.pending.extend_from_slice(&samples[offset..offset + take]);
            offset += take;
            if self.pending.len() == self.frame_len {
                let frame = std::mem::take(&mut self.pending);
                if let Some(segment) = self.push_frame(&frame) {
                    out.push(segment);
                }
            }
        }
        while offset + self.frame_len <= samples.len() {
            let frame = &samples[offset..offset + self.frame_len];
            offset += self.frame_len;
            if let Some(segment) = self.push_frame(frame) {
                out.push(segment);
            }
        }
        if offset < samples.len() {
            self.pending.extend_from_slice(&samples[offset..]);
        }
        out
    }

    /// Finish the recording; returns the tail when it holds speech.
    /// Pure-silence tails are dropped.
    pub(crate) fn finish(&mut self) -> Option<Vec<f32>> {
        if !self.has_speech {
            return None;
        }
        let first = self.speech.iter().position(|&speech| speech);
        let mut tail = std::mem::take(&mut self.samples);
        tail.extend_from_slice(&self.pending);
        self.pending.clear();
        self.rms.clear();
        self.speech.clear();
        self.has_speech = false;
        self.trailing_silence = 0;
        trim_leading_by(tail, first, self.frame_len, self.preroll_frames)
    }

    /// Classify one complete frame and maybe emit a segment.
    fn push_frame(&mut self, frame: &[f32]) -> Option<Vec<f32>> {
        let value = rms(frame);
        let threshold = MIN_THRESHOLD.max(THRESHOLD_RATIO * self.noise_floor);
        let is_speech = value > threshold;
        // Slow-moving minimum: quick down toward new lows, slow drift up
        // on silence only (speech must not raise the floor).
        if value < self.noise_floor {
            self.noise_floor += (value - self.noise_floor) * 0.05;
        } else if !is_speech {
            self.noise_floor += (value - self.noise_floor) * 0.002;
        }
        self.noise_floor = self.noise_floor.clamp(1e-4, 0.5);
        self.samples.extend_from_slice(frame);
        self.rms.push(value);
        self.speech.push(is_speech);
        if is_speech {
            self.has_speech = true;
            self.trailing_silence = 0;
        } else {
            self.trailing_silence += 1;
        }
        if !self.has_speech {
            // Bound leading silence to the pre-roll window.
            let frames = self.speech.len();
            if frames > self.preroll_frames {
                let drop_frames = frames - self.preroll_frames;
                self.samples.drain(..drop_frames * self.frame_len);
                self.rms.drain(..drop_frames);
                self.speech.drain(..drop_frames);
                self.trailing_silence = self.speech.len();
            }
            return None;
        }
        if self.trailing_silence >= self.silence_cut_frames
            && self.samples.len() >= self.min_segment_samples
        {
            return Some(self.cut_pause());
        }
        if self.samples.len() >= self.max_segment_samples {
            return self.cut_force();
        }
        None
    }

    /// Cut in the middle of the trailing silence; the remainder carries over.
    /// The caller guarantees the segment holds speech.
    fn cut_pause(&mut self) -> Vec<f32> {
        let silence = self.trailing_silence;
        let keep_in_first = silence / 2;
        let remainder_frames = silence - keep_in_first;
        let cut = self.samples.len() - remainder_frames * self.frame_len;
        let first_frames = self.speech.len() - remainder_frames;
        let first_speech = self.speech[..first_frames]
            .iter()
            .position(|&speech| speech);
        let head = self.samples[..cut].to_vec();
        // The head holds the segment's speech by construction (the trailing
        // run is silence), so trimming always yields `Some`; fall back to the
        // untrimmed head rather than panic on the unreachable `None`.
        let head = match trim_leading_by(
            head,
            first_speech,
            self.frame_len,
            self.preroll_frames,
        ) {
            Some(head) => head,
            None => self.samples[..cut].to_vec(),
        };
        let remainder = self.samples[cut..].to_vec();
        let remainder_rms = self.rms[first_frames..].to_vec();
        let remainder_speech = self.speech[first_frames..].to_vec();
        self.samples = remainder;
        self.rms = remainder_rms;
        self.speech = remainder_speech;
        self.has_speech = self.speech.iter().any(|&speech| speech);
        self.trailing_silence = trailing_count(&self.speech);
        head
    }

    /// Force-cut a long segment at the end of the quietest frame in the last
    /// 2 s. A first half with no speech frames is dropped (`None`).
    fn cut_force(&mut self) -> Option<Vec<f32>> {
        let frames = self.speech.len();
        let window = self.force_search_frames.min(frames);
        let window_start = frames - window;
        let mut best = window_start;
        for index in window_start + 1..frames {
            if self.rms[index] < self.rms[best] {
                best = index;
            }
        }
        let cut_frames = (best + 1).min(frames);
        let cut = cut_frames * self.frame_len;
        let cut = cut.min(self.samples.len());
        let first_has_speech = self.speech[..cut_frames].iter().any(|&speech| speech);
        let first_speech = self.speech[..cut_frames]
            .iter()
            .position(|&speech| speech);
        let head = self.samples[..cut].to_vec();
        let remainder = self.samples[cut..].to_vec();
        let remainder_rms = self.rms[cut_frames..].to_vec();
        let remainder_speech = self.speech[cut_frames..].to_vec();
        self.samples = remainder;
        self.rms = remainder_rms;
        self.speech = remainder_speech;
        self.has_speech = self.speech.iter().any(|&speech| speech);
        self.trailing_silence = trailing_count(&self.speech);
        if !first_has_speech {
            return None;
        }
        trim_leading_by(head, first_speech, self.frame_len, self.preroll_frames)
    }
}

/// Root mean square of one frame.
fn rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f32 = frame.iter().map(|sample| sample * sample).sum();
    (sum / frame.len() as f32).sqrt()
}

/// Count trailing non-speech frames.
fn trailing_count(speech: &[bool]) -> usize {
    speech.iter().rev().take_while(|&&speech| !speech).count()
}

/// Trim leading silence beyond the pre-roll window, keeping whole frames.
/// `first_speech` is the index of the first speech frame in `samples`.
/// A segment with no speech frames yields `None` (dropped, never sent).
fn trim_leading_by(
    mut samples: Vec<f32>,
    first_speech: Option<usize>,
    frame_len: usize,
    preroll_frames: usize,
) -> Option<Vec<f32>> {
    let first = first_speech?;
    if first > preroll_frames {
        samples.drain(..(first - preroll_frames) * frame_len);
    }
    Some(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    /// Sine burst (speech-like) at 440 Hz, amplitude 0.2.
    fn tone(secs: f32) -> Vec<f32> {
        let len = (secs * RATE as f32) as usize;
        (0..len)
            .map(|i| {
                0.2 * (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / RATE as f32).sin()
            })
            .collect()
    }

    /// Near-silence: faint 440 Hz at amplitude 0.001 (RMS ~0.0007).
    fn silence(secs: f32) -> Vec<f32> {
        let len = (secs * RATE as f32) as usize;
        (0..len)
            .map(|i| {
                0.001 * (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / RATE as f32).sin()
            })
            .collect()
    }

    /// Push `audio` in 100 ms chunks, like the feeder thread.
    fn push_chunked(segmenter: &mut Segmenter, audio: &[f32]) -> Vec<Vec<f32>> {
        let chunk = RATE as usize / 10;
        let mut out = Vec::new();
        for piece in audio.chunks(chunk) {
            out.extend(segmenter.push(piece));
        }
        out
    }

    fn concat(mut segments: Vec<Vec<f32>>, tail: Option<Vec<f32>>) -> Vec<f32> {
        if let Some(tail) = tail {
            segments.push(tail);
        }
        segments.concat()
    }

    #[test]
    fn pause_splits_into_two_segments() {
        let mut audio = tone(2.0);
        audio.extend(silence(0.6));
        audio.extend(tone(2.0));
        let mut segmenter = Segmenter::new(RATE);
        let mut segments = push_chunked(&mut segmenter, &audio);
        let tail = segmenter.finish();
        if tail.is_some() {
            segments.push(tail.unwrap());
        }
        assert_eq!(segments.len(), 2, "a 600 ms pause must split");
        for (index, segment) in segments.iter().enumerate() {
            assert!(
                !segment.is_empty(),
                "segment {index} must hold audio"
            );
        }
    }

    #[test]
    fn short_pause_does_not_split() {
        let mut audio = tone(2.0);
        audio.extend(silence(0.3));
        audio.extend(tone(2.0));
        let mut segmenter = Segmenter::new(RATE);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(
            segments.is_empty(),
            "a 300 ms pause must not split ({} segments)",
            segments.len()
        );
        let tail = segmenter.finish();
        assert!(tail.is_some(), "the whole recording is one tail");
    }

    #[test]
    fn long_tone_force_cuts_at_or_before_eight_seconds() {
        let audio = tone(12.0);
        let mut segmenter = Segmenter::new(RATE);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(
            !segments.is_empty(),
            "12 s of continuous tone must force-cut"
        );
        let limit = RATE as usize * 8 + RATE as usize * FRAME_MS as usize / 1000;
        assert!(
            segments[0].len() <= limit,
            "first segment is {} samples, over 8 s + one frame",
            segments[0].len()
        );
        let tail = segmenter.finish();
        assert!(tail.is_some(), "the remainder is the tail");
    }

    #[test]
    fn pure_silence_yields_nothing() {
        let audio = silence(3.0);
        let mut segmenter = Segmenter::new(RATE);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(segments.is_empty(), "silence must emit no segments");
        assert_eq!(segmenter.finish(), None, "a silent tail is dropped");
    }

    #[test]
    fn outputs_plus_tail_lose_no_speech_samples() {
        let mut audio = tone(2.0);
        audio.extend(silence(0.6));
        audio.extend(tone(2.0));
        let mut segmenter = Segmenter::new(RATE);
        let segments = push_chunked(&mut segmenter, &audio);
        let tail = segmenter.finish();
        let out = concat(segments, tail);
        // Speech is amplitude 0.2, silence 0.001: 0.05 separates them.
        let input_speech = audio.iter().filter(|sample| sample.abs() > 0.05).count();
        let output_speech = out.iter().filter(|sample| sample.abs() > 0.05).count();
        assert!(input_speech > 0, "the fixture must contain speech");
        assert_eq!(
            output_speech, input_speech,
            "every speech sample must survive the split"
        );
    }

    #[test]
    fn leading_silence_is_trimmed_to_preroll() {
        let mut audio = silence(1.0);
        audio.extend(tone(2.0));
        let mut segmenter = Segmenter::new(RATE);
        let segments = push_chunked(&mut segmenter, &audio);
        assert!(segments.is_empty(), "no pause, so no mid-stream cut");
        let tail = segmenter.finish().expect("speech tail");
        // Leading silence kept: 150 ms ± one frame.
        let frame = RATE as usize * FRAME_MS as usize / 1000;
        let preroll = RATE as usize * PREROLL_MS as usize / 1000;
        let leading = tail
            .iter()
            .take_while(|sample| sample.abs() <= 0.05)
            .count();
        assert!(
            leading <= preroll + frame,
            "leading silence is {leading} samples, over 150 ms + one frame"
        );
        assert!(
            leading >= preroll.saturating_sub(frame),
            "pre-roll keeps about 150 ms, got {leading} samples"
        );
    }
}
