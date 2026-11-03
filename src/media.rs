//! Packet-at-a-time media decoding. Stdin is spooled to disk for seekable containers.
use anyhow::Context;
use std::{fs::File, io::Seek, path::Path};
use symphonia::core::{
    audio::SampleBuffer,
    errors::Error,
    formats::FormatOptions,
    io::{MediaSource, MediaSourceStream},
    meta::MetadataOptions,
    probe::Hint,
};

pub(crate) struct Audio {
    format: Box<dyn symphonia::core::formats::FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track: u32,
    decoded_frames: u64,
    expected_frames: Option<u64>,
    pub(crate) rate: u32,
    pub(crate) duration: Option<f64>,
}
impl Audio {
    pub(crate) fn open(path: &str) -> anyhow::Result<Self> {
        Self::open_inner(path).with_context(|| format!("Could not read audio {path}; supported codecs: PCM, ADPCM, MP3, AAC-LC, FLAC, Vorbis (Opus is not supported)"))
    }
    fn open_inner(path: &str) -> anyhow::Result<Self> {
        let source: Box<dyn MediaSource> = if path == "-" {
            let mut spool = tempfile::tempfile().context("Could not create stdin spool")?;
            std::io::copy(&mut std::io::stdin().lock(), &mut spool)
                .context("Could not read stdin")?;
            spool.rewind()?;
            Box::new(spool)
        } else {
            Box::new(File::open(path)?)
        };
        let mut hint = Hint::new();
        if let Some(ext) = Path::new(path).extension().and_then(|s| s.to_str()) {
            hint.with_extension(ext);
        }
        let format = symphonia::default::get_probe()
            .format(
                &hint,
                MediaSourceStream::new(source, Default::default()),
                &FormatOptions {
                    enable_gapless: true,
                    ..Default::default()
                },
                &MetadataOptions::default(),
            )?
            .format;
        let track = format
            .default_track()
            .filter(|t| t.codec_params.sample_rate.is_some())
            .or_else(|| {
                format
                    .tracks()
                    .iter()
                    .find(|t| t.codec_params.sample_rate.is_some())
            })
            .context("No audio track found (Opus is not supported)")?;
        let rate = track
            .codec_params
            .sample_rate
            .context("Audio track has no sample rate")?;
        anyhow::ensure!(rate > 0, "Audio track has invalid sample rate");
        let duration = track.codec_params.n_frames.map(|n| n as f64 / rate as f64);
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions { verify: true })
            .context("Unsupported audio codec; Opus is not supported")?;
        use symphonia::core::codecs::*;
        let expected_frames = match track.codec_params.codec {
            CODEC_TYPE_FLAC | CODEC_TYPE_PCM_S16LE | CODEC_TYPE_PCM_S24LE
            | CODEC_TYPE_PCM_S32LE | CODEC_TYPE_PCM_U8 | CODEC_TYPE_PCM_F32LE
            | CODEC_TYPE_PCM_F64LE => track.codec_params.n_frames,
            _ => None,
        };
        let id = track.id;
        Ok(Self {
            format,
            decoder,
            track: id,
            decoded_frames: 0,
            expected_frames,
            rate,
            duration,
        })
    }
    /// Shared CLI/HTTP resampling step, including the final buffered tail.
    pub(crate) fn resample_packet(
        resampler: &mut Option<crate::audio::StreamResampler>,
        decoded: Option<Vec<f32>>,
    ) -> anyhow::Result<Vec<f32>> {
        match (resampler, decoded) {
            (Some(r), Some(samples)) => r.push(&samples),
            (Some(r), None) => r.finish(),
            (None, Some(samples)) => Ok(samples),
            (None, None) => Ok(Vec::new()),
        }
    }
    pub(crate) fn next(&mut self) -> anyhow::Result<Option<Vec<f32>>> {
        loop {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    if let Some(expected) = self.expected_frames {
                        anyhow::ensure!(
                            self.decoded_frames == expected,
                            "Truncated or corrupt audio: expected {expected} frames, decoded {}",
                            self.decoded_frames
                        );
                    }
                    anyhow::ensure!(
                        self.decoder.finalize().verify_ok != Some(false),
                        "Audio integrity verification failed"
                    );
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            };
            if packet.track_id() != self.track {
                continue;
            }
            let decoded = self.decoder.decode(&packet)?;
            anyhow::ensure!(
                decoded.spec().rate == self.rate,
                "Audio sample rate changed midstream"
            );
            self.decoded_frames += decoded.frames() as u64;
            let channels = decoded.spec().channels.count();
            anyhow::ensure!(channels > 0, "Audio track has no channels");
            let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
            buffer.copy_interleaved_ref(decoded);
            return Ok(Some(crate::mic::downmix(buffer.samples(), channels)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compressed_flac_decodes_without_extension() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), include_bytes!("../tests/fixtures/tone.flac")).unwrap();
        let mut audio = Audio::open(file.path().to_str().unwrap()).unwrap();
        assert_eq!(audio.rate, 16000);
        let mut samples = Vec::new();
        while let Some(packet) = audio.next().unwrap() {
            samples.extend(packet);
        }
        assert_eq!(samples.len(), 1600);
        assert!(samples.iter().any(|s| s.abs() > 0.2));
    }
    #[test]
    fn truncated_flac_fails_instead_of_empty_success() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let data = include_bytes!("../tests/fixtures/tone.flac");
        std::fs::write(file.path(), &data[..data.len() / 2]).unwrap();
        let result = (|| -> anyhow::Result<()> {
            let mut audio = Audio::open(file.path().to_str().unwrap())?;
            while audio.next()?.is_some() {}
            Ok(())
        })();
        assert!(result.is_err());
    }
    #[test]
    fn wav_matches_hound_and_corrupt_names_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut writer = hound::WavWriter::create(
            file.path(),
            hound::WavSpec {
                channels: 2,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..1600 {
            writer.write_sample(1000i16).unwrap();
            writer.write_sample(3000i16).unwrap();
        }
        writer.finalize().unwrap();
        let expected = crate::audio::load_wav_16k_mono(file.path()).unwrap();
        let mut audio = Audio::open(file.path().to_str().unwrap()).unwrap();
        let mut samples = Vec::new();
        while let Some(packet) = audio.next().unwrap() {
            samples.extend(packet);
        }
        assert_eq!(samples, expected);
        std::fs::write(file.path(), b"broken media").unwrap();
        let error = Audio::open(file.path().to_str().unwrap()).err().unwrap();
        assert!(format!("{error:#}").contains(file.path().to_str().unwrap()));
        assert!(format!("{error:#}").contains("Opus"));
    }
}
