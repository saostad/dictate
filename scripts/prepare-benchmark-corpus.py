#!/usr/bin/env python3
"""Prepare the portable 12-clip LibriSpeech benchmark corpus.

The script fetches one fixed row per selected speaker/offset from the Hugging
Face datasets-server `rows` endpoint, downloads the fresh signed FLAC URL,
verifies the pinned FLAC SHA-256, decodes to canonical 16 kHz mono PCM16 WAV,
verifies the pinned WAV SHA-256, and writes a manifest whose clip paths are
relative to the manifest.

This is a one-time setup step and the only part of the benchmark tooling that
uses the network.  `--offline` reuses already-cached FLAC files.  No model is
downloaded here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "openslr/librispeech_asr"
REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "test"
LICENSE = "CC-BY-4.0"
SOURCE_URL = "https://www.openslr.org/12/"
CORPUS_NAME = "LibriSpeech test-clean: fixed six-speaker subset"
SELECTION = (
    "Two 3-12 second clips per speaker, three male and three female speakers, "
    "first qualifying rows at fixed offsets 0,250,500,750,1000,1500. "
    "Small read-English sample, not conversational/accents benchmark."
)
SELECTION_NOTE = (
    "Fixed rows chosen from offsets 0,250,500,750,1000,1500: the first two "
    "3-12 second clips per speaker. This is a small read-English sample, not a "
    "conversational or accent benchmark."
)
USER_AGENT = "dictate-benchmark-corpus/1.0 (+https://github.com/saostad/dictate)"


@dataclasses.dataclass(frozen=True)
class ClipPins:
    row: int
    clip_id: str
    reference: str
    duration_seconds: float
    speaker_id: int
    speaker_gender: str
    flac_sha256: str
    wav_sha256: str


# This table is the reproducibility contract.  Row offsets, references,
# speakers, durations and both hashes are pinned.  The rows endpoint supplies
# fresh signed asset URLs; it is never allowed to change the content silently.
PINS: tuple[ClipPins, ...] = (
    ClipPins(
        0,
        "6930-75918-0000",
        "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS",
        3.505,
        6930,
        "M",
        "9ce35224156f071ab58eb7feb8a5ceae600f6f9f353da2a6cbf797b6b1ac8a23",
        "103c3f15eb3715ebc6243d142244128a3ac39b6bfae315baa6b8dc4a8be14aa8",
    ),
    ClipPins(
        2,
        "6930-75918-0002",
        "CONGRATULATIONS WERE POURED IN UPON THE PRINCESS EVERYWHERE DURING HER JOURNEY",
        5.025,
        6930,
        "M",
        "74ecfd628f49dca71995972c075fcc9904df804d49a4eb5b11fb712b3584f789",
        "ca418189ed1db4ce1bf5e6940d51a9ce870d4d52f3ef9019ccdbb2189d1a0a9c",
    ),
    ClipPins(
        250,
        "260-123288-0018",
        "THE RAFT BEARS ON STILL TO THE SOUTH EAST",
        3.245,
        260,
        "M",
        "0ed15058018063a7fc8c7707fc24686919c0dfbfe8cc21276e5d121a4ff9f7d9",
        "c1a68a9ae9a28289218aca63ae14d10aacdb6dfd091b5463d1faa8135f609062",
    ),
    ClipPins(
        254,
        "260-123288-0022",
        "THEY SEEM TO BE WE ARE LOST BUT I AM NOT SURE",
        3.705,
        260,
        "M",
        "3b04340678b02c90df05b8548b051e9380580d100d6bfe5141c25b2cd69f3a0c",
        "616633e7e8b731e2893be43067ecbea98ab2157fde037ab5c7e1bfe5600c4077",
    ),
    ClipPins(
        501,
        "2961-960-0004",
        "THERE IS NO DANGER OF THE MODERN COMMENTATORS ON THE TIMAEUS FALLING INTO THE ABSURDITIES OF THE NEO PLATONISTS",
        8.22,
        2961,
        "F",
        "c815b1d8d37cd643047430c39f790067c75ee3adb93fdfbbe3da493f71d44615",
        "5161cf90f0fc842d1b5f473ac5b172641aa2678515998a4f315588f874707520",
    ),
    ClipPins(
        502,
        "2961-960-0005",
        "IN THE PRESENT DAY WE ARE WELL AWARE THAT AN ANCIENT PHILOSOPHER IS TO BE INTERPRETED FROM HIMSELF AND BY THE CONTEMPORARY HISTORY OF THOUGHT",
        10.355,
        2961,
        "F",
        "0d0c423dc55c236b5f55ca2fc4ab87b39b85d5d2a34585573afe6038ebf3a091",
        "8cab8ee80366cb73901bb26b5af4103cca895c2644317cc1dcb817cd2df282bf",
    ),
    ClipPins(
        750,
        "1995-1826-0000",
        "IN THE DEBATE BETWEEN THE SENIOR SOCIETIES HER DEFENCE OF THE FIFTEENTH AMENDMENT HAD BEEN NOT ONLY A NOTABLE BIT OF REASONING BUT DELIVERED WITH REAL ENTHUSIASM",
        9.485,
        1995,
        "F",
        "74596627a026feb72e3acba17ee9268f120ee25e9f4afb935bb2f3d3ba779e57",
        "03ac3f6a371c4ef5fd518588c0df859e6de2b82ca1cbe464533d53be303f5e5a",
    ),
    ClipPins(
        751,
        "1995-1826-0001",
        "THE SOUTH SHE HAD NOT THOUGHT OF SERIOUSLY AND YET KNOWING OF ITS DELIGHTFUL HOSPITALITY AND MILD CLIMATE SHE WAS NOT AVERSE TO CHARLESTON OR NEW ORLEANS",
        10.17,
        1995,
        "F",
        "56d8ebeb4138c7cc33c7f468aec078a3d76694b8759b1e860d876288f475e8ee",
        "660488d1124a2d3ecdc76133799b6b1006c65e83af5952a9acb91bfb82b991dc",
    ),
    ClipPins(
        1000,
        "8455-210777-0028",
        "JACK WOULD BECOME EVA'S HAPPY HUSBAND AND WOULD REMAIN AMIDST THE HURRIED DUTIES OF THE EAGER WORLD",
        7.735,
        8455,
        "M",
        "61e1697bdc0de5142e0cd76c4a14ed38086b66b9a0ab7722bd1c02aa3cebd261",
        "9f4ec23f805b2f9f67671a780864c8ea4d40d678da2edd65f8f24e799c085641",
    ),
    ClipPins(
        1001,
        "8455-210777-0029",
        "THINKING OF ALL THIS I WENT TO SLEEP",
        3.085,
        8455,
        "M",
        "289ad7e37196db91419a006a00b121b4ccb7eba5980b02be4941e174c3d201c4",
        "a8c9b3fcfa4d8d491bdc73c337611a82f9a0e43e558c1243e3dfb9f639714969",
    ),
    ClipPins(
        1502,
        "4507-16021-0033",
        "DO WE REALLY KNOW THE MOUNTAIN WELL WHEN WE ARE NOT ACQUAINTED WITH THE CAVERN",
        5.545,
        4507,
        "F",
        "1a95f7d0613cb1b284ce711f82f46383f9f2e3969d63d356906b7e6ea5a15a62",
        "95242d5933a7e9d4272b21008310cc888769f73e05a120bf771582a36540449a",
    ),
    ClipPins(
        1503,
        "4507-16021-0034",
        "THEY CONSTITUTE TWO DIFFERENT ORDERS OF FACTS WHICH CORRESPOND TO EACH OTHER WHICH ARE ALWAYS INTERLACED AND WHICH OFTEN BRING FORTH RESULTS",
        11.27,
        4507,
        "F",
        "82d1c6989ed82652706793e5449d59106eb0cd0fcc9fc2d0b53fae601834d832",
        "2bc8d7291a3f3a1dc60091d54718473d6fb46f3fd8e12f3aa4c70bdff681c38e",
    ),
)

PINNED_BY_ID = {clip.clip_id: clip for clip in PINS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def normalize_ws(text: str) -> str:
    return " ".join(str(text).split())


def http_bytes(url: str, timeout: float, attempts: int = 4) -> bytes:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, audio/flac, application/octet-stream, */*",
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in {408, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"HTTP {error.code} for {url}: {error.reason}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"network error for {url}: {error}") from error
    raise RuntimeError(f"network error for {url}: {last_error}")


def fetch_row(row: int, timeout: float) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "offset": row,
            "length": 1,
        }
    )
    url = f"{HF_ROWS_URL}?{query}"
    payload = json.loads(http_bytes(url, timeout).decode("utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError(f"rows endpoint returned no single row for offset {row}")
    entry = rows[0]
    value = entry.get("row") if isinstance(entry, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError(f"rows endpoint returned no row object for offset {row}")
    if entry.get("row_idx") not in (None, row) and entry.get("row_index") not in (None, row):
        raise RuntimeError(
            f"rows endpoint returned row {entry.get('row_idx', entry.get('row_index'))} "
            f"for requested offset {row}"
        )
    return value


def audio_source(row: dict[str, Any]) -> str:
    audio = row.get("audio")
    source = None
    if isinstance(audio, list) and audio:
        first = audio[0]
        if isinstance(first, dict):
            source = first.get("src")
    elif isinstance(audio, dict):
        source = audio.get("src")
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"row {row.get('id', '?')} has no audio source URL")
    return source


def validate_row(pins: ClipPins, row: dict[str, Any]) -> None:
    if str(row.get("id")) != pins.clip_id:
        raise RuntimeError(
            f"row {pins.row} returned id {row.get('id')!r}, expected {pins.clip_id!r}"
        )
    text = row.get("text")
    if not isinstance(text, str) or normalize_ws(text).casefold() != pins.reference.casefold():
        raise RuntimeError(
            f"row {pins.row} text changed for {pins.clip_id}: {text!r}"
        )
    speaker = row.get("speaker_id")
    if speaker is not None and int(speaker) != pins.speaker_id:
        raise RuntimeError(
            f"row {pins.row} speaker is {speaker}, expected {pins.speaker_id}"
        )
    audio_source(row)


def canonicalize_wav(source: Path, destination: Path) -> None:
    with wave.open(str(source), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
        if channels != 1 or width != 2 or rate != 16_000:
            raise RuntimeError(
                f"{source} decoded to {rate} Hz, {channels} channel(s), {width * 8}-bit; "
                "expected 16 kHz mono PCM16"
            )
        if not frames:
            raise RuntimeError(f"{source} decoded to empty audio")
    tmp = destination.with_name(destination.name + ".canonical.tmp")
    try:
        with wave.open(str(tmp), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(frames)
        os.replace(tmp, destination)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def wav_info(path: Path) -> tuple[float, str]:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2 or reader.getframerate() != 16_000:
            raise RuntimeError(f"{path} is not 16 kHz mono PCM16")
        frames = reader.getnframes()
        rate = reader.getframerate()
    if frames <= 0:
        raise RuntimeError(f"{path} is empty")
    return frames / rate, sha256_file(path)


def decode_audio(source: Path, destination: Path, decoder: str, timeout: float) -> str:
    flac = shutil.which("flac")
    ffmpeg = shutil.which("ffmpeg")
    if decoder == "flac" and not flac:
        raise RuntimeError("--decoder flac requested but the `flac` executable is not installed")
    if decoder == "ffmpeg" and not ffmpeg:
        raise RuntimeError("--decoder ffmpeg requested but the `ffmpeg` executable is not installed")
    if decoder == "auto":
        decoder = "flac" if flac else ("ffmpeg" if ffmpeg else "none")
    if decoder == "none":
        raise RuntimeError("neither `flac` nor `ffmpeg` is installed; install one to decode the corpus")
    tmp = destination.with_name(destination.name + ".decode.tmp")
    try:
        if decoder == "flac":
            completed = subprocess.run(
                [flac, "-d", "--force", "--silent", "-o", str(tmp), str(source)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"flac failed for {source.name}: {completed.stderr.strip() or completed.returncode}"
                )
            # Verify the exact flac output first.  If it differs, canonicalize
            # and try once more; the pinned hash is the final arbiter.
            os.replace(tmp, destination)
            if sha256_file(destination) == PINNED_BY_ID[source.stem].wav_sha256:
                return decoder
            canonicalize_wav(destination, destination)
            return decoder
        completed = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(tmp),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {source.name}: {completed.stderr.strip() or completed.returncode}"
            )
        canonicalize_wav(tmp, destination)
        return decoder
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def download_flac(source_url: str, destination: Path, expected_sha256: str, timeout: float) -> None:
    temporary = destination.with_name(destination.name + ".download.tmp")
    try:
        data = http_bytes(source_url, timeout)
        actual = hashlib.sha256(data).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"downloaded FLAC SHA-256 is {actual}, expected {expected_sha256}; "
                "the dataset revision or asset changed"
            )
        temporary.write_bytes(data)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def manifest_for(
    output: Path,
    audio_subdir: str,
    clips: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": CORPUS_NAME,
        "source": {
            "url": SOURCE_URL,
            "dataset": DATASET,
            "revision": REVISION,
            "license": LICENSE,
            "selection": SELECTION,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "rows": [pins.row for pins in PINS],
            "selection_note": SELECTION_NOTE,
        },
        "clips": clips,
    }


def attribution_text() -> str:
    rows = ", ".join(str(pins.row) for pins in PINS)
    return f"""LibriSpeech test-clean excerpts used by Dictate's model benchmark
====================================================================

Corpus: {CORPUS_NAME}
Source: OpenSLR LibriSpeech, {SOURCE_URL}
Hugging Face dataset: {DATASET} (config {DATASET_CONFIG}, split {DATASET_SPLIT})
Pinned dataset revision: {REVISION}
Selected rows: {rows}
License: {LICENSE} (Creative Commons Attribution 4.0 International)
https://creativecommons.org/licenses/by/4.0/

LibriSpeech is based on public-domain LibriVox audio books and was introduced
in "LibriSpeech: An ASR corpus based on public domain audio books" by Vassil
Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur (ICASSP 2015).

The selected audio and reference transcripts remain subject to CC BY 4.0.
The manifest pins each source FLAC SHA-256 and the canonical 16 kHz mono PCM16
WAV SHA-256. Selection is two 3-12 second clips from each of six speakers
(three male, three female), at fixed dataset rows; it is a small read-English
sample and is not a conversational, accented, or noisy-speech benchmark.
"""


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify the fixed 12-clip LibriSpeech benchmark corpus, "
            "decode it to canonical 16 kHz mono PCM16 WAV, and write the manifest."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "tests/fixtures",
        help="directory for the manifest and audio/ (default: tests/fixtures)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=(Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
                 / "dictate-benchmarks" / "corpus"),
        help="FLAC cache directory (default: $XDG_CACHE_HOME/dictate-benchmarks/corpus)",
    )
    parser.add_argument("--offline", action="store_true", help="use cached FLAC files; do not contact the network")
    parser.add_argument("--force", action="store_true", help="re-download and re-decode every clip")
    parser.add_argument("--decoder", choices=["auto", "flac", "ffmpeg"], default="auto", help="FLAC decoder preference")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP/decode timeout in seconds (default 60)")
    parser.add_argument("--audio-subdir", default="audio", help="audio subdirectory relative to --output (default audio)")
    parser.add_argument("--manifest-name", default="benchmark-corpus.json", help="manifest file name")
    parser.add_argument("--dry-run", action="store_true", help="print the fixed plan without downloading")
    args = parser.parse_args(list(argv))
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    subdir = Path(args.audio_subdir)
    if subdir.is_absolute() or ".." in subdir.parts or not subdir.parts:
        parser.error("--audio-subdir must be a simple relative path")
    args.audio_subdir = str(subdir)
    args.output = args.output.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.dry_run:
        plan = {
            "output": str(args.output),
            "cache": str(args.cache),
            "audio_subdir": args.audio_subdir,
            "offline": args.offline,
            "decoder": args.decoder,
            "rows": [pins.row for pins in PINS],
            "clips": [pins.clip_id for pins in PINS],
            "total_seconds": sum(pins.duration_seconds for pins in PINS),
        }
        json.dump(plan, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.output.exists() and not args.output.is_dir():
        raise SystemExit(f"output path exists and is not a directory: {args.output}")
    audio_dir = args.output / args.audio_subdir
    audio_dir.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    clips: list[dict[str, Any]] = []
    decoders: set[str] = set()
    total_seconds = 0.0
    for position, pins in enumerate(PINS, start=1):
        flac_path = args.cache / f"{pins.clip_id}.flac"
        wav_path = audio_dir / f"{pins.clip_id}.wav"
        source_url = None

        if not args.offline:
            row = fetch_row(pins.row, args.timeout)
            validate_row(pins, row)
            source_url = audio_source(row)

        if args.force or not flac_path.is_file():
            if args.offline:
                raise SystemExit(
                    f"offline and missing cached FLAC: {flac_path}\n"
                    "Run once without --offline to populate the cache."
                )
            assert source_url is not None
            download_flac(source_url, flac_path, pins.flac_sha256, args.timeout)
        else:
            actual_flac = sha256_file(flac_path)
            if actual_flac.lower() != pins.flac_sha256.lower():
                if args.offline:
                    raise SystemExit(
                        f"cached FLAC SHA-256 mismatch for {flac_path}: {actual_flac}"
                    )
                assert source_url is not None
                print(
                    f"  cached FLAC hash changed for {pins.clip_id}; re-downloading",
                    file=sys.stderr,
                )
                download_flac(source_url, flac_path, pins.flac_sha256, args.timeout)
        if sha256_file(flac_path).lower() != pins.flac_sha256.lower():
            raise SystemExit(f"FLAC SHA-256 verification failed for {flac_path}")

        need_decode = args.force or not wav_path.is_file()
        if not need_decode:
            try:
                _, existing_sha = wav_info(wav_path)
            except (OSError, RuntimeError):
                need_decode = True
            else:
                if existing_sha.lower() != pins.wav_sha256.lower():
                    need_decode = True
        if need_decode:
            decoder = decode_audio(flac_path, wav_path, args.decoder, args.timeout)
            decoders.add(decoder)
        duration, actual_wav = wav_info(wav_path)
        if actual_wav.lower() != pins.wav_sha256.lower():
            raise SystemExit(
                f"WAV SHA-256 verification failed for {wav_path}: got {actual_wav}, expected {pins.wav_sha256}. "
                "Install `flac` and retry, or pass --decoder flac."
            )
        if abs(duration - pins.duration_seconds) > 0.02:
            raise SystemExit(
                f"{wav_path} is {duration:.3f} s but the pinned duration is {pins.duration_seconds:.3f} s"
            )
        total_seconds += duration
        clips.append(
            {
                "id": pins.clip_id,
                "file": f"{args.audio_subdir}/{wav_path.name}",
                "reference": pins.reference,
                "duration_seconds": pins.duration_seconds,
                "speaker_id": pins.speaker_id,
                "speaker_gender": pins.speaker_gender,
                "dataset_row": pins.row,
                "source_flac_sha256": pins.flac_sha256,
                "wav_sha256": pins.wav_sha256,
            }
        )
        print(
            f"[{position}/{len(PINS)}] {pins.clip_id}: {duration:.3f} s, "
            f"wav {actual_wav[:12]}...",
            file=sys.stderr,
            flush=True,
        )

    manifest = manifest_for(args.output, args.audio_subdir, clips)
    manifest_path = args.output / args.manifest_name
    attribution_path = args.output / "BENCHMARK-CORPUS-LICENSE.txt"
    write_json_atomic(manifest_path, manifest)
    write_text_atomic(attribution_path, attribution_text())
    print(
        f"Prepared {len(clips)} clips, {total_seconds:.3f} s total.\n"
        f"Manifest: {manifest_path}\n"
        f"Attribution: {attribution_path}\n"
        f"Decoder(s): {', '.join(sorted(decoders)) or 'reused existing canonical WAVs'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
