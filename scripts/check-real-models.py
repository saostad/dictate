"""Transcribe a real clip with every model, using a built dictate binary.

Usage: python3 scripts/check-real-models.py <dictate binary> [16-bit WAV]

Needs the models downloaded first (dictate download, --live, --fluid).
Runs the binary that ships, so CI checks it without compiling tests.
"""

import os
import subprocess
import sys
import tempfile
import wave

EXPECTED = "ask not what your country can do for you"
FULL_SENTENCE = (
    "and so, my fellow americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)


def run(binary, *args):
    result = subprocess.run(
        [binary, "transcribe", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        sys.exit(f"dictate transcribe {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout, result.stderr


def expect_sentence(name, stdout):
    if EXPECTED not in stdout.lower():
        sys.exit(f"unexpected {name} transcription: {stdout}")
    print(f"ok: {name}: {stdout.strip()}")


def read_mono(path):
    with wave.open(path) as w:
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: only 16-bit WAV is supported")
        channels, rate = w.getnchannels(), w.getframerate()
        data = w.readframes(w.getnframes())
    samples = [
        int.from_bytes(data[i : i + 2], "little", signed=True) / 32768.0
        for i in range(0, len(data), 2)
    ]
    mono = [
        sum(samples[i : i + channels]) / channels
        for i in range(0, len(samples), channels)
    ]
    return mono, rate


def resample_linear(mono, from_rate, to_rate):
    if not mono or from_rate == to_rate:
        return list(mono)
    out_len = len(mono) * to_rate // from_rate
    out = []
    for i in range(out_len):
        pos = i * from_rate / to_rate
        lo = int(pos)
        hi = min(lo + 1, len(mono) - 1)
        frac = pos - lo
        out.append(mono[lo] * (1.0 - frac) + mono[hi] * frac)
    return out


def write_wav(path, samples, rate):
    frames = b"".join(
        int(max(-1.0, min(1.0, s)) * 32767.0).to_bytes(2, "little", signed=True)
        for s in samples
    )
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def normalize(text):
    return "".join(c for c in text.lower() if c not in ".,")


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    binary = sys.argv[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav = sys.argv[2] if len(sys.argv) == 3 else os.path.join(root, "tests", "fixtures", "jfk.wav")

    expect_sentence("whole", run(binary, wav)[0])
    expect_sentence("fluid", run(binary, "--fluid", wav)[0])

    stdout, stderr = run(binary, "--live", wav)
    expect_sentence("live", stdout)
    rtf = next((line for line in stderr.splitlines() if line.startswith("live: avg")), None)
    if rtf is None:
        sys.exit(f"expected a live RTF line, stderr: {stderr}")
    print(rtf)

    # Device-rate path: the fixture at 48 kHz through the streaming daemon's feeder.
    mono, rate = read_mono(wav)
    with tempfile.TemporaryDirectory() as tmp:
        wav48 = os.path.join(tmp, "seg48.wav")
        write_wav(wav48, resample_linear(mono, rate, 48_000), 48_000)
        stdout, stderr = run(binary, "--segmented", wav48)
    if normalize(stdout.strip()) != normalize(FULL_SENTENCE):
        sys.exit(f"segmented transcription differs: {stdout}")
    # The onset check only means something if the clip was actually split.
    if "segment 1:" not in stderr or "segment 2:" not in stderr:
        sys.exit(f"expected at least two segments, stderr: {stderr}")
    print(f"ok: segmented at 48 kHz: {stdout.strip()}")


if __name__ == "__main__":
    main()
