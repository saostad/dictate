#!/usr/bin/env python3
"""Run Dictate's offline ASR benchmark matrix and build the reports.

One engine and one thread setting run per `dictate benchmark` process. This
script starts those processes sequentially, keeps the raw JSON and stderr from
each one, computes WER/CER and summary statistics, and writes JSON, CSV,
Markdown and a self-contained HTML report.

Nothing here downloads models, changes user configuration, or types into an
application. Child processes get an isolated XDG tree and an explicit model
root; model hashes and the dirty Git tree are recorded so the result can be
audited later.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

TOOL_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent

ENGINE_LABELS = {
    "standard": "Parakeet TDT 0.6B v2 int8",
    "fluid": "Fluid Speech Mini int8",
    "whisper": "Whisper Tiny English Q8_0 (whisper.cpp)",
    "live": "Nemotron streaming en 0.6B int8",
    "moonshine": "Moonshine Tiny Streaming (native C)",
}

# Only the settings the native APIs really expose. Moonshine's pinned C API
# has no exact two-thread setting, so it is never listed with "2".
THREAD_MATRIX = {
    "standard": ["default", "1", "2"],
    "fluid": ["default", "1", "2"],
    "whisper": ["default", "1", "2"],
    "live": ["default", "1", "2"],
    "moonshine": ["1", "runtime-default"],
}

ENGINE_MODEL_DIRS = {
    "standard": ["parakeet-tdt-0.6b-v2-int8"],
    "fluid": ["fluid-speech-mini-onnx"],
    "whisper": ["whisper-tiny-en"],
    "live": ["nemotron-streaming-en-0.6b-int8"],
    "moonshine": ["moonshine-tiny-streaming-en", "moonshine-runtime-0.1.5"],
}

# The paused pipeline uses the same Silero VAD for all whole/pauses engines.
VAD_MODEL_DIR = "silero-vad-v6"
PUBLIC_MODEL_DIRS = [
    "parakeet-tdt-0.6b-v2-int8",
    "fluid-speech-mini-onnx",
    "whisper-tiny-en",
    "nemotron-streaming-en-0.6b-int8",
    "moonshine-tiny-streaming-en",
    "moonshine-runtime-0.1.5",
    VAD_MODEL_DIR,
]

JFK_REFERENCE = (
    "And so my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)

DEFAULT_CORPUS = REPO_ROOT / "tests/fixtures/benchmark-corpus.json"
DEFAULT_JFK = REPO_ROOT / "tests/fixtures/jfk.wav"
DEFAULT_BINARY = REPO_ROOT / "target/release/dictate"

BOUNDARY_PUNCT = re.compile(r"^[\s\.\,\!\?\;\:\)\(\[\]]+")
BRACKET_TAG = re.compile(
    r"\[\s*(?:blank_audio|silence|music|noise|applause|laughter|inaudible|"
    r"no[ _]speech|background[ _]noise|unintelligible)\s*\]"
    r"|\(\s*(?:blank_audio|silence|music|noise|applause|laughter|inaudible|"
    r"no[ _]speech|background[ _]noise|unintelligible)\s*\)"
)
HYPHENS = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014\u2015]+")
SPACE = re.compile(r"\s+")
UNSAFE_ID = re.compile(r"[^a-z0-9._-]+")

UNIT = {
    "seconds": 1000.0,
}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fmt_number(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = as_float(value)
    if number is None:
        return "-"
    if digits == 0:
        return f"{number:.0f}{suffix}"
    return f"{number:.{digits}f}{suffix}"


def fmt_ms(value: Any, digits: int = 1) -> str:
    return fmt_number(value, digits, " ms")


def fmt_bytes(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    index = 0
    while number >= 1024.0 and index < len(units) - 1:
        number /= 1024.0
        index += 1
    if index == 0:
        return f"{number:.0f} B"
    return f"{number:.1f} {units[index]}"


def fmt_ratio(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def fmt_percent(value: Any, digits: int = 1) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return f"{number * 100.0:.{digits}f}%"


def safe_id(text: str) -> str:
    cleaned = UNSAFE_ID.sub("-", text.lower()).strip("-")
    return cleaned or "config"


def normalize_text(text: Any) -> str:
    """Normalize ASR text for WER and CER.

    NFKC, case-fold, remove known bracketed non-speech annotations, turn all hyphen
    variants into spaces, then keep only alphanumeric characters and spaces.
    Apostrophes and other punctuation are deleted (Eva's -> evas), matching
    common LibriSpeech scoring practice. No number expansion or inverse text
    normalization is performed. Ordinary bracketed words remain scoreable.
    """
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = BRACKET_TAG.sub(" ", value)
    # Apostrophes are deleted rather than turned into a word break, so
    # "Eva's" scores as "evas", matching common LibriSpeech practice.
    value = value.replace("'", "").replace("\u2018", "").replace("\u2019", "").replace("\u02bc", "")
    value = HYPHENS.sub(" ", value)
    value = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in value)
    return SPACE.sub(" ", value).strip()


def word_tokens(text: Any) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def char_tokens(text: Any) -> list[str]:
    return list(normalize_text(text).replace(" ", ""))


@dataclasses.dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0
    hypothesis_length: int = 0

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float | None:
        if self.reference_length == 0:
            return None
        return self.edits / self.reference_length

    def as_dict(self) -> dict[str, Any]:
        return {
            "edits": self.edits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.reference_length,
            "hypothesis_length": self.hypothesis_length,
            "wer": self.wer,
        }


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Levenshtein alignment with explicit S/D/I counts.

    Ties prefer substitution, then deletion, then insertion. Any tie-break
    yields the same total edit distance; the split is documented so reports
    remain comparable.
    """
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distance = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(columns):
        distance[0][j] = j
    for i in range(1, rows):
        ref_item = reference[i - 1]
        row = distance[i]
        previous = distance[i - 1]
        for j in range(1, columns):
            cost = 0 if ref_item == hypothesis[j - 1] else 1
            row[j] = min(
                previous[j] + 1,
                row[j - 1] + 1,
                previous[j - 1] + cost,
            )

    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_length=len(reference),
        hypothesis_length=len(hypothesis),
    )


def word_error_counts(reference: Any, hypothesis: Any) -> EditCounts:
    return edit_counts(word_tokens(reference), word_tokens(hypothesis))


def character_error_counts(reference: Any, hypothesis: Any) -> EditCounts:
    return edit_counts(char_tokens(reference), char_tokens(hypothesis))


def aggregate_edits(items: Iterable[EditCounts]) -> EditCounts:
    values = list(items)
    return EditCounts(
        substitutions=sum(item.substitutions for item in values),
        deletions=sum(item.deletions for item in values),
        insertions=sum(item.insertions for item in values),
        reference_length=sum(item.reference_length for item in values),
        hypothesis_length=sum(item.hypothesis_length for item in values),
    )


def describe(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [number for number in (as_float(value) for value in values) if number is not None]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stdev": None,
            "p90": None,
        }
    ordered = sorted(numbers)
    index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "stdev": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "p90": ordered[index],
    }


def config_matrix(engines: Sequence[str], thread_filter: Sequence[str] | None = None) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for engine in engines:
        for thread in THREAD_MATRIX[engine]:
            if thread_filter and thread not in thread_filter:
                continue
            selected.append(
                {
                    "config_id": f"{engine}-{safe_id(thread)}",
                    "engine": engine,
                    "engine_label": ENGINE_LABELS[engine],
                    "threads": thread,
                }
            )
    return selected


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def read_os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        return result
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip("\"'")
    return result


def cpu_model_name() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            value = value.strip()
            if value:
                return value
    return None


def physical_core_count() -> int | None:
    identifiers = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology"):
        try:
            core = (path / "core_id").read_text(encoding="utf-8").strip()
            package = (path / "physical_package_id").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        identifiers.add((package, core))
    return len(identifiers) or None


def memory_total_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def scaling_governor() -> str | None:
    for path in (
        Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"),
    ):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def host_metadata() -> dict[str, Any]:
    uname = platform.uname()
    release = read_os_release()
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {
        "hostname": uname.node,
        "machine": uname.machine,
        "platform": uname.system,
        "kernel_release": uname.release,
        "os_pretty_name": release.get("PRETTY_NAME"),
        "os_id": release.get("ID"),
        "os_version": release.get("VERSION_ID"),
        "cpu_model": cpu_model_name(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": physical_core_count(),
        "memory_total_bytes": memory_total_bytes(),
        "scaling_governor": scaling_governor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "load_average": load,
        "timezone": time.tzname[0] if time.tzname else None,
    }


def run_command(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_metadata(repo: Path) -> dict[str, Any] | None:
    if not (repo / ".git").exists():
        return None

    def text(arguments: Sequence[str]) -> str | None:
        completed = run_command(["git", *arguments], repo)
        if completed.returncode != 0:
            return None
        return completed.stdout.decode("utf-8", "replace").strip()

    def raw(arguments: Sequence[str]) -> bytes:
        completed = run_command(["git", *arguments], repo)
        return completed.stdout if completed.returncode == 0 else b""

    head = text(["rev-parse", "HEAD"])
    branch = text(["rev-parse", "--abbrev-ref", "HEAD"])
    status = text(["status", "--porcelain"])
    diffstat = text(["diff", "--stat", "HEAD"])
    diff = raw(["diff", "--binary", "HEAD"])
    untracked_text = text(["ls-files", "--others", "--exclude-standard"]) or ""
    untracked = [line for line in untracked_text.splitlines() if line.strip()]

    digest = hashlib.sha256()
    digest.update(b"dictate-benchmark-dirty-tree-v1\0")
    digest.update((head or "").encode())
    digest.update(b"\0")
    digest.update((status or "").encode())
    digest.update(b"\0")
    digest.update(diff)
    for relative in sorted(untracked):
        digest.update(b"\0untracked\0")
        digest.update(relative.encode("utf-8", "replace"))
        path = repo / relative
        try:
            if path.is_file():
                if path.stat().st_size <= 64 * 1024 * 1024:
                    digest.update(bytes.fromhex(sha256_file(path)))
                else:
                    digest.update(b"large-file")
            else:
                digest.update(b"not-a-file")
        except OSError:
            digest.update(b"unreadable")
    return {
        "head": head,
        "branch": branch,
        "dirty": bool((status or "").strip()),
        "status_porcelain": status or "",
        "diffstat": diffstat or "",
        "diff_sha256": sha256_bytes(diff),
        "dirty_tree_sha256": digest.hexdigest(),
        "untracked_files": untracked,
    }


def normalize_models_root(path: Path) -> Path:
    """Accept either the models root or the Parakeet child directory."""
    resolved = path.expanduser().resolve()
    if resolved.name == "parakeet-tdt-0.6b-v2-int8":
        return resolved.parent
    return resolved


def required_model_dirs(engines: Sequence[str]) -> list[str]:
    required: list[str] = []
    for engine in engines:
        for name in ENGINE_MODEL_DIRS[engine]:
            if name not in required:
                required.append(name)
    if any(engine in {"standard", "fluid", "whisper"} for engine in engines):
        if VAD_MODEL_DIR not in required:
            required.append(VAD_MODEL_DIR)
    return required


def check_models(root: Path, engines: Sequence[str]) -> None:
    if not root.is_dir():
        raise SystemExit(f"model directory does not exist: {root}")
    missing = [name for name in required_model_dirs(engines) if not (root / name).is_dir()]
    if missing:
        raise SystemExit(
            "model directory is missing: "
            + ", ".join(missing)
            + f"\nLooked in: {root}\n"
            + "Pass --model-dir <models> where the Parakeet and sibling model folders live."
        )


def model_pins(root: Path, hash_files: bool = True) -> dict[str, Any]:
    directories = []
    for name in PUBLIC_MODEL_DIRS:
        directory = root / name
        if directory.is_dir():
            directories.append(directory)
    files: list[dict[str, Any]] = []
    total = 0
    for directory in sorted(directories):
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            record = {
                "path": str(path.relative_to(root)),
                "bytes": size,
                "sha256": sha256_file(path) if hash_files else None,
            }
            files.append(record)
            total += size
    return {
        "root": str(root),
        "hashed": hash_files,
        "directories": [directory.name for directory in sorted(directories)],
        "files": files,
        "file_count": len(files),
        "total_bytes": total,
    }


def validate_wav(path: Path) -> tuple[int, int]:
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
    except (OSError, wave.Error) as error:
        raise ValueError(f"could not read WAV {path}: {error}") from error
    if channels != 1 or width != 2 or rate != 16_000:
        raise ValueError(
            f"{path} must be 16 kHz mono PCM16, found {rate} Hz, {channels} channel(s), {width * 8}-bit"
        )
    if frames <= 0:
        raise ValueError(f"{path} is empty")
    return frames, rate


def load_corpus(path: Path, verify_hashes: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"corpus manifest does not exist: {path}")
    document = read_json(path)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise SystemExit(f"{path}: expected a schema_version 1 corpus manifest")
    clips = document.get("clips")
    if not isinstance(clips, list) or not clips:
        raise SystemExit(f"{path}: corpus has no clips")
    root = path.parent
    by_id: dict[str, dict[str, Any]] = {}
    checked: list[dict[str, Any]] = []
    total_seconds = 0.0
    errors: list[str] = []
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            errors.append(f"clip {index} is not an object")
            continue
        clip_id = str(clip.get("id") or "")
        file_name = clip.get("file")
        reference = clip.get("reference")
        if not clip_id:
            errors.append(f"clip {index} has no id")
            continue
        if clip_id in by_id:
            errors.append(f"duplicate clip id {clip_id}")
        if not isinstance(file_name, str) or not file_name:
            errors.append(f"clip {clip_id} has no file")
            continue
        if not isinstance(reference, str):
            errors.append(f"clip {clip_id} has no string reference")
            continue
        audio_path = Path(file_name)
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        try:
            frames, rate = validate_wav(audio_path)
        except ValueError as error:
            errors.append(str(error))
            continue
        duration = frames / rate
        expected_duration = as_float(clip.get("duration_seconds"))
        if expected_duration is not None and abs(duration - expected_duration) > 0.02:
            errors.append(
                f"clip {clip_id} is {duration:.3f} s but the manifest says {expected_duration:.3f} s"
            )
        expected_sha = clip.get("wav_sha256")
        actual_sha = None
        if verify_hashes and isinstance(expected_sha, str) and expected_sha:
            actual_sha = sha256_file(audio_path)
            if actual_sha.lower() != expected_sha.lower():
                errors.append(
                    f"clip {clip_id} WAV SHA-256 is {actual_sha}, expected {expected_sha}"
                )
        record = dict(clip)
        record["audio_path"] = str(audio_path)
        record["duration_seconds_actual"] = duration
        record["wav_sha256_actual"] = actual_sha
        total_seconds += duration
        checked.append(record)
        by_id[clip_id] = record
    if errors:
        raise SystemExit("corpus validation failed:\n- " + "\n- ".join(errors))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest": document,
        "clips": checked,
        "by_id": by_id,
        "audio_seconds": total_seconds,
        "name": document.get("name"),
        "source": document.get("source") or {},
        "license": (document.get("source") or {}).get("license"),
    }


def validate_jfk(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"JFK reference does not exist: {path}")
    frames, rate = validate_wav(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "audio_seconds": frames / rate,
        "reference": JFK_REFERENCE,
    }


def parse_json_document(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    text = stdout.strip()
    if not text:
        return None, "empty stdout"
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value, None
        return None, f"stdout JSON is {type(value).__name__}, expected object"
    except json.JSONDecodeError as error:
        start = text.find("{")
        if start < 0:
            return None, f"no JSON object in stdout: {error}"
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None, f"invalid JSON in stdout: {error}"
        if isinstance(value, dict):
            return value, None
        return None, f"stdout JSON is {type(value).__name__}, expected object"


@dataclasses.dataclass(frozen=True)
class ChildUsage:
    user_seconds: float
    system_seconds: float
    maxrss_kib: int | None

    @property
    def total_seconds(self) -> float:
        return self.user_seconds + self.system_seconds


def child_usage() -> ChildUsage:
    try:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (AttributeError, OSError):
        return ChildUsage(0.0, 0.0, None)
    maxrss = int(usage.ru_maxrss) if usage.ru_maxrss else None
    if maxrss is not None and sys.platform == "darwin":
        maxrss //= 1024
    return ChildUsage(float(usage.ru_utime), float(usage.ru_stime), maxrss)


def usage_delta(before: ChildUsage, after: ChildUsage) -> dict[str, Any]:
    return {
        "user_seconds": max(0.0, after.user_seconds - before.user_seconds),
        "system_seconds": max(0.0, after.system_seconds - before.system_seconds),
        "total_seconds": max(0.0, after.total_seconds - before.total_seconds),
    }


def child_environment(work: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("DICTATE_"):
            env.pop(key, None)
    for key in (
        "MOONSHINE_ORT_SINGLE_THREAD",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.pop(key, None)
    for name in ("config", "data", "cache", "tmp"):
        (work / name).mkdir(parents=True, exist_ok=True)
    env["XDG_CONFIG_HOME"] = str(work / "config")
    env["XDG_DATA_HOME"] = str(work / "data")
    env["XDG_CACHE_HOME"] = str(work / "cache")
    env["TMPDIR"] = str(work / "tmp")
    env["RUST_BACKTRACE"] = "0"
    env["NO_COLOR"] = "1"
    return env


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            process.kill()
        except OSError:
            pass


def build_command(
    binary: Path,
    config: dict[str, str],
    corpus: Path,
    jfk: Path,
    model_root: Path,
    runs: int,
    paced_runs: int,
    quick: bool,
    sandbox: Path | None,
) -> list[str]:
    arguments = [
        str(binary),
        "benchmark",
        "--engine",
        config["engine"],
        "--threads",
        config["threads"],
        "--corpus",
        str(corpus),
        "--jfk",
        str(jfk),
        "--runs",
        str(runs),
        "--paced-runs",
        str(paced_runs),
        "--model-dir",
        str(model_root),
    ]
    if quick:
        arguments.append("--quick")
    if sandbox is None:
        return arguments
    if sandbox.suffix.lower() == ".py":
        return [sys.executable, str(sandbox), *arguments]
    return [str(sandbox), *arguments]


def stderr_tail(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    tail = "\n".join(text.splitlines()[-20:])
    return tail[-limit:]


def run_config(
    index: int,
    total: int,
    config: dict[str, str],
    output: Path,
    binary: Path,
    corpus: Path,
    jfk: Path,
    model_root: Path,
    runs: int,
    paced_runs: int,
    quick: bool,
    sandbox: Path | None,
    timeout: float,
    work: Path,
) -> dict[str, Any]:
    config_id = config["config_id"]
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = raw_dir / f"{index:02d}-{config_id}.stdout.txt"
    stderr_path = raw_dir / f"{index:02d}-{config_id}.stderr.log"
    command = build_command(
        binary,
        config,
        corpus,
        jfk,
        model_root,
        runs,
        paced_runs,
        quick,
        sandbox,
    )
    print(
        f"[{index}/{total}] {config['engine']} threads={config['threads']} ...",
        file=sys.stderr,
        flush=True,
    )
    environment = child_environment(work)
    before_usage = child_usage()
    started = time.monotonic()
    timed_out = False
    launch_error = None
    stdout = ""
    stderr = ""
    returncode: int | None = None
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process(process)
            stdout, stderr = process.communicate()
        except KeyboardInterrupt:
            stop_process(process)
            stdout, stderr = process.communicate()
            raise
        returncode = process.returncode
    except OSError as error:
        launch_error = f"{error}"
    elapsed_ms = (time.monotonic() - started) * 1000.0
    after_usage = child_usage()
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    document, parse_error = parse_json_document(stdout)
    if timed_out:
        status = "timeout"
    elif launch_error:
        status = "error"
    elif document is None:
        status = "error"
    else:
        status = str(document.get("status") or "error")
        if returncode not in (0, None) and status == "ok":
            status = "error"
    record: dict[str, Any] = {
        "config_id": config_id,
        "engine": config["engine"],
        "engine_label": config["engine_label"],
        "threads": config["threads"],
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "wall_ms": elapsed_ms,
        "cpu": usage_delta(before_usage, after_usage),
        "command": command,
        "stdout": str(stdout_path.relative_to(output)),
        "stderr": str(stderr_path.relative_to(output)),
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        "parse_error": parse_error,
        "document": document,
    }
    if status not in {"ok", "partial"}:
        record["stderr_tail"] = stderr_tail(stderr)
    print(
        f"[{index}/{total}] {config['engine']} threads={config['threads']}: "
        f"{status} in {elapsed_ms / 1000.0:.1f} s",
        file=sys.stderr,
        flush=True,
    )
    return record


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _quality_clip(row: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    clip_id = str(row.get("clip_id") or "")
    known = corpus["by_id"].get(clip_id, {})
    reference = row.get("reference")
    if not isinstance(reference, str):
        reference = known.get("reference", "")
    hypothesis = row.get("transcript")
    if not isinstance(hypothesis, str):
        hypothesis = ""
    words = word_error_counts(reference, hypothesis)
    characters = character_error_counts(reference, hypothesis)
    duration = as_float(row.get("duration_seconds"))
    if duration is None:
        duration = as_float(known.get("duration_seconds"))
    decode_ms = as_float(row.get("decode_ms"))
    rtf = as_float(row.get("rtf"))
    if rtf is None and decode_ms is not None and duration and duration > 0:
        rtf = decode_ms / (duration * 1000.0)
    return {
        "clip_id": clip_id,
        "reference": reference,
        "hypothesis": hypothesis,
        "status": row.get("status") or ("error" if row.get("error") else "ok"),
        "error": row.get("error"),
        "duration_seconds": duration,
        "decode_ms": decode_ms,
        "rtf": rtf,
        "wer": words.wer,
        "edits": words.edits,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "reference_words": words.reference_length,
        "hypothesis_words": words.hypothesis_length,
        "cer": characters.wer,
        "character_edits": characters.edits,
        "reference_characters": characters.reference_length,
        "speaker_id": row.get("speaker_id"),
        "speaker_gender": row.get("speaker_gender"),
        "dataset_row": row.get("dataset_row"),
    }


def summarize_quality(document: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    quality = _dict(document.get("quality"))
    rows = [row for row in _list(quality.get("runs")) if isinstance(row, dict)]
    clips = [_quality_clip(row, corpus) for row in rows]
    word_counts = [
        EditCounts(
            substitutions=int(clip["substitutions"]),
            deletions=int(clip["deletions"]),
            insertions=int(clip["insertions"]),
            reference_length=int(clip["reference_words"]),
            hypothesis_length=int(clip["hypothesis_words"]),
        )
        for clip in clips
    ]
    words = aggregate_edits(word_counts)
    # Character edits are already exact for each clip; sum them directly.
    character_edits = sum(int(clip["character_edits"]) for clip in clips)
    character_reference = sum(int(clip["reference_characters"]) for clip in clips)
    character_wer = character_edits / character_reference if character_reference else None
    audio_seconds = sum(clip["duration_seconds"] or 0.0 for clip in clips)
    decode_values = [clip["decode_ms"] for clip in clips if clip["decode_ms"] is not None]
    decode_ms = sum(decode_values) if decode_values else None
    rtf = decode_ms / (audio_seconds * 1000.0) if decode_ms is not None and audio_seconds > 0 else None
    return {
        "clips": clips,
        "clip_count": len(clips),
        "failed_clips": sum(1 for clip in clips if clip["status"] != "ok"),
        "audio_seconds": audio_seconds,
        "decode_ms": decode_ms,
        "rtf": rtf,
        "reference_words": words.reference_length,
        "hypothesis_words": words.hypothesis_length,
        "edits": words.edits,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "wer": words.wer,
        "character_edits": character_edits,
        "reference_characters": character_reference,
        "cer": character_wer,
        "corpus": quality.get("corpus"),
    }


def summarize_jfk(document: dict[str, Any]) -> dict[str, Any]:
    jfk = _dict(document.get("jfk"))
    runs = [row for row in _list(jfk.get("runs")) if isinstance(row, dict)]
    prepared = []
    counts = []
    for row in runs:
        hypothesis = row.get("transcript")
        if not isinstance(hypothesis, str):
            hypothesis = ""
        words = word_error_counts(JFK_REFERENCE, hypothesis)
        counts.append(words)
        item = dict(row)
        item["wer"] = words.wer
        item["edits"] = words.edits
        item["substitutions"] = words.substitutions
        item["deletions"] = words.deletions
        item["insertions"] = words.insertions
        prepared.append(item)
    times = [as_float(row.get("decode_ms")) for row in runs if row.get("status") == "ok"]
    rtfs = [as_float(row.get("rtf")) for row in runs if row.get("status") == "ok"]
    aggregate = aggregate_edits(counts)
    return {
        "runs": prepared,
        "run_count": len(runs),
        "failed_runs": sum(1 for row in runs if row.get("status") != "ok"),
        "decode_ms": describe(times),
        "rtf": describe(rtfs),
        "wer": aggregate.wer,
        "edits": aggregate.edits,
        "substitutions": aggregate.substitutions,
        "deletions": aggregate.deletions,
        "insertions": aggregate.insertions,
        "reference_words": aggregate.reference_length,
        "audio_seconds": as_float(jfk.get("audio_seconds")),
        "path": jfk.get("path"),
    }


def summarize_stress(document: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in _list(document.get("stress")) if isinstance(row, dict)]
    cases = []
    for row in rows:
        transcript = row.get("transcript")
        if not isinstance(transcript, str):
            transcript = ""
        item = dict(row)
        item["transcript"] = transcript
        item["recognized_words"] = len(word_tokens(transcript))
        item["false_words"] = (
            item["recognized_words"] if row.get("kind") in {"silence", "noise"} else None
        )
        cases.append(item)
    silence = [case for case in cases if case.get("kind") == "silence"]
    noise = [case for case in cases if case.get("kind") == "noise"]
    return {
        "cases": cases,
        "silence_false_words": sum(int(case["false_words"]) for case in silence) if silence else None,
        "noise_false_words": sum(int(case["false_words"]) for case in noise) if noise else None,
    }


def summarize_paced(document: dict[str, Any]) -> dict[str, Any] | None:
    paced = document.get("paced")
    if not isinstance(paced, dict):
        return None
    runs = [row for row in _list(paced.get("runs")) if isinstance(row, dict)]
    ok_runs = [row for row in runs if row.get("status") == "ok"]
    first_committed = [as_float(row.get("first_committed_ms")) for row in ok_runs]
    stop_to_final = [as_float(row.get("stop_to_final_ms")) for row in ok_runs]
    finalization = [as_float(row.get("finalization_compute_ms")) for row in ok_runs]
    backlog = [as_float(row.get("max_backlog_ms")) for row in runs]
    lag = [as_float(row.get("max_lag_ms")) for row in runs]
    backlog_numbers = [value for value in backlog if value is not None]
    lag_numbers = [value for value in lag if value is not None]
    return {
        "applies": bool(paced.get("applies")),
        "method": paced.get("method"),
        "runs": runs,
        "run_count": len(runs),
        "failed_runs": sum(1 for row in runs if row.get("status") != "ok"),
        "first_committed_ms": describe(first_committed),
        "stop_to_final_ms": describe(stop_to_final),
        "finalization_compute_ms": describe(finalization),
        "max_backlog_ms": max(backlog_numbers) if backlog_numbers else None,
        "max_lag_ms": max(lag_numbers) if lag_numbers else None,
        "first_provisional_count": sum(
            1 for row in runs if row.get("first_provisional_ms") is not None
        ),
    }


def summarize_paused(document: dict[str, Any]) -> dict[str, Any] | None:
    paused = document.get("whole_vs_paused_jfk")
    if not isinstance(paused, dict):
        return None
    whole = paused.get("whole_transcript")
    if not isinstance(whole, str):
        whole = ""
    paused_text = paused.get("paused_transcript")
    if not isinstance(paused_text, str):
        paused_text = ""
    whole_words = word_error_counts(JFK_REFERENCE, whole)
    paused_words = word_error_counts(JFK_REFERENCE, paused_text)
    result = dict(paused)
    result["whole_wer"] = whole_words.wer
    result["whole_edits"] = whole_words.edits
    result["paused_wer"] = paused_words.wer
    result["paused_edits"] = paused_words.edits
    return result


def build_summary(record: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    document = record.get("document")
    summary: dict[str, Any] = {
        "config_id": record["config_id"],
        "engine": record["engine"],
        "engine_label": record["engine_label"],
        "threads": record["threads"],
        "status": record["status"],
        "returncode": record.get("returncode"),
        "wall_ms": record.get("wall_ms"),
        "cpu": record.get("cpu"),
        "command": record.get("command"),
        "stdout": record.get("stdout"),
        "stderr": record.get("stderr"),
        "parse_error": record.get("parse_error"),
        "stderr_tail": record.get("stderr_tail"),
        "threads_effective": None,
        "threads_policy": None,
        "model_load_ms": None,
        "first_decode_ms": None,
        "first_chunk_ms": None,
        "warmup_total_ms": None,
        "warmup_transcript": None,
        "memory": None,
        "runtime": None,
        "jfk": None,
        "quality": None,
        "stress": None,
        "paced": None,
        "paused": None,
        "engine_errors": [],
        "document_status": None,
    }
    if not isinstance(document, dict):
        return summary

    summary["document_status"] = document.get("status")
    summary["engine_label"] = document.get("engine_label") or summary["engine_label"]
    threads = _dict(document.get("threads"))
    summary["threads_effective"] = as_int(threads.get("effective"))
    summary["threads_policy"] = threads.get("policy")
    if threads.get("moonshine_policy"):
        summary["threads_policy"] = (
            f"{summary['threads_policy']}; {threads.get('moonshine_policy')}"
        )

    load = _dict(document.get("load"))
    summary["model_load_ms"] = as_float(load.get("model_load_ms"))
    summary["first_decode_ms"] = as_float(load.get("first_decode_ms"))
    summary["first_chunk_ms"] = as_float(load.get("first_chunk_ms"))
    summary["warmup_total_ms"] = as_float(load.get("warmup_total_ms"))
    summary["warmup_transcript"] = load.get("warmup_transcript")

    memory = _dict(document.get("memory"))
    after = _dict(memory.get("after_load"))
    end = _dict(memory.get("at_end"))
    rss_values = [as_int(after.get("rss_bytes")), as_int(end.get("rss_bytes"))]
    peak_values = [as_int(after.get("peak_rss_bytes")), as_int(end.get("peak_rss_bytes"))]
    rss_values = [value for value in rss_values if value is not None]
    peak_values = [value for value in peak_values if value is not None]
    summary["memory"] = {
        "rss_after_load_bytes": as_int(after.get("rss_bytes")),
        "rss_at_end_bytes": as_int(end.get("rss_bytes")),
        "peak_rss_bytes": max(peak_values) if peak_values else None,
        "units": memory.get("units"),
    }

    runtime = _dict(document.get("runtime"))
    summary["runtime"] = {
        "package_version": runtime.get("package_version"),
        "model_dir": runtime.get("model_dir"),
        "model_bytes": as_int(runtime.get("model_bytes")),
        "file_count": len(_list(runtime.get("model_files"))),
    }

    summary["jfk"] = summarize_jfk(document)
    summary["quality"] = summarize_quality(document, corpus)
    summary["stress"] = summarize_stress(document)
    summary["paced"] = summarize_paced(document)
    summary["paused"] = summarize_paused(document)

    errors = []
    for error in _list(document.get("engine_errors")):
        if isinstance(error, dict):
            errors.append(error)
    summary["engine_errors"] = errors
    if errors and summary["status"] == "ok":
        summary["status"] = "partial"
    return summary


SUMMARY_FIELDS = [
    "config_id",
    "engine",
    "engine_label",
    "threads",
    "threads_effective",
    "status",
    "document_status",
    "model_load_ms",
    "first_decode_ms",
    "first_chunk_ms",
    "warmup_total_ms",
    "jfk_runs",
    "jfk_failed_runs",
    "jfk_decode_ms_mean",
    "jfk_decode_ms_median",
    "jfk_decode_ms_min",
    "jfk_decode_ms_max",
    "jfk_rtf_median",
    "jfk_wer",
    "corpus_clips",
    "corpus_failed_clips",
    "corpus_audio_seconds",
    "corpus_decode_ms",
    "corpus_rtf",
    "corpus_reference_words",
    "corpus_edits",
    "corpus_substitutions",
    "corpus_deletions",
    "corpus_insertions",
    "corpus_wer",
    "corpus_character_edits",
    "corpus_reference_characters",
    "corpus_cer",
    "silence_false_words",
    "noise_false_words",
    "rss_after_load_bytes",
    "rss_at_end_bytes",
    "peak_rss_bytes",
    "model_bytes",
    "paced_runs",
    "paced_failed_runs",
    "paced_first_committed_ms_median",
    "paced_stop_to_final_ms_median",
    "paced_finalization_compute_ms_median",
    "paced_max_backlog_ms",
    "paced_max_lag_ms",
    "paced_first_provisional_count",
    "wall_ms",
    "cpu_user_seconds",
    "cpu_system_seconds",
    "cpu_total_seconds",
    "command",
    "stdout",
    "stderr",
]

CLIP_FIELDS = [
    "config_id",
    "engine",
    "threads",
    "clip_id",
    "status",
    "duration_seconds",
    "decode_ms",
    "rtf",
    "reference_words",
    "hypothesis_words",
    "edits",
    "substitutions",
    "deletions",
    "insertions",
    "wer",
    "reference_characters",
    "character_edits",
    "cer",
    "reference",
    "hypothesis",
    "error",
]

STRESS_FIELDS = [
    "config_id",
    "engine",
    "threads",
    "case_id",
    "kind",
    "status",
    "duration_seconds",
    "decode_ms",
    "rtf",
    "recognized_words",
    "false_words",
    "transcript",
    "error",
    "label",
]


def summary_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    jfk = summary.get("jfk") or {}
    jfk_stats = jfk.get("decode_ms") or {}
    jfk_rtf = jfk.get("rtf") or {}
    quality = summary.get("quality") or {}
    stress = summary.get("stress") or {}
    memory = summary.get("memory") or {}
    runtime = summary.get("runtime") or {}
    paced = summary.get("paced") or {}
    paced_commit = (paced.get("first_committed_ms") or {}) if paced else {}
    paced_stop = (paced.get("stop_to_final_ms") or {}) if paced else {}
    paced_final = (paced.get("finalization_compute_ms") or {}) if paced else {}
    cpu = summary.get("cpu") or {}
    return {
        "config_id": summary.get("config_id"),
        "engine": summary.get("engine"),
        "engine_label": summary.get("engine_label"),
        "threads": summary.get("threads"),
        "threads_effective": summary.get("threads_effective"),
        "status": summary.get("status"),
        "document_status": summary.get("document_status"),
        "model_load_ms": summary.get("model_load_ms"),
        "first_decode_ms": summary.get("first_decode_ms"),
        "first_chunk_ms": summary.get("first_chunk_ms"),
        "warmup_total_ms": summary.get("warmup_total_ms"),
        "jfk_runs": jfk.get("run_count"),
        "jfk_failed_runs": jfk.get("failed_runs"),
        "jfk_decode_ms_mean": jfk_stats.get("mean"),
        "jfk_decode_ms_median": jfk_stats.get("median"),
        "jfk_decode_ms_min": jfk_stats.get("min"),
        "jfk_decode_ms_max": jfk_stats.get("max"),
        "jfk_rtf_median": jfk_rtf.get("median"),
        "jfk_wer": jfk.get("wer"),
        "corpus_clips": quality.get("clip_count"),
        "corpus_failed_clips": quality.get("failed_clips"),
        "corpus_audio_seconds": quality.get("audio_seconds"),
        "corpus_decode_ms": quality.get("decode_ms"),
        "corpus_rtf": quality.get("rtf"),
        "corpus_reference_words": quality.get("reference_words"),
        "corpus_edits": quality.get("edits"),
        "corpus_substitutions": quality.get("substitutions"),
        "corpus_deletions": quality.get("deletions"),
        "corpus_insertions": quality.get("insertions"),
        "corpus_wer": quality.get("wer"),
        "corpus_character_edits": quality.get("character_edits"),
        "corpus_reference_characters": quality.get("reference_characters"),
        "corpus_cer": quality.get("cer"),
        "silence_false_words": stress.get("silence_false_words"),
        "noise_false_words": stress.get("noise_false_words"),
        "rss_after_load_bytes": memory.get("rss_after_load_bytes"),
        "rss_at_end_bytes": memory.get("rss_at_end_bytes"),
        "peak_rss_bytes": memory.get("peak_rss_bytes"),
        "model_bytes": runtime.get("model_bytes"),
        "paced_runs": paced.get("run_count"),
        "paced_failed_runs": paced.get("failed_runs"),
        "paced_first_committed_ms_median": paced_commit.get("median"),
        "paced_stop_to_final_ms_median": paced_stop.get("median"),
        "paced_finalization_compute_ms_median": paced_final.get("median"),
        "paced_max_backlog_ms": paced.get("max_backlog_ms"),
        "paced_max_lag_ms": paced.get("max_lag_ms"),
        "paced_first_provisional_count": paced.get("first_provisional_count"),
        "wall_ms": summary.get("wall_ms"),
        "cpu_user_seconds": cpu.get("user_seconds"),
        "cpu_system_seconds": cpu.get("system_seconds"),
        "cpu_total_seconds": cpu.get("total_seconds"),
        "command": " ".join(summary.get("command") or []),
        "stdout": summary.get("stdout"),
        "stderr": summary.get("stderr"),
    }


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def quality_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    quality = summary.get("quality") or {}
    for clip in quality.get("clips") or []:
        row = dict(clip)
        row.update(
            {
                "config_id": summary.get("config_id"),
                "engine": summary.get("engine"),
                "threads": summary.get("threads"),
            }
        )
        rows.append(row)
    return rows


def stress_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    stress = summary.get("stress") or {}
    for case in stress.get("cases") or []:
        row = dict(case)
        row.update(
            {
                "config_id": summary.get("config_id"),
                "engine": summary.get("engine"),
                "threads": summary.get("threads"),
            }
        )
        rows.append(row)
    return rows


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    host = metadata["host"]
    git = metadata["git"] or {}
    summaries = [summary for summary in report["summaries"] if summary.get("status") in {"ok", "partial"}]
    failures = [summary for summary in report["summaries"] if summary.get("status") not in {"ok", "partial"}]
    lines: list[str] = []
    lines.append("# Dictate ASR model benchmarks")
    lines.append("")
    lines.append(
        f"Generated {report['generated_utc']} on {host.get('hostname') or 'unknown host'} "
        f"({host.get('cpu_model') or 'unknown CPU'}, {host.get('logical_cpus') or '?'} logical CPUs, "
        f"{fmt_bytes(host.get('memory_total_bytes'))} RAM)."
    )
    lines.append("")
    if git:
        dirty = "dirty" if git.get("dirty") else "clean"
        lines.append(
            f"Tree: `{str(git.get('head') or 'unknown')[:12]}` ({git.get('branch')}, {dirty}), "
            f"dirty-tree SHA-256 `{str(git.get('dirty_tree_sha256') or '')[:16]}`."
        )
    lines.append(
        f"Binary: `{metadata['binary']['path']}` (SHA-256 `{metadata['binary']['sha256'][:16]}`). "
        f"Corpus: `{metadata['corpus']['path']}` (SHA-256 `{metadata['corpus']['sha256'][:16]}`)."
    )
    lines.append("")
    lines.append(
        "> One engine per process, loaded once and kept resident. JFK runs are warmed and unpaced; "
        "streaming paced runs feed audio at real time. RTF is processing time divided by audio time. "
        "RSS is the whole process from `/proc/self/status`, not weights-only memory. "
        "The 12-clip LibriSpeech subset is small read English, not conversational or accent coverage."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    headers = [
        "Engine",
        "Threads",
        "Status",
        "Load ms",
        "First JFK decode ms",
        "JFK median ms",
        "JFK RTF",
        "Corpus WER",
        "CER",
        "Peak RSS",
        "Paced stop ms",
    ]
    rows = []
    for summary in summaries:
        jfk = summary.get("jfk") or {}
        jfk_stats = jfk.get("decode_ms") or {}
        jfk_rtf = jfk.get("rtf") or {}
        quality = summary.get("quality") or {}
        memory = summary.get("memory") or {}
        paced = summary.get("paced") or {}
        paced_stop = (paced.get("stop_to_final_ms") or {}) if paced else {}
        rows.append(
            [
                summary.get("engine"),
                summary.get("threads"),
                summary.get("status"),
                fmt_number(summary.get("model_load_ms"), 0),
                fmt_number(summary.get("first_decode_ms"), 0),
                fmt_number(jfk_stats.get("median"), 0),
                fmt_ratio(jfk_rtf.get("median"), 3),
                fmt_percent(quality.get("wer"), 2),
                fmt_percent(quality.get("cer"), 2),
                fmt_bytes(memory.get("peak_rss_bytes")),
                fmt_number(paced_stop.get("median"), 0),
            ]
        )
    lines.append(markdown_table(headers, rows))
    lines.append("")
    lines.append("## Quality per clip")
    lines.append("")
    clip_headers = ["Engine", "Threads", "Clip", "Duration s", "Decode ms", "WER", "S", "D", "I"]
    clip_rows = []
    for summary in summaries:
        quality = summary.get("quality") or {}
        for clip in quality.get("clips") or []:
            clip_rows.append(
                [
                    summary.get("engine"),
                    summary.get("threads"),
                    clip.get("clip_id"),
                    fmt_number(clip.get("duration_seconds"), 3),
                    fmt_number(clip.get("decode_ms"), 0),
                    fmt_percent(clip.get("wer"), 1),
                    clip.get("substitutions"),
                    clip.get("deletions"),
                    clip.get("insertions"),
                ]
            )
    lines.append(markdown_table(clip_headers, clip_rows))
    lines.append("")
    lines.append("## Stress and false words")
    lines.append("")
    stress_headers = ["Engine", "Threads", "5 s silence words", "5 s noise words", "Paced stop ms", "Max backlog ms"]
    stress_rows = []
    for summary in summaries:
        stress = summary.get("stress") or {}
        paced = summary.get("paced") or {}
        paced_stop = (paced.get("stop_to_final_ms") or {}) if paced else {}
        stress_rows.append(
            [
                summary.get("engine"),
                summary.get("threads"),
                stress.get("silence_false_words"),
                stress.get("noise_false_words"),
                fmt_number(paced_stop.get("median"), 0),
                fmt_number(paced.get("max_backlog_ms"), 0),
            ]
        )
    lines.append(markdown_table(stress_headers, stress_rows))
    lines.append("")
    if failures:
        lines.append("## Failed configurations")
        lines.append("")
        failure_headers = ["Config", "Status", "Return code", "Parse error", "stderr tail"]
        failure_rows = []
        for summary in failures:
            failure_rows.append(
                [
                    summary.get("config_id"),
                    summary.get("status"),
                    summary.get("returncode"),
                    summary.get("parse_error") or "",
                    (summary.get("stderr_tail") or "").replace("\n", " ")[:300],
                ]
            )
        lines.append(markdown_table(failure_headers, failure_rows))
        lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "- One process per engine/thread configuration, model loaded once. "
        "The standard, Fluid, Live and Whisper paths use their native 1/2-thread constructor settings; "
        "the default setting is the app's real default. Moonshine exposes only single-thread mode or "
        "the ONNX Runtime default, so it is never reported as an exact two-thread run."
    )
    lines.append(
        "- Audio is preloaded before timers start. One warmup JFK decode is excluded from the measured JFK runs; "
        "it covers the whole first recording for every engine, and streaming also reports the first chunk separately. "
        "Five warmed JFK runs by default; the 12-clip corpus runs once after warmup."
    )
    lines.append(
        "- WER normalization: NFKC, case-fold, remove bracketed non-speech tags, hyphens to spaces, "
        "keep alphanumerics and spaces, delete other punctuation (apostrophes included). "
        "Corpus WER is total edits / total reference words; S/D/I are from a word Levenshtein alignment. "
        "CER uses the same normalization without spaces. Numbers are not expanded."
    )
    lines.append(
        "- Silence and noise false words are reported separately and are not part of corpus WER. "
        "Synthetic JFK repeats are labeled synthetic and never enter corpus WER."
    )
    lines.append(
        "- Live and Moonshine paced runs feed 100 ms (Moonshine) or the model's native chunk (Live) at "
        "real-time arrival, then flush exactly as the production lifecycle does. Slower-than-real-time "
        "is recorded, not gated. `stop_to_final_ms` includes the flush/finalization compute."
    )
    lines.append(
        "- Measurements are one machine, warm file cache, no GUI latency, and no cold-disk claim. "
        "CPU time is parent-side `RUSAGE_CHILDREN`; peak RSS comes from the child. "
        "Thermal state, load, and CPU governor can move results between runs."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `benchmark.json`: full raw documents and summaries.")
    lines.append("- `benchmark.csv`: one summary row per configuration.")
    lines.append("- `benchmark-clips.csv`: per-clip references, hypotheses, WER, CER, S/D/I.")
    lines.append("- `benchmark-stress.csv`: silence/noise/JFK stress transcripts.")
    lines.append("- `benchmark.html`: self-contained, offline report with engine/thread filters.")
    lines.append("- `raw/`: one stdout JSON and stderr log per process.")
    lines.append("")
    write_text_atomic(path, "\n".join(lines))


def _html_data(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report["metadata"]
    models = metadata.get("models") or {}
    return {
        "generated_utc": report.get("generated_utc"),
        "host": metadata.get("host"),
        "git": metadata.get("git"),
        "binary": metadata.get("binary"),
        "corpus": metadata.get("corpus"),
        "models": {
            "root": models.get("root"),
            "hashed": models.get("hashed"),
            "file_count": models.get("file_count"),
            "total_bytes": models.get("total_bytes"),
            "directories": models.get("directories"),
        },
        "flags": metadata.get("flags"),
        "isolation": metadata.get("isolation"),
        "summaries": report.get("summaries") or [],
        "jfk_reference": JFK_REFERENCE,
    }


HTML_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #f4f6f8;
  --panel: #ffffff;
  --fg: #15202b;
  --muted: #5a6876;
  --line: #d8dee5;
  --accent: #1f6feb;
  --accent-soft: #dbeafe;
  --good: #137333;
  --warn: #9a6700;
  --bad: #b42318;
  --bar: #e3e8ee;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --fg: #e6edf3;
    --muted: #8b949e;
    --line: #30363d;
    --accent: #58a6ff;
    --accent-soft: #16263d;
    --good: #3fb950;
    --warn: #d29922;
    --bad: #f85149;
    --bar: #21262d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
header {
  background: linear-gradient(135deg, var(--panel), var(--accent-soft));
  border-bottom: 1px solid var(--line);
}
.wrap { max-width: 1440px; margin: 0 auto; padding: 22px; }
h1 { margin: 0 0 6px; font-size: 26px; letter-spacing: -0.02em; }
h2 { margin: 0 0 12px; font-size: 18px; }
h3 { margin: 18px 0 8px; font-size: 15px; }
p { margin: 6px 0; }
.sub { color: var(--muted); max-width: 1000px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 10px; margin: 18px 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.card .value { font-size: 20px; font-weight: 650; margin-top: 4px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin: 16px 0; overflow: hidden; }
.toolbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-bottom: 12px; }
.toolbar label { display: grid; gap: 4px; color: var(--muted); font-size: 12px; }
.toolbar select { min-width: 150px; padding: 7px 9px; border: 1px solid var(--line); border-radius: 7px; background: var(--panel); color: var(--fg); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); background: var(--panel); position: sticky; top: 0; }
td.num, th.num { text-align: right; white-space: nowrap; }
tbody tr:hover { background: color-mix(in srgb, var(--accent-soft) 36%, transparent); }
.status-ok { color: var(--good); font-weight: 600; }
.status-partial { color: var(--warn); font-weight: 600; }
.status-error, .status-timeout { color: var(--bad); font-weight: 600; }
.bar { width: 76px; height: 7px; border-radius: 99px; background: var(--bar); overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 6px; }
.bar > span { display: block; height: 100%; background: var(--accent); border-radius: 99px; }
details { border: 1px solid var(--line); border-radius: 9px; padding: 9px 11px; margin: 8px 0; background: var(--panel); }
details > summary { cursor: pointer; font-weight: 600; }
details details { background: transparent; }
pre { margin: 6px 0; padding: 9px; background: var(--bg); border: 1px solid var(--line); border-radius: 7px; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.muted { color: var(--muted); }
.small { font-size: 12px; }
.err { color: var(--bad); }
.method ul { margin: 6px 0 0 20px; padding: 0; }
.method li { margin: 5px 0; }
.badge { display: inline-block; padding: 1px 7px; border: 1px solid var(--line); border-radius: 99px; font-size: 12px; color: var(--muted); }
.footer { color: var(--muted); font-size: 12px; margin: 20px 0 40px; }
"""

HTML_JS = r"""
const DATA = JSON.parse(document.getElementById("report-data").textContent);
const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
};
const pct = (value, digits = 2) => value === null || value === undefined ? "-" : `${(Number(value) * 100).toFixed(digits)}%`;
const bytes = (value) => {
  if (value === null || value === undefined) return "-";
  let number = Number(value), index = 0;
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  while (number >= 1024 && index < units.length - 1) { number /= 1024; index += 1; }
  return `${number.toFixed(index ? 1 : 0)} ${units[index]}`;
};
const statusClass = (status) => `status-${String(status || "error").replace(/[^a-z]/g, "")}`;
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "colspan") node.colSpan = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
function statusCell(status) {
  return h("td", {}, h("span", { class: statusClass(status) }, status || "error"));
}
function bar(value, max, label) {
  const wrapper = h("span", { class: "bar" });
  const span = h("span");
  const ratio = value !== null && value !== undefined && max > 0 ? Math.min(1, Number(value) / max) : 0;
  span.style.width = `${Math.max(2, ratio * 100)}%`;
  wrapper.append(span);
  return h("span", {}, label, wrapper);
}
function table(headers, rows) {
  const thead = h("thead", {}, h("tr", {}, ...headers.map((header, index) =>
    h("th", { class: typeof header === "object" && header.num ? "num" : "" }, typeof header === "object" ? header.label : header)
  )));
  const tbody = h("tbody", {}, ...rows.map((row) => h("tr", {}, ...row.map((cell, index) =>
    cell instanceof HTMLTableCellElement ? cell : h("td", { class: typeof headers[index] === "object" && headers[index].num ? "num" : "" }, cell ?? "")
  ))));
  return h("table", {}, thead, tbody);
}
function transcriptBlock(text) {
  return h("pre", {}, text === null || text === undefined || text === "" ? "(empty)" : String(text));
}
function configLabel(summary) {
  return `${summary.engine} / ${summary.threads}`;
}
function filtered() {
  const engine = $("engine-filter").value;
  const thread = $("thread-filter").value;
  const status = $("status-filter").value;
  return DATA.summaries.filter((summary) =>
    (engine === "all" || summary.engine === engine) &&
    (thread === "all" || summary.threads === thread) &&
    (status === "all" || summary.status === status)
  );
}
function renderSummary() {
  const rows = filtered();
  const maxWer = Math.max(0.0001, ...rows.map((summary) => (summary.quality || {}).wer || 0));
  const maxRtf = Math.max(0.0001, ...rows.map((summary) => ((summary.jfk || {}).rtf || {}).median || 0));
  const maxRss = Math.max(1, ...rows.map((summary) => ((summary.memory || {}).peak_rss_bytes) || 0));
  const headers = [
    "Engine", "Threads", "Status",
    { label: "Load ms", num: true }, { label: "First JFK ms", num: true },
    { label: "JFK med ms", num: true }, { label: "JFK RTF", num: true },
    "Corpus WER", "CER", "Peak RSS", { label: "Stop ms", num: true }
  ];
  const body = rows.map((summary) => {
    const quality = summary.quality || {};
    const jfk = summary.jfk || {};
    const jfkStats = jfk.decode_ms || {};
    const jfkRtf = jfk.rtf || {};
    const memory = summary.memory || {};
    const paced = summary.paced || {};
    const stop = (paced.stop_to_final_ms || {}).median;
    return [
      summary.engine_label || summary.engine,
      summary.threads,
      statusCell(summary.status),
      fmt(summary.model_load_ms, 0),
      fmt(summary.first_decode_ms, 0),
      fmt(jfkStats.median, 0),
      bar(jfkRtf.median, maxRtf, fmt(jfkRtf.median, 3)),
      bar(quality.wer, maxWer, pct(quality.wer, 2)),
      pct(quality.cer, 2),
      bar(memory.peak_rss_bytes, maxRss, bytes(memory.peak_rss_bytes)),
      fmt(stop, 0)
    ];
  });
  const container = $("summary-table");
  container.replaceChildren(table(headers, body));
  $("summary-count").textContent = `${rows.length} configuration${rows.length === 1 ? "" : "s"}`;
}
function renderDetails() {
  const container = $("details");
  const rows = filtered();
  if (!rows.length) {
    container.replaceChildren(h("p", { class: "muted" }, "No configurations match the filters."));
    return;
  }
  container.replaceChildren(...rows.map((summary) => {
    const quality = summary.quality || {};
    const jfk = summary.jfk || {};
    const stress = summary.stress || {};
    const paced = summary.paced || {};
    const paused = summary.paused;
    const details = h("details", {},
      h("summary", {},
        `${configLabel(summary)} - `,
        h("span", { class: statusClass(summary.status) }, summary.status),
        ` - WER ${pct(quality.wer, 2)}, JFK ${fmt((jfk.decode_ms || {}).median, 0)} ms`
      )
    );
    details.append(h("p", { class: "muted small" },
      `Load ${fmt(summary.model_load_ms, 0)} ms; first JFK decode ${fmt(summary.first_decode_ms, 0)} ms; ` +
      `first streaming chunk ${fmt(summary.first_chunk_ms, 1)} ms; ` +
      `peak RSS ${bytes((summary.memory || {}).peak_rss_bytes)}; model data ${bytes((summary.runtime || {}).model_bytes)}. ` +
      `Threads: requested ${summary.threads}, effective ${summary.threads_effective ?? "runtime default"}${summary.threads_policy ? ` (${summary.threads_policy})` : ""}.`
    ));

    details.append(h("h3", {}, "JFK warmed runs"));
    const jfkHeaders = ["Run", "Status", "Decode ms", "RTF", "WER", "Transcript"];
    const jfkRows = (jfk.runs || []).map((run) => [
      run.run, statusCell(run.status), fmt(run.decode_ms, 1), fmt(run.rtf, 3),
      pct(run.wer, 2), transcriptBlock(run.transcript)
    ]);
    details.append(jfkRows.length ? table(jfkHeaders, jfkRows) : h("p", { class: "muted" }, "No JFK runs."));

    details.append(h("h3", {}, "Corpus clips"));
    const clipHeaders = ["Clip", "Status", "Dur s", "Decode ms", "WER", "CER", "S", "D", "I", "Reference / hypothesis"];
    const clipRows = (quality.clips || []).map((clip) => [
      clip.clip_id, statusCell(clip.status), fmt(clip.duration_seconds, 3),
      fmt(clip.decode_ms, 0), pct(clip.wer, 2), pct(clip.cer, 2),
      clip.substitutions, clip.deletions, clip.insertions,
      h("details", {}, h("summary", {}, "show text"),
        h("p", {}, h("strong", {}, "Reference: "), clip.reference || ""),
        h("p", {}, h("strong", {}, "Hypothesis: "), clip.hypothesis || "")
      )
    ]);
    details.append(clipRows.length ? table(clipHeaders, clipRows) : h("p", { class: "muted" }, "No corpus clips."));

    if (paused) {
      details.append(h("h3", {}, "JFK whole vs paused pipeline"));
      details.append(h("p", { class: "muted small" },
        `Whole WER ${pct(paused.whole_wer, 2)}; paused WER ${pct(paused.paused_wer, 2)}; ` +
        `paused total ${fmt(paused.paused_total_ms, 0)} ms; decode ${fmt(paused.paused_decode_ms, 0)} ms; ` +
        `VAD load ${fmt(paused.vad_load_ms, 0)} ms (fresh load, excluded from paused total); ` +
        `segments ${paused.paused_segments ?? "-"}; trials ${paused.trials ?? 1}.`
      ));
      details.append(h("p", { class: "muted small" }, paused.note || "Single paired warm JFK trial; not a statistical distribution."));
      details.append(h("p", {}, h("strong", {}, "Whole: "), paused.whole_transcript || "(empty)"));
      details.append(h("p", {}, h("strong", {}, "Paused: "), paused.paused_transcript || "(empty)"));
    }

    details.append(h("h3", {}, "Stress cases"));
    const stressHeaders = ["Case", "Kind", "Status", "Dur s", "Decode ms", "False words", "Transcript"];
    const stressRows = (stress.cases || []).map((item) => [
      item.case_id, item.kind, statusCell(item.status), fmt(item.duration_seconds, 2),
      fmt(item.decode_ms, 0), item.false_words ?? "-", transcriptBlock(item.transcript)
    ]);
    details.append(stressRows.length ? table(stressHeaders, stressRows) : h("p", { class: "muted" }, "No stress cases."));

    if (paced) {
      details.append(h("h3", {}, "Paced JFK runs"));
      const pacedHeaders = ["Run", "Status", "First chunk ms", "First commit ms", "Stop to final ms", "Finalization ms", "Backlog ms", "Lag ms", "Transcript"];
      const pacedRows = (paced.runs || []).map((run) => [
        run.run, statusCell(run.status), fmt(run.first_chunk_ms, 1),
        fmt(run.first_committed_ms, 1), fmt(run.stop_to_final_ms, 1),
        fmt(run.finalization_compute_ms, 1), fmt(run.max_backlog_ms, 1),
        fmt(run.max_lag_ms, 1), transcriptBlock(run.transcript)
      ]);
      details.append(pacedRows.length ? table(pacedHeaders, pacedRows) : h("p", { class: "muted" }, "No paced runs."));
    }

    if ((summary.engine_errors || []).length) {
      details.append(h("h3", {}, "Engine errors"));
      details.append(h("ul", {}, ...summary.engine_errors.map((error) =>
        h("li", { class: "err" }, `${error.stage || "error"} / ${error.id || ""}: ${error.error || ""}`)
      )));
    }
    if (summary.stderr_tail) {
      details.append(h("h3", {}, "stderr tail"));
      details.append(transcriptBlock(summary.stderr_tail));
    }
    details.append(h("p", { class: "muted small" },
      `Raw: ${summary.stdout || "-"} and ${summary.stderr || "-"}. ` +
      `Wall ${fmt((summary.wall_ms || 0) / 1000, 1)} s; child CPU ${fmt((summary.cpu || {}).total_seconds, 1)} s.`
    ));
    return details;
  }));
}
function renderCards() {
  const rows = DATA.summaries;
  const ok = rows.filter((summary) => summary.status === "ok").length;
  const partial = rows.filter((summary) => summary.status === "partial").length;
  const failed = rows.length - ok - partial;
  const clips = Math.max(0, ...rows.map((summary) => ((summary.quality || {}).clip_count) || 0));
  const cards = [
    ["Configurations", `${rows.length}`],
    ["OK / partial / failed", `${ok} / ${partial} / ${failed}`],
    ["Corpus clips per config", `${clips}`],
    ["Corpus audio", `${fmt((DATA.corpus || {}).audio_seconds, 1)} s`],
  ];
  $("cards").replaceChildren(...cards.map(([label, value]) =>
    h("div", { class: "card" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value))
  ));
}
function init() {
  const engines = [...new Set(DATA.summaries.map((summary) => summary.engine))].sort();
  const threads = [...new Set(DATA.summaries.map((summary) => summary.threads))];
  for (const engine of engines) $("engine-filter").append(h("option", { value: engine }, engine));
  for (const thread of threads) $("thread-filter").append(h("option", { value: thread }, thread));
  for (const id of ["engine-filter", "thread-filter", "status-filter"]) {
    $(id).addEventListener("change", () => { renderSummary(); renderDetails(); });
  }
  renderCards();
  renderSummary();
  renderDetails();
}
init();
"""


def render_html(report: dict[str, Any]) -> str:
    import html as _html

    data = _html_data(report)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    host = data.get("host") or {}
    git = data.get("git") or {}
    binary = data.get("binary") or {}
    corpus = data.get("corpus") or {}
    dirty = "dirty" if git.get("dirty") else "clean"
    title = "Dictate ASR model benchmarks"
    method_items = [
        "One process per engine and thread setting. The model is loaded once and kept resident.",
        "Audio is preloaded before timers start. The first JFK decode is a warmup and is excluded from measured runs; it covers the whole recording for every engine, while the streaming first chunk is reported separately.",
        "JFK: five warmed runs by default. Corpus: every clip once after warmup. Streaming quality is unpaced.",
        "RTF = process time / audio time. Live and Moonshine paced runs feed at real-time arrival; slower than real time is a recorded result, not a gate.",
        "Memory is whole-process VmRSS/VmHWM from /proc/self/status, never weights-only memory.",
        "WER: NFKC, case-fold, remove bracketed non-speech tags, hyphens to spaces, keep alphanumerics and spaces, delete other punctuation. Corpus WER is total edits / total reference words; S/D/I from word Levenshtein. CER is character-level without spaces.",
        "Silence and noise false words are separate from corpus WER. Synthetic JFK repeats are labeled and excluded from corpus WER.",
        "Moonshine exposes single-thread mode or the ONNX Runtime default; there is no exact two-thread setting in its pinned C API.",
        "Single machine, warm file cache, no cold-disk claim, no GUI latency. Thermal state, load and governor can move results.",
    ]
    method_html = "".join(f"<li>{_html.escape(item)}</li>" for item in method_items)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>{HTML_STYLE}</style>
</head>
<body>
<header><div class="wrap">
<h1>{_html.escape(title)}</h1>
<p class="sub">Generated {_html.escape(str(data.get("generated_utc") or "unknown"))} on {_html.escape(str(host.get("hostname") or "unknown host"))}.
CPU: {_html.escape(str(host.get("cpu_model") or "unknown"))}; {_html.escape(str(host.get("logical_cpus") or "?"))} logical CPUs;
{_html.escape(fmt_bytes(host.get("memory_total_bytes")))} RAM.
Tree: {_html.escape(str((git.get("head") or "unknown")[:12]))} ({_html.escape(str(git.get("branch") or "?"))}, {_html.escape(dirty)}).
Binary SHA-256: {_html.escape(str((binary.get("sha256") or "")[:16]))}.
Corpus: {_html.escape(str(corpus.get("name") or "?"))} ({_html.escape(str(corpus.get("license") or "license unknown"))}).</p>
</div></header>
<main class="wrap">
<section class="cards" id="cards"></section>
<section class="panel">
<h2>Summary <span class="muted small" id="summary-count"></span></h2>
<div class="toolbar">
<label>Engine<select id="engine-filter"><option value="all">all</option></select></label>
<label>Threads<select id="thread-filter"><option value="all">all</option></select></label>
<label>Status<select id="status-filter"><option value="all">all</option><option value="ok">ok</option><option value="partial">partial</option><option value="error">error</option><option value="timeout">timeout</option></select></label>
</div>
<div class="table-wrap" id="summary-table"></div>
</section>
<section class="panel">
<h2>Per-clip, stress and paced transcripts</h2>
<p class="muted small">Bars are relative to the largest value in the current filter. Lower is better for WER, RTF and RSS.</p>
<div id="details"></div>
</section>
<section class="panel method">
<h2>Method and caveats</h2>
<ul>{method_html}</ul>
<p class="muted small">JFK reference: {_html.escape(JFK_REFERENCE)}</p>
</section>
<p class="footer">Offline self-contained report. No remote scripts, styles, fonts or images. Raw process logs and full JSON are next to this file.</p>
</main>
<script id="report-data" type="application/json">{encoded}</script>
<script>{HTML_JS}</script>
</body>
</html>
"""


def build_report(
    metadata: dict[str, Any],
    corpus: dict[str, Any],
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    interrupted: bool,
) -> dict[str, Any]:
    ok = sum(1 for summary in summaries if summary.get("status") == "ok")
    partial = sum(1 for summary in summaries if summary.get("status") == "partial")
    failures = [summary for summary in summaries if summary.get("status") not in {"ok", "partial"}]
    with_wer = [summary for summary in summaries if (summary.get("quality") or {}).get("wer") is not None]
    with_rtf = [
        summary
        for summary in summaries
        if ((summary.get("jfk") or {}).get("rtf") or {}).get("median") is not None
    ]
    best_wer = min(with_wer, key=lambda summary: summary["quality"]["wer"], default=None)
    best_rtf = min(
        with_rtf,
        key=lambda summary: summary["jfk"]["rtf"]["median"],
        default=None,
    )
    return {
        "schema_version": 1,
        "kind": "dictate-model-benchmark-report",
        "generated_utc": utc_now(),
        "interrupted": interrupted,
        "metadata": metadata,
        "corpus_manifest": corpus["manifest"],
        "configs": records,
        "summaries": summaries,
        "aggregate": {
            "configurations": len(summaries),
            "ok": ok,
            "partial": partial,
            "failed": len(failures),
            "failed_config_ids": [summary.get("config_id") for summary in failures],
            "best_corpus_wer": (
                {
                    "config_id": best_wer.get("config_id"),
                    "engine": best_wer.get("engine"),
                    "threads": best_wer.get("threads"),
                    "wer": best_wer["quality"]["wer"],
                }
                if best_wer
                else None
            ),
            "best_jfk_rtf": (
                {
                    "config_id": best_rtf.get("config_id"),
                    "engine": best_rtf.get("engine"),
                    "threads": best_rtf.get("threads"),
                    "median_rtf": best_rtf["jfk"]["rtf"]["median"],
                }
                if best_rtf
                else None
            ),
        },
    }


def write_reports(output: Path, report: dict[str, Any]) -> None:
    summaries = report["summaries"]
    write_json_atomic(output / "benchmark.json", report)
    write_csv(
        output / "benchmark.csv",
        SUMMARY_FIELDS,
        [summary_csv_row(summary) for summary in summaries],
    )
    clip_rows = []
    for summary in summaries:
        clip_rows.extend(quality_csv_rows(summary))
    write_csv(output / "benchmark-clips.csv", CLIP_FIELDS, clip_rows)
    stress_rows = []
    for summary in summaries:
        stress_rows.extend(stress_csv_rows(summary))
    write_csv(output / "benchmark-stress.csv", STRESS_FIELDS, stress_rows)
    write_markdown(output / "benchmark.md", report)
    write_text_atomic(output / "benchmark.html", render_html(report))
    commands = []
    for summary in summaries:
        commands.append(
            f"# {summary.get('config_id')} ({summary.get('status')})\n"
            + " ".join(subprocess.list2cmdline([str(part)]) for part in (summary.get("command") or []))
        )
    write_text_atomic(output / "commands.txt", "\n\n".join(commands) + ("\n" if commands else ""))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every Dictate ASR engine sequentially through `dictate benchmark`, "
            "then write JSON, CSV, Markdown and a self-contained HTML report."
        )
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help="dictate binary (default: target/release/dictate)")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="prepared corpus manifest")
    parser.add_argument("--jfk", type=Path, default=DEFAULT_JFK, help="public-domain JFK WAV")
    parser.add_argument("--model-dir", type=Path, help="models root (or the Parakeet child); defaults to DICTATE_MODEL_DIR")
    parser.add_argument("--output", type=Path, help="output directory (default: benchmark-results/<UTC timestamp>)")
    parser.add_argument("--runs", type=int, default=5, help="warmed JFK runs per config (default 5)")
    parser.add_argument("--paced-runs", type=int, default=3, help="real-time JFK runs for Live/Moonshine (default 3)")
    parser.add_argument("--quick", action="store_true", help="one JFK run, one paced run, first two corpus clips")
    parser.add_argument("--timeout", type=float, default=2400.0, help="seconds allowed per config (default 2400)")
    parser.add_argument("--engines", help="comma-separated subset: standard,fluid,whisper,live,moonshine")
    parser.add_argument("--threads", help="comma-separated subset: default,1,2,runtime-default")
    parser.add_argument("--sandbox", type=Path, help="optional executable or .py libseccomp/no-network wrapper")
    parser.add_argument("--no-model-hash", action="store_true", help="record model sizes but skip SHA-256 hashing")
    parser.add_argument("--no-corpus-hash", action="store_true", help="skip corpus WAV SHA-256 verification")
    parser.add_argument("--force", action="store_true", help="write into a non-empty output directory")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any configuration fails")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and commands without running")
    args = parser.parse_args(list(argv))

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.paced_runs < 1:
        parser.error("--paced-runs must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    engines = parse_csv_list(args.engines) or list(THREAD_MATRIX)
    unknown = [engine for engine in engines if engine not in THREAD_MATRIX]
    if unknown:
        parser.error(f"unknown engine(s): {', '.join(unknown)}")
    thread_filter = parse_csv_list(args.threads)
    if thread_filter:
        known_threads = {thread for values in THREAD_MATRIX.values() for thread in values}
        unknown_threads = [value for value in thread_filter if value not in known_threads]
        if unknown_threads:
            parser.error(f"unknown thread setting(s): {', '.join(unknown_threads)}")
    configs = config_matrix(engines, thread_filter)
    if not configs:
        parser.error("the engine/thread filters select no configurations")
    args.engines = engines
    args.thread_filter = thread_filter
    args.configs = configs

    if args.model_dir is None:
        env_model = os.environ.get("DICTATE_MODEL_DIR")
        if env_model:
            args.model_dir = Path(env_model)
    if args.model_dir is None:
        parser.error("--model-dir is required (or set DICTATE_MODEL_DIR)")

    args.binary = args.binary.expanduser().resolve()
    args.corpus = args.corpus.expanduser().resolve()
    args.jfk = args.jfk.expanduser().resolve()
    args.model_dir = normalize_models_root(args.model_dir)
    if args.sandbox is not None:
        args.sandbox = args.sandbox.expanduser().resolve()
    if args.output is None:
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = REPO_ROOT / "benchmark-results" / stamp
    args.output = args.output.expanduser().resolve()
    if args.quick:
        args.runs = 1
        args.paced_runs = 1
    return args


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if not args.binary.is_file():
        raise SystemExit(f"binary does not exist: {args.binary}\nBuild it with: cargo build --release")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    if args.sandbox is not None and not args.sandbox.is_file():
        raise SystemExit(f"sandbox wrapper does not exist: {args.sandbox}")
    check_models(args.model_dir, args.engines)
    corpus = load_corpus(args.corpus, verify_hashes=not args.no_corpus_hash)
    jfk = validate_jfk(args.jfk)
    return {"corpus": corpus, "jfk": jfk}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    validated = validate_inputs(args)
    corpus = validated["corpus"]
    jfk = validated["jfk"]

    if args.dry_run:
        plan = {
            "binary": str(args.binary),
            "corpus": str(args.corpus),
            "jfk": str(args.jfk),
            "model_dir": str(args.model_dir),
            "quick": args.quick,
            "runs": args.runs,
            "paced_runs": args.paced_runs,
            "timeout_seconds": args.timeout,
            "sandbox": str(args.sandbox) if args.sandbox else None,
            "configurations": [
                {
                    **config,
                    "command": build_command(
                        args.binary,
                        config,
                        args.corpus,
                        args.jfk,
                        args.model_dir,
                        args.runs,
                        args.paced_runs,
                        args.quick,
                        args.sandbox,
                    ),
                }
                for config in args.configs
            ],
        }
        json.dump(plan, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.output.exists() and not args.output.is_dir():
        raise SystemExit(f"output path exists and is not a directory: {args.output}")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise SystemExit(
            f"output directory is not empty: {args.output}\nUse --force or choose another --output."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw").mkdir(parents=True, exist_ok=True)
    work = args.output / "_work"
    work.mkdir(parents=True, exist_ok=True)

    print(f"Hashing and inventorying models under {args.model_dir} ...", file=sys.stderr, flush=True)
    models = model_pins(args.model_dir, hash_files=not args.no_model_hash)
    binary_stat = args.binary.stat()
    metadata = {
        "tool": {
            "name": Path(__file__).name,
            "version": TOOL_VERSION,
            "argv": sys.argv,
        },
        "host": host_metadata(),
        "git": git_metadata(REPO_ROOT),
        "binary": {
            "path": str(args.binary),
            "sha256": sha256_file(args.binary),
            "bytes": binary_stat.st_size,
            "modified_unix_s": binary_stat.st_mtime,
        },
        "corpus": {
            "path": str(args.corpus),
            "sha256": corpus["sha256"],
            "clip_count": len(corpus["clips"]),
            "audio_seconds": corpus["audio_seconds"],
            "name": corpus.get("name"),
            "license": corpus.get("license"),
            "source": corpus.get("source"),
        },
        "jfk": jfk,
        "models": models,
        "flags": {
            "engines": args.engines,
            "threads": args.thread_filter,
            "quick": args.quick,
            "runs": args.runs,
            "paced_runs": args.paced_runs,
            "timeout_seconds": args.timeout,
            "sandbox": str(args.sandbox) if args.sandbox else None,
            "hash_models": not args.no_model_hash,
            "verify_corpus_hashes": not args.no_corpus_hash,
        },
        "isolation": {
            "xdg_config_home": str(work / "config"),
            "xdg_data_home": str(work / "data"),
            "xdg_cache_home": str(work / "cache"),
            "tmpdir": str(work / "tmp"),
            "removed_env_prefix": "DICTATE_*",
            "removed_thread_env": [
                "MOONSHINE_ORT_SINGLE_THREAD",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ],
            "note": (
                "The model root is passed with --model-dir. Child processes never read user config; "
                "DICTATE_MOONSHINE_RUNTIME_DIR and all DICTATE_* overrides are removed."
            ),
        },
    }

    records: list[dict[str, Any]] = []
    interrupted = False
    try:
        for index, config in enumerate(args.configs, start=1):
            record = run_config(
                index=index,
                total=len(args.configs),
                config=config,
                output=args.output,
                binary=args.binary,
                corpus=args.corpus,
                jfk=args.jfk,
                model_root=args.model_dir,
                runs=args.runs,
                paced_runs=args.paced_runs,
                quick=args.quick,
                sandbox=args.sandbox,
                timeout=args.timeout,
                work=work,
            )
            records.append(record)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; writing the configurations that completed.", file=sys.stderr)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    summaries = [build_summary(record, corpus) for record in records]
    report = build_report(metadata, corpus, records, summaries, interrupted)
    write_reports(args.output, report)

    aggregate = report["aggregate"]
    print(
        f"Wrote {args.output}/benchmark.{{json,csv,md,html}} "
        f"({aggregate['ok']} ok, {aggregate['partial']} partial, {aggregate['failed']} failed)",
        file=sys.stderr,
    )
    if args.strict and aggregate["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
