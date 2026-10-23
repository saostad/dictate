#!/usr/bin/env python3
"""Light, offline tests for the benchmark tooling.

Run with: python3 scripts/test_benchmark_tools.py
These tests do not load models, download audio, or run a benchmark.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bm = load_module("benchmark_models", REPO_ROOT / "scripts/benchmark-models.py")
prep = load_module("prepare_benchmark_corpus", REPO_ROOT / "scripts/prepare-benchmark-corpus.py")


def write_sine_wav(path: Path, seconds: float = 0.05, rate: int = 16_000) -> None:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)


class NormalizationTests(unittest.TestCase):
    def test_ordinary_bracketed_words_are_scored(self) -> None:
        self.assertEqual(
            bm.normalize_text("(Thank you) [hello] [MUSIC] (background noise)"),
            "thank you hello",
        )
        self.assertEqual(bm.word_error_counts("", "(Thank you)").insertions, 2)

    def test_punctuation_apostrophe_hyphen_and_tags(self) -> None:
        text = "  Hello,   WORLD! This is Eva's well-known [BLANK_AUDIO] test. "
        self.assertEqual(
            bm.normalize_text(text),
            "hello world this is evas well known test",
        )

    def test_casefold_and_unicode(self) -> None:
        self.assertEqual(bm.normalize_text("STRASSE Straße"), "strasse strasse")
        self.assertEqual(bm.normalize_text("“quoted” — dash"), "quoted dash")

    def test_empty(self) -> None:
        self.assertEqual(bm.normalize_text(None), "")
        self.assertEqual(bm.normalize_text("   "), "")


class WerTests(unittest.TestCase):
    def test_substitution_insertion(self) -> None:
        counts = bm.word_error_counts("a b c d", "a x c e f")
        self.assertEqual(counts.substitutions, 2)
        self.assertEqual(counts.deletions, 0)
        self.assertEqual(counts.insertions, 1)
        self.assertEqual(counts.edits, 3)
        self.assertAlmostEqual(counts.wer, 0.75)

    def test_deletion(self) -> None:
        counts = bm.word_error_counts("a b c", "a c")
        self.assertEqual((counts.substitutions, counts.deletions, counts.insertions), (0, 1, 0))
        self.assertAlmostEqual(counts.wer, 1 / 3)

    def test_insertion(self) -> None:
        counts = bm.word_error_counts("a", "a b")
        self.assertEqual((counts.substitutions, counts.deletions, counts.insertions), (0, 0, 1))
        self.assertAlmostEqual(counts.wer, 1.0)

    def test_empty_reference_has_undefined_wer(self) -> None:
        counts = bm.word_error_counts("", "x y")
        self.assertEqual(counts.insertions, 2)
        self.assertIsNone(counts.wer)

    def test_empty_hypothesis_is_all_deletions(self) -> None:
        counts = bm.word_error_counts("x y", "")
        self.assertEqual(counts.deletions, 2)
        self.assertAlmostEqual(counts.wer, 1.0)

    def test_aggregate_is_total_edits_over_total_reference(self) -> None:
        counts = [
            bm.word_error_counts("a b", "a"),
            bm.word_error_counts("c d", "x d y"),
        ]
        total = bm.aggregate_edits(counts)
        self.assertEqual(total.reference_length, 4)
        self.assertEqual(total.edits, 3)
        self.assertAlmostEqual(total.wer, 0.75)

    def test_cer(self) -> None:
        counts = bm.character_error_counts("abc", "axc")
        self.assertEqual(counts.edits, 1)
        self.assertAlmostEqual(counts.wer, 1 / 3)


class StatisticsTests(unittest.TestCase):
    def test_describe(self) -> None:
        stats = bm.describe([1, 2, 3, 4, 5])
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["median"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["p90"], 5.0)
        self.assertAlmostEqual(stats["stdev"], 2**0.5)

    def test_describe_empty(self) -> None:
        stats = bm.describe([])
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["median"])


class MatrixTests(unittest.TestCase):
    def test_all_engines_and_moonshine_never_exact_two(self) -> None:
        configs = bm.config_matrix(list(bm.THREAD_MATRIX))
        engines = {config["engine"] for config in configs}
        self.assertEqual(engines, set(bm.THREAD_MATRIX))
        self.assertEqual(len(configs), 14)
        moonshine = [config for config in configs if config["engine"] == "moonshine"]
        self.assertEqual({config["threads"] for config in moonshine}, {"1", "runtime-default"})
        self.assertNotIn("2", {config["threads"] for config in moonshine})

    def test_thread_filter(self) -> None:
        configs = bm.config_matrix(["standard", "moonshine"], ["1"])
        self.assertEqual([config["config_id"] for config in configs], ["standard-1", "moonshine-1"])


class JsonTests(unittest.TestCase):
    def test_plain_and_prefixed_json(self) -> None:
        document = {"schema_version": 1, "status": "ok"}
        parsed, error = bm.parse_json_document(json.dumps(document))
        self.assertIsNone(error)
        self.assertEqual(parsed, document)
        parsed, error = bm.parse_json_document("warning\n" + json.dumps(document) + "\ntrailer")
        self.assertIsNone(error)
        self.assertEqual(parsed, document)

    def test_invalid(self) -> None:
        parsed, error = bm.parse_json_document("not json")
        self.assertIsNone(parsed)
        self.assertIsNotNone(error)


class FixtureTests(unittest.TestCase):
    def test_portable_manifest(self) -> None:
        path = REPO_ROOT / "tests/fixtures/benchmark-corpus.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        clips = manifest["clips"]
        self.assertEqual(len(clips), 12)
        self.assertEqual(sum(1 for clip in clips if clip["speaker_gender"] == "M"), 3 * 2)
        self.assertEqual(sum(1 for clip in clips if clip["speaker_gender"] == "F"), 3 * 2)
        self.assertEqual(len({clip["id"] for clip in clips}), 12)
        self.assertEqual(len({clip["dataset_row"] for clip in clips}), 12)
        total = sum(float(clip["duration_seconds"]) for clip in clips)
        self.assertAlmostEqual(total, 81.345, places=3)
        rows = manifest["source"]["rows"]
        self.assertEqual(rows, [0, 2, 250, 254, 501, 502, 750, 751, 1000, 1001, 1502, 1503])
        for clip in clips:
            self.assertNotIn("source_url", clip)
            self.assertEqual(len(clip["source_flac_sha256"]), 64)
            self.assertEqual(len(clip["wav_sha256"]), 64)
            self.assertRegex(clip["file"], r"^audio/[^/]+\.wav$")

    def test_prepare_pins_match_fixture(self) -> None:
        manifest = json.loads(
            (REPO_ROOT / "tests/fixtures/benchmark-corpus.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(prep.PINS), 12)
        for expected, pins in zip(manifest["clips"], prep.PINS):
            self.assertEqual(expected["id"], pins.clip_id)
            self.assertEqual(expected["dataset_row"], pins.row)
            self.assertEqual(expected["reference"], pins.reference)
            self.assertEqual(expected["source_flac_sha256"], pins.flac_sha256)
            self.assertEqual(expected["wav_sha256"], pins.wav_sha256)
            self.assertAlmostEqual(expected["duration_seconds"], pins.duration_seconds, places=3)


class PrepareTests(unittest.TestCase):
    def test_validate_row_accepts_pinned_content(self) -> None:
        pins = prep.PINS[0]
        prep.validate_row(
            pins,
            {
                "id": pins.clip_id,
                "text": pins.reference,
                "speaker_id": pins.speaker_id,
                "audio": [{"src": "https://example.invalid/audio.flac"}],
            },
        )

    def test_validate_row_rejects_wrong_text(self) -> None:
        pins = prep.PINS[0]
        with self.assertRaises(RuntimeError):
            prep.validate_row(
                pins,
                {
                    "id": pins.clip_id,
                    "text": "DIFFERENT TEXT",
                    "speaker_id": pins.speaker_id,
                    "audio": [{"src": "https://example.invalid/audio.flac"}],
                },
            )

    def test_manifest_is_relative_and_complete(self) -> None:
        clips = [
            {
                "id": prep.PINS[0].clip_id,
                "file": f"audio/{prep.PINS[0].clip_id}.wav",
                "reference": prep.PINS[0].reference,
                "duration_seconds": prep.PINS[0].duration_seconds,
                "speaker_id": prep.PINS[0].speaker_id,
                "speaker_gender": prep.PINS[0].speaker_gender,
                "dataset_row": prep.PINS[0].row,
                "source_flac_sha256": prep.PINS[0].flac_sha256,
                "wav_sha256": prep.PINS[0].wav_sha256,
            }
        ]
        manifest = prep.manifest_for(Path("/tmp"), "audio", clips)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["source"]["revision"], prep.REVISION)
        self.assertEqual(manifest["source"]["rows"], [pins.row for pins in prep.PINS])
        self.assertEqual(manifest["clips"][0]["file"], f"audio/{prep.PINS[0].clip_id}.wav")
        self.assertIn("CC-BY-4.0", prep.attribution_text())

    def test_canonicalize_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            destination = Path(directory) / "destination.wav"
            write_sine_wav(source)
            prep.canonicalize_wav(source, destination)
            duration, digest = prep.wav_info(destination)
            self.assertAlmostEqual(duration, 0.05, places=3)
            self.assertEqual(len(digest), 64)
            self.assertGreater(destination.stat().st_size, 44)


class HtmlTests(unittest.TestCase):
    def fake_report(self, transcript: str) -> dict:
        clip = {
            "clip_id": "test-clip",
            "reference": "a b",
            "hypothesis": transcript,
            "status": "ok",
            "duration_seconds": 1.0,
            "decode_ms": 10.0,
            "rtf": 0.01,
            "wer": 0.5,
            "edits": 1,
            "substitutions": 1,
            "deletions": 0,
            "insertions": 0,
            "reference_words": 2,
            "hypothesis_words": 2,
            "cer": 0.5,
            "character_edits": 1,
            "reference_characters": 2,
        }
        summary = {
            "config_id": "standard-1",
            "engine": "standard",
            "engine_label": "Parakeet",
            "threads": "1",
            "status": "ok",
            "model_load_ms": 1.0,
            "first_decode_ms": 2.0,
            "threads_effective": 1,
            "threads_policy": "explicit",
            "memory": {"peak_rss_bytes": 1000},
            "runtime": {"model_bytes": 500},
            "jfk": {
                "runs": [{"run": 1, "status": "ok", "decode_ms": 1.0, "rtf": 0.1, "wer": 0.0, "transcript": "a"}],
                "decode_ms": {"median": 1.0},
                "rtf": {"median": 0.1},
            },
            "quality": {"clips": [clip], "wer": 0.5, "cer": 0.5, "clip_count": 1},
            "stress": {"cases": [], "silence_false_words": 0, "noise_false_words": 0},
            "paced": None,
            "paused": None,
            "engine_errors": [],
            "stdout": "raw/x.stdout.txt",
            "stderr": "raw/x.stderr.log",
            "wall_ms": 1000.0,
            "cpu": {"total_seconds": 0.5},
        }
        return {
            "generated_utc": "2026-01-01T00:00:00+00:00",
            "metadata": {
                "host": {
                    "hostname": "test",
                    "cpu_model": "cpu",
                    "logical_cpus": 8,
                    "memory_total_bytes": 1024**3,
                },
                "git": {"head": "a" * 40, "branch": "main", "dirty": True},
                "binary": {"path": "/bin/dictate", "sha256": "b" * 64},
                "corpus": {"path": "/corpus.json", "sha256": "c" * 64, "name": "test", "license": "CC-BY-4.0"},
                "models": {"root": "/models", "hashed": True, "file_count": 1, "total_bytes": 10, "directories": ["x"]},
                "flags": {},
                "isolation": {},
            },
            "summaries": [summary],
        }

    def test_html_is_offline_and_escapes_transcript(self) -> None:
        report = self.fake_report("</script><img src=x onerror=bad()>")
        html = bm.render_html(report)
        self.assertIn("<style>", html)
        self.assertIn("<script>", html)
        for pattern in ('src="http', 'href="http', "@import", "url(http"):
            self.assertNotIn(pattern, html)
        self.assertNotIn("</script><img", html)
        # The embedded JSON is still parseable after escaping closing tags.
        start = html.index('<script id="report-data" type="application/json">') + len(
            '<script id="report-data" type="application/json">'
        )
        end = html.index("</script>", start)
        data = json.loads(html[start:end].replace("<\\/", "</"))
        self.assertEqual(data["summaries"][0]["quality"]["clips"][0]["hypothesis"], "</script><img src=x onerror=bad()>")

    def test_write_reports(self) -> None:
        report = self.fake_report("hello")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bm.write_reports(output, report)
            for name in (
                "benchmark.json",
                "benchmark.csv",
                "benchmark-clips.csv",
                "benchmark-stress.csv",
                "benchmark.md",
                "benchmark.html",
                "commands.txt",
            ):
                self.assertTrue((output / name).is_file(), name)
            with (output / "benchmark.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["config_id"], "standard-1")
            self.assertEqual(rows[0]["corpus_wer"], "0.5")


class SummaryTests(unittest.TestCase):
    def test_summary_computes_wer_and_stress(self) -> None:
        corpus = {
            "by_id": {
                "c1": {"reference": "a b", "duration_seconds": 1.0},
            }
        }
        document = {
            "status": "ok",
            "engine_label": "Parakeet",
            "threads": {"effective": 1, "policy": "explicit"},
            "load": {"model_load_ms": 10, "first_decode_ms": 20, "first_chunk_ms": 7, "warmup_total_ms": 30},
            "memory": {
                "after_load": {"rss_bytes": 100, "peak_rss_bytes": 200},
                "at_end": {"rss_bytes": 150, "peak_rss_bytes": 250},
            },
            "runtime": {"package_version": "1", "model_bytes": 300, "model_files": [{}, {}]},
            "jfk": {
                "audio_seconds": 1.0,
                "runs": [{"run": 1, "status": "ok", "decode_ms": 100, "rtf": 0.1, "transcript": "a"}],
            },
            "quality": {
                "corpus": {},
                "runs": [
                    {
                        "clip_id": "c1",
                        "reference": "a b",
                        "transcript": "a x",
                        "status": "ok",
                        "duration_seconds": 1.0,
                        "decode_ms": 1000,
                    }
                ],
            },
            "stress": [
                {"case_id": "silence-5s", "kind": "silence", "transcript": "", "status": "ok"},
                {"case_id": "noise-5s", "kind": "noise", "transcript": "you", "status": "ok"},
                {"case_id": "short-speech", "kind": "short_speech", "transcript": "a b", "status": "ok"},
            ],
            "paced": None,
            "whole_vs_paused_jfk": None,
            "engine_errors": [],
        }
        record = {
            "config_id": "standard-1",
            "engine": "standard",
            "engine_label": "Parakeet",
            "threads": "1",
            "status": "ok",
            "document": document,
            "wall_ms": 1000,
            "cpu": {"total_seconds": 0.5},
        }
        summary = bm.build_summary(record, corpus)
        self.assertEqual(summary["quality"]["clips"][0]["substitutions"], 1)
        self.assertAlmostEqual(summary["quality"]["wer"], 0.5)
        self.assertEqual(summary["quality"]["reference_words"], 2)
        self.assertEqual(summary["stress"]["silence_false_words"], 0)
        self.assertEqual(summary["stress"]["noise_false_words"], 1)
        self.assertEqual(summary["stress"]["cases"][2]["recognized_words"], 2)
        self.assertIsNone(summary["stress"]["cases"][2]["false_words"])
        self.assertEqual(summary["memory"]["peak_rss_bytes"], 250)
        self.assertEqual(summary["first_chunk_ms"], 7)
        self.assertEqual(summary["runtime"]["file_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
