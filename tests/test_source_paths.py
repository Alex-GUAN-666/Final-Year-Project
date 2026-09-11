"""Source renaming must preserve recorded identities and immutable payloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fyp.sources import ROOT, source_path


class SourcePathTests(unittest.TestCase):
    def test_every_source_alias_retains_the_declared_publication_hash(self):
        entries = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
        bundled = [entry for entry in entries if entry.get("repository_path")]
        self.assertEqual(len(bundled), 18)
        renamed = [entry for entry in bundled if entry.get("previous_repository_path")]
        self.assertEqual(len(renamed), 11)
        for entry in bundled:
            aliases = [entry["file"], entry["repository_path"], Path(entry["repository_path"]).name]
            if entry.get("previous_repository_path"):
                aliases.append(entry["previous_repository_path"])
                self.assertFalse((ROOT / entry["previous_repository_path"]).exists())
            for alias in aliases:
                with self.subTest(alias=alias):
                    path = source_path(alias)
                    self.assertEqual(path, ROOT / entry["repository_path"])
                    payload = path.read_bytes()
                    self.assertEqual(len(payload), entry.get("published_bytes", entry["bytes"]))
                    self.assertEqual(hashlib.sha256(payload).hexdigest(),
                                     entry.get("published_sha256", entry["sha256"]))

    def test_duplicate_exports_remain_separate_files_with_original_identities(self):
        for first, second in (
            ("evaluation_results_base_cot_v5(1).csv", "evaluation_results_merged_cot_v5(1).csv"),
            ("Qwen3_(4B)_SFFT_code_0811.ipynb", "Qwen3_(4B)_SFFT_code_0811(2).ipynb"),
        ):
            with self.subTest(first=first):
                left, right = source_path(first), source_path(second)
                self.assertNotEqual(left, right)
                self.assertEqual(left.read_bytes(), right.read_bytes())

    def test_unknown_alias_never_falls_back_to_an_unregistered_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source_manifest.json").write_text("[]", encoding="utf-8")
            (root / "unregistered.csv").write_text("some data", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                source_path("unregistered.csv", root)

    def test_ambiguous_alias_is_rejected_instead_of_selecting_first_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = [
                {"file": "first.csv", "repository_path": "a/shared.csv"},
                {"file": "second.csv", "repository_path": "b/shared.csv"},
            ]
            (root / "source_manifest.json").write_text(json.dumps(entries), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                source_path("shared.csv", root)

    def test_manifest_paths_cannot_escape_the_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for unsafe in ("../outside.csv", str(root.parent / "outside.csv")):
                entries = [{"file": "original.csv", "repository_path": unsafe}]
                (root / "source_manifest.json").write_text(json.dumps(entries), encoding="utf-8")
                with self.subTest(unsafe=unsafe), self.assertRaisesRegex(ValueError, "inside the repository"):
                    source_path("original.csv", root)


if __name__ == "__main__":
    unittest.main()
