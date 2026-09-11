"""Small independent contracts that do not download a model or require ML packages."""
import json
from pathlib import Path
import tempfile
import unittest

from fyp.merge import resolve_base
from fyp.data import file_hash
from fyp.train import pad_examples, read_candidates


class TrainingContractTests(unittest.TestCase):
    def test_real_eos_is_supervised_when_eos_equals_padding(self):
        batch = pad_examples([{"input_ids": [3, 9, 4, 9]}, {"input_ids": [5, 9]}], pad_token_id=9)
        self.assertEqual(batch["labels"], [[3, 9, 4, 9], [5, 9, -100, -100]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(batch["input_ids"][1], [5, 9, 9, 9])

    def test_data_must_be_prepared_and_ids_unique(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            with self.assertRaises(FileNotFoundError):
                read_candidates(folder)
            row = {"id": "repeated", "messages": [{"role": role, "content": "text"} for role in ("system", "user", "assistant")]}
            (folder / "train_candidates.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row))
            (folder / "manifest.json").write_text(json.dumps({"prepared_hashes": {
                "train_candidates.jsonl": file_hash(folder / "train_candidates.jsonl")}}))
            with self.assertRaisesRegex(ValueError, "unique"):
                read_candidates(folder)

    def test_missing_or_changed_training_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            row = {"id": "one", "messages": [{"role": role, "content": "text"}
                   for role in ("system", "user", "assistant")]}
            data = folder / "train_candidates.jsonl"
            data.write_text(json.dumps(row) + "\n")
            for manifest in ({}, {"prepared_hashes": {}}, {"prepared_hashes": []}):
                (folder / "manifest.json").write_text(json.dumps(manifest))
                with self.subTest(manifest=manifest), self.assertRaisesRegex(ValueError, "checksum"):
                    read_candidates(folder)
            (folder / "manifest.json").write_text(json.dumps({"prepared_hashes": {
                data.name: file_hash(data)}}))
            self.assertEqual(len(read_candidates(folder)[0]), 1)
            data.write_text(json.dumps(row).replace('text', 'changed') + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_candidates(folder)

    def test_merge_rejects_base_and_revision_substitutions(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B-Base", "revision": "abc123"}))
            (folder / "reproduction_manifest.json").write_text(json.dumps({"status": "completed", "base_model": "Qwen/Qwen3-4B-Base", "resolved_revision": "abc123", "requested_revision": "main"}))
            self.assertEqual(resolve_base(folder)[1], "abc123")
            with self.assertRaisesRegex(ValueError, "model"):
                resolve_base(folder, model="different/model")
            with self.assertRaisesRegex(ValueError, "revision"):
                resolve_base(folder, revision="main")


if __name__ == "__main__":
    unittest.main()
