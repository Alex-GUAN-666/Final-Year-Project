"""Deterministic question identities and source readers; no model execution."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("$Calculate#", "Calculate")
    return re.sub(r"\s+", " ", text).strip().casefold()


def question_id(text: str) -> str:
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_prepared_manifest(folder: Path, required_file: str) -> dict:
    """Require the consumed dataset's checksum; an empty manifest is not verification."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("prepared_hashes") if isinstance(manifest, dict) else None
    if not isinstance(hashes, dict) or required_file not in hashes:
        raise ValueError(f"Prepared manifest must include a checksum for {required_file}.")
    for name, expected in hashes.items():
        if (not isinstance(name, str) or not name or Path(name).name != name
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("Invalid prepared file checksum entry.")
        if file_hash(folder / name) != expected:
            raise ValueError(f"Prepared file hash mismatch: {name}; regenerate preparation after edits.")
    return manifest


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                raise ValueError(f"{path.name}: empty row {line}")
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}: row {line} is not an object")
            records.append(value)
    return records


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        records = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in records):
        raise ValueError(f"{path.name}: malformed CSV row")
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
