"""Resolve original and current source names through the provenance manifest."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_path(identifier: str, root: Path = ROOT) -> Path:
    """Find one bundled source without changing names recorded in old evidence.

    Original upload names, previous repository paths, and current paths/names
    are aliases for the same manifest entry. Unknown or ambiguous aliases fail;
    there is no fallback that could silently select a different source file.
    """
    root = Path(root).resolve()
    entries = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(identifier, str) or not identifier or not isinstance(entries, list):
        raise ValueError("Source identifier and manifest must be valid")
    matches = []
    for entry in entries:
        relative = entry.get("repository_path")
        if relative is None:
            continue
        previous = entry.get("previous_repository_path")
        aliases = {entry["file"], relative, Path(relative).name}
        if previous:
            aliases.update((previous, Path(previous).name))
        if identifier in aliases:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(f"Source identifier must resolve to exactly one bundled file: {identifier!r}")
    relative = Path(matches[0]["repository_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest source path must stay inside the repository")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Manifest source path must stay inside the repository")
    if not resolved.is_file():
        raise FileNotFoundError(f"Bundled source is missing: {relative}")
    return resolved
