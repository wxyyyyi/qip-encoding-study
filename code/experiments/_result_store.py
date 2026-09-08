"""Small, dependency-free helpers for experiment result directories."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path


class ResultMetadataError(ValueError):
    """Raised when an existing result does not describe the requested run."""


def run_mode_root(output_root: Path, smoke_test: bool) -> Path:
    """Keep quick execution checks separate from full-protocol results."""
    return Path(output_root) / ("smoke" if smoke_test else "full")


def iqp_pattern_slug(pattern_name: str) -> str:
    """Return the stable directory name used for an IQP interaction pattern."""
    slugs = {
        "all-to-all": "all_to_all",
        "nearest-neighbor": "nearest_neighbor",
    }
    try:
        return slugs[pattern_name]
    except KeyError as exc:
        raise ValueError(f"Unknown IQP pattern: {pattern_name}") from exc


def read_matching_metrics(path: Path, expected: Mapping[str, object]) -> dict:
    """Load a result only when its identifying metadata matches the request."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultMetadataError(f"Cannot read result metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultMetadataError(f"Result metadata must be a JSON object: {path}")

    mismatches = []
    for key, expected_value in expected.items():
        actual_value = payload.get(key, "<missing>")
        if actual_value != expected_value:
            mismatches.append(f"{key}={actual_value!r} (expected {expected_value!r})")
    if mismatches:
        details = ", ".join(mismatches)
        raise ResultMetadataError(f"Existing result does not match requested run: {details}; {path}")
    return payload


def collect_metrics(
    root: Path,
    required_fields: Iterable[str],
    identity_fields: Iterable[str],
) -> list[dict]:
    """Collect and deterministically order all completed runs below ``root``."""
    root = Path(root)
    required = tuple(required_fields)
    identity = tuple(identity_fields)
    rows = []
    seen = set()
    for path in sorted(root.rglob("metrics.json"), key=lambda item: item.as_posix()):
        row = read_matching_metrics(path, {})
        missing = [field for field in required if field not in row]
        if missing:
            raise ResultMetadataError(
                f"Result metadata is missing {', '.join(missing)}: {path}"
            )
        key = tuple(row.get(field) for field in identity)
        if key in seen:
            raise ResultMetadataError(
                f"Duplicate result identity {dict(zip(identity, key))}: {path}"
            )
        seen.add(key)
        rows.append(row)
    return sorted(rows, key=lambda row: tuple(row[field] for field in identity))
