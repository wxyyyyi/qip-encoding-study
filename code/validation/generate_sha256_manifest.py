"""Generate a deterministic SHA256 manifest for repository files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "SHA256SUMS.txt"
EXCLUDED_PARTS = {".git", "__pycache__", "runs", "tmp"}
EXCLUDED_SUFFIXES = {".pth"}
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
TEXT_NAMES = {".gitattributes", ".gitignore", "LICENSE"}


def sha256(path):
    digest = hashlib.sha256()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == OUTPUT or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def main():
    lines = []
    for path in sorted(included_files(), key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{sha256(path)}  {relative}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
