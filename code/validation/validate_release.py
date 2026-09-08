"""Run dependency-free integrity checks for the public release package."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "README.md",
    "README_zh.md",
    "CITATION.cff",
    "REPRODUCIBILITY_CHECKLIST.md",
    "configs/protocol.md",
    "data/README.md",
    "results/RESULTS_MANIFEST.md",
    "results/final_results/final_summary.csv",
    "code/experiments/continuous_encoding_experiment.py",
    "code/experiments/iqp_all_experiment.py",
    "code/experiments/basis_ste_experiment.py",
    "code/experiments/_result_store.py",
    "code/validation/test_repository.py",
)
EXPECTED_ACCURACY = {
    ("mnist", "amplitude"): (0.982625, 0.000976281209488338),
    ("mnist", "angle"): (0.97625, 0.0024109126902482265),
    ("mnist", "iqp-all"): (0.9755, 0.00275),
    ("mnist", "basis-ste"): (0.5014583333333333, 0.09250045044935366),
    ("mnist", "iqp-nn"): (0.9756666666666667, 0.0004018187817080474),
    ("fashion", "amplitude"): (0.88125, 0.002260392665003168),
    ("fashion", "angle"): (0.8817083333333334, 0.002525907427704623),
    ("fashion", "iqp-all"): (0.869625, 0.0025093574874856276),
    ("fashion", "basis-ste"): (0.598625, 0.07651358131077121),
    ("fashion", "iqp-nn"): (0.8724583333333333, 0.003923752455664485),
}
TEXT_SUFFIXES = {
    ".cff", ".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:\\(?:Users|home)\\"),
    re.compile(r"/(?:Users|home)/[^/\s]+/"),
)
LEGACY_CODE_TERMS = (
    "paper_analysis",
    "Amplitude encoding",
    "angle encoding",
    "IQP encoding",
    "basis encoding",
    "fashion-MNIST",
)


class Report:
    def __init__(self):
        self.failures = []
        self.warnings = []
        self.passes = []

    def fail(self, message):
        self.failures.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def ok(self, message):
        self.passes.append(message)


def iter_text_files():
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", "SHA256SUMS"}:
            yield path


def check_required(report):
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        report.fail("Missing required files: " + ", ".join(missing))
    else:
        report.ok(f"Required file set present ({len(REQUIRED)} files)")


def check_python(report):
    errors = []
    files = sorted(ROOT.rglob("*.py"))
    for path in files:
        try:
            source = path.read_text(encoding="utf-8-sig")
            compile(source, str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    if errors:
        report.fail("Python compile failures: " + " | ".join(errors))
    else:
        report.ok(f"Python syntax valid ({len(files)} files)")


def check_raw_data(report):
    forbidden_dirs = [ROOT / "data" / name for name in ("raw", "MNIST", "FashionMNIST")]
    found = [str(path.relative_to(ROOT)) for path in forbidden_dirs if path.exists()]
    extensions = {".gz", ".ubyte", ".idx", ".npz"}
    found.extend(
        str(path.relative_to(ROOT))
        for path in (ROOT / "data").rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if found:
        report.fail("Raw dataset payload found: " + ", ".join(found))
    else:
        report.ok("No raw MNIST/Fashion-MNIST payload found")


def check_sensitive_text(report):
    secrets = []
    private_paths = []
    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            continue
        relative = str(path.relative_to(ROOT))
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            secrets.append(relative)
        if any(pattern.search(text) for pattern in PRIVATE_PATH_PATTERNS):
            private_paths.append(relative)
    if secrets:
        report.fail("Possible secret material in: " + ", ".join(secrets))
    else:
        report.ok("No common secret patterns found")
    if private_paths:
        report.fail("Possible private absolute paths in: " + ", ".join(private_paths))
    else:
        report.ok("No personal absolute paths found")


def check_standalone_code_paths(report):
    affected = []
    for path in (ROOT / "code").rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8-sig")
        if any(term in text for term in LEGACY_CODE_TERMS):
            affected.append(str(path.relative_to(ROOT)))
    if affected:
        report.fail("Code still depends on legacy project paths: " + ", ".join(affected))
    else:
        report.ok("Tracked code has no legacy project-directory dependencies")


def check_main_results(report):
    path = ROOT / "results" / "final_results" / "final_summary.csv"
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["dataset"], row["model"]): row for row in rows}
    errors = []
    for key, (expected_mean, expected_sd) in EXPECTED_ACCURACY.items():
        row = indexed.get(key)
        if row is None:
            errors.append(f"missing {key[0]}/{key[1]}")
            continue
        actual_mean = float(row["final_test_accuracy_mean"])
        actual_sd = float(row["final_test_accuracy_sd"])
        if abs(actual_mean - expected_mean) > 1e-12 or abs(actual_sd - expected_sd) > 1e-12:
            errors.append(
                f"{key[0]}/{key[1]} expected ({expected_mean}, {expected_sd}) "
                f"but found ({actual_mean}, {actual_sd})"
            )
    if errors:
        report.fail("Fixed main-result mismatch: " + " | ".join(errors))
    else:
        report.ok("Fixed-budget main accuracy values match the manuscript audit")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest(report):
    path = ROOT / "SHA256SUMS.txt"
    if not path.is_file():
        report.warn("SHA256SUMS.txt has not been generated")
        return
    errors = []
    entries = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = ROOT / Path(relative)
        entries += 1
        if not target.is_file():
            errors.append(f"missing {relative}")
        elif sha256(target) != expected:
            errors.append(f"hash mismatch {relative}")
    if errors:
        report.fail("SHA256 manifest errors: " + " | ".join(errors[:10]))
    else:
        report.ok(f"SHA256 manifest verified ({entries} files)")


def check_release_decisions(report):
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    combined = citation + "\n" + (ROOT / "README.md").read_text(encoding="utf-8")
    if "[PERMANENT DOI OR URL]" in combined or "TO BE ADDED" in combined:
        report.warn("Persistent repository URL/DOI is still unresolved")
    if "repository-code:" not in citation:
        report.warn("CITATION.cff intentionally has no repository URL before publication")
    if not any((ROOT / name).is_file() for name in ("LICENSE", "LICENSE.txt", "LICENSE.md")):
        report.warn("Software license has not been selected by the authors")
    checkpoint_files = list(ROOT.rglob("*.pth"))
    if checkpoint_files:
        report.fail(
            "Binary checkpoints are present although redistribution is unresolved: "
            + ", ".join(str(path.relative_to(ROOT)) for path in checkpoint_files[:5])
        )
    else:
        report.ok("No unapproved checkpoint binaries included")


def main():
    report = Report()
    check_required(report)
    check_python(report)
    check_raw_data(report)
    check_sensitive_text(report)
    check_standalone_code_paths(report)
    check_main_results(report)
    check_manifest(report)
    check_release_decisions(report)
    for message in report.passes:
        print(f"PASS: {message}")
    for message in report.warnings:
        print(f"WARN: {message}")
    for message in report.failures:
        print(f"FAIL: {message}")
    print(
        f"SUMMARY: {len(report.passes)} pass, "
        f"{len(report.warnings)} warning, {len(report.failures)} failure"
    )
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
