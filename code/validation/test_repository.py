"""Dependency-free regression tests for repository result handling."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENT_DIR))

from _result_store import (  # noqa: E402
    ResultMetadataError,
    collect_metrics,
    iqp_pattern_slug,
    read_matching_metrics,
    run_mode_root,
)


class ResultStoreTests(unittest.TestCase):
    def test_smoke_and_full_roots_are_distinct(self):
        root = Path("runs")
        self.assertEqual(run_mode_root(root, True), root / "smoke")
        self.assertEqual(run_mode_root(root, False), root / "full")

    def test_iqp_topologies_have_distinct_directories(self):
        self.assertEqual(iqp_pattern_slug("all-to-all"), "all_to_all")
        self.assertEqual(iqp_pattern_slug("nearest-neighbor"), "nearest_neighbor")
        self.assertNotEqual(
            iqp_pattern_slug("all-to-all"),
            iqp_pattern_slug("nearest-neighbor"),
        )

    def test_existing_metadata_must_match_requested_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_text(
                json.dumps({"dataset": "mnist", "seed": 42, "iqp_pattern": "all_to_all"}),
                encoding="utf-8",
            )
            with self.assertRaises(ResultMetadataError):
                read_matching_metrics(
                    path,
                    {"dataset": "mnist", "seed": 42, "iqp_pattern": "nearest_neighbor"},
                )

    def test_partial_runs_are_merged_into_aggregate_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (456, 42, 123):
                path = root / "mnist" / f"seed_{seed}" / "metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps({"dataset": "mnist", "model": "Basis-STE", "seed": seed}),
                    encoding="utf-8",
                )
            rows = collect_metrics(
                root,
                required_fields=("dataset", "model", "seed"),
                identity_fields=("dataset", "seed"),
            )
            self.assertEqual([row["seed"] for row in rows], [42, 123, 456])


if __name__ == "__main__":
    unittest.main()
