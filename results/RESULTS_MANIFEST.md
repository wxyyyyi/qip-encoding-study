# Results manifest

The directories below are the processed result families used by the manuscript.

| Directory | Manuscript role |
|---|---|
| `final_results/` | Independent 8,000-example final-holdout summaries and figures |
| `main/` | Main metric consistency and per-run audit |
| `quantum_resources/` | Decomposed gate counts, depths, and parameter counts |
| `basis_budget_sensitivity/` | Fixed versus extended Basis-STE budgets |
| `basis_correction/` | Basis-Hard and Basis-STE correction analysis |
| `basis_information/` | Bit-pattern, entropy, drift, and gradient-flow diagnostics |
| `iqp_topology_ablation/` | IQP-All versus IQP-NN topology comparison |
| `fashion_robustness_combined/` | Frozen-checkpoint finite-shot and depolarizing-noise summaries |
| `baselines_validation_selected/` | Validation-selected classical-control metrics and logs; binary weights are excluded |

The package intentionally excludes smoke-test directories, raw downloaded datasets, and run-level robustness JSON archives. If an editor or reviewer requests those archives, add them as a versioned repository release asset and update this manifest.

Some processed CSV and JSON records retain historical `checkpoint` strings from the private source tree. These strings document which source run was summarized; they are not package-local download paths, and the corresponding binaries are not distributed in this candidate.

## Reproduction boundary

The files above are publication records and can be audited directly. The standardized experiment entry points write new outputs to `runs/`, which is intentionally excluded from version control. Historical report builders that depended on omitted run-level archives remain in a local private provenance archive and are not represented as standalone public code.

The fixed-budget headline accuracy values are checked by `code/validation/validate_release.py`. Extended-budget Basis-STE results remain a separate sensitivity analysis and must not replace the fixed-budget rows in `final_results/final_summary.csv`.
