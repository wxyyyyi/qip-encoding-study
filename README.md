# Reproducibility package for Quantum Data Encoding Trade-offs

This is the public reproducibility repository for the manuscript *Quantum Data Encoding Trade-offs in Parallel Hybrid Image Classification*. It contains standardized training entry points, processed results, figures, and the fixed data-split protocol.

The reviewed package is publicly available at [wxyyyyi/qip-encoding-study](https://github.com/wxyyyyi/qip-encoding-study), and the repository has been enabled in Zenodo's GitHub integration. A versioned archival DOI has not yet been issued. Repository-authored contents are available under the MIT License. Unverified legacy checkpoints are excluded from the initial public release.

## Reproducibility scope

The package supports two distinct tasks:

1. **Audit the reported evidence.** The processed tables and figures in `results/` are the records used by the manuscript. Run `python code/validation/validate_release.py` to check their fixed-budget headline values, Python syntax, raw-data exclusion, common secret patterns, and the checksum manifest.
2. **Rerun the standardized models.** The scripts below use the same deterministic split, optimizer settings, epoch budgets, seeds, four-branch architecture, and validation-selected checkpoint policy described in `configs/protocol.md`.

The processed result tables are the manuscript-facing publication records. Historical report builders that required omitted run-level archives are retained privately rather than presented here as standalone code. The exact public-package boundary is recorded in `results/RESULTS_MANIFEST.md` and `CODE_AUDIT.md`.

## Contents

- `code/experiments/continuous_encoding_experiment.py`: standardized Amplitude and Angle training.
- `code/experiments/iqp_all_experiment.py`: standardized IQP-All or IQP-NN topology-control training.
- `code/experiments/basis_ste_experiment.py`: standardized Basis-STE training and diagnostics.
- `code/experiments/classical_bottleneck_baseline.py`: validation-selected classical bottleneck controls.
- `code/analysis/quantum_resource_analysis.py`: standalone circuit resource accounting.
- `code/validation/`: release checks, checksum generation, and the Amplitude input-gradient audit.
- `CODE_AUDIT.md`: public-code findings, corrections, and acceptance evidence.
- `LICENSE`: MIT terms for repository-authored contents.
- `results/`: processed CSV, JSON, Markdown, and manuscript result figures.
- `figures/`: copies of the principal manuscript figures.
- `checkpoints/README.md`: explains why unverified legacy weights are not part of this candidate.
- `data/README.md`: dataset source and deterministic split information; raw datasets are not included.

## Environment

The source result metadata record PennyLane `0.44.1`, PyTorch `2.6.0+cu124`, and torchvision `0.21.0+cu124`. The pinned `requirements.txt` and `environment.yml` describe the intended environment. A CPU-only PyTorch build can be used, but full quantum training is computationally expensive.

The scripts use `download=False`. Download MNIST and Fashion-MNIST separately as described in `data/README.md`, then provide the containing directory with `--data-root`.

## Quick validation

```bash
python code/validation/validate_release.py
python -m unittest discover -s code/validation -p "test_*.py" -v
python code/validation/generate_sha256_manifest.py
python code/validation/validate_release.py
```

The first command uses only the Python standard library. It can therefore validate the package before scientific dependencies are installed. SHA256 values normalize text-file CRLF line endings to LF so the manifest is stable across Windows and Linux; binary files are hashed byte for byte. The pending archival DOI is a documented release task and is not treated as a local-integrity failure.

## Standardized smoke tests

```bash
python code/experiments/continuous_encoding_experiment.py --encodings amplitude angle --datasets mnist --seeds 42 --smoke-test --data-root path/to/data
python code/experiments/iqp_all_experiment.py --datasets mnist --seeds 42 --smoke-test --data-root path/to/data
python code/experiments/basis_ste_experiment.py --datasets mnist --seeds 42 --smoke-test --data-root path/to/data
```

Smoke-test outputs and full reruns are written to separate `smoke/` and `full/` trees below `runs/`, which is excluded from version control. IQP-All and IQP-NN also use separate topology directories. A smoke test verifies execution, not numerical reproduction of the reported full-budget results.

## Full fixed-budget runs

The defaults run MNIST for 10 epochs, Fashion-MNIST for 25 epochs, and seeds `42 123 456`:

```bash
python code/experiments/continuous_encoding_experiment.py --data-root path/to/data
python code/experiments/iqp_all_experiment.py --data-root path/to/data
python code/experiments/basis_ste_experiment.py --data-root path/to/data
python code/experiments/classical_bottleneck_baseline.py --data-root path/to/data
```

The custom evaluation protocol uses official training indices `0:10000`, official test indices `0:2000` for validation, and official test indices `2000:10000` as the independent final holdout. This is not the canonical full-test benchmark.

These scripts are standardized rerun entry points reconstructed from the fixed protocol and source implementations. They are not claimed to reproduce every reported floating-point value exactly across hardware, package builds, batching modes, or historical random-number consumption.

## Interpretation boundary

The evidence supports an architecture-specific, resource-aware comparison of encoding pipelines. It does not establish quantum advantage, hardware robustness, a universal encoding ranking, or a causal effect attributable to the straight-through estimator alone.

## Release and citation

The reviewed repository is public and enabled in Zenodo. The remaining release steps are to create the fixed `v1.0.0` release, allow Zenodo to archive that exact release, verify the resulting DOI, and then insert the permanent DOI into the manuscript, both READMEs, and `CITATION.cff`. No DOI is stated before Zenodo issues and resolves it.

Repository-authored contents are licensed under the MIT License. MNIST, Fashion-MNIST, and all third-party dependencies retain their own licenses and citation requirements; their raw data and source code are not redistributed here.
