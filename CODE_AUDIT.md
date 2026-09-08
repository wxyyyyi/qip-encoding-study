# Code audit for the public release candidate

Audit date: 2026-09-08

## Release decision

The original compact package preserved the reported evidence, but it was not
ready to publish as code. The public candidate was reduced to standalone entry
points and given dependency-free regression checks before repository creation.

## Blocking findings corrected

- IQP-All and IQP-NN previously shared the same per-seed paths. Existing
  IQP-All metadata could therefore be mistaken for an IQP-NN run, and `--force`
  could overwrite the other topology. Output roots now include the topology.
- Smoke tests previously used the same paths as full-protocol runs. Every
  trainer now separates `smoke/` and `full/` results.
- Existing `metrics.json` files were skipped without checking their identity.
  Dataset, seed, model or encoding, run mode, and IQP topology are now checked
  before reuse.
- Aggregate CSV files previously contained only the tasks selected in the most
  recent command. They are now rebuilt from every completed run in the active
  result tree, so partial invocations do not erase earlier rows.
- Historical analysis and evaluation scripts referenced directories absent
  from the compact package. They were moved to a local private provenance
  archive instead of being presented as runnable public code.
- Legacy checkpoint files were incomplete and did not share a verified loading
  contract with the standardized trainers. They were moved to a local private
  artifact archive pending an explicit author redistribution decision.

## Remaining intentional duplication

The quantum trainers retain separate model definitions. This makes each
encoding circuit auditable in one file and avoids changing scientific behavior
through a late, high-risk architecture refactor. Shared result-directory and
metadata logic is centralized in `code/experiments/_result_store.py`.

## Automated acceptance checks

Run:

```bash
python code/validation/validate_release.py
python -m unittest discover -s code/validation -p "test_*.py" -v
python -m compileall -q code
```

The repository workflow runs the same dependency-free checks on every push and
pull request. Scientific smoke tests additionally require the pinned packages
and locally downloaded MNIST/Fashion-MNIST data.

## Acceptance results

- Ten public Python files, approximately 2,514 lines; the original candidate
  contained 19 files and approximately 5,159 lines.
- Python compilation: pass.
- Pyflakes unused/undefined-name scan: zero findings.
- Dependency-free regression tests: 4/4 pass.
- Clean Python 3.12 CPU environment: PennyLane 0.44.1, PyTorch 2.6.0,
  torchvision 0.21.0, and scikit-learn 1.8.0 installed successfully.
- Real one-epoch smoke runs: classical bottleneck, Amplitude, Angle, IQP-All,
  IQP-NN, and Basis-STE all completed checkpoint save/load and aggregation.
- MNIST and Fashion-MNIST local data loading: pass.
- Resource-analysis regeneration: all four generated files are byte-identical
  to the committed publication records.
- Maintainability index: nine files grade A and Basis-STE grade B. The remaining
  C-complexity functions are the explicit training and circuit-accounting
  procedures documented above, not hidden publication blockers.
