# Analysis code

`quantum_resource_analysis.py` reconstructs the circuit variants and writes the
resource-accounting tables to `results/quantum_resources/`.

The compact public package intentionally omits historical report builders that
depended on run-level archives and legacy directory names not included here.
Their processed, manuscript-facing outputs remain under `results/` and are
protected by `SHA256SUMS.txt` and the fixed-value checks in
`code/validation/validate_release.py`.
