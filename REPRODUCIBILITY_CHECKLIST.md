# Reproducibility and public-release checklist

## Completed locally on 2026-09-08

- [x] Check required repository files and Python syntax with the dependency-free validator.
- [x] Confirm that raw MNIST/Fashion-MNIST files are not included.
- [x] Scan text files for common secret patterns and personal absolute paths.
- [x] Check the fixed-budget headline accuracies against the manuscript audit values.
- [x] Provide standardized Amplitude, Angle, IQP-All/IQP-NN, and Basis-STE training entry points using `download=False`.
- [x] Add a GitHub Actions workflow for repository integrity checks.
- [x] Generate a repository SHA256 manifest.
- [x] Separate smoke/full and IQP topology output directories and test metadata reuse.
- [x] Remove non-standalone legacy scripts and unverified checkpoint binaries from the public candidate.
- [x] Run all standardized entry points in a clean Python 3.12 CPU environment and verify both dataset loaders.
- [x] Create the private GitHub staging repository and push the reviewed `main` branch.
- [x] Replace the public commit identity with the GitHub noreply email.
- [x] License repository-authored contents under the MIT License.
- [x] Record the verified GitHub repository URL in both READMEs and `CITATION.cff`.

## Author decisions required before public release

- [ ] Confirm author names, order, affiliations, email addresses, and any ORCID identifiers.
- [ ] Decide whether a later release should redistribute verified checkpoints; the initial public candidate omits them.
- [ ] Create a public GitHub repository and a fixed release tag, preferably `v1.0.0`.
- [ ] Archive that exact release in Zenodo or another trusted repository and obtain a DOI.
- [ ] Test the DOI and repository while logged out.
- [ ] Add the final archival DOI to `CITATION.cff`, both READMEs, and the manuscript.
- [ ] Keep Data availability and Code availability synchronized with the files actually released.

## Final verification after any change

- [ ] Regenerate `SHA256SUMS.txt` with `python code/validation/generate_sha256_manifest.py`.
- [ ] Run `python code/validation/validate_release.py` and resolve all failures.
- [ ] Confirm that remaining warnings correspond only to deliberate, documented author decisions.
