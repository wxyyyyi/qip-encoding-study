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
- [x] Create the GitHub staging repository and push the reviewed `main` branch.
- [x] Replace the public commit identity with the GitHub noreply email.
- [x] License repository-authored contents under the MIT License.
- [x] Record the verified GitHub repository URL in both READMEs and `CITATION.cff`.
- [x] Make the reviewed repository public.
- [x] Enable the repository in Zenodo's GitHub integration.
- [x] Confirm that the initial release omits unverified legacy checkpoints.
- [x] Confirm author order, affiliations, corresponding author email, release date, and omission of unverified ORCID identifiers.
- [x] Add version-aligned `CITATION.cff` and `.zenodo.json` metadata for `v1.0.0`.
- [x] Regenerate `SHA256SUMS.txt` and complete the final local verification.

## Versioned archive and DOI follow-up

- [x] Create and push the fixed `v1.0.0` release tag.
- [x] Archive that exact release in Zenodo and obtain DOI `10.5281/zenodo.22658398`.
- [x] Test the public DOI record, repository, and archived download without Zenodo account credentials.
- [x] Add the final archival DOI to `CITATION.cff`, both READMEs, and the manuscript.
- [x] Keep Data availability and Code availability synchronized with the files actually released.

## Verification required after future changes

- [ ] Regenerate `SHA256SUMS.txt` with `python code/validation/generate_sha256_manifest.py`.
- [ ] Run `python code/validation/validate_release.py` and resolve all failures.
- [ ] Confirm that remaining warnings correspond only to deliberate, documented author decisions.
