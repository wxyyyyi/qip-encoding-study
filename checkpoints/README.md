# Checkpoint policy

The public `v1.0.0` release does not contain binary checkpoints. The legacy files were incomplete for the full model-by-dataset-by-seed matrix, and their state-dictionary names were not verified against the standardized training entry points. Presenting them as reusable weights would therefore be misleading.

The original files remain in a local private artifact archive. They may be added to a later versioned release only after every file has a model/dataset/seed manifest, loads through a documented entry point with `weights_only=True`, and has author and institutional approval for redistribution. The manuscript Code availability statement must not claim that model checkpoints are public while they are absent.
