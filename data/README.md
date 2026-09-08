# Dataset access and deterministic split

The experiments use the public MNIST and Fashion-MNIST datasets through `torchvision.datasets.MNIST` and `torchvision.datasets.FashionMNIST`. No raw dataset files are redistributed in this package.

Download both datasets with the torchvision dataset classes and keep the downloaded files under a local data root. The scripts are run with `download=False` so that a reproduction does not silently change the dataset state during an experiment.

For each dataset:

- optimization subset: official training partition indices `0:10,000`;
- validation subset: official test partition indices `0:2,000`;
- final holdout: official test partition indices `2,000:10,000`;
- image shape: one channel, 28 x 28 pixels;
- classes: 10;
- normalization: mean `0.5`, standard deviation `0.5`.

The final holdout is evaluated only after validation-selected checkpoint loading. It is not used for optimization, checkpoint selection, finite-shot analysis, or depolarizing-noise analysis. The resulting 8,000-example split is a study-specific holdout and should not be described as the canonical full-test benchmark.

MNIST and Fashion-MNIST are third-party public datasets. The public repository record should cite their authoritative sources and the torchvision release used for reproduction. Do not apply a new licence to the redistributed source data.

The source environment contains torchvision `0.21.0+cu124`; record the exact release used for the public reproduction and update this line if a CPU-only or later torchvision build changes the downloaded archive or metadata.
