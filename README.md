# Universal scale-free representations in human visual cortex

This repository contains code required to reproduce the experimental results [published in PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1013714) and as [a preprint on arXiv](https://arxiv.org/abs/2409.06843v1).

## System requirements

### Hardware

These analyses were run on a workstation with

- Processor: [13th Gen Intel(R) Core(TM) i9-13900K](https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html)
- RAM: 128 GB
- GPU: [NVIDIA GeForce RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) (24 GB VRAM)
- Storage: 2 TB NVMe SSD

If you plan to reproduce these results, I'd recommend using a machine with

- \>= 64 GB RAM
- a GPU with >= 8 GB VRAM
- \>= 1.5 TB of free disk space

> [!CAUTION]
> RAM usage peaks when preparing the datasets for use the first time: we concatenate 40 sessions of fMRI data (!) for each subject in the Natural Scenes Dataset, which uses almost 64 GB of memory.

> [!TIP]
> Throughout the code, there are `batch_size` parameters that control the amount of GPU memory used. You might want to adjust this especially if you run permutation tests or bootstrap resampling when computing covariance spectra.

### Software

The code has been tested on RHEL 9.3 (Linux 5.14.0-503.23.1.el9_5.x86_64). Any standard Linux distribution should work. Other operating systems are not supported.

We use Python 3.13.7 for all analyses; the other required Python dependencies are specified in `uv.lock`. Do not attempt to install them directly; follow the installation guide below.

## Installation (~5 min)

1. Clone this repository.

   ```shell
    git clone https://github.com/BonnerLab/scale-free-visual-cortex.git
   ```

2. Edit `.env`.
    - `AWS_SHARED_CREDENTIALS_FILE` should be the path to an [AWS credentials file](https://docs.aws.amazon.com/sdkref/latest/guide/file-location.html) that gives you [access to the Natural Scenes Dataset](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform).
    - `PROJECT_HOME` controls where figures and results are saved. If you leave this unset, it will default to `~/scale-free-visual-cortex`.
    - The other environment variables can be left unset: simply delete those lines.

3. Set up the Python environment.
   - Install the [uv](https://docs.astral.sh/uv/) package manager.
   - Run `uv sync --no-sources --extra {cpu, cu126, cu128, rocm}` from `$PROJECT_HOME` to create the environment at `$PROJECT_HOME/.venv`.
        - If you have no GPU, use `cpu`.
        - If you have an Nvidia GPU, use `cu126`, `cu128`, or `cu129` depending on your driver version.
        - If you have an AMD GPU, use `rocm` (note that this hasn't been tested).
   - Activate the environment (e.g. `source $PROJECT_HOME/.venv/bin/activate` if you're using `bash`)

## Demo (~5 min)

We provide a simple high-level overview of the analysis in `demo.ipynb` using a small subset of the data. After installing this package, simply run the notebook file: this will automatically download ~300 MB of data and run the within- and between-subject analyses for one pair of subjects.

## Reproducing the analyses

> [!IMPORTANT]
> You will need [access to the Natural Scenes Dataset](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform) to reproduce the analyses in the paper. Specifically, you will need to obtain an AWS credentials file and set the `AWS_SHARED_CREDENTIALS_FILE` environment variable (see Step 2 of Installation).

`manuscript/notebooks` contains Jupyter notebooks that generate the figures in the paper, PDFs of which are located in `manuscript/figures`:

| Figure | PDF                                                             | Notebook                     |
| -------|-----------------------------------------------------------------|------------------------------|
| 1      | `schematic.pdf`                                                 | `schematic.ipynb`            |
| 2      | `general.pdf`                                                   | `general-region.ipynb`       |
| 3      | `significance-test.pdf`                                         | `significance-test.ipynb`    |
| 4      | `cross-correlations.pdf`                                        | `cross-similarity.ipynb`     |
| 5      | `visual-regions.pdf`                                            | `visual-regions.ipynb`       |
| 6      | `between-region-heatmaps.pdf`                                   | `visual-regions.ipynb`       |
| 7      | `rsa.pdf`                                                       | `rsa.ipynb`                  |
| S1     | `vary-n-voxels.pdf`                                             | `vary-n-voxels.ipynb`        |
| S2     | `vary-n-stimuli.pdf`                                            | `vary-n-stimuli.ipynb`       |
| S3     | `cross-detectability.pdf`                                       | `cross-similarity.ipynb`     |
| S4     | `gabor-model.pdf`                                               | `gabor-filter-bank.ipynb`    |
| S5     | `between-region-spectra.pdf`                                    | `visual-regions.ipynb`       |
| S6     | `general-all.pdf`                                               | `general-region.ipynb`       |
| S7     | `cross-correlations-all.pdf`                                    | `cross-similarity.ipynb`     |
| S8     | `singular-vectors.pdf`                                          | `singular-vectors.ipynb`     |
| S9     | `dimensions-V1.pdf` and `dimensions-V1-synthetic.pdf`           | `visualize-dimensions.ipynb` |
| S10    | `dimensions-general.pdf` and `dimensions-general-synthetic.pdf` | `visualize-dimensions.ipynb` |
| S11    | `things.pdf`                                                    | `things.ipynb`               |
| S12    | `between-monkey.pdf`                                            | `tvsd.ipynb`                 |
| S13    | `dimensions-general-semantic.pdf`                               | `visualize-dimensions.ipynb` |
| S14    | `things-dimensions.pdf`                                         | `things.ipynb`               |

> [!WARNING]
> Running a notebook for the first time will likely take ages since the datasets used will be downloaded, processed and cached. Subsequent runs will be much faster.

The manuscript and responses to reviewers can be found in `manuscript`, with compiled PDFs (`.pdf`) and source files in Markdown (`.qmd`) and LaTeX (`.tex`) format.

## Citation

If you find this work useful in your own research, please cite our manuscript [published in PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1013714).

A BibTeX entry is provided below for your convenience.

### BibTeX

```bibtex
@article{Gauthaman2025,
  title = {Universal scale-free representations in human visual cortex},
  volume = {21},
  ISSN = {1553-7358},
  url = {http://dx.doi.org/10.1371/journal.pcbi.1013714},
  DOI = {10.1371/journal.pcbi.1013714},
  number = {11},
  journal = {PLOS Computational Biology},
  publisher = {Public Library of Science (PLoS)},
  author = {Gauthaman,  Raj Magesh and Ménard,  Brice and Bonner,  Michael F.},
  editor = {Christian Kietzmann,  Tim},
  year = {2025},
  month = nov,
  pages = {e1013714}
}
```

## Contributing

If you run into any trouble, notice any bugs, or have any feedback, please create an issue in this repository or contact me by email at [science@raj-magesh.org](mailto:science@raj-magesh.org)! Thanks for your interest in our work :)
