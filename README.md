# Universal scale-free representations in human visual cortex

This repository contains code required to reproduce the experimental results published as [a preprint on arXiv](https://arxiv.org/abs/2409.06843v1).

## System requirements

### Hardware

These analyses were run on a workstation with

- Processor: [13th Gen Intel(R) Core(TM) i9-13900K](https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html)
- RAM: 128 GB
- GPU: [NVIDIA GeForce RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)  (24 GB VRAM)
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

The code has been tested on RHEL 9.3. Any standard Linux distribution should work.

We use Python 3.12.4 for all analyses; the other required Python dependencies are described in `requirements.txt`. Do not attempt to install them directly; follow the installation guide below.

## Installation (~5 min)

1. Clone this repository.

```
git clone https://github.com/BonnerLab/scale-free-visual-cortex.git
```

2. Edit `.env`
    - `PROJECT_HOME` should be the path of the cloned repository (e.g. `/home/$USER/scale-free-visual-cortex`)
    - `AWS_SHARED_CREDENTIALS_FILE` should be the path to an [AWS credentials file](https://docs.aws.amazon.com/sdkref/latest/guide/file-location.html) that gives you [access to the Natural Scenes Dataset](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform)
    - The other environment variables can be left unset: simply delete those lines. Cache directories will be created at `~/.cache/bonner-*` by default.

3. Set up the Python environment
   - Option 1: Use [`conda`](https://docs.conda.io/)
        - Install the environment (`conda env create -f $PROJECT_HOME/environment.yml`)
        - Activate the environment (`conda activate scale-free-visual-cortex`)
   - Option 2: Use your favorite package manager
        - Install Python 3.12.4
        - Create a virtual environment (`python -m venv <path-to-venv>`)
        - Activate your virtual environment (e.g. `<path-to-venv>/bin/activate` if you're using `bash`)
        - Install the required packages (`pip install -r requirements.txt`)

## Demo (~5 min)

We provide a simple high-level overview of the analysis in `demo.ipynb` using a small subset of the data. After installing this package, simply run the notebook file: this will automatically download ~300 MB of data and run the within- and between-subject analyses for one pair of subjects.

## Reproducing the analyses

> [!IMPORTANT]
> You will need [access to the Natural Scenes Dataset](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform) to reproduce the analyses in the paper. Specifically, you will need to obtain an AWS credentials file and set the `AWS_SHARED_CREDENTIALS_FILE` environment variable (see Step 2 of Installation).

`manuscript/notebooks` contains Jupyter notebooks that generate the figures in the paper.

- `schematic.ipynb` shows the comparison between different spectral estimators (Figure 1)
- `spectra.ipynb` computes all the power-law covariance spectra (Figures 2, 4, S1, S3, and S4)
- `singular_vectors.ipynb` generates brain maps of some example singular vectors (Figure S2)
- `cross_correlations.ipynb` compares functional and anatomical alignment (Figures 3, S5, and S6)
- `rsa.ipynb` demonstrates the insensitivity of RSA to high-dimensional structure (Figure S7)

> [!WARNING]
> Running a notebook for the first time will likely take ages since the datasets used will be downloaded, processed and cached. Subsequent runs will be much faster.
