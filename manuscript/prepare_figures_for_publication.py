#!/usr/bin/env python

import subprocess
from pathlib import Path

# TODO ensure that inkscape and imagemagick are installed with flake.nix

FIGURES = [
    "schematic",
    "general",
    "significance-test",
    "cross-correlations",
    "visual-regions",
    "between-region-heatmaps",
    "rsa",
]

SUPPLEMENTARY_FIGURES = [
    "vary-n-voxels",
    "vary-n-stimuli",
    "cross-detectability",
    "gabor-model",
    "between-region-spectra",
    "general-all",
    "cross-correlations-all",
    "singular-vectors",
    "dimensions-V1-synthetic",
    "dimensions-general-synthetic",
    "things",
    "between-monkey",
    "dimensions-general-semantic",
    "things-dimensions",
]

if __name__ == "__main__":
    FIGURES_HOME = Path("figures")
    (FIGURES_HOME / "publication-ready").mkdir(exist_ok=True, parents=True)

    mapping: dict[Path, Path] = {}
    for i_figure, label in enumerate(FIGURES):
        mapping[FIGURES_HOME / f"{label}.pdf"] = (
            FIGURES_HOME / "publication-ready" / f"Fig{1 + i_figure}.png"
        )
    for i_figure, label in enumerate(SUPPLEMENTARY_FIGURES):
        mapping[FIGURES_HOME / f"{label}.pdf"] = (
            FIGURES_HOME / "publication-ready" / f"S{1 + i_figure}_fig.png"
        )

    for source, target in mapping.items():
        _ = subprocess.run(
            [
                "inkscape",
                str(source),
                "--export-dpi=600",
                f"--export-filename={target}",
            ],
            check=True,
        )
        _ = subprocess.run(
            [
                "magick",
                str(target),
                "-compress",
                "lzw",
                f"{target.with_suffix('').with_suffix('.tif')}",
            ],
            check=True,
        )
        target.unlink()
