__all__ = (
    "BONNER_BRAINIO_HOME",
    "BONNER_CACHING_HOME",
    "BONNER_DATASETS_HOME",
    "BONNER_MODELS_HOME",
    "JOURNAL_MATPLOTLIBRC",
    "MICRO",
    "PROJECT_HOME",
    "TICK_LABELS",
    "apply_gabor_filter_bank",
    "assign_bins",
    "bin_data",
    "compute_best_fit",
    "create_gabor_filter_bank",
    "extract_geometrically_spaced_bins",
    "extract_uniformly_spaced_bins",
    "mathtext_exponent_label",
)

import warnings

from lib.utilities._binning import (
    assign_bins,
    bin_data,
    extract_geometrically_spaced_bins,
    extract_uniformly_spaced_bins,
)
from lib.utilities._environment import (
    BONNER_BRAINIO_HOME,
    BONNER_CACHING_HOME,
    BONNER_DATASETS_HOME,
    BONNER_MODELS_HOME,
    PROJECT_HOME,
)
from lib.utilities._exponent import mathtext_exponent_label
from lib.utilities._gabor import apply_gabor_filter_bank, create_gabor_filter_bank
from lib.utilities._plotting import JOURNAL_MATPLOTLIBRC
from lib.utilities._power_laws import compute_best_fit

PLANES = {
    "yz": {"elev": 0, "azim": 0, "roll": 0},
    "xy": {"elev": 90, "azim": 0, "roll": 90},
    "xz": {"elev": 0, "azim": 90, "roll": 0},
}

MICRO = "μ"

TICK_LABELS = {
    1e-6: f"1 {MICRO}m",
    1e-5: f"10 {MICRO}m",
    1e-4: f"100 {MICRO}m",
    1e-3: "1 mm",
    1e-2: "10 mm",
    1e-1: "100 mm",
    1e0: "1 m",
}

warnings.filterwarnings("ignore", module="lib.utilities._binning")
