from bonner.plotting import DEFAULT_FIGURE_OPTIONS, DEFAULT_FONTS, DEFAULT_SIZES

JOURNAL_FONTS = DEFAULT_FONTS | {
    "font.serif": ["NewComputerModernMath"],
    "font.sans-serif": ["Liberation Sans"],
    "mathtext.fontset": "custom",
    "mathtext.sf": "Liberation Sans",
    "mathtext.it": "NewComputerModernMath",
    "mathtext.rm": "NewComputerModernMath",
    "mathtext.tt": "NewComputerModernMath",
    "mathtext.bf": "NewComputerModernMath",
    "mathtext.bfit": "NewComputerModernMath",
    "mathtext.cal": "NewComputerModernMath",
    "mathtext.fallback": "cm",
}

JOURNAL_MATPLOTLIBRC = JOURNAL_FONTS | DEFAULT_FIGURE_OPTIONS | DEFAULT_SIZES
