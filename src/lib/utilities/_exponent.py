def mathtext_exponent_label(exponent: int) -> str:
    return f"$\\mathdefault{{10^{{{exponent}}}}}$"


def arbitary_float(x: float) -> str:
    significand, exponent = f"{x:.0E}".split("E")
    if float(significand) == 1:
        label = mathtext_exponent_label(int(exponent))
    else:
        label = f"$\\mathdefault{{{significand} \\times 10^{{{int(exponent)}}}}}$"
    raise NotImplementedError
    return label
