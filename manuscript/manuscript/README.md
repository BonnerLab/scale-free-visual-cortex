This is a variant of the preprint submitted to arxiv so that it satisfies PLOS Computational Biology's stupid formatting requirements.

In particular, (i) supplementary figures are to be referred to as "Sk Figure" in the text and captions, and (ii) the supplementary figures---but not the captions---should be excluded from the PDF (this is absolutely ridiculous---why can't they simply do what `arxiv` does? Maybe I should create an open-source scientific publishing system that doesn't make me want to kill myself smh).

Anyway, I've achieved (i) by

- adding `\renewcommand{\thefigure}{S\arabic{figure} Figure}` before the supplementary figures, which adds the "S" prefix and " Figure" suffix
- prefixing my `@fig` references to supplementary figures with `-` (i.e. `-@fig`), which suppresses the prefix "Figure" in the main text
- adding this to the preamble
    ```latex
    \usepackage{caption}
    \DeclareCaptionLabelFormat{supplementary-figure}{#2}
    ```
- adding `\captionsetup[figure]{labelformat=supplementary-figure}` just before the supplementary figures

and (ii) by simply adding `width=0%` to my figure metadata in my Markdown source.

I've done this manually but I should really automate it so I have a single source of truth. But it's such a pain. :( I hate scientific publishing.
