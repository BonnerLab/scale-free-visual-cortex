all: manuscript.tex manuscript.pdf

manuscript.tex:
	pandoc -d defaults.yml manuscript.qmd -o manuscript.tex

manuscript.pdf: manuscript.tex
	latexmk -pdf manuscript.tex
	latexmk -c
