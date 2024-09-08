all: response.tex response.pdf

response.tex:
	pandoc -d defaults.yml response.qmd -o response.tex

response.pdf: response.tex
	latexmk -pdf response.tex
	latexmk -c
