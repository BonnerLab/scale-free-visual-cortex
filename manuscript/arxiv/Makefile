all: preprint.tex preprint.pdf diff.pdf arxiv-submission.tar.gz

preprint.tex:
	pandoc -d defaults.yml preprint.qmd -o preprint.tex
	latexdiff submitted_1.tex preprint.tex > diff.tex

preprint.pdf: preprint.tex
	latexmk -pdf preprint.tex
	latexmk -c

diff.pdf: preprint.tex
	latexmk -pdf diff.tex
	latexmk -c

arxiv-submission.tar.gz: preprint.tex
	python create_arxiv_submission.py
