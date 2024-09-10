quarto render
mv arxiv/_tex/* arxiv
rm -r arxiv/_tex
mv preprint.bbl arxiv
mv arxiv/preprint.pdf .
tar -czvf arxiv.tar.gz arxiv
rm preprint.aux
rm preprint.bcf
rm preprint.blg
rm preprint.log
rm preprint.out
rm preprint.run.xml
rm -r .quarto
rm arxiv.sty
rm orcidlink.sty
