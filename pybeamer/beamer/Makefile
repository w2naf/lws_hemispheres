TEX = pdflatex
BIB = bibtex

PAPER = pres_latex

all: $(PAPER).tex
#	$(TEX) -draftmode $(PAPER).tex
##	$(BIB) $(PAPER).aux
#	$(TEX) -draftmode $(PAPER).tex
	$(TEX) $(PAPER).tex

short: $(PAPER).tex
	$(TEX) $(PAPER).tex

view: $(PAPER).pdf
	open $(PAPER).pdf

spell::
	ispell -t *.tex

clean::
	rm -fv *.aux *.log *.bbl *.blg *.toc *.out *.lot *.lof *.nav *.snm *.pdf
