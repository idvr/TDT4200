chrischa_ps.pdf: chrischa_ps2.tex
	pdflatex chrischa_ps2.tex
	#bibtex chrischa_ps2.aux
	pdflatex chrischa_ps2.tex
	pdflatex chrischa_ps2.tex #Two runs for references

all: clean chrischa_ps2.pdf

clean:
	rm -f *.aux *.log *.pdf *.loc *.lot *.toc *.lof *.bbl *.blg
