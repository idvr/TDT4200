chrischa_ps.pdf: chrischa_ps3.tex
	pdflatex chrischa_ps3.tex
	#bibtex chrischa_ps3.aux
	pdflatex chrischa_ps3.tex
	pdflatex chrischa_ps3.tex #Two runs for references

all: clean chrischa_ps3.pdf

clean:
	rm -f *.aux *.log *.pdf *.loc *.lot *.toc *.lof *.bbl *.blg
