report: chrischa_ps5.tex
	pdflatex chrischa_ps5.tex
	#bibtex chrischa_ps5.aux
	pdflatex chrischa_ps5.tex
	pdflatex chrischa_ps5.tex #Two runs for references

all: clean report

clean:
	rm -f *.aux *.log *.pdf *.loc *.lot *.toc *.lof *.bbl *.blg
