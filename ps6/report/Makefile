report: chrischa_ps6.tex
	pdflatex chrischa_ps6.tex
	#bibtex chrischa_ps6.aux
	pdflatex chrischa_ps6.tex
	pdflatex chrischa_ps6.tex #Two runs for references

all: clean report

clean:
	rm -f *.aux *.log *.pdf *.loc *.lot *.toc *.lof *.bbl *.blg
