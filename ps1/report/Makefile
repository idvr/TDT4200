chrischa_ps1.pdf: chrischa_ps1.tex
	pdflatex chrischa_ps1.tex
	#bibtex chrischa_ps1.aux
	pdflatex chrischa_ps1.tex
	pdflatex chrischa_ps1.tex #Two runs for references

all: clean chrischa_ps1.pdf

clean:
	rm -f *.aux *.log *.pdf *.loc *.lot *.toc *.lof *.bbl *.blg