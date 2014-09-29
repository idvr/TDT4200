report: chrischa_ps4.tex
	pdflatex chrischa_ps4.tex
	#bibtex chrischa_ps4.aux
	pdflatex chrischa_ps4.tex
	pdflatex chrischa_ps4.tex #Two runs for references

all: clean report

clean:
	rm -f *.aux *.log *.pdf *.loc *.lot *.toc *.lof *.bbl *.blg
