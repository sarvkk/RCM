# Makefile for SaycuredAI Documentation
# =====================================

.PHONY: all docs clean help

# Default target
all: docs

# Build technical paper PDF
docs: docs/technical_paper.pdf

docs/technical_paper.pdf: docs/technical_paper.tex
	@echo "Building technical paper..."
	cd docs && pdflatex -interaction=nonstopmode technical_paper.tex
	cd docs && pdflatex -interaction=nonstopmode technical_paper.tex
	@echo "Done: docs/technical_paper.pdf"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f docs/*.aux docs/*.log docs/*.out docs/*.toc docs/*.bbl docs/*.blg
	@echo "Done."

# Clean everything including PDF
distclean: clean
	rm -f docs/*.pdf

# Help
help:
	@echo "SaycuredAI Documentation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make docs      - Build technical paper PDF"
	@echo "  make clean     - Remove auxiliary files"
	@echo "  make distclean - Remove all generated files"
	@echo "  make help      - Show this help"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex (TeX Live or similar)"
	@echo "  - LaTeX packages: tikz, pgfplots, tcolorbox, etc."
