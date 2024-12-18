include Makefile.venv

.PHONY: all
all: setup test

.PHONY: setup
setup: venv
	$(VENV)/pre-commit install

