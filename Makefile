include Makefile.venv

.PHONY: all
all: setup test

.PHONY: setup
setup: venv
	$(VENV)/pre-commit install
	go mod download

.PHONY: test
test:
	dagger run go test -v