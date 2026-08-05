BINARY  := llm-fit
BIN_DIR := bin
PKG     := ./cmd/llm-fit

.PHONY: all build test tidy lint clean demo help

.DEFAULT_GOAL := help

## help: show this help
help:
	@awk '/^## [a-zA-Z0-9_-]+:/ { l=$$0; sub(/^## /,"",l); i=index(l,":"); \
	         printf "  %-14s %s\n", substr(l,1,i-1), substr(l,i+2); next } \
	     /^[a-zA-Z0-9_-]+:.*## / { i=index($$0,":"); j=index($$0,"## "); \
	         printf "  %-14s %s\n", substr($$0,1,i-1), substr($$0,j+3) }' $(MAKEFILE_LIST)

all: build

## build: compile the binary into ./bin
build:
	go build -o $(BIN_DIR)/$(BINARY) $(PKG)

## test: run tests
test:
	go test -race -count=1 ./...

## lint: vet and formatting check
lint:
	go vet ./...
	@test -z "$$(gofmt -l . )" || { echo "gofmt needed:"; gofmt -l .; exit 1; }

## tidy: tidy modules
tidy:
	go mod tidy

## demo: what this machine can run
demo: build
	./$(BIN_DIR)/$(BINARY) detect
	@echo
	./$(BIN_DIR)/$(BINARY) suggest

## clean: remove build artifacts
clean:
	rm -rf $(BIN_DIR)
