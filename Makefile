BINARY  := llm-fit
BIN_DIR := bin
PKG     := ./cmd/llm-fit

.PHONY: all build test tidy lint clean demo

all: build

## build: compile the binary into ./bin
build:
	go build -o $(BIN_DIR)/$(BINARY) $(PKG)

## test: run tests
test:
	go test ./...

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
