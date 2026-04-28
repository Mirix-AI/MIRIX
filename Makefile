.PHONY: install format lint test all check proto

# Define variables
PYTHON = python3
POETRY = poetry
PYTEST = $(POETRY) run pytest
RUFF = $(POETRY) run ruff
PYRIGHT = $(POETRY) run pyright

# Default target
all: format lint test

# Install dependencies
install:
	$(POETRY) install

# Format code
format:
	$(RUFF) check --select I --fix
	$(RUFF) format

# Lint code
lint:
	$(RUFF) check --fix
	$(PYRIGHT) .

# Run tests
test:
	$(PYTEST)

# Run format, lint, and test
check: format lint test

# Regenerate protobuf gencode for mirix/queue/*.proto.
# Uses grpcio-tools pinned in pyproject.toml (>=1.66.0,<1.67.0) so the
# checked-in *_pb2.py / *_pb2.pyi / *_pb2_grpc.py files are reproducible.
proto:
	$(POETRY) run python -m grpc_tools.protoc \
		-I. \
		--python_out=. \
		--pyi_out=. \
		--grpc_python_out=. \
		mirix/queue/message.proto
